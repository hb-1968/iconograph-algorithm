"""Stage 3 L2 — structural compound classification of L1 misses.

Pipeline position:
    align (L0) → segment_prim_classify (L1) → L2_compound_classify (THIS, L2) → L3 (TBD)

L1 hands L2 the connected components no single primitive could fit. L2 decides,
per structure, whether it is built from LOCAL features (an independent shape) or
GLOBAL features (a subdivided ring/disc — Chrome, Spotify), then rebuilds it from
{Ellipse, Rectangle, Triangle} primitives combined by UNION and SUBTRACTION.

Router:
  • global-ring — ≥2 misses sharing a centre whose closed combined mask is annular
    (wide angular coverage + a central hole) are decomposed once on that combined
    mask → one annulus (outer circle − inner circle). The central hub (already a
    Classified circle from L1) stays a pass-through. cat.routing=="burst" /
    color_essential relaxes the coverage gate; geometry is the primary gate.
  • local      — each remaining miss is decomposed on its own mask.

Negation is INTELLIGENT: subtraction fires only where findContours(RETR_CCOMP)
finds a real interior hole (the "A" counter, an "O"/ring, Instagram's nested
squares). Engine, per (combined or local) mask:
    holes  = interior holes (RETR_CCOMP children, area ≥ HOLE_AREA_FRAC·filled)
    filled = mask with holes filled
    adds   = greedy residual cover of `filled` (recall-biased gain)
    subs   = best inscribed primitive fit to each hole (skipped if it matches poorly)
    compound = OR(adds) AND NOT OR(subs);  iou = IoU(compound, mask)

Vocabulary {Ellipse, Rectangle, Triangle}: ellipse subsumes the circle. A compound
that still misses ACCEPT_IOU within the part budget → L2Defer (forwarded to L3 for
the structurally-irreducible remainder). Determinism (H8): analytic fits only
(distance transform, histogram max-rect, min-enclosing, moments), fixed candidate
order, no RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from iconograph.application.compress.common import (
    Ellipse,
    Primitive,
    Rectangle,
    Triangle,
    _fit_ellipse,
    _fit_rectangle,
    _fit_triangle,
    iou_analytical,
)
from iconograph.application.compress.segment_prim_classify import (
    Classified,
    L1Miss,
    SegmentClassification,
)

if TYPE_CHECKING:
    from iconograph.application.categorize import Categorization


# ── local-decomposition knobs ──────────────────────────────────────────
ACCEPT_IOU = 0.90
MAX_PARTS = 6
SPILL_LAMBDA = 1.0
MIN_GAIN_FRAC = 0.02
MIN_FEATURE_DIVISOR = 48
HOLE_AREA_FRAC = 0.02       # a hole must be ≥ this fraction of the filled area to carve
HOLE_FIT_MIN_IOU = 0.50     # only subtract a hole a primitive actually matches

# ── global-ring router knobs ───────────────────────────────────────────
RING_MIN_MISSES = 2
RING_CENTER_TOL = 0.35      # a member centroid sits within this × S of the global centroid
RING_MIN_COVERAGE = 0.66    # angular coverage of the closed combined mask around the centre
RING_ANGULAR_BINS = 36
RING_BURST_RELAX = 0.15     # coverage gate relaxed by this when cat says burst
RING_CLOSE_FRAC = 0.04      # morphological-close kernel (× S) to bridge inter-sector gaps
RING_HOLE_CENTER_TOL = 0.25  # the annulus's central hole sits within this × S of the centre

Op = Literal["add", "subtract"]
L2Part = Ellipse | Rectangle | Triangle


@dataclass(frozen=True)
class Part:
    primitive: L2Part
    op: Op                               # "add" | "subtract"


@dataclass(frozen=True)
class L2Compound:
    parts: tuple[Part, ...]              # full-image coords; adds OR'd then subs carved
    bbox: tuple[int, int, int, int]      # (x, y, w, h) of the source miss / combined group
    color_rgb: tuple[int, int, int]
    iou: float                           # reconstruction IoU vs the (combined) mask
    structure: Literal["local", "global_ring"]


@dataclass(frozen=True)
class L2Defer:
    bbox: tuple[int, int, int, int]
    color_rgb: tuple[int, int, int]
    area_px: int
    mask: np.ndarray                     # bbox-cropped mask, forwarded to L3
    reason: str                          # "no_gain" | "accuracy_floor" | "part_budget"
    iou: float                           # best attempted reconstruction IoU (how close it got)
    n_holes: int                         # interior holes detected in the mask


@dataclass
class CompoundClassification:
    image_shape: tuple[int, int, int]
    segments: list[Classified | L2Compound | L2Defer]


# --------------------------------------------------------------------------- #
# Largest inscribed axis-aligned rectangle.
#
# H2 note: the recommended library (largestinteriorrectangle / lir) is built on
# numba, which has no wheel for this interpreter (Python 3.14) — importing it
# fails. OpenCV and scikit-image have no inscribed-rectangle primitive either.
# The fallback the plan anticipated is the histogram maximal-rectangle DP; we
# run it on a downscaled proxy so the per-row monotonic-stack scan stays bounded
# (~PROXY^2) and fast, then scale the rectangle back. Deterministic, no RNG.
# --------------------------------------------------------------------------- #
_INSCRIBED_PROXY = 160


def _max_rect_histogram(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest all-True axis-aligned rectangle in a bool mask, via the classic
    per-row histogram + monotonic-stack DP. Returns (x, y, w, h) or None."""
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int64)
    best_area = 0
    best = (0, 0, 0, 0)
    for r in range(h):
        heights = np.where(mask[r], heights + 1, 0)
        stack: list[tuple[int, int]] = []  # (start_col, height), strictly increasing
        for i in range(w + 1):
            cur = int(heights[i]) if i < w else 0
            start = i
            while stack and stack[-1][1] > cur:
                idx, ht = stack.pop()
                area = ht * (i - idx)
                if area > best_area:
                    best_area = area
                    best = (idx, r - ht + 1, i - idx, ht)
                start = idx
            stack.append((start, cur))
    return best if best_area > 0 else None


def _inscribed_rect_xywh(region: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest inscribed axis-aligned rectangle of a binary region (uint8
    {0,255}), as (x, y, w, h) in region coords, or None if empty."""
    h, w = region.shape[:2]
    if not np.any(region):
        return None
    scale = min(1.0, _INSCRIBED_PROXY / max(h, w))
    if scale < 1.0:
        sw = max(1, int(round(w * scale)))
        sh = max(1, int(round(h * scale)))
        small = cv2.resize(region, (sw, sh), interpolation=cv2.INTER_NEAREST)
        rect = _max_rect_histogram(small > 0)
        if rect is None:
            return None
        inv = 1.0 / scale
        inset = max(1, int(round(inv)))  # guard the upscale boundary against spill
        x = int(round(rect[0] * inv)) + inset
        y = int(round(rect[1] * inv)) + inset
        rw = int(round(rect[2] * inv)) - 2 * inset
        rh = int(round(rect[3] * inv)) - 2 * inset
    else:
        rect = _max_rect_histogram(region > 0)
        if rect is None:
            return None
        x, y, rw, rh = rect
    if rw < 1 or rh < 1:
        return None
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


# --------------------------------------------------------------------------- #
# mask / geometry helpers
# --------------------------------------------------------------------------- #
def _min_feature_px(s: int) -> int:
    """Smallest primitive extent worth emitting at canonical resolution s."""
    return max(2, math.ceil(s / MIN_FEATURE_DIVISOR))


def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """(cx, cy) of a uint8 mask via image moments, or None if empty."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0.0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def _angular_coverage(mask: np.ndarray, cx: float, cy: float) -> float:
    """Fraction of angular bins around (cx, cy) that the mask occupies."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0
    ang = np.arctan2(ys - cy, xs - cx)
    counts, _ = np.histogram(ang, bins=RING_ANGULAR_BINS, range=(-np.pi, np.pi))
    return float(np.count_nonzero(counts)) / RING_ANGULAR_BINS


def _largest_lobe(mask: np.ndarray) -> np.ndarray | None:
    """The largest 8-connected foreground component, as its own {0,255} mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S,
    )
    if num <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8)) * 255


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if c.shape[0] < 3:
        return None
    return c


def _extent_ok(fit: np.ndarray, min_feat: int) -> bool:
    x, y, w, h = cv2.boundingRect(fit)
    return min(w, h) >= min_feat


# --------------------------------------------------------------------------- #
# hole detection / filling (negation)
# --------------------------------------------------------------------------- #
def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Solid outer shape: redraw external contours filled (holes vanish)."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def _detect_holes(mask: np.ndarray) -> list[np.ndarray]:
    """Interior holes as filled {0,255} masks. RETR_CCOMP gives a 2-level tree;
    a hole is any contour with a parent (hierarchy[i][3] != -1). Holes below
    HOLE_AREA_FRAC of the filled area are ignored (sub-feature noise)."""
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    filled_area = float(np.count_nonzero(_fill_holes(mask)))
    if filled_area <= 0:
        return []
    holes: list[np.ndarray] = []
    for i, c in enumerate(contours):
        if hier[i][3] == -1:                       # no parent → outer boundary
            continue
        if cv2.contourArea(c) < HOLE_AREA_FRAC * filled_area:
            continue
        hm = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(hm, [c], -1, 255, thickness=cv2.FILLED)
        holes.append(hm)
    return holes


# --------------------------------------------------------------------------- #
# primitive candidates (crop-local coordinates: ox = oy = 0)
# --------------------------------------------------------------------------- #
def _cand_inscribed_circle(
    region: np.ndarray, h: int, w: int
) -> tuple[Ellipse, np.ndarray, float] | None:
    """Largest circle inside region (distance-transform peak), as an axis-aligned
    Ellipse with rx == ry."""
    dist = cv2.distanceTransform(region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, r, _, center = cv2.minMaxLoc(dist)
    if r < 1.0:
        return None
    cx, cy = float(center[0]), float(center[1])
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fit, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return Ellipse(cx=cx, cy=cy, rx=float(r), ry=float(r)), fit, float(math.pi * r * r)


def _cand_inscribed_rectangle(
    region: np.ndarray, h: int, w: int
) -> tuple[Rectangle, np.ndarray, float] | None:
    xywh = _inscribed_rect_xywh(region)
    if xywh is None:
        return None
    x, y, rw, rh = xywh
    if rw < 1 or rh < 1:
        return None
    fit = np.zeros((h, w), dtype=np.uint8)
    fit[y:y + rh, x:x + rw] = 255
    return Rectangle(x=int(x), y=int(y), w=int(rw), h=int(rh)), fit, float(rw * rh)


def _all_candidates(
    region: np.ndarray, contour: np.ndarray, h: int, w: int, min_feat: int
) -> list[tuple[Primitive, np.ndarray, float]]:
    """Inside-fitting candidate set for greedy. Fixed order encodes the tie-break
    preference (ellipse-family before rectangle before triangle). Enclosing
    fitters are excluded — union cannot subtract spill, so a primitive that
    overshoots the target permanently caps the IoU; min-enclosing triangle stays
    in because on a genuinely sharp lobe it barely overshoots and SPILL_LAMBDA
    rejects it on round lobes."""
    raw: list[tuple[Primitive, np.ndarray, float] | None] = [
        _cand_inscribed_circle(region, h, w),
        _fit_ellipse(contour, h, w, 0, 0),
        _cand_inscribed_rectangle(region, h, w),
        _fit_triangle(contour, h, w, 0, 0),
    ]
    return [c for c in raw if c is not None and _extent_ok(c[1], min_feat)]


def _best_step(
    target_b: np.ndarray,
    union_b: np.ndarray,
    cands: list[tuple[Primitive, np.ndarray, float]],
    tarea: int,
) -> tuple[Primitive, np.ndarray] | None:
    """Pick the candidate with the best recall-biased marginal gain, or None when
    nothing clears the minimum-gain floor."""
    best: tuple[Primitive, np.ndarray] | None = None
    best_gain = MIN_GAIN_FRAC * tarea
    for prim, fit, _area in cands:
        fb = fit.astype(bool)
        new_cover = int(np.count_nonzero(fb & target_b & ~union_b))
        spill = int(np.count_nonzero(fb & ~target_b))
        gain = new_cover - SPILL_LAMBDA * spill
        if gain > best_gain:
            best_gain = gain
            best = (prim, fit)
    return best


def _fit_hole(hole_mask: np.ndarray, h: int, w: int, min_feat: int) -> Primitive | None:
    """Best single primitive matching a hole, to subtract it. None when the hole
    is sub-feature-size or no primitive matches it well enough to carve."""
    if not _extent_ok(hole_mask, min_feat):
        return None
    contour = _largest_contour(hole_mask)
    if contour is None:
        return None
    cands: list[tuple[Primitive, np.ndarray, float] | None] = [
        _cand_inscribed_circle(hole_mask, h, w),
        _cand_inscribed_rectangle(hole_mask, h, w),
        _fit_ellipse(contour, h, w, 0, 0),
        _fit_rectangle(contour, h, w, 0, 0),
        _fit_triangle(contour, h, w, 0, 0),
    ]
    best: Primitive | None = None
    best_iou = HOLE_FIT_MIN_IOU
    for cand in cands:
        if cand is None:
            continue
        prim, fit, area = cand
        score = iou_analytical(hole_mask, fit, area)
        if score > best_iou:
            best_iou = score
            best = prim
    return best


# --------------------------------------------------------------------------- #
# compound rasterization + scoring
# --------------------------------------------------------------------------- #
def _raster_part(p: Primitive, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if isinstance(p, Ellipse):
        cv2.ellipse(
            m, (int(round(p.cx)), int(round(p.cy))),
            (max(1, int(round(p.rx))), max(1, int(round(p.ry)))),
            0, 0, 360, 255, -1,
        )
    elif isinstance(p, Rectangle):
        m[p.y:p.y + p.h, p.x:p.x + p.w] = 255
    elif isinstance(p, Triangle):
        pts = np.array([p.v1, p.v2, p.v3], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 255)
    return m


def _compound_mask(parts: list[Part] | tuple[Part, ...], h: int, w: int) -> np.ndarray:
    """OR every add part, then AND-NOT every subtract part (order-independent)."""
    union = np.zeros((h, w), dtype=np.uint8)
    for part in parts:
        if part.op == "add":
            union |= _raster_part(part.primitive, h, w)
    for part in parts:
        if part.op == "subtract":
            union &= ~_raster_part(part.primitive, h, w)
    return union


def _compound_iou(parts: list[Part], target: np.ndarray, h: int, w: int) -> float:
    return _iou_bool(_compound_mask(parts, h, w).astype(bool), target.astype(bool))


def _offset(p: Primitive, ox: int, oy: int) -> Primitive:
    if isinstance(p, Ellipse):
        return Ellipse(cx=p.cx + ox, cy=p.cy + oy, rx=p.rx, ry=p.ry)
    if isinstance(p, Rectangle):
        return Rectangle(x=p.x + ox, y=p.y + oy, w=p.w, h=p.h)
    return Triangle(
        v1=(p.v1[0] + ox, p.v1[1] + oy),
        v2=(p.v2[0] + ox, p.v2[1] + oy),
        v3=(p.v3[0] + ox, p.v3[1] + oy),
    )


def _offset_part(part: Part, ox: int, oy: int) -> Part:
    return Part(primitive=_offset(part.primitive, ox, oy), op=part.op)


# --------------------------------------------------------------------------- #
# decomposition engine (shared by local + global-ring)
# --------------------------------------------------------------------------- #
def _greedy_union(target: np.ndarray, h: int, w: int, min_feat: int) -> list[Primitive]:
    """Greedy residual cover of a (hole-filled) target — additive only."""
    target_b = target.astype(bool)
    tarea = int(target_b.sum())
    union_b = np.zeros((h, w), dtype=bool)
    parts: list[Primitive] = []
    while len(parts) < MAX_PARTS:
        residual = ((target_b & ~union_b).astype(np.uint8)) * 255
        lobe = _largest_lobe(residual)
        if lobe is None:
            break
        contour = _largest_contour(lobe)
        if contour is None:
            break
        pick = _best_step(target_b, union_b, _all_candidates(lobe, contour, h, w, min_feat), tarea)
        if pick is None:
            break
        prim, fit = pick
        parts.append(prim)
        union_b |= fit.astype(bool)
        if _iou_bool(union_b, target_b) >= ACCEPT_IOU:
            break
    return parts


def _decompose(target: np.ndarray, h: int, w: int, min_feat: int) -> tuple[list[Part], float, int]:
    """Hole-aware decomposition. Fill holes, greedily union the solid outer, then
    subtract a fitted primitive per detected hole. Returns (parts, IoU vs target,
    #holes detected)."""
    holes = _detect_holes(target)
    filled = _fill_holes(target)
    add_prims = _greedy_union(filled, h, w, min_feat)
    parts: list[Part] = [Part(p, "add") for p in add_prims]
    for hole in holes:
        sub = _fit_hole(hole, h, w, min_feat)
        if sub is not None:
            parts.append(Part(sub, "subtract"))
    return parts, _compound_iou(parts, target, h, w), len(holes)


# --------------------------------------------------------------------------- #
# global-ring router
# --------------------------------------------------------------------------- #
def _is_burst(cat: "Categorization | None") -> bool:
    return bool(cat is not None and (getattr(cat, "routing", None) == "burst"
                                     or getattr(cat, "color_essential", False)))


def _miss_centroid_full(miss: L1Miss) -> tuple[float, float] | None:
    c = _centroid(miss.mask)
    if c is None:
        return None
    x, y, _w, _h = miss.bbox
    return c[0] + x, c[1] + y


def _group_global_ring(
    sc: SegmentClassification, misses: list[L1Miss], s: int, cat: "Categorization | None"
) -> tuple[list[int], np.ndarray, tuple[int, int, int, int]] | None:
    """Detect a subdivided-ring group among the misses. Returns
    (member indices, closed combined mask at S, combined bbox) or None.

    Gates: ≥2 misses near the global centroid, wide angular coverage, and (the
    annulus signature) a central hole in the gap-closed combined mask."""
    if len(misses) < RING_MIN_MISSES:
        return None
    gc = _centroid(sc.binarized)
    if gc is None:
        return None
    gcx, gcy = gc
    near = []
    for i, m in enumerate(misses):
        c = _miss_centroid_full(m)
        if c is not None and math.hypot(c[0] - gcx, c[1] - gcy) <= RING_CENTER_TOL * s:
            near.append(i)
    if len(near) < RING_MIN_MISSES:
        return None

    combined = np.zeros((s, s), dtype=np.uint8)
    for i in near:
        x, y, w, h = misses[i].bbox
        combined[y:y + h, x:x + w] |= misses[i].mask
    k = max(3, int(round(s * RING_CLOSE_FRAC)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    coverage_floor = RING_MIN_COVERAGE - (RING_BURST_RELAX if _is_burst(cat) else 0.0)
    if _angular_coverage(closed, gcx, gcy) < coverage_floor:
        return None

    central_hole = False
    for hole in _detect_holes(closed):
        hc = _centroid(hole)
        if hc is not None and math.hypot(hc[0] - gcx, hc[1] - gcy) <= RING_HOLE_CENTER_TOL * s:
            central_hole = True
            break
    if not central_hole:
        return None

    ys, xs = np.where(closed > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return near, closed, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def decompose_miss(mask: np.ndarray, image_size: int) -> tuple[list[Part], float]:
    """Hole-aware decomposition of one mask. mask: uint8 (h, w) {0,255}.
    image_size: S (for the min-feature floor). Returns (parts in CROP-LOCAL
    coords, reconstruction IoU). Used by the testbed and for debugging."""
    h, w = mask.shape[:2]
    parts, iou, _n_holes = _decompose(mask, h, w, _min_feature_px(image_size))
    return parts, iou


def render_compound(
    parts: list[Part] | tuple[Part, ...], h: int, w: int,
    color: tuple[int, int, int] = (40, 40, 40),
    bg: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Flat-fill the union-minus-holes of crop-local parts onto an (h, w, 3) RGB
    canvas."""
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    canvas[_compound_mask(parts, h, w) > 0] = color
    return canvas


def _avg_color(misses: list[L1Miss]) -> tuple[int, int, int]:
    cols = np.array([m.color_rgb for m in misses], dtype=float).mean(axis=0)
    return tuple(int(round(c)) for c in cols)


def _defer(miss: L1Miss, parts: list[Part], iou: float, n_holes: int) -> L2Defer:
    n_add = sum(1 for p in parts if p.op == "add")
    if not parts:
        reason = "no_gain"
    elif n_add >= MAX_PARTS:
        reason = "part_budget"
    else:
        reason = "accuracy_floor"
    return L2Defer(
        bbox=miss.bbox,
        color_rgb=tuple(int(c) for c in miss.color_rgb),
        area_px=miss.area_px,
        mask=miss.mask,
        reason=reason,
        iou=float(iou),
        n_holes=int(n_holes),
    )


def L2_compound_classify(
    sc: SegmentClassification, cat: "Categorization | None" = None
) -> CompoundClassification:
    """L2 — pass Classified segments through; route the L1 misses to a global-ring
    annulus (when a subdivided ring is detected) or per-miss local decomposition;
    each yields an L2Compound (IoU ≥ ACCEPT_IOU) or an L2Defer (→ L3)."""
    s = int(sc.image_shape[0])
    min_feat = _min_feature_px(s)
    misses = [seg for seg in sc.segments if isinstance(seg, L1Miss)]

    # Global-ring: decompose the combined mask once; consume its members on success.
    ring_compound: L2Compound | None = None
    consumed_ids: set[int] = set()
    grp = _group_global_ring(sc, misses, s, cat)
    if grp is not None:
        idxs, closed, (cx, cy, cw, ch) = grp
        crop = closed[cy:cy + ch, cx:cx + cw]
        parts_local, iou, _n_holes = _decompose(crop, ch, cw, min_feat)
        if parts_local and iou >= ACCEPT_IOU:
            members = [misses[i] for i in idxs]
            ring_compound = L2Compound(
                parts=tuple(_offset_part(p, cx, cy) for p in parts_local),
                bbox=(cx, cy, cw, ch),
                color_rgb=_avg_color(members),
                iou=float(iou),
                structure="global_ring",
            )
            consumed_ids = {id(m) for m in members}

    out: list[Classified | L2Compound | L2Defer] = []
    ring_emitted = False
    for seg in sc.segments:
        if isinstance(seg, Classified):
            out.append(seg)
            continue
        if id(seg) in consumed_ids:
            if ring_compound is not None and not ring_emitted:
                out.append(ring_compound)
                ring_emitted = True
            continue
        # local miss
        x, y, _w, _h = seg.bbox
        mh, mw = seg.mask.shape[:2]
        parts_local, iou, n_holes = _decompose(seg.mask, mh, mw, min_feat)
        if parts_local and iou >= ACCEPT_IOU:
            out.append(L2Compound(
                parts=tuple(_offset_part(p, x, y) for p in parts_local),
                bbox=seg.bbox,
                color_rgb=tuple(int(c) for c in seg.color_rgb),
                iou=float(iou),
                structure="local",
            ))
        else:
            out.append(_defer(seg, parts_local, iou, n_holes))

    return CompoundClassification(image_shape=sc.image_shape, segments=out)
