"""Stage 3 L2 — feature segmentation (split, don't fit).

Pipeline position:
    align (L0) → segment_prim_classify (L1) → L2_feature_segment (THIS) → fitting (deferred)

L1 hands L2 the connected components no single primitive could fit (`L1Miss`).
L2 does NOT fit them — it SPLITS each into minimal, near-fittable sub-features and
tags each with a `shape_hint` for the deferred fitting layer. Per miss:

  1. global read (no categorize internals, H6) — a structure single-primitive
     carving shouldn't shatter, kept WHOLE and tagged `global`:
       • ring   — surrounds an empty centre (centroid over background + the
                  foreground wraps it), open band or closed annulus (Edge, Chrome);
       • holed  — an interior counterform/hole (the "A", Instagram) → needs negation;
       • sunburst — many radial spikes (Claude, Firefox).
  2. local — a body with thin appendages → **body + appendages** split:
     morphological opening (adaptive kernel from the body inradius) removes the
     thin protrusions; the body is kept EXACT (mask − appendage components); each
     part is classified: a flat-chord-plus-arc body → `half_circle`, an otherwise
     single-primitive part → `primitive`, anything still complex → `defer`.
A narrow cross-component pre-pass groups shared-centre misses that tile a ring
(Chrome's sectors) into one `global` ring — the only grouping L2 does.

Library scrutiny: ring/half-disc/appendage signals all reuse primitives already
here (distance transform, concavity, contour, min-enclosing circle via `common`);
`MORPH_OPEN` only seeds the appendages (body kept exact). Deterministic (H8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from iconograph.application.compress.common import (
    RingGeom,
    _centroid,
    _detect_holes,
    _fit_circle,
    _fit_ellipse,
    _fit_rectangle,
    _fit_rotated_ellipse,
    _fit_rotated_rect,
    _fit_triangle,
    _largest_contour,
    iou_analytical,
)
from iconograph.application.compress.segment_prim_classify import (
    Classified,
    L1Miss,
    SegmentClassification,
)

if TYPE_CHECKING:
    from iconograph.application.categorize import Categorization


# ── classification knobs ───────────────────────────────────────────────
SEG_FIT_IOU = 0.90               # a part is a single-primitive when best fit IoU ≥ this
# global: sunburst
SUNBURST_MIN_VTX = 20
SUNBURST_POLY_TOL = 0.01
# global: ring / open band (surrounds an empty centre)
RING_WRAP_COVERAGE = 0.80        # angular coverage of foreground around the centroid
RING_CENTER_RADIUS_FRAC = 0.06   # centroid-patch radius as a fraction of the mask extent
RING_CENTER_FG_MAX = 0.50        # centroid patch must be < this foreground (empty centre)
RING_ANGULAR_BINS = 36
RING_MAX_BAND = 0.42             # max inscribed-r / outer-r — a true ring is a THIN band (Edge 0.387)
RING_MAX_SOLIDITY = 0.95         # reject a solid blob whose slight centre-hollow tripped the gate
# half-disc
HALFDISC_RATIO_LO = 0.40         # area / (π · minEnclosingCircle_r²) ≈ 0.5 for a half-disc
HALFDISC_RATIO_HI = 0.55         # tightened — rejects filled roundish shapes (obsidian 0.62)
HALFDISC_CONCAVITY_MAX = 0.08    # tightened — rejects angular fills (telegram/figma ≈0.12); dome 0.005
HALFDISC_MAX_ASPECT = 2.8        # round (dome ≈2.2); rejects thin capsules/antennae (≈5+)
# body / appendage split
OPEN_K_INRADIUS_FRAC = 0.45      # opening kernel ≈ this × body inradius (≈ 0.085 · S)
APPENDAGE_AREA_FRAC = 0.02       # an appendage component ≥ this fraction of the mask area
# cross-miss ring pre-pass
RING_MIN_MISSES = 2
RING_CENTER_TOL = 0.35
RING_MIN_COVERAGE = 0.66
RING_CLOSE_FRAC = 0.04
RING_HOLE_CENTER_TOL = 0.25
# polar-unroll ring resolver
R_BINS = 256                     # radial resolution (polar columns: radius 0..maxRadius)
THETA_BINS = 720                 # angular resolution (polar rows: θ 0..2π, ≈ 0.5°/row)
RING_CENTRE_MARGIN = 2.0         # maxRadius = outer_r + this, so the outer edge isn't clipped
POLAR_MIN_AREA_FRAC = 0.003      # drop polar components below this fraction of the polar image
ANNULAR_BAND_COVERAGE = 0.90     # angular row-span ≥ this → annular_band, else annular_sector

Kind = Literal["local", "global", "defer"]
ShapeHint = Literal[
    "half_circle", "ring", "holed", "sunburst",
    "circle", "ellipse", "rectangle", "triangle", "rotated_rect", "rotated_ellipse",
    "annular_band", "annular_sector", "gapped_annulus",
] | None


@dataclass(frozen=True)
class Feature:
    mask: np.ndarray                     # (h, w) uint8 {0,255}, cropped to bbox
    bbox: tuple[int, int, int, int]      # (x, y, w, h) full-image coords
    kind: Kind                           # "local" | "global" | "defer"
    color_rgb: tuple[int, int, int]
    best_fit_iou: float                  # best single-primitive IoU (informational)
    shape_hint: ShapeHint                # how the deferred fitter should fit this feature
    ring_geom: "RingGeom | None" = None   # ring center/radii/angles (annular hints only)


@dataclass
class FeatureSegmentation:
    image_shape: tuple[int, int, int]
    segments: list[Classified | Feature]


def _concavity(mask: np.ndarray) -> float:
    """1 − area / convex_hull_area (0 = convex, →1 = deeply concave)."""
    c = _largest_contour(mask)
    if c is None:
        return 0.0
    area = cv2.contourArea(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    return 1.0 - area / hull_area if hull_area > 0.0 else 0.0


def _angular_coverage(mask: np.ndarray, cx: float, cy: float) -> float:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0
    counts, _ = np.histogram(np.arctan2(ys - cy, xs - cx),
                             bins=RING_ANGULAR_BINS, range=(-np.pi, np.pi))
    return float(np.count_nonzero(counts)) / RING_ANGULAR_BINS


# --------------------------------------------------------------------------- #
# the fittable metric (REUSE common's min-enclosing fitters — H5)
# --------------------------------------------------------------------------- #
def _best_fit(mask: np.ndarray, h: int, w: int) -> tuple[str, float]:
    """(winning shape name, best IoU) over the axis-aligned + rotated fitters — the
    per-feature identification metric."""
    c = _largest_contour(mask)
    if c is None:
        return "none", 0.0
    cands: list[tuple[str, np.ndarray, float]] = [
        ("circle", *_fit_circle(c, h, w, 0, 0)[1:]),
        ("rectangle", *_fit_rectangle(c, h, w, 0, 0)[1:]),
        ("triangle", *_fit_triangle(c, h, w, 0, 0)[1:]),
        ("ellipse", *_fit_ellipse(c, h, w, 0, 0)[1:]),
    ]
    rr = _fit_rotated_rect(c, h, w)
    if rr is not None:
        cands.append(("rotated_rect", rr[1], rr[2]))
    re = _fit_rotated_ellipse(c, h, w)
    if re is not None:
        cands.append(("rotated_ellipse", re[1], re[2]))
    best_name, best_iou = "none", 0.0
    for name, fit, area in cands:
        iou = iou_analytical(mask, fit, area)
        if iou > best_iou:
            best_iou, best_name = iou, name
    return best_name, best_iou


def _best_fit_iou(mask: np.ndarray, h: int, w: int) -> float:
    return _best_fit(mask, h, w)[1]


# --------------------------------------------------------------------------- #
# global structure reads
# --------------------------------------------------------------------------- #
def _is_sunburst(mask: np.ndarray) -> bool:
    c = _largest_contour(mask)
    if c is None:
        return False
    approx = cv2.approxPolyDP(c, SUNBURST_POLY_TOL * cv2.arcLength(c, True), True)
    return len(approx) >= SUNBURST_MIN_VTX


def _is_ring(mask: np.ndarray) -> bool:
    """A THIN band around an empty centre. Gates: centroid over background, wide angular
    wrap, band-thinness (max inscribed radius ≪ outer radius), and not-solid. An open
    swoosh (Edge) or annulus passes; a dome/solid blob/thick outline fails."""
    c = _centroid(mask)
    if c is None:
        return False
    cx, cy = c
    h, w = mask.shape
    ix, iy = int(round(cx)), int(round(cy))
    centre_empty = True
    if 0 <= ix < w and 0 <= iy < h:
        rad = max(3, int(RING_CENTER_RADIUS_FRAC * max(h, w)))
        patch = mask[max(0, iy - rad):iy + rad + 1, max(0, ix - rad):ix + rad + 1]
        centre_empty = patch.size > 0 and np.count_nonzero(patch) / patch.size < RING_CENTER_FG_MAX
    if not centre_empty or _angular_coverage(mask, cx, cy) < RING_WRAP_COVERAGE:
        return False
    cnt = _largest_contour(mask)
    if cnt is None:
        return False
    (_, _), outer_r = cv2.minEnclosingCircle(cnt)
    inradius = float(cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).max())
    if outer_r <= 0 or inradius / outer_r >= RING_MAX_BAND:
        return False                                   # thick band ⇒ not a ring
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 1.0
    return solidity < RING_MAX_SOLIDITY                 # solid blob ⇒ not a ring


def _global_shape(mask: np.ndarray) -> ShapeHint:
    """A coherent structure to keep whole, or None (split it locally). Most-specific
    first: a counterform hole, then a radial sunburst, then a thin ring band."""
    if _detect_holes(mask):
        return "holed"
    if _is_sunburst(mask):
        return "sunburst"
    if _is_ring(mask):
        return "ring"
    return None


# --------------------------------------------------------------------------- #
# local: half-disc + body/appendage split
# --------------------------------------------------------------------------- #
def _is_half_disc(mask: np.ndarray) -> bool:
    """area / (π·r²) ≈ 0.5 (half the min-enclosing circle), near-convex, AND round — the
    elongation gate (minAreaRect aspect) rejects thin capsules/antennae that also hit ≈0.5."""
    c = _largest_contour(mask)
    if c is None:
        return False
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > HALFDISC_MAX_ASPECT:
        return False
    (_, _), r = cv2.minEnclosingCircle(c)
    if r <= 0:
        return False
    ratio = cv2.contourArea(c) / (math.pi * r * r)
    return HALFDISC_RATIO_LO <= ratio <= HALFDISC_RATIO_HI and _concavity(mask) < HALFDISC_CONCAVITY_MAX


def _body_appendages(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Morphological-opening split: (body kept exact, [appendage masks]).
    Opening removes thin protrusions whose width < the kernel; the body is then
    mask − the appendage components, so its outline is exact (no rounding)."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, inradius, _, _ = cv2.minMaxLoc(dist)
    k = max(3, int(round(inradius * OPEN_K_INRADIUS_FRAC)))
    k |= 1                                              # odd kernel
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    if not opened.any():
        return mask.copy(), []
    removed = cv2.subtract(mask, opened)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(removed, connectivity=8)
    mask_area = float(np.count_nonzero(mask))
    appendages = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= APPENDAGE_AREA_FRAC * mask_area:
            appendages.append(((lbl == i).astype(np.uint8)) * 255)
    if not appendages:
        return mask.copy(), []
    union = np.zeros_like(mask)
    for ap in appendages:
        union |= ap
    body = cv2.subtract(mask, union)
    return body, appendages


# --------------------------------------------------------------------------- #
# cross-miss global ring pre-pass (the only grouping L2 does)
# --------------------------------------------------------------------------- #
def _group_global_ring(
    sc: SegmentClassification, misses: list[L1Miss], s: int
) -> tuple[list[int], np.ndarray, tuple[int, int, int, int]] | None:
    if len(misses) < RING_MIN_MISSES:
        return None
    gc = _centroid(sc.binarized)
    if gc is None:
        return None
    gcx, gcy = gc
    near = []
    for i, m in enumerate(misses):
        c = _centroid(m.mask)
        if c is None:
            continue
        if math.hypot(c[0] + m.bbox[0] - gcx, c[1] + m.bbox[1] - gcy) <= RING_CENTER_TOL * s:
            near.append(i)
    if len(near) < RING_MIN_MISSES:
        return None
    combined = np.zeros((s, s), np.uint8)
    for i in near:
        x, y, w, h = misses[i].bbox
        combined[y:y + h, x:x + w] |= misses[i].mask
    k = max(3, int(round(s * RING_CLOSE_FRAC)))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if _angular_coverage(closed, gcx, gcy) < RING_MIN_COVERAGE:
        return None
    central = any(
        (_centroid(hole) is not None
         and math.hypot(_centroid(hole)[0] - gcx, _centroid(hole)[1] - gcy) <= RING_HOLE_CENTER_TOL * s)
        for hole in _detect_holes(closed)
    )
    if not central:
        return None
    # Return the UN-closed union: the close was only to gate coverage/central-hole; the polar
    # resolver needs the sectors still separated (closing them merges Chrome's 3 into one band).
    ys, xs = np.where(combined > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return near, combined, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


# --------------------------------------------------------------------------- #
# polar-unroll ring resolver — split a detected ring into its sub-features
# --------------------------------------------------------------------------- #
def _ring_centre(mask: np.ndarray) -> tuple[float, float, float] | None:
    """(cx, cy, outer_r) of the ring — the min-enclosing circle over ALL foreground points
    (not the largest contour): a grouped multi-sector ring has several contours, and the
    largest is one sector, whose centre/radius would mis-map the unroll. Enclosing every
    point gives the shared outer circle for grouped sectors and single rings alike, and stays
    centred for open arcs (a centroid would drift off an open band)."""
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    (cx, cy), outer_r = cv2.minEnclosingCircle(pts)
    if outer_r <= 0.0:
        return None
    return float(cx), float(cy), float(outer_r)


def _seam_merged_labels(polar: np.ndarray) -> np.ndarray:
    """connectedComponents on a polar image, with labels stitched across the θ=0/2π seam:
    row 0 and the last row are physically the same direction, so a sector straddling it is
    one feature, not two. Union-find merges any labels whose pixels touch across the wrap
    (8-connectivity → also the diagonal column neighbours). Robust when no empty angular row
    exists (a spiral-divided ring like Chrome has foreground at every angle, so rolling a gap
    to the seam is impossible). Returns the merged label image (background 0)."""
    n, lbl, _, _ = cv2.connectedComponentsWithStats(polar, connectivity=8)
    parent = list(range(n))

    def find(a: int) -> int:
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:                           # path-compress
            parent[a], a = r, parent[a]
        return r

    top, bot = lbl[0], lbl[-1]
    width = lbl.shape[1]
    for c in range(width):
        a = int(top[c])
        if a == 0:
            continue
        for cc in (c - 1, c, c + 1):                    # 8-connectivity across the wrap
            if 0 <= cc < width:
                b = int(bot[cc])
                if b:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    return roots[lbl]


def _resolve_ring(mask: np.ndarray, ox: int, oy: int,
                  color: tuple[int, int, int]) -> list[Feature]:
    """Unroll a detected ring about its centre (warpPolar) and split the polar image into
    connected components — each is one ring sub-feature. In polar space (rows = θ∈[0,2π),
    cols = radius∈[0,maxR)) an angular sector is a band of ROWS, a concentric ring is a band
    of COLUMNS, a full annulus spans all rows. Each component is inverse-warped back to a
    cartesian sub-mask, tagged annular_band (near-complete annulus) or annular_sector (wedge/arc).

    The θ=0/2π seam (the +x direction) would split a sector that crosses it into two polar
    components; `_seam_merged_labels` stitches them back (row 0 and the last row are physically
    adjacent). INTER_NEAREST keeps the mask binary through both warps; centre + maxR are
    identical forward and inverse so the round-trip lines up."""
    cen = _ring_centre(mask)
    if cen is None:
        return []
    cx, cy, outer_r = cen
    h, w = mask.shape
    max_r = outer_r + RING_CENTRE_MARGIN
    centre = (cx, cy)
    fwd = cv2.WARP_POLAR_LINEAR + cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS

    polar = cv2.warpPolar(mask, (R_BINS, THETA_BINS), centre, max_r, fwd)  # (θ rows, r cols)
    merged = _seam_merged_labels(polar)

    min_area = POLAR_MIN_AREA_FRAC * polar.size
    inv = fwd + cv2.WARP_INVERSE_MAP
    feats: list[Feature] = []
    for label in np.unique(merged):
        if label == 0:
            continue
        comp = merged == label
        if int(np.count_nonzero(comp)) < min_area:
            continue
        coverage = int(np.count_nonzero(comp.any(axis=1))) / THETA_BINS
        hint: ShapeHint = "annular_band" if coverage >= ANNULAR_BAND_COVERAGE else "annular_sector"
        cart = cv2.warpPolar(comp.astype(np.uint8) * 255, (w, h), centre, max_r, inv)
        if not cart.any():
            continue
        bx, by, bw, bh = cv2.boundingRect(cart)
        if comp.any():
            inner_r = float(np.flatnonzero(comp.any(axis=0)).min()) / R_BINS * max_r
            inner_r = min(inner_r, outer_r - 1.0)
        else:
            inner_r = 0.0
        if coverage >= ANNULAR_BAND_COVERAGE:
            theta0, theta1 = 0.0, 2.0 * math.pi
        else:
            present = comp.any(axis=1)
            if present[0] and present[-1]:                  # sector crosses the θ=0/2π seam
                absent = np.flatnonzero(~present)
                t0, t1 = int(absent.max()) + 1, int(absent.min()) + THETA_BINS
            else:
                rows = np.flatnonzero(present)
                t0, t1 = int(rows.min()), int(rows.max())
            theta0 = t0 / THETA_BINS * 2.0 * math.pi
            theta1 = t1 / THETA_BINS * 2.0 * math.pi
        ring_geom = RingGeom(cx + ox, cy + oy, inner_r, outer_r, theta0, theta1)
        feats.append(Feature(
            mask=cart[by:by + bh, bx:bx + bw].copy(),
            bbox=(bx + ox, by + oy, bw, bh),
            kind="global", color_rgb=color, best_fit_iou=0.0, shape_hint=hint,
            ring_geom=ring_geom,
        ))

    # ≥2 sectors from one ring → a GAPPED RING (Chrome): emit ONE `gapped_annulus` Feature for the
    # whole ring mask (the gaps stay in the mask) instead of N separate blade-sectors. D2 then builds
    # Circle(outer) − Circle(inner) − a thin rect per detected gap (the user's redline). Gated on ≥2
    # sectors, so single-sector rings (whatsapp / signal) and full bands are unchanged.
    if sum(1 for f in feats if f.shape_hint == "annular_sector") >= 2:
        cols = polar.any(axis=0)
        inner_r = float(np.flatnonzero(cols).min()) / R_BINS * max_r if cols.any() else 0.0
        inner_r = min(inner_r, outer_r - 1.0)
        bx, by, bw, bh = cv2.boundingRect(mask)
        return [Feature(
            mask=mask[by:by + bh, bx:bx + bw].copy(),
            bbox=(bx + ox, by + oy, bw, bh),
            kind="global", color_rgb=color, best_fit_iou=0.0, shape_hint="gapped_annulus",
            ring_geom=RingGeom(cx + ox, cy + oy, inner_r, outer_r, 0.0, 2.0 * math.pi),
        )]
    return feats


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _feature(part: np.ndarray, ox: int, oy: int, h: int, w: int,
             color: tuple[int, int, int]) -> Feature:
    """Identify one local part by its best-fitting shape; crop to its tight bbox."""
    name, iou = _best_fit(part, h, w)
    if iou >= SEG_FIT_IOU:
        kind, hint = "local", name              # a clean single primitive (incl. rotated)
    elif _is_half_disc(part):
        kind, hint = "local", "half_circle"     # not single-primitive, but a round half-disc
    else:
        kind, hint = "defer", None
    x, y, pw, ph = cv2.boundingRect(part)
    return Feature(
        mask=part[y:y + ph, x:x + pw].copy(),
        bbox=(x + ox, y + oy, pw, ph),
        kind=kind, color_rgb=color, best_fit_iou=float(iou), shape_hint=hint,
    )


def segment_features(
    sc: SegmentClassification, cat: "Categorization | None" = None
) -> FeatureSegmentation:
    """L2 — pass Classified through; split each L1Miss into near-fittable features."""
    s = int(sc.image_shape[0])
    misses = [seg for seg in sc.segments if isinstance(seg, L1Miss)]

    # Cross-miss ring pre-pass: a shared-centre annular group → one ring, polar-resolved into
    # its sub-features (Chrome's sectors). Degenerate unroll falls back to one whole ring.
    consumed: set[int] = set()
    ring_features: list[Feature] = []
    grp = _group_global_ring(sc, misses, s)
    if grp is not None:
        idxs, group_mask, (cx, cy, cw, ch) = grp
        color = tuple(int(c) for c in np.mean([misses[i].color_rgb for i in idxs], axis=0).round())
        crop = group_mask[cy:cy + ch, cx:cx + cw]
        ring_features = _resolve_ring(crop, cx, cy, color)
        if not ring_features:
            ring_features = [Feature(crop.copy(), (cx, cy, cw, ch), "global", color,
                                     _best_fit_iou(crop, ch, cw), "ring")]
        consumed = {id(misses[i]) for i in idxs}

    out: list[Classified | Feature] = []
    ring_emitted = False
    for seg in sc.segments:
        if isinstance(seg, Classified):
            out.append(seg)
            continue
        if id(seg) in consumed:
            if ring_features and not ring_emitted:
                out.extend(ring_features)
                ring_emitted = True
            continue
        x, y, mw, mh = seg.bbox
        color = tuple(int(c) for c in seg.color_rgb)

        gshape = _global_shape(seg.mask)
        if gshape == "ring":
            resolved = _resolve_ring(seg.mask, x, y, color)
            if resolved:                                # split the ring into its sub-features
                out.extend(resolved)
                continue                                # (else fall through → keep whole below)
        if gshape is not None:
            out.append(Feature(seg.mask.copy(), seg.bbox, "global", color,
                               _best_fit_iou(seg.mask, mh, mw), gshape))
            continue

        body, appendages = _body_appendages(seg.mask)
        out.append(_feature(body, x, y, mh, mw, color))
        for ap in appendages:
            out.append(_feature(ap, x, y, mh, mw, color))

    return FeatureSegmentation(image_shape=sc.image_shape, segments=out)


# --------------------------------------------------------------------------- #
# overlay renderer (testbed / H7 strip + the l2seg_debug --dump share this)
# --------------------------------------------------------------------------- #
_FEATURE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0xE6, 0x19, 0x4B), (0x3C, 0xB4, 0x4B), (0x43, 0x63, 0xD8), (0xF5, 0x82, 0x31),
    (0x91, 0x1E, 0xB4), (0x42, 0xD4, 0xF4), (0xF0, 0x32, 0xE6), (0xBF, 0xEF, 0x45),
    (0xFA, 0xBE, 0xD4), (0x46, 0x99, 0x90), (0x9A, 0x63, 0x24), (0x80, 0x00, 0x00),
)


def draw_feature_overlay(aligned: np.ndarray, fs: FeatureSegmentation) -> np.ndarray:
    """Render the L2 split for review: each Feature's mask in a distinct palette
    colour (so the decomposition is legible), blended over the aligned image.
    Classified pass-throughs (L1 hits) are shown in the L1 panel, not here.

    aligned: (H, W, 3) uint8 RGB. Returns: (H, W, 3) uint8 RGB.
    Where two features' cropped bboxes overlap, the later one wins the pixels —
    a display-only artifact; the per-feature masks themselves stay disjoint."""
    overlay = np.full_like(aligned, 255)
    ci = 0
    for seg in fs.segments:
        if not isinstance(seg, Feature):
            continue
        x, y, w, h = seg.bbox
        col = _FEATURE_PALETTE[ci % len(_FEATURE_PALETTE)]
        ci += 1
        sub = overlay[y:y + h, x:x + w]
        sub[seg.mask > 0] = col
    return (0.40 * aligned + 0.60 * overlay).astype(np.uint8)
