"""Stage 3 compress — shared primitive vocabulary, rasterizing fitters, IoU.

Lifted out of segment_prim_classify.py (L1) so both L1 and the later L2/L3
share one definition of the {Circle, Rectangle, Triangle, Ellipse} primitives,
the per-primitive rasterizing fitters, and the analytical-area IoU (H5 — reuse
across the stage; one source of truth, not N copies).

L1 fits ONE min-enclosing / bounding primitive per connected component; later
stages (L2 feature-segmentation and the deferred shape-fitting layer) reuse
these dataclasses + iou_analytical. _axis_symmetry stays in L1 — it gates
single-primitive classification and is not needed downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Triangle:
    v1: tuple[float, float]
    v2: tuple[float, float]
    v3: tuple[float, float]


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    rx: float
    ry: float


@dataclass(frozen=True)
class RotatedRectangle:
    cx: float
    cy: float
    w: float
    h: float
    angle: float                          # degrees, cv2.minAreaRect convention


@dataclass(frozen=True)
class RotatedEllipse:
    cx: float
    cy: float
    rx: float                             # SEMI-axis (cv2.fitEllipse full-axis ÷ 2)
    ry: float                             # SEMI-axis
    angle: float                          # degrees, clockwise


Primitive = (
    Circle | Rectangle | Triangle | Ellipse | RotatedRectangle | RotatedEllipse
)


def iou_analytical(roi: np.ndarray, fit: np.ndarray, true_fit_area: float) -> float:
    """IoU(CC, primitive) using the primitive's ANALYTICAL area as the union
    denominator instead of the rasterized-and-clipped count. Without this,
    a primitive whose true geometry extends beyond the CC bbox (most often
    the min-enclosing triangle's apex) gets its excess area silently
    clipped by cv2.fillPoly and the IoU is wildly inflated.
    """
    roi_bool = roi.astype(bool)
    fit_bool = fit.astype(bool)
    inter = int(np.logical_and(roi_bool, fit_bool).sum())
    roi_area = int(roi_bool.sum())
    union = roi_area + true_fit_area - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _fit_circle(
    contour: np.ndarray, h: int, w: int, ox: int, oy: int
) -> tuple[Circle, np.ndarray, float]:
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fit, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    true_area = float(np.pi * r * r)
    return Circle(cx=float(cx + ox), cy=float(cy + oy), r=float(r)), fit, true_area


def _fit_rectangle(
    contour: np.ndarray, h: int, w: int, ox: int, oy: int
) -> tuple[Rectangle, np.ndarray, float]:
    xr, yr, wr, hr = cv2.boundingRect(contour)
    fit = np.zeros((h, w), dtype=np.uint8)
    # numpy slicing — avoids cv2.rectangle's inclusive-endpoint off-by-one.
    fit[yr:yr + hr, xr:xr + wr] = 255
    true_area = float(wr * hr)
    return Rectangle(x=int(xr + ox), y=int(yr + oy), w=int(wr), h=int(hr)), fit, true_area


def _fit_triangle(
    contour: np.ndarray, h: int, w: int, ox: int, oy: int
) -> tuple[Triangle, np.ndarray, float]:
    area, tri = cv2.minEnclosingTriangle(contour)
    if tri is None:
        return Triangle(v1=(0.0, 0.0), v2=(0.0, 0.0), v3=(0.0, 0.0)), np.zeros((h, w), dtype=np.uint8), 0.0
    pts = tri.astype(np.int32)
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fit, [pts], 255)
    flat = tri.reshape(3, 2)
    return Triangle(
        v1=(float(flat[0, 0] + ox), float(flat[0, 1] + oy)),
        v2=(float(flat[1, 0] + ox), float(flat[1, 1] + oy)),
        v3=(float(flat[2, 0] + ox), float(flat[2, 1] + oy)),
    ), fit, float(area)


def _fit_ellipse(
    contour: np.ndarray, h: int, w: int, ox: int, oy: int
) -> tuple[Ellipse, np.ndarray, float]:
    xr, yr, wr, hr = cv2.boundingRect(contour)
    rx = wr / 2.0
    ry = hr / 2.0
    cx = xr + rx
    cy = yr + ry
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        fit,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(rx))), max(1, int(round(ry)))),
        0, 0, 360, 255, -1,
    )
    true_area = float(np.pi * rx * ry)
    return Ellipse(
        cx=float(cx + ox), cy=float(cy + oy),
        rx=float(rx), ry=float(ry),
    ), fit, true_area


# --------------------------------------------------------------------------- #
# Rotated-primitive fitters (lifted from L2 so L1/L2/L3 share one definition,
# H5). Each returns (Primitive, rasterized mask, analytical area) | None — the
# same 3-tuple shape as the axis-aligned fitters above. A rotated primitive only
# competes when it earns it (elongation / ≥5 not-too-angular points); a blob is
# left to the axis-aligned fitters.
# --------------------------------------------------------------------------- #
ROT_RECT_MIN_ASPECT = 2.0                 # rotated rect only when elongated
ROT_ELLIPSE_POLY_TOL = 0.02
ROT_ELLIPSE_MAX_VERTICES = 8


def _fit_rotated_rect(
    contour: np.ndarray, h: int, w: int
) -> tuple[RotatedRectangle, np.ndarray, float] | None:
    """Min-area rotated rectangle. boxPoints rasterization is angle-agnostic."""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) < ROT_RECT_MIN_ASPECT:
        return None
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fit, [cv2.boxPoints(rect).astype(np.int32)], 255)
    prim = RotatedRectangle(
        cx=float(cx), cy=float(cy), w=float(rw), h=float(rh), angle=float(angle)
    )
    return prim, fit, float(rw * rh)


def _fit_rotated_ellipse(
    contour: np.ndarray, h: int, w: int
) -> tuple[RotatedEllipse, np.ndarray, float] | None:
    """Rotated least-squares ellipse (≥5 points, not highly-angular). fitEllipse
    returns FULL axis lengths → halve for cv2.ellipse's semi-axes."""
    if contour.shape[0] < 5:
        return None
    approx = cv2.approxPolyDP(contour, ROT_ELLIPSE_POLY_TOL * cv2.arcLength(contour, True), True)
    if len(approx) > ROT_ELLIPSE_MAX_VERTICES:
        return None
    (cx, cy), (a_full, b_full), angle = cv2.fitEllipse(contour)
    if b_full < 1.0:
        return None
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(fit, (int(round(cx)), int(round(cy))),
                (max(1, int(a_full / 2)), max(1, int(b_full / 2))), angle, 0, 360, 255, -1)
    prim = RotatedEllipse(
        cx=float(cx), cy=float(cy), rx=float(a_full / 2.0), ry=float(b_full / 2.0),
        angle=float(angle),
    )
    return prim, fit, float(math.pi * (a_full / 2.0) * (b_full / 2.0))


# --------------------------------------------------------------------------- #
# Construction vocabulary (signed primitives) + mask algebra.
#
# Lifted from the L2 compound-classify prototype (H5 — one source of truth for
# L1/L2/L3). A *construction* is a list of Parts: every "add" is OR'd, then every
# "subtract" is carved (AND-NOT). Order-independent within each op group. The
# fitting layer (L3) builds constructions; these helpers rasterize and score them.
# --------------------------------------------------------------------------- #
MIN_FEATURE_DIVISOR = 48                  # smallest meaningful extent ≈ S / divisor

Op = Literal["add", "subtract"]


@dataclass(frozen=True)
class Part:
    primitive: Primitive
    op: Op                                # "add" | "subtract"


def _min_feature_px(s: int) -> int:
    """Smallest primitive extent worth emitting at canonical resolution s."""
    return max(2, math.ceil(s / MIN_FEATURE_DIVISOR))


def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks (uint8 {0,255} or bool)."""
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """(cx, cy) of a uint8 mask via image moments, or None if empty."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0.0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


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
    """Largest external contour (≥3 points) of a mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if c.shape[0] < 3:
        return None
    return c


def _extent_ok(mask: np.ndarray, min_feat: int) -> bool:
    """True when the mask's bbox spans at least min_feat in both dimensions."""
    x, y, w, h = cv2.boundingRect(mask)
    return min(w, h) >= min_feat


def _raster_part(p: Primitive, h: int, w: int) -> np.ndarray:
    """Rasterize one primitive to an (h, w) uint8 {0,255} mask."""
    m = np.zeros((h, w), dtype=np.uint8)
    if isinstance(p, Circle):
        cv2.circle(m, (int(round(p.cx)), int(round(p.cy))), max(1, int(round(p.r))), 255, -1)
    elif isinstance(p, Ellipse):
        cv2.ellipse(
            m, (int(round(p.cx)), int(round(p.cy))),
            (max(1, int(round(p.rx))), max(1, int(round(p.ry)))),
            0, 0, 360, 255, -1,
        )
    elif isinstance(p, RotatedEllipse):
        cv2.ellipse(
            m, (int(round(p.cx)), int(round(p.cy))),
            (max(1, int(round(p.rx))), max(1, int(round(p.ry)))),
            float(p.angle), 0, 360, 255, -1,
        )
    elif isinstance(p, Rectangle):
        m[p.y:p.y + p.h, p.x:p.x + p.w] = 255
    elif isinstance(p, RotatedRectangle):
        box = ((float(p.cx), float(p.cy)),
               (max(1.0, float(p.w)), max(1.0, float(p.h))), float(p.angle))
        cv2.fillPoly(m, [cv2.boxPoints(box).astype(np.int32)], 255)
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


def _compound_iou(
    parts: list[Part] | tuple[Part, ...], target: np.ndarray, h: int, w: int
) -> float:
    """IoU of a construction's rasterized mask against a target mask."""
    return _iou_bool(_compound_mask(parts, h, w), target)


def _offset(p: Primitive, ox: int, oy: int) -> Primitive:
    """Translate a primitive from crop-local to full-image coordinates."""
    if isinstance(p, Circle):
        return Circle(cx=p.cx + ox, cy=p.cy + oy, r=p.r)
    if isinstance(p, Ellipse):
        return Ellipse(cx=p.cx + ox, cy=p.cy + oy, rx=p.rx, ry=p.ry)
    if isinstance(p, RotatedEllipse):
        return RotatedEllipse(cx=p.cx + ox, cy=p.cy + oy, rx=p.rx, ry=p.ry, angle=p.angle)
    if isinstance(p, Rectangle):
        return Rectangle(x=p.x + ox, y=p.y + oy, w=p.w, h=p.h)
    if isinstance(p, RotatedRectangle):
        return RotatedRectangle(cx=p.cx + ox, cy=p.cy + oy, w=p.w, h=p.h, angle=p.angle)
    return Triangle(
        v1=(p.v1[0] + ox, p.v1[1] + oy),
        v2=(p.v2[0] + ox, p.v2[1] + oy),
        v3=(p.v3[0] + ox, p.v3[1] + oy),
    )


def _offset_part(part: Part, ox: int, oy: int) -> Part:
    """Translate a Part's primitive to full-image coordinates."""
    return Part(primitive=_offset(part.primitive, ox, oy), op=part.op)


def _cand_inscribed_circle(
    region: np.ndarray, h: int, w: int
) -> tuple[Ellipse, np.ndarray, float] | None:
    """Largest inscribed circle (distance-transform peak) as an axis-aligned
    Ellipse with rx == ry, plus its rasterized mask and analytical area."""
    dist = cv2.distanceTransform(region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, r, _, center = cv2.minMaxLoc(dist)
    if r < 1.0:
        return None
    cx, cy = float(center[0]), float(center[1])
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fit, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return Ellipse(cx=cx, cy=cy, rx=float(r), ry=float(r)), fit, float(math.pi * r * r)


# --------------------------------------------------------------------------- #
# Largest inscribed axis-aligned rectangle (histogram maximal-rectangle DP on a
# downscaled proxy — the one sanctioned hand-rolled loop; no OpenCV/skimage
# inscribed-rect primitive exists, and the `lir` lib needs numba which has no
# wheel for this interpreter). Used to fit a rectangular hole tightly without
# the bounding-rect overshoot. Lifted from the L2 compound-classify prototype.
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
        stack: list[tuple[int, int]] = []          # (start_col, height), strictly increasing
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
    """Largest inscribed axis-aligned rectangle of a binary region (uint8 {0,255}),
    as (x, y, w, h), or None. Runs the DP on a downscaled proxy (bounded cost) then
    scales the rectangle back with an inset guard against upscale spill."""
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
        inset = max(1, int(round(inv)))
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


def _cand_inscribed_rectangle(
    region: np.ndarray, h: int, w: int
) -> tuple[Rectangle, np.ndarray, float] | None:
    """Largest inscribed axis-aligned rectangle as a Rectangle + rasterized mask + area."""
    xywh = _inscribed_rect_xywh(region)
    if xywh is None:
        return None
    x, y, rw, rh = xywh
    if rw < 1 or rh < 1:
        return None
    fit = np.zeros((h, w), dtype=np.uint8)
    fit[y:y + rh, x:x + rw] = 255
    return Rectangle(x=int(x), y=int(y), w=int(rw), h=int(rh)), fit, float(rw * rh)


# --------------------------------------------------------------------------- #
# Hole detection / filling / fitting (negation support, shared by L2's detection
# gate and L3's holed/annular templates). Lifted from the L2 prototype + L2 itself.
# --------------------------------------------------------------------------- #
HOLE_AREA_FRAC = 0.03           # a hole must be ≥ this fraction of the filled area to count
HOLE_FIT_MIN_IOU = 0.50         # only subtract a hole a primitive actually matches


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Solid outer shape: redraw external contours filled (interior holes vanish)."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def _detect_holes(mask: np.ndarray) -> list[np.ndarray]:
    """Interior holes as filled {0,255} masks. RETR_CCOMP gives a 2-level tree; a hole
    is any contour with a parent. Holes below HOLE_AREA_FRAC of the filled area are
    ignored (sub-feature noise)."""
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    filled_area = float(np.count_nonzero(_fill_holes(mask)))
    if filled_area <= 0:
        return []
    holes: list[np.ndarray] = []
    for i, c in enumerate(contours):
        if hier[i][3] == -1:                        # no parent → outer boundary
            continue
        if cv2.contourArea(c) < HOLE_AREA_FRAC * filled_area:
            continue
        hm = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(hm, [c], -1, 255, thickness=cv2.FILLED)
        holes.append(hm)
    return holes


def _fit_hole(hole_mask: np.ndarray, h: int, w: int, min_feat: int) -> Primitive | None:
    """Best single primitive matching a hole (to subtract it). None when the hole is
    sub-feature-size or no primitive matches it well enough to carve."""
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
# D1 contour-parse vocabulary — a shape's outer boundary as typed runs + corners.
# Shared so the contour parser (D1), edge-tracer (D2), and junction tagger (D3)
# read one definition (H5).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Corner:
    """An approx-polygon vertex: 'corner' (sharp turn) vs 'smooth', plus convexity —
    convex = a peak that juts OUT of the shape, concave = a valley that dips IN."""
    point: tuple[float, float]
    kind: Literal["corner", "smooth"]
    convex: bool


@dataclass(frozen=True)
class ContourSegment:
    """A maximal boundary run between two corners. 'straight' when the dense contour hugs its
    chord; 'arc' when it bows. convex = bulges OUTWARD (an additive primitive in D2); concave =
    curves INWARD (a subtractive primitive / junction)."""
    kind: Literal["straight", "arc"]
    points: np.ndarray                   # (K, 2) dense contour points of this run
    start: tuple[float, float]
    end: tuple[float, float]
    convex: bool


@dataclass(frozen=True)
class ParsedContour:
    """One closed boundary as an ordered, cyclic sequence of typed runs + corner tags.
    is_hole marks an interior hole (D2 subtracts it) vs the outer boundary."""
    segments: tuple[ContourSegment, ...]
    corners: tuple[Corner, ...]
    is_hole: bool = False


@dataclass(frozen=True)
class ParsedShape:
    """A feature's full boundary: the outer contour + its interior holes (RETR_CCOMP top level;
    a nested hole-within-a-hole appears as foreground inside its hole mask, not separately)."""
    outer: ParsedContour
    holes: tuple[ParsedContour, ...]
