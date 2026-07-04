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


@dataclass(frozen=True)
class RoundedRectangle:
    cx: float
    cy: float
    w: float                              # FULL width  (cv2.minAreaRect convention)
    h: float                              # FULL height
    angle: float                          # degrees, cv2.minAreaRect convention
    corner_radius: float                  # ≥ 0; clamped to ≤ min(w, h)/2 at raster time


@dataclass(frozen=True)
class Wedge:
    """Filled circular sector (pie slice): circle(c, r) ∩ angular span [a0, a1].
    Angles in DEGREES, cv2 y-down convention (0° = +x, increasing clockwise on
    screen — same convention as the rotated primitives' angles). a1 is ALWAYS
    > a0 (sweep = a1 − a0) and may exceed 360 to cross the 0° seam: cv2.ellipse
    does not modulo-normalize, so 350→30 means a huge arc, 350→390 means 40°.
    Normalize at construction sites, never at raster time."""
    cx: float
    cy: float
    r: float
    a0: float
    a1: float


@dataclass(frozen=True)
class Polygon:
    """Arbitrary closed polygon (≥3 vertices, canvas px coords, pixel-center
    convention like every other primitive). RENDER-TIER ONLY: produced by the
    stage-3 R1 renderer's degree ladder (straighten.py) — no deterministic
    fitter constructs one, and Polygon-bearing Parts never enter the diff_fit
    seed pool or the L3 baseline (their primitive dispatches would reject it).
    Dormant for the decomposer by design, same pattern as Wedge before its
    producer. Rasterized at half-pixel precision (fillPoly shift=1). fillPoly is
    OUTLINE-INCLUSIVE with round-half-up vertex placement: integer-corner boxes
    (a..b) fill pixels a..b exactly (b−a+1 wide), and fractional edges round
    outward — so producers emit geometry INSET by ~0.5 px per edge (straighten.py
    does) to land on the pixel-centers-inside coverage rule. The render tier's
    grid snapping targets integer vertex coordinates, the crisp case.
    pts is a tuple of tuples (not ndarray) to stay frozen/hashable."""
    pts: tuple[tuple[float, float], ...]


Primitive = (
    Circle | Rectangle | Triangle | Ellipse | RotatedRectangle | RotatedEllipse
    | RoundedRectangle | Wedge | Polygon
)


@dataclass(frozen=True)
class RingGeom:
    """L2 polar-unroll ring geometry, threaded to the annular fitter (full-image px)."""
    cx: float
    cy: float
    inner_r: float                        # 0.0 if solid (no inner hole)
    outer_r: float
    theta0: float                         # sector bounds, radians (warpPolar y-down); band → (0, 2π)
    theta1: float


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


def _fit_arc_circle(points: np.ndarray) -> tuple[Circle, float] | None:
    """Kasa least-squares circle through an arc run's dense points + RMSE (px).
    Solves [2x 2y 1]·[a b c]ᵀ = x²+y² (np.linalg.lstsq, float64): center (a, b),
    r = sqrt(c + a² + b²). cv2 has no LSQ circle fit — minEnclosingCircle is a
    BOUNDING circle, not a least-squares one, and cv2.fitEllipse is ill-conditioned
    on partial arcs (a half-circle run balloons). PURE numerics — the caller owns
    the policy gates (bbox cap, centre escape, RMSE acceptance). None when the run
    is too short, collinear (rank-deficient), or degenerate. Deterministic (H8)."""
    if len(points) < 5:
        return None
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a_mat = np.column_stack([2.0 * pts[:, 0], 2.0 * pts[:, 1], np.ones(len(pts))])
    b_vec = pts[:, 0] ** 2 + pts[:, 1] ** 2
    sol, _res, rank, _sv = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    if rank < 3:                                          # collinear run — circle undefined
        return None
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0.0:
        return None
    r = math.sqrt(r2)
    if not (np.isfinite(cx) and np.isfinite(cy)) or r < 1.0:
        return None
    rmse = float(np.sqrt(np.mean((np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) ** 2)))
    return Circle(cx=cx, cy=cy, r=r), rmse


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
ROUNDRECT_MIN_SOLIDITY = 0.85             # rounded-rect fit only on a near-convex region (a ring/star's area gap is spurious)
ROUNDRECT_MIN_RADIUS_FRAC = 0.10          # corner_radius below this × short-side → ~sharp; None so _fit_rotated_rect wins
ROUNDRECT_MAX_RADIUS_FRAC = 0.45          # corner_radius above this × short-side → ≈ a circle; the rrect-annular caller
#                                           rejects such a shell so a true circular ring (youtube) keeps its Circle annulus


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


def _fit_rounded_rect(
    contour: np.ndarray, h: int, w: int, min_aspect: float = ROT_RECT_MIN_ASPECT
) -> tuple[RoundedRectangle, np.ndarray, float] | None:
    """Min-area rotated rectangle WITH rounded corners. corner_radius from the box-vs-contour area gap
    (a rounded rect of w×h has area w·h − (4−π)r², so r = sqrt(gap/(4−π))). Returns None when the
    region isn't elongated enough (aspect < min_aspect — the default blocks near-square blobs, but the
    rrect-annular caller passes 1.0 to admit a squircle), isn't near-convex (a ring/star's gap is
    spurious), or is effectively sharp (r < ROUNDRECT_MIN_RADIUS → caller falls back to
    _fit_rotated_rect). Same 3-tuple as the siblings."""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) < min_aspect:
        return None
    c_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0 or c_area / hull_area < ROUNDRECT_MIN_SOLIDITY:   # concave → not a rounded rect
        return None
    gap = max(0.0, rw * rh - c_area - (rw + rh))                       # − minAreaRect's ≈½px padding (≈ perimeter)
    r = math.sqrt(gap / (4.0 - math.pi)) if gap >= 1.0 else 0.0
    r = min(r, min(rw, rh) / 2.0)
    if r < ROUNDRECT_MIN_RADIUS_FRAC * min(rw, rh):                    # ~sharp / negligible rounding → _fit_rotated_rect wins
        return None
    prim = RoundedRectangle(cx=float(cx), cy=float(cy), w=float(rw), h=float(rh),
                            angle=float(angle), corner_radius=float(r))
    return prim, _raster_part(prim, h, w), float(rw * rh - (4.0 - math.pi) * r * r)


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
    group: int = 0                        # CSG group: subtracts scope to their own group, groups then
    #                                       union. Default 0 = one global group = the flat (∪add)−(∪sub)
    #                                       every existing construction relies on. Distinct groups let a
    #                                       union of per-shape annuli (outer−inner) keep their woven
    #                                       interiors (react's 3 orbit rings) instead of carving each
    #                                       other away under a single global subtract. NEGATIVE group ⇒
    #                                       the whole group's compound mask is carved FROM the positive
    #                                       groups' union (a counter whose own construction contains
    #                                       subtracts — recursive hole decomposition, mastodon's m).


def _min_feature_px(s: int) -> int:
    """Smallest primitive extent worth emitting at canonical resolution s."""
    return max(2, math.ceil(s / MIN_FEATURE_DIVISOR))


def _iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks (uint8 {0,255} or bool)."""
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def _nbcount(mask: np.ndarray) -> np.ndarray:
    """Per-pixel count (int16, 0..8) of FOREGROUND 8-neighbours, center excluded.
    BORDER_CONSTANT pads with 0 (off-canvas = background); normalise to {0,1} first."""
    b = (mask > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return cv2.filter2D(b, cv2.CV_16S, k, borderType=cv2.BORDER_CONSTANT)


def _nb4count(mask: np.ndarray) -> np.ndarray:
    """Per-pixel count (int16, 0..4) of FOREGROUND 4-neighbours (shared-edge only, center
    excluded). BORDER_CONSTANT pads with 0 (off-canvas = background); normalise to {0,1} first."""
    b = (mask > 0).astype(np.uint8)
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
    return cv2.filter2D(b, cv2.CV_16S, k, borderType=cv2.BORDER_CONSTANT)


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


ROUNDED_ARC_DELTA = 2     # degrees per ellipse2Poly step for rounded-rect corners (fixed → H8 determinism)


def _rounded_rect_poly(cx: float, cy: float, w: float, h: float, angle: float,
                       r: float) -> np.ndarray:
    """Filled outline of a ROTATED rounded rectangle as an (N,1,2) int32 polygon for fillPoly: 4 corner
    quarter-arcs (cv2.ellipse2Poly, fixed ROUNDED_ARC_DELTA → deterministic) at the inset corners,
    concatenated in order and rotated as one. corner_radius clamped to ≤ min(w,h)/2 (else the arc
    centres cross and the polygon bow-ties)."""
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    hw, hh = w / 2.0, h / 2.0
    corners = (((hw - r, -hh + r), -90, 0), ((hw - r, hh - r), 0, 90),
               ((-hw + r, hh - r), 90, 180), ((-hw + r, -hh + r), 180, 270))
    arcs = [cv2.ellipse2Poly((int(round(icx)), int(round(icy))),
                             (max(1, int(round(r))), max(1, int(round(r)))), 0, a0, a1, ROUNDED_ARC_DELTA)
            for (icx, icy), a0, a1 in corners]
    local = np.vstack(arcs).astype(np.float64)
    th = math.radians(angle)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    world = (rot @ local.T).T + np.array([cx, cy])
    return world.round().astype(np.int32).reshape(-1, 1, 2)


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
    elif isinstance(p, RoundedRectangle):
        cv2.fillPoly(m, [_rounded_rect_poly(p.cx, p.cy, p.w, p.h, p.angle, p.corner_radius)], 255)
    elif isinstance(p, Wedge):
        # filled partial-angle ellipse = the true pie INCLUDING the center; axes int ≥ 1
        cv2.ellipse(
            m, (int(round(p.cx)), int(round(p.cy))),
            (max(1, int(round(p.r))), max(1, int(round(p.r)))),
            0, float(p.a0), float(p.a1), 255, -1,
        )
    elif isinstance(p, Triangle):
        pts = np.array([p.v1, p.v2, p.v3], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 255)
    elif isinstance(p, Polygon):
        # shift=2 fixed point: coords ×4 → quarter-pixel precision. Never LINE_AA
        # (gray fringe corrupts {0,255}). shift=1 would quantize x.3/x.7 onto the
        # x.5 round-half-up ties of the outline-inclusive fill, bloating right/
        # bottom edges by a pixel; quarter-px keeps fractional vertices off ties.
        pts = (np.asarray(p.pts, dtype=np.float64) * 4.0).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 255, shift=2)
    return m


def _half_plane_rect(a, b, outside_dir: np.ndarray, h: int, w: int) -> RotatedRectangle:
    """A huge rotated rectangle covering the half-plane on the `outside_dir` side of line a→b — the
    spill-carve / sector-cut primitive (subtract). Lifted from decompose.py (H5: generic CSG geometry,
    shared by the spill-carve and the annular sector cuts)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    u = b - a
    u = u / (np.linalg.norm(u) + 1e-9)
    nrm = np.array([-u[1], u[0]])
    if float(np.dot(nrm, outside_dir)) < 0:
        nrm = -nrm
    diag = float(np.hypot(h, w))
    mid = (a + b) / 2.0 + nrm * (diag / 2.0)
    angle = float(np.degrees(np.arctan2(u[1], u[0])))
    return RotatedRectangle(cx=float(mid[0]), cy=float(mid[1]), w=2.0 * diag, h=diag, angle=angle)


def _compound_from_masks(items: list[tuple["Op", int, np.ndarray]] | tuple,
                         h: int, w: int) -> np.ndarray:
    """The grouped-CSG algebra over PRE-RASTERIZED member masks — the single
    source of truth _compound_mask delegates to (H5). Exists so raster-only
    members (the R1 renderer's mirror-flipped followers, which have no exact
    Part) can join a compound without forking the algebra.
    items: (op, group, (h, w) uint8 mask) per member."""
    out = np.zeros((h, w), dtype=np.uint8)
    neg = np.zeros((h, w), dtype=np.uint8)
    for g in sorted({g for _op, g, _m in items}):
        union = np.zeros((h, w), dtype=np.uint8)
        for op, gg, m in items:
            if gg == g and op == "add":
                union |= m
        for op, gg, m in items:
            if gg == g and op == "subtract":
                union &= ~m
        if g >= 0:
            out |= union
        else:
            neg |= union
    return out & ~neg


def _compound_mask(parts: list[Part] | tuple[Part, ...], h: int, w: int) -> np.ndarray:
    """Per-GROUP CSG: within each group OR its adds then AND-NOT its subtracts; then OR the
    POSITIVE groups and AND-NOT the NEGATIVE groups (group < 0 ⇒ the whole group's compound
    mask carves the result — how a counter whose own construction needs subtracts, e.g. a
    letterform arch = ring − inner − half-plane, is negated; there is no add-back otherwise).
    A subtract only carves its OWN group, so a union of per-shape annuli keeps each shape's
    interior (react's woven orbit rings). Order-independent and, when every part is in the
    default group 0, byte-identical to the old flat (∪add)−(∪sub)."""
    return _compound_from_masks(
        [(p.op, p.group, _raster_part(p.primitive, h, w)) for p in parts], h, w)


def _compound_iou(
    parts: list[Part] | tuple[Part, ...], target: np.ndarray, h: int, w: int
) -> float:
    """IoU of a construction's rasterized mask against a target mask."""
    return _iou_bool(_compound_mask(parts, h, w), target)


PRUNE_TOL = 0.001        # _post_prune free pass: drop a part if removing it costs < this min-IoU
STRUCT_PRUNE_STEP = 0.02  # _post_prune budget pass: a structure-first removal may cost up to this
#                           min-IoU per step (the structured-over-fidelity trade, bounded per part)


def _post_prune(parts: list["Part"], target: np.ndarray, h: int, w: int,
                target_n: int | None = None) -> list["Part"]:
    """Greedy hard-score cleanup. FREE pass: drop any part whose removal costs
    < PRUNE_TOL of min(area, boundary) IoU — kills redundant pileup (refine-loop
    triangle tails, degenerate slivers a soft optimizer flattened) without touching
    load-bearing structure. Then, when `target_n` is given and the construction is
    still larger, a BUDGET pass: repeatedly remove the part whose removal hurts
    least, while each step costs ≤ STRUCT_PRUNE_STEP — clean few-part structure is
    worth bounded fidelity (the chrome-annulus doctrine); stop when the cheapest
    removal exceeds the step budget (genuinely irreducible sprawl stays oversized
    for the caller's cap to judge). Deterministic (H8): area/index ordering, stable
    arg-max. Shared by the trace, the recursive hole carve, and the diff refiner (H5)."""
    if len(parts) <= 1:
        return parts
    base = _compound_score(parts, target, h, w)[2]
    order = sorted(range(len(parts)),
                   key=lambda i: (int(np.count_nonzero(_raster_part(parts[i].primitive, h, w))), i))
    keep = list(parts)
    for idx in order:
        if len(keep) <= 1:
            break
        if parts[idx] not in keep:
            continue
        trial = [p for p in keep if p is not parts[idx]]
        sc = _compound_score(trial, target, h, w)[2]
        if sc >= base - PRUNE_TOL:
            keep = trial
            base = max(base, sc)
    if target_n is not None:
        while len(keep) > max(1, target_n):
            best_i, best_sc = -1, -1.0
            for i in range(len(keep)):
                sc = _compound_score(keep[:i] + keep[i + 1:], target, h, w)[2]
                if sc > best_sc:
                    best_sc, best_i = sc, i
            if best_sc < base - STRUCT_PRUNE_STEP:
                break
            keep = keep[:best_i] + keep[best_i + 1:]
            base = best_sc
    return keep


# ── boundary-aware scoring (contour fidelity) ───────────────────────────
# Pure area IoU hides corner/edge error (a sharp rect scores ~0.93 against a
# rounded "pill"; an ellipse scores high against a square). Boundary IoU
# (Cheng 2021) restricts the IoU to a d-px band along each boundary, so wrong
# corners/edges register. Fused with area IoU by min() — the honest harsher
# number, no blend weight (mirrors L1's _axis_symmetry min-of-two-IoUs). It
# reads RASTERIZED boundaries only, so it is immune to D1 run-typing.
# Deterministic (H8): cv2.erode + numpy, no RNG.
BOUNDARY_FRAC = 0.02     # boundary-band width as a fraction of the bbox diagonal (Cheng 2021 default)
MIN_BAND_PX = 8          # target band below this ⇒ degenerate → fall back to area IoU


def _band_width(target: np.ndarray) -> int:
    """Boundary-band width d = BOUNDARY_FRAC × bbox diagonal (Cheng 2021's dilation-ratio form)."""
    _x, _y, w, h = cv2.boundingRect(target)
    return max(2, int(round(BOUNDARY_FRAC * math.hypot(w, h))))


def _boundary_band(m: np.ndarray, d: int) -> np.ndarray:
    """The d-px rim of a binary mask: m AND-NOT erode(m, d) (uint8 {0,255}).
    One erode with a (2d+1)² RECT kernel — byte-identical to d iterations of the 3×3
    (rect SEs compose exactly), but ~d× faster; the iterated form dominated stage-3
    profiles (boundary IoU is called per candidate per prune trial)."""
    k = 2 * int(d) + 1
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    eroded = cv2.erode(m, se)
    return m & ~eroded


def _boundary_iou(a: np.ndarray, b: np.ndarray, d: int) -> float:
    """IoU restricted to the d-px boundary bands of a and b (Cheng 2021 Boundary IoU). Degenerate
    fallback to full-mask area IoU when either band is below MIN_BAND_PX OR equals its whole mask
    (erosion wiped a thin interior, so the band IS the body) — otherwise a rim would be scored
    against a whole body, deflating the result. Every caller (scoring + the subtractor pick)
    inherits this guard (H5)."""
    band_a, band_b = _boundary_band(a, d), _boundary_band(b, d)
    na, nb = int(np.count_nonzero(band_a)), int(np.count_nonzero(band_b))
    if (na < MIN_BAND_PX or nb < MIN_BAND_PX
            or na == int(np.count_nonzero(a)) or nb == int(np.count_nonzero(b))):
        return _iou_bool(a, b)
    return _iou_bool(band_a, band_b)


def _compound_boundary_iou(
    parts: list[Part] | tuple[Part, ...], target: np.ndarray, h: int, w: int,
    d: int | None = None,
) -> float:
    """Boundary IoU of a construction's compound mask against target (d auto from target)."""
    if d is None:
        d = _band_width(target)
    return _boundary_iou(_compound_mask(parts, h, w), target, d)


def _compound_score(
    parts: list[Part] | tuple[Part, ...], target: np.ndarray, h: int, w: int
) -> tuple[float, float, float]:
    """(area_iou, boundary_iou, score) of a construction vs target, where score = min(area, boundary)
    — the honest harsher number (no blend weight). Boundary falls back to area when the target's
    boundary band is degenerate (a tiny/thin feature whose band is the whole mask)."""
    recon = _compound_mask(parts, h, w)
    area = _iou_bool(recon, target)
    boundary = _boundary_iou(recon, target, _band_width(target))   # self-handles degenerate bands
    return area, boundary, min(area, boundary)


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
    if isinstance(p, RoundedRectangle):
        return RoundedRectangle(cx=p.cx + ox, cy=p.cy + oy, w=p.w, h=p.h,
                                angle=p.angle, corner_radius=p.corner_radius)
    if isinstance(p, Wedge):
        return Wedge(cx=p.cx + ox, cy=p.cy + oy, r=p.r, a0=p.a0, a1=p.a1)
    if isinstance(p, Polygon):
        return Polygon(pts=tuple((x + ox, y + oy) for x, y in p.pts))
    return Triangle(
        v1=(p.v1[0] + ox, p.v1[1] + oy),
        v2=(p.v2[0] + ox, p.v2[1] + oy),
        v3=(p.v3[0] + ox, p.v3[1] + oy),
    )


def _offset_part(part: Part, ox: int, oy: int) -> Part:
    """Translate a Part's primitive to full-image coordinates (carrying its CSG group, so a grouped
    construction — react's woven orbit rings — survives the offset instead of flattening to group 0)."""
    return Part(primitive=_offset(part.primitive, ox, oy), op=part.op, group=part.group)


def _scale(p: Primitive, f: float) -> Primitive:
    """Scale a primitive about the ORIGIN by factor f (coordinate-space rescale,
    the canvas→target-grid mapping of the render tier). Mirrors _offset's per-kind
    dispatch. Angles are scale-invariant; Rectangle's int fields round after
    scaling (sub-half-pixel error at the supersampled reference resolution)."""
    if isinstance(p, Circle):
        return Circle(cx=p.cx * f, cy=p.cy * f, r=p.r * f)
    if isinstance(p, Ellipse):
        return Ellipse(cx=p.cx * f, cy=p.cy * f, rx=p.rx * f, ry=p.ry * f)
    if isinstance(p, RotatedEllipse):
        return RotatedEllipse(cx=p.cx * f, cy=p.cy * f, rx=p.rx * f, ry=p.ry * f, angle=p.angle)
    if isinstance(p, Rectangle):
        return Rectangle(x=int(round(p.x * f)), y=int(round(p.y * f)),
                         w=max(1, int(round(p.w * f))), h=max(1, int(round(p.h * f))))
    if isinstance(p, RotatedRectangle):
        return RotatedRectangle(cx=p.cx * f, cy=p.cy * f, w=p.w * f, h=p.h * f, angle=p.angle)
    if isinstance(p, RoundedRectangle):
        return RoundedRectangle(cx=p.cx * f, cy=p.cy * f, w=p.w * f, h=p.h * f,
                                angle=p.angle, corner_radius=p.corner_radius * f)
    if isinstance(p, Wedge):
        return Wedge(cx=p.cx * f, cy=p.cy * f, r=p.r * f, a0=p.a0, a1=p.a1)
    if isinstance(p, Polygon):
        return Polygon(pts=tuple((x * f, y * f) for x, y in p.pts))
    return Triangle(
        v1=(p.v1[0] * f, p.v1[1] * f),
        v2=(p.v2[0] * f, p.v2[1] * f),
        v3=(p.v3[0] * f, p.v3[1] * f),
    )


def _scale_part(part: Part, f: float) -> Part:
    """Scale a Part's primitive about the origin, carrying op and CSG group."""
    return Part(primitive=_scale(part.primitive, f), op=part.op, group=part.group)


# ── mirror symmetry (lifted from diff_fit.py per H5 — shared by the diff
# refiner's grid-reflection tying and the R1 renderer's symmetrization; pure
# numpy/cv2, NO torch, so render.py can import without pulling the refiner) ──
ANGLE_SNAP_DEG = 1.5     # reflect/decode: angle within this of 0 (mod 90°) → exact axis-aligned type
SYM_MIN_IOU = 0.90       # mask-vs-reflected-mask IoU to accept a mirror axis
SYM_PAIR_IOU = 0.50      # part-vs-reflected-partner raster IoU to call two parts a mirrored pair


def _snap(theta: float) -> tuple[bool, int]:
    """(within ANGLE_SNAP_DEG of a multiple of 90°?, that multiple's parity —
    1 means the 90° leg, i.e. swap the in-plane extents)."""
    per = math.pi / 2.0
    k = round(theta / per)
    return abs(theta - k * per) <= math.radians(ANGLE_SNAP_DEG), k % 2


def _reflection_m(px: float, py: float, ux: float, uy: float) -> np.ndarray:
    """2×3 affine reflecting across the line through (px, py) with unit dir (ux, uy)
    (Householder R = 2uuᵀ − I; translation keeps the line invariant)."""
    r2 = 2.0 * np.array([[ux * ux, ux * uy], [ux * uy, uy * uy]], np.float64) - np.eye(2)
    p = np.array([px, py], np.float64)
    t = p - r2 @ p
    return np.hstack([r2, t.reshape(2, 1)])


def _mirror_axis(target: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Best of {vertical, horizontal, PCA major, PCA minor} axes through the centroid,
    accepted iff reflect-and-IoU ≥ SYM_MIN_IOU. Returns (point, unit_dir) or None."""
    m = cv2.moments(target, binaryImage=True)
    if m["m00"] == 0.0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    theta = 0.5 * math.atan2(2.0 * m["mu11"], m["mu20"] - m["mu02"])
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_iou = SYM_MIN_IOU
    for ux, uy in ((0.0, 1.0), (1.0, 0.0),
                   (math.cos(theta), math.sin(theta)),
                   (-math.sin(theta), math.cos(theta))):
        mat = _reflection_m(cx, cy, ux, uy)
        ref = cv2.warpAffine(target, mat, (target.shape[1], target.shape[0]),
                             flags=cv2.INTER_NEAREST, borderValue=0)
        iou = _iou_bool(target, ref)
        if iou > best_iou:
            best_iou = iou
            best = (np.array([cx, cy], np.float64), np.array([ux, uy], np.float64))
    return best


def _reflect_pt(x: float, y: float, p: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    r2 = 2.0 * np.outer(u, u) - np.eye(2)
    q = p + r2 @ (np.array([x, y], np.float64) - p)
    return float(q[0]), float(q[1])


def _reflect_part(part: Part, p: np.ndarray, u: np.ndarray) -> Part:
    """Analytic mirror of one Part across the axis (point p, unit dir u). A direction
    at angle φ reflects to 2β − φ (β = axis angle); reflected near-axis-aligned
    angles snap back to the exact axis-aligned types."""
    beta = math.degrees(math.atan2(u[1], u[0]))
    pr = part.primitive

    def _ref_angle(angle: float) -> float:
        return (2.0 * beta - angle) % 180.0

    if isinstance(pr, Circle):
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        out: object = Circle(cx=x, cy=y, r=pr.r)
    elif isinstance(pr, (Ellipse, RotatedEllipse)):
        ang0 = pr.angle if isinstance(pr, RotatedEllipse) else 0.0
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        na = _ref_angle(ang0)
        snapped, parity = _snap(math.radians(na))
        if snapped:
            rx, ry = (pr.ry, pr.rx) if parity else (pr.rx, pr.ry)
            out = Ellipse(cx=x, cy=y, rx=rx, ry=ry)
        else:
            out = RotatedEllipse(cx=x, cy=y, rx=pr.rx, ry=pr.ry, angle=na)
    elif isinstance(pr, Rectangle):
        ccx, ccy = pr.x + pr.w / 2.0, pr.y + pr.h / 2.0
        x, y = _reflect_pt(ccx, ccy, p, u)
        na = _ref_angle(0.0)
        snapped, parity = _snap(math.radians(na))
        if snapped:
            wd, ht = (pr.h, pr.w) if parity else (pr.w, pr.h)
            out = Rectangle(x=int(round(x - wd / 2.0)), y=int(round(y - ht / 2.0)),
                            w=int(round(wd)), h=int(round(ht)))
        else:
            out = RotatedRectangle(cx=x, cy=y, w=float(pr.w), h=float(pr.h), angle=na)
    elif isinstance(pr, RotatedRectangle):
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        out = RotatedRectangle(cx=x, cy=y, w=pr.w, h=pr.h, angle=_ref_angle(pr.angle))
    elif isinstance(pr, RoundedRectangle):
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        out = RoundedRectangle(cx=x, cy=y, w=pr.w, h=pr.h, angle=_ref_angle(pr.angle),
                               corner_radius=pr.corner_radius)
    elif isinstance(pr, Wedge):
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        half = (pr.a1 - pr.a0) / 2.0
        nb = (2.0 * beta - (pr.a0 + pr.a1) / 2.0) % 360.0
        a0 = (nb - half) % 360.0
        out = Wedge(cx=x, cy=y, r=pr.r, a0=a0, a1=a0 + 2.0 * half)
    elif isinstance(pr, Polygon):
        # straightened forms (R1 render tier); reverse winding so the reflected
        # polygon keeps the original orientation convention
        out = Polygon(pts=tuple(reversed([_reflect_pt(x, y, p, u) for x, y in pr.pts])))
    else:
        v1 = _reflect_pt(pr.v1[0], pr.v1[1], p, u)
        v2 = _reflect_pt(pr.v2[0], pr.v2[1], p, u)
        v3 = _reflect_pt(pr.v3[0], pr.v3[1], p, u)
        out = Triangle(v1=v1, v2=v2, v3=v3)
    return Part(out, part.op, part.group)


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
