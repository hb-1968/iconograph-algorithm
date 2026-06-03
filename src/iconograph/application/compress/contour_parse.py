"""Stage 3 D1 — contour parse: a mask's outer boundary → typed runs + corners.

Pipeline position (the contour-driven decomposer that replaces the monolithic L3 beam):
    D1 contour_parse (THIS) → D2 edge-tracing → D3 junction + residual

D1 turns a feature mask into a `ParsedContour`. The full-resolution ordered outer contour
(`_largest_contour`, CHAIN_APPROX_NONE) is reduced to dominant vertices (`approxPolyDP`); each
vertex is tagged CORNER (sharp path turn) or SMOOTH (shallow, partway along a curve); the boundary
is split at corners into maximal RUNS, each classified STRAIGHT (hugs its chord) or ARC (bows). No
fitting happens here — D1 only produces the typed boundary that D2 traces (corner/straight run → a
triangle or rect edge; arc run → an ellipse). Deterministic (H8): approxPolyDP (Ramer–Douglas–
Peucker) + angle / chord-deviation tests, no RNG.
"""

from __future__ import annotations

import cv2
import numpy as np

from iconograph.application.compress.common import (
    ContourSegment,
    Corner,
    ParsedContour,
    ParsedShape,
    _detect_holes,
    _largest_contour,
)

CONTOUR_EPS_FRAC = 0.01        # approxPolyDP epsilon as a fraction of the contour perimeter
CORNER_TURN_DEG = 40.0         # path turn ≥ this at a vertex ⇒ a sharp corner (else smooth)
STRAIGHT_DEV_FRAC = 0.12       # run is 'arc' when max chord-deviation > this·chord (or > 2 px).
#                                Sagitta ratio (max_dev/chord) of a circular arc: semicircle=0.50,
#                                60°-arc=0.13, 45°-arc=0.10 — so 0.12 keeps genuine arcs (dome, fins)
#                                while abstracting gentle bows as straight (triangle) edges.


def _turn_deg(prev_pt: np.ndarray, vtx: np.ndarray, next_pt: np.ndarray) -> float:
    """Path turn angle at vtx in degrees: 0 = collinear (straight through), 90 = right-angle
    corner, →180 = hairpin. = 180 − interior angle between the two outgoing edges."""
    v1 = prev_pt.astype(float) - vtx.astype(float)
    v2 = next_pt.astype(float) - vtx.astype(float)
    cos_t = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    interior = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    return 180.0 - interior


def _run_kind(run_pts: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> str:
    """'straight' if the dense run hugs the chord p0→p1 (max perpendicular deviation small),
    else 'arc'."""
    p0 = p0.astype(float)
    chord = p1.astype(float) - p0
    length = float(np.linalg.norm(chord)) + 1e-9
    u = chord / length
    rel = run_pts.astype(float) - p0
    perp = rel - np.outer(rel @ u, u)
    max_dev = float(np.max(np.hypot(perp[:, 0], perp[:, 1]))) if len(rel) else 0.0
    return "straight" if max_dev <= max(2.0, STRAIGHT_DEV_FRAC * length) else "arc"


def _signed_area(poly: np.ndarray) -> float:
    """Shoelace signed area of an auto-closed polygon. In y-down image coords a CW loop is
    POSITIVE (cv2 convention)."""
    x = poly[:, 0].astype(float)
    y = poly[:, 1].astype(float)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def parse_contour(mask: np.ndarray, is_hole: bool = False) -> ParsedContour | None:
    """One closed boundary (uint8 {0,255} mask) → ParsedContour with convex/concave tags, or
    None when degenerate (< 3 dominant vertices). The contour is normalized CW-in-image so the
    convexity signs are consistent (external boundary = CW = positive area)."""
    c = _largest_contour(mask)
    if c is None:
        return None
    if cv2.contourArea(c, oriented=True) < 0.0:          # normalize winding → CW (positive)
        c = c[::-1].copy()
    pts = c[:, 0, :]
    n = len(pts)
    approx = cv2.approxPolyDP(c, CONTOUR_EPS_FRAC * cv2.arcLength(c, True), True)[:, 0, :]
    m = len(approx)
    if m < 3:
        return None

    dense_idx = [int(np.argmin(np.sum((pts - v) ** 2, axis=1))) for v in approx]
    corners: list[Corner] = []
    kinds: list[str] = []
    for i in range(m):
        prev, vtx, nxt = approx[(i - 1) % m], approx[i], approx[(i + 1) % m]
        e_in = (vtx - prev).astype(float)
        e_out = (nxt - vtx).astype(float)
        cross = e_in[0] * e_out[1] - e_in[1] * e_out[0]   # >0 = convex peak (CW-normalized)
        kind = "corner" if _turn_deg(prev, vtx, nxt) >= CORNER_TURN_DEG else "smooth"
        kinds.append(kind)
        corners.append(Corner(point=(float(vtx[0]), float(vtx[1])), kind=kind, convex=bool(cross > 0)))

    # split the boundary at corners into runs (a fully-smooth contour ⇒ one arc run)
    breaks = [i for i in range(m) if kinds[i] == "corner"] or [0]
    segments: list[ContourSegment] = []
    for k in range(len(breaks)):
        i0, i1 = breaks[k], breaks[(k + 1) % len(breaks)]
        a, b = dense_idx[i0], dense_idx[i1]
        run = pts[np.arange(a, b if b > a else b + n) % n]   # modular slice across the seam
        if len(run) < 2:
            continue
        segments.append(ContourSegment(
            kind=_run_kind(run, approx[i0], approx[i1]),
            points=run.copy(),
            start=(float(approx[i0][0]), float(approx[i0][1])),
            end=(float(approx[i1][0]), float(approx[i1][1])),
            convex=bool(_signed_area(run) > 0),          # run+chord loop >0 = outward bulge
        ))
    if not segments:
        return None
    return ParsedContour(segments=tuple(segments), corners=tuple(corners), is_hole=is_hole)


def parse_shape(mask: np.ndarray) -> ParsedShape | None:
    """A feature's full boundary: outer contour + interior holes, each typed + convexity-tagged.
    Holes come from `_detect_holes` (RETR_CCOMP) and are parsed as their own contours."""
    outer = parse_contour(mask, is_hole=False)
    if outer is None:
        return None
    holes = tuple(
        h for h in (parse_contour(hm, is_hole=True) for hm in _detect_holes(mask)) if h is not None
    )
    return ParsedShape(outer=outer, holes=holes)


# --------------------------------------------------------------------------- #
# overlay (testbed / debug)
# --------------------------------------------------------------------------- #
_STRAIGHT_COL = (0x43, 0x63, 0xD8)       # blue — straight run
_CONVEX_ARC_COL = (0xE6, 0x19, 0x4B)     # red — convex arc (additive)
_CONCAVE_ARC_COL = (0x91, 0x1E, 0xB4)    # purple — concave arc (subtractive)
_HOLE_COL = (0xF5, 0xC2, 0x18)           # amber — interior hole boundary
_CONVEX_CORNER = (0x3C, 0xB4, 0x4B)      # green — convex corner (peak)
_CONCAVE_CORNER = (0x42, 0xD4, 0xF4)     # cyan — concave corner (valley)
_SMOOTH_COL = (0xF5, 0x82, 0x31)         # orange — smooth vertex


def _draw_contour(out: np.ndarray, pc: ParsedContour, ox: int, oy: int) -> None:
    for seg in pc.segments:
        if pc.is_hole:
            col = _HOLE_COL
        elif seg.kind == "straight":
            col = _STRAIGHT_COL
        else:
            col = _CONVEX_ARC_COL if seg.convex else _CONCAVE_ARC_COL
        poly = (seg.points + np.array([ox, oy])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], False, col, 2)
    for cor in pc.corners:
        p = (int(round(cor.point[0])) + ox, int(round(cor.point[1])) + oy)
        if cor.kind == "smooth":
            cv2.circle(out, p, 3, _SMOOTH_COL, -1)
        else:
            cv2.circle(out, p, 5, _CONVEX_CORNER if cor.convex else _CONCAVE_CORNER, -1)


def draw_shape_overlay(aligned: np.ndarray, shapes_by_feature) -> np.ndarray:
    """Render parsed shapes over the aligned image: straight=blue, convex arc=red, concave
    arc=purple, hole boundary=amber; convex corner=green, concave corner=cyan, smooth=orange.
    `shapes_by_feature` is a list of (ParsedShape | None, (ox, oy))."""
    out = aligned.copy()
    for shape, (ox, oy) in shapes_by_feature:
        if shape is None:
            continue
        _draw_contour(out, shape.outer, ox, oy)
        for hole in shape.holes:
            _draw_contour(out, hole, ox, oy)
    return out
