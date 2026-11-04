"""Map primitives to area-matched forms along a curvature ladder.

Degree 0 is rectilinear; degree 3 retains the true curve. Preserve operation
and group labels. Rotated forms snap angles at lower degrees; wedges snap
start and sweep together so arcs crossing 360 degrees remain valid.

Rasterization uses outline-inclusive fillPoly with round-half-up vertices.
Polygon edges are inset by roughly half a pixel to match continuous extents.
This module supplies geometry; the renderer scores candidate feasibility.
"""

from __future__ import annotations

import math

import numpy as np

from iconograph.application.compress.common import (
    Circle,
    Ellipse,
    Part,
    Polygon,
    Primitive,
    Rectangle,
    RotatedEllipse,
    RotatedRectangle,
    RoundedRectangle,
    Triangle,
    Wedge,
)

DEG_TRUE = 3                      # ladder top: render the true curve

# Area-match circumradius factors: N-gon area = (N/2)·sin(2π/N)·R² ≡ πr².
SQUARE_SIDE_F = math.sqrt(math.pi)                          # side = r·√π  (IoU vs disc ≈ 0.83)
OCT_F = math.sqrt(math.pi / (2.0 * math.sqrt(2.0)))         # ≈ 1.0539
GON16_F = math.sqrt(math.pi / (8.0 * math.sin(math.pi / 8.0)))  # ≈ 1.0130

TRI_SNAP_TOL = (15.0, 13.5)       # D0 / D1 edge-angle snap tolerance, degrees
#   round 6: D1 was 8.0, which left DEAD ZONES — the farthest any angle sits
#   from Θ is 13.285° (mid of the 0↔26.57° gap), so edges in (8°, 18.57°),
#   (34.57°, 37°), (53°, 55.43°), (71.43°, 82°) kept their TRUE angle while
#   the record said deg=1 (gmail's V ~72-82°, vscode's chevron hypotenuses
#   rendered as irregular 1-2px bump trains). 13.5 covers the full grid.


# polygon builders


def _ngon(cx: float, cy: float, rx: float, ry: float, n: int,
          offset_deg: float, rot_deg: float = 0.0) -> tuple[tuple[float, float], ...]:
    """Affine N-gon: unit-circle vertices scaled by (rx, ry), rotated, translated.
    cv2 y-down screen coords (0° = +x, +sin = downward); endpoint=False keeps the
    vertex set deterministic and non-repeating (H8)."""
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) + math.radians(offset_deg)
    xs = rx * np.cos(ang)
    ys = ry * np.sin(ang)
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    wx = cx + xs * c - ys * s
    wy = cy + xs * s + ys * c
    return tuple((float(x), float(y)) for x, y in zip(wx, wy))


def _box_pts(cx: float, cy: float, w: float, h: float,
             angle_deg: float) -> tuple[tuple[float, float], ...]:
    """The 4 corners of a centered w×h box rotated by angle_deg (float boxPoints)."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    hw, hh = w / 2.0, h / 2.0
    local = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    return tuple((cx + x * c - y * s, cy + x * s + y * c) for x, y in local)


def _chamfer_pts(cx: float, cy: float, w: float, h: float, angle_deg: float,
                 r: float) -> tuple[tuple[float, float], ...]:
    """Chamfered rect: each corner arc replaced by the chord between its two arc
    endpoints (corner inset r along both adjacent edges). The chord lies strictly
    inside the arc, so the chamfer never grows past the rounded original."""
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    hw, hh = w / 2.0, h / 2.0
    local = (
        (-hw + r, -hh), (hw - r, -hh), (hw, -hh + r), (hw, hh - r),
        (hw - r, hh), (-hw + r, hh), (-hw, hh - r), (-hw, -hh + r),
    )
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    return tuple((cx + x * c - y * s, cy + x * s + y * c) for x, y in local)


def _snap_angle(angle: float, grid: float) -> float:
    """Nearest multiple of `grid` degrees."""
    return round(angle / grid) * grid


# Digital slope grid: atan(0), atan(1/2), atan(1), atan(2), 90°, … mod 180 —
# every member rasters as a periodic digital line with step runs ≥ 2 px (or
# pure axis); 45° is in the grid but J2 prices its 1-px teeth beyond K45.
SLOPE_GRID_DEG: tuple[float, ...] = (
    0.0, 26.565051177077994, 45.0, 63.43494882292201, 90.0,
    116.56505117707799, 135.0, 153.43494882292202,
)
# Round 6 — the run-≥2 sub-grid: Θ minus the 45° class. Every member rasters
# with step runs ≥ 2 px (or pure axis) — NO 1-px teeth at any length. Built
# by filtering SLOPE_GRID_DEG so both grids share the exact float members
# (snap idempotence: D1's snap_slope re-snap of a run-2 base is identity).
SLOPE_GRID_RUN2: tuple[float, ...] = tuple(
    m for m in SLOPE_GRID_DEG if abs(m % 90.0 - 45.0) > 1e-9)
RUN2_TRI_TOL = 18.5   # max distance to SLOPE_GRID_RUN2 is 18.435° (at 45°);
#                       this tol makes the run-2 triangle snap total — the
#                       fid floor at the call site arbitrates affordability
K45 = 3.0   # max 45° edge length (in DIAGONAL MOVES) that is a free chamfer


def _snap_to_grid(angle: float, grid: tuple[float, ...]) -> float:
    """Nearest grid member, mod 180 (wrap-aware); deterministic ties toward
    the smaller grid angle."""
    a = angle % 180.0
    best = grid[0]
    d0 = abs(a - grid[0])
    best_d = min(d0, 180.0 - d0)
    for m in grid[1:]:
        d = abs(a - m)
        d = min(d, 180.0 - d)
        if d < best_d - 1e-12:
            best_d = d
            best = m
    return best


def snap_slope(angle: float) -> float:
    """Nearest SLOPE_GRID_DEG member, mod 180 (wrap-aware); deterministic
    ties toward the smaller grid angle."""
    return _snap_to_grid(angle, SLOPE_GRID_DEG)


def snap_slope_run2(angle: float) -> float:
    """Round 6 — nearest RUN-≥2 slope (Θ minus the 45° class): the
    'geometric extension' target for diagonals whose nearest Θ member is
    45° and would carry 1-px teeth. Idempotent on run-2 members (the D1
    ladder re-snap is identity by shared grid floats)."""
    return _snap_to_grid(angle, SLOPE_GRID_RUN2)


def _rrect_pts(cx: float, cy: float, w: float, h: float, angle_deg: float,
               r: float, step_deg: float = 6.0) -> tuple[tuple[float, float], ...]:
    """Dense rounded-rect outline (float corner arcs at step_deg) — the DEG_TRUE
    form. Replaces _rounded_rect_poly's int-rounded raster path: at quarter-px
    polygon precision the arc placement noise that caps boundary IoU disappears."""
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    hw, hh = w / 2.0, h / 2.0
    corners = (((hw - r, -hh + r), -90.0, 0.0), ((hw - r, hh - r), 0.0, 90.0),
               ((-hw + r, hh - r), 90.0, 180.0), ((-hw + r, -hh + r), 180.0, 270.0))
    pts = []
    for (icx, icy), a0, a1 in corners:
        n = max(1, math.ceil((a1 - a0) / step_deg))
        for k in range(n + 1):
            th = math.radians(a0 + (a1 - a0) * k / n)
            pts.append((icx + r * math.cos(th), icy + r * math.sin(th)))
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    return tuple((cx + x * c - y * s, cy + x * s + y * c) for x, y in pts)


_CHAM_ALPHA = 1.0 / (2.0 + math.sqrt(2.0))   # regular-octagon corner-cut ratio c/s
_CHAM_SIDE_F = math.sqrt(math.pi / (1.0 - 2.0 * _CHAM_ALPHA * _CHAM_ALPHA))
#                       side = r·this when the cut is proportional (≈ 1.947 —
#                       the regular octagon's across-flats width, area-matched)


def _circle_ladder(cx: float, cy: float, rx: float, ry: float, degree: int,
                   rot_deg: float = 0.0) -> tuple[tuple[float, float], ...]:
    """Shared circle/ellipse straightened forms (area-matched per axis, then
    raster-inset: −1 per box dim / −0.5 of apothem for N-gons). D1 is the
    CHAMFERED SQUARE (round 3): at small r it reproduces the regular octagon's
    proportions; at larger r the 45° cuts cap at K45 px so the form never
    carries a long 1-px-tooth staircase — the doctrine-legal octagon."""
    if degree == 0:
        return _box_pts(cx, cy, max(0.5, rx * SQUARE_SIDE_F - 1.0),
                        max(0.5, ry * SQUARE_SIDE_F - 1.0), rot_deg)
    if degree == 1:
        sx = rx * _CHAM_SIDE_F
        sy = ry * _CHAM_SIDE_F
        c = min(K45, _CHAM_ALPHA * min(sx, sy))
        if c >= K45:                      # cuts capped: re-match the area with
            s2 = 2.0 * c * c              # fixed-size corner cuts
            sx = math.sqrt(math.pi * rx * rx + s2 * (rx / max(ry, 1e-9)))
            sy = math.sqrt(math.pi * ry * ry + s2 * (ry / max(rx, 1e-9)))
        return _chamfer_pts(cx, cy, max(0.5, sx - 1.0), max(0.5, sy - 1.0),
                            rot_deg, c)
    inset = 0.5 / math.cos(math.pi / 16.0)
    return _ngon(cx, cy, max(0.3, rx * GON16_F - inset),
                 max(0.3, ry * GON16_F - inset), 16, 11.25, rot_deg)


def _wedge_fan(p: Wedge, degree: int) -> tuple[tuple[float, float], ...]:
    """Apex + chord-fan vertices. (a0, sweep) snap to the degree's grid as a PAIR;
    per-segment radius r' = r·√(φ/sinφ) area-matches each chord to its arc segment.
    Raster-inset: arc radius −0.5, apex nudged 0.5 along the sector bisector."""
    sweep = p.a1 - p.a0
    if degree == 0:
        a0 = _snap_angle(p.a0, 45.0)
        sweep = max(45.0, _snap_angle(sweep, 45.0))
        n = max(1, math.ceil(sweep / 120.0))
    elif degree == 1:
        a0 = _snap_angle(p.a0, 22.5)
        sweep = max(22.5, _snap_angle(sweep, 22.5))
        n = max(2, math.ceil(sweep / 60.0))
    elif degree == 2:
        a0 = p.a0
        n = max(4, math.ceil(sweep / 30.0))
    else:                                 # DEG_TRUE: dense fan (~6° chords)
        a0 = p.a0
        n = max(8, math.ceil(sweep / 6.0))
    seg = math.radians(sweep) / n
    r = p.r * math.sqrt(seg / math.sin(seg)) if 0.0 < seg < math.pi else p.r
    r = max(0.5, r - 0.5)
    mid = math.radians(a0 + sweep / 2.0)
    pts = [(p.cx + 0.5 * math.cos(mid), p.cy + 0.5 * math.sin(mid))]
    for k in range(n + 1):
        th = math.radians(a0) + k * seg
        pts.append((p.cx + r * math.cos(th), p.cy + r * math.sin(th)))
    return tuple(pts)


def _snap_triangle(p: Triangle, tol: float,
                   grid: tuple[float, ...] | None = None
                   ) -> tuple[tuple[float, float], ...] | None:
    """Snap edge directions within tol° of a Θ member (snap_slope, mod 180) —
    or of a `grid` member when one is given (round 6: SLOPE_GRID_RUN2);
    vertices are re-derived as intersections of consecutive edge lines through
    the original edge midpoints. None when lines go near-parallel or the area
    collapses — caller falls back to the original triangle."""
    v = [np.asarray(p.v1, float), np.asarray(p.v2, float), np.asarray(p.v3, float)]
    centroid = (v[0] + v[1] + v[2]) / 3.0
    lines = []                                   # (point-on-line, unit direction)
    for i in range(3):
        a, b = v[i], v[(i + 1) % 3]
        d = b - a
        theta = math.degrees(math.atan2(d[1], d[0])) % 180.0
        m = (snap_slope(theta) if grid is None   # nearest digital-slope member
             else _snap_to_grid(theta, grid))
        diff = theta - m
        diff = diff - 180.0 if diff > 90.0 else (diff + 180.0 if diff < -90.0 else diff)
        ang = m if abs(diff) <= tol else theta
        u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        mid = (a + b) / 2.0
        nrm = np.array([-u[1], u[0]])            # raster-inset: edge line 0.5 inward
        if float(np.dot(nrm, centroid - mid)) < 0:
            nrm = -nrm
        lines.append((mid + 0.5 * nrm, u))
    out = []
    for i in range(3):
        (p0, u0), (p1, u1) = lines[i - 1], lines[i]   # vertex i = lines (i-1) ∩ (i)
        det = u0[0] * u1[1] - u0[1] * u1[0]
        if abs(det) < 1e-9:
            return None
        t = ((p1[0] - p0[0]) * u1[1] - (p1[1] - p0[1]) * u1[0]) / det
        q = p0 + t * u0
        out.append((float(q[0]), float(q[1])))
    a2 = abs((out[1][0] - out[0][0]) * (out[2][1] - out[0][1])
             - (out[2][0] - out[0][0]) * (out[1][1] - out[0][1]))
    orig2 = abs((v[1][0] - v[0][0]) * (v[2][1] - v[0][1])
                - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]))
    if orig2 > 0 and not (0.25 <= a2 / orig2 <= 4.0):
        return None
    return tuple(out)


def snap_triangle_run2(p: Triangle) -> tuple[tuple[float, float], ...] | None:
    """Round 6 — the triangle 'geometric extension': every edge snapped to
    the nearest RUN-≥2 slope (total at RUN2_TRI_TOL — max grid distance is
    18.435°). None on degenerate re-intersection; the caller skips the base."""
    return _snap_triangle(p, RUN2_TRI_TOL, grid=SLOPE_GRID_RUN2)


def _snap_polygon(verts: tuple[tuple[float, float], ...], tol: float,
                  grid: tuple[float, ...] | None = None
                  ) -> tuple[tuple[float, float], ...] | None:
    """N-gon generalization of _snap_triangle (kept SEPARATE from the byte-locked
    triangle path): snap each edge direction within tol° of a Θ member (snap_slope)
    or a `grid` member, then re-derive every vertex as the intersection of its two
    adjacent (raster-inset 0.5 px inward) edge lines. None on a near-parallel
    re-intersection or an area collapse/blowup (0.25..4.0× guard). Used by the
    cubify triangle-label path for flat-top (4-vertex) silhouettes."""
    v = [np.asarray(p, float) for p in verts]
    n = len(v)
    if n < 3:
        return None
    centroid = sum(v) / n

    def _area2(poly) -> float:                       # |shoelace| (winding-agnostic)
        s = 0.0
        for i in range(len(poly)):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % len(poly)]
            s += x0 * y1 - x1 * y0
        return abs(s)

    lines = []                                       # (point-on-line, unit direction)
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        d = b - a
        theta = math.degrees(math.atan2(d[1], d[0])) % 180.0
        m = (snap_slope(theta) if grid is None else _snap_to_grid(theta, grid))
        diff = theta - m
        diff = diff - 180.0 if diff > 90.0 else (diff + 180.0 if diff < -90.0 else diff)
        ang = m if abs(diff) <= tol else theta
        u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        mid = (a + b) / 2.0
        nrm = np.array([-u[1], u[0]])                # raster-inset: edge line 0.5 inward
        if float(np.dot(nrm, centroid - mid)) < 0:
            nrm = -nrm
        lines.append((mid + 0.5 * nrm, u))
    out = []
    for i in range(n):
        (p0, u0), (p1, u1) = lines[i - 1], lines[i]  # vertex i = lines (i-1) ∩ (i)
        det = u0[0] * u1[1] - u0[1] * u1[0]
        if abs(det) < 1e-9:
            return None
        t = ((p1[0] - p0[0]) * u1[1] - (p1[1] - p0[1]) * u1[0]) / det
        q = p0 + t * u0
        out.append((float(q[0]), float(q[1])))
    orig2 = _area2([tuple(x) for x in v])
    if orig2 > 0 and not (0.25 <= _area2(out) / orig2 <= 4.0):
        return None
    return tuple(out)


def snap_polygon_run2(verts: tuple[tuple[float, float], ...]
                      ) -> tuple[tuple[float, float], ...] | None:
    """Run-≥2 slope snap of an arbitrary closed polygon (the N-gon analogue of
    snap_triangle_run2 — a flat-top triangle is a 4-gon). None on degenerate
    re-intersection."""
    return _snap_polygon(verts, RUN2_TRI_TOL, grid=SLOPE_GRID_RUN2)


def stroke_axis_runs(x0: float, y0: float, x1: float, y1: float,
                     width: float, form: str,
                     band_axis: str | None = None,
                     band_at: float | None = None,
                     run_center: float | None = None
                     ) -> tuple[tuple[float, float], ...] | None:
    """Round 5 — the L-CORNER doctrine (user-drawn ideal): a thin diagonal
    stroke at tiny T reduces to AXIS-ALIGNED runs joined by single L corners,
    the coarsest digital line — "diagonals reduced into L corners for minimal
    jank and heavy 1px stepping". Forms (endpoint-relative, invariant under
    the normalizations below):
      "l"  — dominant run at P0's transverse level, corner + stub at P1;
      "lr" — stub at P0, corner + dominant run at P1's transverse level;
      "z"  — transverse stubs at BOTH ends, dominant run through the
             midpoint (a crossing pair renders as the drawn X: center band
             + four stubs; the local-pair flip supplies the symmetry);
      "z2" — DOUBLE-CORNER arms (user-locked for near-square strokes whose
             "z" stubs would grow past recognizability): three dominant
             runs (tip tread, center band, tip tread; breaks at 25 %/75 %
             of the span) joined by transverse risers — the coarsest
             3-run digital line. The four corners of a crossing pair stay
             DISTINCT; whether the assembled axis-run form reads as an X
             at near-square proportions is an OPEN strip call (grader
             finding: thin_x@16 reads closer to an H-frame — README
             round-5 open item (e)).
    band_axis/band_at: crossing strokes share ONE band — the drawn X's
    center band lies on the PAIR axis, not on each stroke's own midline.
    band_axis ("x"|"y") forces the run direction (overriding the stroke's
    own dominance); band_at fixes the band's transverse coordinate.
    Returns rectilinear Polygon pts (6 for L, 8 for Z, 12 for Z2), raster-
    inset per the integer-corner fill convention (box spans use the −1
    half-extent, stroke end caps inset 0.5), or None when the reduction
    degenerates (either span ≤ 1.5×width — that territory is D0's box)."""
    dx, dy = x1 - x0, y1 - y0
    horiz = (band_axis == "x") if band_axis else (abs(dx) >= abs(dy))
    if not horiz:                         # normalize: dominant horizontal
        pts = stroke_axis_runs(y0, x0, y1, x1, width, form,
                               band_axis="x" if band_axis else None,
                               band_at=band_at, run_center=run_center)
        return None if pts is None else tuple((y, x) for x, y in pts)
    if x0 > x1:                           # normalize: dominant ascending
        pts = stroke_axis_runs(x1, y1, x0, y0, width,
                               {"l": "lr", "lr": "l", "z": "z",
                                "z2": "z2"}[form],
                               band_axis="x" if band_axis else None,
                               band_at=band_at, run_center=run_center)
        return pts
    if dy < 0:                            # normalize: transverse descending
        pts = stroke_axis_runs(x0, -y0, x1, -y1, width, form,
                               band_axis="x" if band_axis else None,
                               band_at=None if band_at is None else -band_at,
                               run_center=run_center)
        return None if pts is None else tuple((x, -y) for x, y in pts)
    if abs(dy) <= 1.5 * width or (x1 - x0) <= 1.5 * width:
        return None
    h = max(0.0, (width - 1.0) / 2.0)
    if form == "z2":
        # all derived coordinates round to the (target-space) integer grid:
        # a fractional break straddles a pixel boundary and fattens its run
        yc = round((y0 + y1) / 2.0 if band_at is None else band_at)
        # quartile breaks symmetric about the CROSSING center when given —
        # stroke-relative breaks land the left/right risers of a crossing
        # pair one row apart (pinwheel chirality, not the drawn mirror X)
        rc = (x0 + x1) / 2.0 if run_center is None else run_center
        xq1 = float(round(rc - 0.25 * (x1 - x0)))
        xq2 = float(round(rc + 0.25 * (x1 - x0)))
        if xq2 - xq1 <= 2.0 * h + 1.0:    # treads would fuse — fall back
            form = "z"
        else:
            xs, xe = x0 + 0.5, x1 - 0.5
            return ((xs, y0 - h), (xq1 + h, y0 - h),
                    (xq1 + h, yc - h), (xq2 + h, yc - h),
                    (xq2 + h, y1 - h), (xe, y1 - h),
                    (xe, y1 + h), (xq2 - h, y1 + h),
                    (xq2 - h, yc + h), (xq1 - h, yc + h),
                    (xq1 - h, y0 + h), (xs, y0 + h))
    if form == "z":
        yc = round((y0 + y1) / 2.0 if band_at is None else band_at)
        return ((x0 - h, y0 + 0.5), (x0 + h, y0 + 0.5),
                (x0 + h, yc - h), (x1 + h, yc - h),
                (x1 + h, y1 - 0.5), (x1 - h, y1 - 0.5),
                (x1 - h, yc + h), (x0 - h, yc + h))
    if form == "l":                       # run at y0, stub down at x1
        return ((x0 + 0.5, y0 - h), (x1 + h, y0 - h),
                (x1 + h, y1 - 0.5), (x1 - h, y1 - 0.5),
                (x1 - h, y0 + h), (x0 + 0.5, y0 + h))
    if form == "lr":                      # stub down at x0, run at y1
        return ((x0 - h, y0 + 0.5), (x0 + h, y0 + 0.5),
                (x0 + h, y1 - h), (x1 - 0.5, y1 - h),
                (x1 - 0.5, y1 + h), (x0 - h, y1 + h))
    raise ValueError(f"unknown stroke_axis_runs form {form!r}")


def star_polygon(cx: float, cy: float, n: int, r_outer: float, r_inner: float,
                 phase_deg: float,
                 spoke_halfwidth_deg: float | None = None
                 ) -> tuple[tuple[float, float], ...]:
    """N-point star: points at phase_deg + k·(360/n) on r_outer. Two flavors:
    classic (spoke_halfwidth_deg None) — 2n vertices, valleys HALFWAY between
    points on r_inner; ASTERISK — 3n vertices, the valleys hug each spike at
    ±spoke_halfwidth_deg, for THIN radiating blades (claude's ~7° spokes vs the
    classic star's fat half-pitch triangles, which over-fill the gaps to area
    IoU ≈ 0.44). cv2 y-down angles, raster-inset 0.5 on both radii. Already
    straight — straighten() passes Polygon through; size/snap tweaks come free
    via the Polygon branches of the transforms. Phase snapping is the caller's
    job (modulo the pitch 360/n, never the absolute 45° grid). Float vertices —
    Polygon rasters at quarter-px."""
    if n < 3:
        raise ValueError(f"star needs n >= 3 points, got {n}")
    outer = max(0.5, r_outer - 0.5)
    inner = max(0.5, r_inner - 0.5)
    if inner >= outer:
        raise ValueError(f"degenerate star: r_inner {r_inner} >= r_outer {r_outer}")
    pitch = 360.0 / n
    pts = []
    if spoke_halfwidth_deg is None:
        for k in range(2 * n):
            th = math.radians(phase_deg + k * pitch / 2.0)
            r = outer if k % 2 == 0 else inner
            pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
        return tuple(pts)
    hw = min(max(0.5, spoke_halfwidth_deg), pitch / 2.0 - 0.5)
    for k in range(n):
        center = phase_deg + k * pitch
        for ang, r in ((center - hw, inner), (center, outer), (center + hw, inner)):
            th = math.radians(ang)
            pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return tuple(pts)


def asterisk_polygon(cx: float, cy: float,
                     spikes: tuple[tuple[float, float], ...], r_inner: float,
                     width: float) -> tuple[tuple[float, float], ...]:
    """NON-UNIFORM asterisk: one 4-vertex CONSTANT-WIDTH quad spike per
    (angle_deg, r_tip) entry, valleys at r_inner. The radial-family collapse
    form for ORGANIC sunbursts whose blades sit on no uniform pitch (claude
    deviates ±8° from any 30° comb — a uniform star can never align, best
    sweep area IoU 0.32) AND are constant-width capsules (a triangular spike
    tapering to a point covers only the blade's inner sliver — measured 60%
    of the silhouette missed). The flat tip chord is the natural D0
    straightening of a capsule's rounded end. spikes must be sorted by angle;
    raster-inset 0.5 on radii and width."""
    if len(spikes) < 3:
        raise ValueError(f"asterisk needs >= 3 spikes, got {len(spikes)}")
    inner = max(0.5, r_inner - 0.5)
    hw_lin = max(0.5, width / 2.0 - 0.5)
    pts = []
    for ang, r_tip in spikes:
        outer = max(inner + 0.5, r_tip - 0.5)
        hw_in = math.degrees(math.atan2(hw_lin, inner))
        hw_tip = math.degrees(math.atan2(hw_lin, outer))
        for a, r in ((ang - hw_in, inner), (ang - hw_tip, outer),
                     (ang + hw_tip, outer), (ang + hw_in, inner)):
            th = math.radians(a)
            pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return tuple(pts)


# the ladder


def straighten(p: Primitive, degree: int) -> Primitive:
    """One primitive's straightened form at `degree` (0..DEG_TRUE). DEG_TRUE keeps
    the true curve but RASTER-INSETS it (radius −0.5 / box dims −1): cv2's
    center+radius drawing is boundary-inclusive (a radius-r circle covers 2r+1
    px), the same outward bloat the polygon forms inset for — without it every
    true-curve candidate runs ~1 px fat against the supersampled reference and
    boundary IoU sinks. Stage-3 fitters never call straighten, so their raster
    behavior is untouched. Failed snaps return the primitive unchanged."""
    d = max(0, min(DEG_TRUE, int(degree)))
    if isinstance(p, Polygon):
        return p
    if isinstance(p, Circle):
        if d == DEG_TRUE:
            return Polygon(pts=_ngon(p.cx, p.cy, max(0.5, p.r - 0.5),
                                     max(0.5, p.r - 0.5), 64, 0.0))
        return Polygon(pts=_circle_ladder(p.cx, p.cy, p.r, p.r, d))
    if isinstance(p, Ellipse):
        if d == DEG_TRUE:
            return Polygon(pts=_ngon(p.cx, p.cy, max(0.5, p.rx - 0.5),
                                     max(0.5, p.ry - 0.5), 64, 0.0))
        return Polygon(pts=_circle_ladder(p.cx, p.cy, p.rx, p.ry, d))
    if isinstance(p, RotatedEllipse):
        if d == DEG_TRUE:
            return Polygon(pts=_ngon(p.cx, p.cy, max(0.5, p.rx - 0.5),
                                     max(0.5, p.ry - 0.5), 64, 0.0, p.angle))
        rot = (_snap_angle(p.angle, 90.0) if d == 0
               else snap_slope(p.angle) if d == 1 else p.angle)
        return Polygon(pts=_circle_ladder(p.cx, p.cy, p.rx, p.ry, d, rot))
    if isinstance(p, Rectangle):
        if d == DEG_TRUE:
            return p                              # numpy-slice raster is exact
        # already straight below — integer-corner polygon: outline-inclusive
        # fill covers x..x+w−1 exactly, matching the numpy-slice raster
        x0, y0 = float(p.x), float(p.y)
        x1, y1 = float(p.x + p.w - 1), float(p.y + p.h - 1)
        return Polygon(pts=((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    if isinstance(p, RotatedRectangle):
        rot = (_snap_angle(p.angle, 90.0) if d == 0
               else snap_slope(p.angle) if d == 1 else p.angle)
        # d ≥ 2 keeps the original angle; the polygon path replaces the
        # boxPoints int-truncation raster, so DEG_TRUE goes through it too
        return Polygon(pts=_box_pts(p.cx, p.cy, max(0.5, p.w - 1.0),
                                    max(0.5, p.h - 1.0), rot))
    if isinstance(p, RoundedRectangle):
        if d == 0:
            return Polygon(pts=_box_pts(p.cx, p.cy, max(0.5, p.w - 1.0),
                                        max(0.5, p.h - 1.0),
                                        _snap_angle(p.angle, 90.0)))
        if d == 1:
            # chamfer leg capped at K45 (round 3): one short cut per corner;
            # larger radii escalate to D2 via the budget rather than carrying
            # a long 45° staircase. Round 7 — a THIRD cap by the inset short
            # dimension: when the corner radius approaches half the short
            # side (a capsule/pill — figma's top pill, cr ≈ h/2), an
            # unclamped leg eats the whole vertical side and the D1 form
            # degenerates to a hexagon/triangle, not a rounded bar. Capping
            # the leg at (short-1)/2 leaves a ≥1-px straight run on the short
            # axis, so the pill keeps a flat middle and reads round+clean at
            # 16 (the represented-geometry ideal). Bites ONLY for small short
            # dims (< ~7 px); larger RRects keep min(cr, K45) unchanged.
            wi, hi = max(0.5, p.w - 1.0), max(0.5, p.h - 1.0)
            leg = min(p.corner_radius, K45, max(1.0, (min(wi, hi) - 1.0) / 2.0))
            return Polygon(pts=_chamfer_pts(p.cx, p.cy, wi, hi,
                                            snap_slope(p.angle), leg))
        # D2/D3 = true rrect as a dense float polygon, raster-inset (int-px
        # radius snap is the grid-fit's size snap, not geometry)
        return Polygon(pts=_rrect_pts(p.cx, p.cy, max(0.5, p.w - 1.0),
                                      max(0.5, p.h - 1.0), p.angle,
                                      max(0.0, p.corner_radius - 0.5)))
    if isinstance(p, Triangle):
        if d >= 2:
            # no snapping, but still raster-inset (replaces _raster_part's
            # int32-truncation of float vertices with the exact polygon path)
            inset = _snap_triangle(p, 0.0)
            return Polygon(pts=inset) if inset is not None else p
        snapped = _snap_triangle(p, TRI_SNAP_TOL[d])
        return Polygon(pts=snapped) if snapped is not None else p
    if isinstance(p, Wedge):
        return Polygon(pts=_wedge_fan(p, d))
    return p


def straighten_part(part: Part, degree: int) -> Part:
    """Straighten a Part's primitive, carrying op and CSG group."""
    return Part(primitive=straighten(part.primitive, degree), op=part.op, group=part.group)


# transforms the grid-fit search tweaks candidates with


def prim_center(p: Primitive) -> tuple[float, float]:
    """Continuous-space center (pixel-center convention; pixel i spans i±0.5)."""
    if isinstance(p, (Circle, Ellipse, RotatedEllipse, RotatedRectangle,
                      RoundedRectangle, Wedge)):
        return float(p.cx), float(p.cy)
    if isinstance(p, Rectangle):
        return p.x + p.w / 2.0 - 0.5, p.y + p.h / 2.0 - 0.5
    if isinstance(p, Triangle):
        return ((p.v1[0] + p.v2[0] + p.v3[0]) / 3.0,
                (p.v1[1] + p.v2[1] + p.v3[1]) / 3.0)
    xs = [q[0] for q in p.pts]
    ys = [q[1] for q in p.pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def translate_prim(p: Primitive, dx: float, dy: float) -> Primitive:
    """Float translate. Rectangle (int fields) rounds the offset — the search only
    applies sub-pixel moves to polygonized forms, where they are exact."""
    if isinstance(p, Circle):
        return Circle(cx=p.cx + dx, cy=p.cy + dy, r=p.r)
    if isinstance(p, Ellipse):
        return Ellipse(cx=p.cx + dx, cy=p.cy + dy, rx=p.rx, ry=p.ry)
    if isinstance(p, RotatedEllipse):
        return RotatedEllipse(cx=p.cx + dx, cy=p.cy + dy, rx=p.rx, ry=p.ry, angle=p.angle)
    if isinstance(p, Rectangle):
        return Rectangle(x=p.x + int(round(dx)), y=p.y + int(round(dy)), w=p.w, h=p.h)
    if isinstance(p, RotatedRectangle):
        return RotatedRectangle(cx=p.cx + dx, cy=p.cy + dy, w=p.w, h=p.h, angle=p.angle)
    if isinstance(p, RoundedRectangle):
        return RoundedRectangle(cx=p.cx + dx, cy=p.cy + dy, w=p.w, h=p.h,
                                angle=p.angle, corner_radius=p.corner_radius)
    if isinstance(p, Wedge):
        return Wedge(cx=p.cx + dx, cy=p.cy + dy, r=p.r, a0=p.a0, a1=p.a1)
    if isinstance(p, Polygon):
        return Polygon(pts=tuple((x + dx, y + dy) for x, y in p.pts))
    return Triangle(v1=(p.v1[0] + dx, p.v1[1] + dy),
                    v2=(p.v2[0] + dx, p.v2[1] + dy),
                    v3=(p.v3[0] + dx, p.v3[1] + dy))


def scale_about_center(p: Primitive, f: float) -> Primitive:
    """Uniform size scale about the primitive's own center. Rectangle with f≠1
    goes float via RotatedRectangle(angle=0) — the degree ladder polygonizes it
    anyway, so no int-rounding precision is lost."""
    if f == 1.0:
        return p
    if isinstance(p, Circle):
        return Circle(cx=p.cx, cy=p.cy, r=p.r * f)
    if isinstance(p, Ellipse):
        return Ellipse(cx=p.cx, cy=p.cy, rx=p.rx * f, ry=p.ry * f)
    if isinstance(p, RotatedEllipse):
        return RotatedEllipse(cx=p.cx, cy=p.cy, rx=p.rx * f, ry=p.ry * f, angle=p.angle)
    if isinstance(p, Rectangle):
        cx, cy = prim_center(p)
        return RotatedRectangle(cx=cx, cy=cy, w=p.w * f, h=p.h * f, angle=0.0)
    if isinstance(p, RotatedRectangle):
        return RotatedRectangle(cx=p.cx, cy=p.cy, w=p.w * f, h=p.h * f, angle=p.angle)
    if isinstance(p, RoundedRectangle):
        return RoundedRectangle(cx=p.cx, cy=p.cy, w=p.w * f, h=p.h * f,
                                angle=p.angle, corner_radius=p.corner_radius * f)
    if isinstance(p, Wedge):
        return Wedge(cx=p.cx, cy=p.cy, r=p.r * f, a0=p.a0, a1=p.a1)
    cx, cy = prim_center(p)
    if isinstance(p, Polygon):
        return Polygon(pts=tuple((cx + (x - cx) * f, cy + (y - cy) * f) for x, y in p.pts))
    return Triangle(
        v1=(cx + (p.v1[0] - cx) * f, cy + (p.v1[1] - cy) * f),
        v2=(cx + (p.v2[0] - cx) * f, cy + (p.v2[1] - cy) * f),
        v3=(cx + (p.v3[0] - cx) * f, cy + (p.v3[1] - cy) * f),
    )


def scale_about_center_xy(p: Primitive, fx: float, fy: float) -> Primitive:
    """Anisotropic size scale about the primitive's center — canvas axes for
    axis-aligned kinds, LOCAL frame for rotated ones (w×fx, h×fy). Wedges have
    no anisotropic form; they fall back to uniform fx. Lets the grid-fit snap
    each axis extent to the pixel grid independently (a 23.3×10 capsule whose
    reference renders 23×11)."""
    if fx == fy:
        return scale_about_center(p, fx)
    if isinstance(p, Circle):
        return Ellipse(cx=p.cx, cy=p.cy, rx=p.r * fx, ry=p.r * fy)
    if isinstance(p, Ellipse):
        return Ellipse(cx=p.cx, cy=p.cy, rx=p.rx * fx, ry=p.ry * fy)
    if isinstance(p, RotatedEllipse):
        return RotatedEllipse(cx=p.cx, cy=p.cy, rx=p.rx * fx, ry=p.ry * fy,
                              angle=p.angle)
    if isinstance(p, Rectangle):
        cx, cy = prim_center(p)
        return RotatedRectangle(cx=cx, cy=cy, w=p.w * fx, h=p.h * fy, angle=0.0)
    if isinstance(p, RotatedRectangle):
        return RotatedRectangle(cx=p.cx, cy=p.cy, w=p.w * fx, h=p.h * fy,
                                angle=p.angle)
    if isinstance(p, RoundedRectangle):
        return RoundedRectangle(cx=p.cx, cy=p.cy, w=p.w * fx, h=p.h * fy,
                                angle=p.angle,
                                corner_radius=p.corner_radius * min(fx, fy))
    if isinstance(p, Wedge):
        return scale_about_center(p, fx)
    cx, cy = prim_center(p)
    if isinstance(p, Polygon):
        return Polygon(pts=tuple((cx + (x - cx) * fx, cy + (y - cy) * fy)
                                 for x, y in p.pts))
    return Triangle(
        v1=(cx + (p.v1[0] - cx) * fx, cy + (p.v1[1] - cy) * fy),
        v2=(cx + (p.v2[0] - cx) * fx, cy + (p.v2[1] - cy) * fy),
        v3=(cx + (p.v3[0] - cx) * fx, cy + (p.v3[1] - cy) * fy),
    )


def prim_bbox(p: Primitive) -> tuple[float, float, float, float]:
    """Continuous-space (x0, y0, x1, y1) bounds. Analytic where cheap; Wedge
    samples its fan at 5° (deterministic). RoundedRectangle uses the full box
    (corner rounding only shrinks it — fine for grid snapping)."""
    if isinstance(p, Circle):
        return p.cx - p.r, p.cy - p.r, p.cx + p.r, p.cy + p.r
    if isinstance(p, Ellipse):
        return p.cx - p.rx, p.cy - p.ry, p.cx + p.rx, p.cy + p.ry
    if isinstance(p, RotatedEllipse):
        th = math.radians(p.angle)
        hx = math.hypot(p.rx * math.cos(th), p.ry * math.sin(th))
        hy = math.hypot(p.rx * math.sin(th), p.ry * math.cos(th))
        return p.cx - hx, p.cy - hy, p.cx + hx, p.cy + hy
    if isinstance(p, Rectangle):
        return p.x - 0.5, p.y - 0.5, p.x + p.w - 0.5, p.y + p.h - 0.5
    if isinstance(p, (RotatedRectangle, RoundedRectangle)):
        pts = _box_pts(p.cx, p.cy, p.w, p.h, p.angle)
    elif isinstance(p, Wedge):
        sweep = p.a1 - p.a0
        n = max(2, math.ceil(sweep / 5.0))
        pts = [(p.cx, p.cy)] + [
            (p.cx + p.r * math.cos(math.radians(p.a0 + k * sweep / n)),
             p.cy + p.r * math.sin(math.radians(p.a0 + k * sweep / n)))
            for k in range(n + 1)
        ]
    elif isinstance(p, Triangle):
        pts = (p.v1, p.v2, p.v3)
    else:
        pts = p.pts
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    return min(xs), min(ys), max(xs), max(ys)
