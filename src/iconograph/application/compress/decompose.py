"""Stage 3 D2+D3 — contour-driven decomposer: trace primitives + tag junctions/residual.

Pipeline position (the contour-driven decomposer that supersedes the monolithic L3 beam):
    D1 contour_parse (parse_shape) → D2+D3 decompose (THIS)

Given a feature mask, D2 traces the D1 boundary into a small set of signed primitives and D3 tags
the leftovers. The mechanism is **edge-tracing by convexity** (the locked design):

  - a CONVEX arc run → an ellipse that ADDS;  a CONCAVE arc run → an ellipse that SUBTRACTS
    (so Apple's bite / a ring's inner edge / a notch carve, generically — no hint needed);
  - each arc-ellipse fit is constrained to the feature's own bbox (AXIS_BBOX_FRAC) so a partial,
    ill-conditioned arc can't balloon into a giant circle;
  - the convex-arc ellipses are carved by half-planes from the straight runs they cross (the whale
    hull's flat top → ellipse − rect = half-disc), then accepted AS A GROUP iff the carved
    construction stays within S — spill ≤ ACCEPT_SPILL, judged AFTER the carve — so a true half-disc
    passes while an un-carvable runaway that engulfs background is demoted to a junction label;
  - the straight-edged remainder is covered by best-fit primitives; interior HOLES subtract a fitted
    primitive; what no primitive explains (coverage < ADEQUATE_COV) is the small **residual** mask.

Operates on a PADDED canvas so an enclosing ellipse can spill into background and be carved.
Deterministic (H8): cv2 fitters + numpy, fixed run order, integer-keyed carve sort, no RNG.
Output is a `Decomposition`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from iconograph.application.compress.common import (
    Circle,
    Op,
    Part,
    RingGeom,
    RotatedEllipse,
    RotatedRectangle,
    Triangle,
    _band_width,
    _boundary_iou,
    _cand_inscribed_circle,
    _centroid,
    _compound_mask,
    _compound_score,
    _detect_holes,
    _extent_ok,
    _fill_holes,
    _fit_circle,
    _fit_ellipse,
    _fit_hole,
    _fit_rectangle,
    _fit_rotated_ellipse,
    _fit_rotated_rect,
    _fit_rounded_rect,
    _fit_triangle,
    _half_plane_rect,
    _iou_bool,
    _largest_contour,
    _raster_part,
)
from iconograph.application.compress.contour_parse import CONTOUR_EPS_FRAC, parse_shape

# ── knobs ───────────────────────────────────────────────────────────────
HARD_RUN_CAP = 16        # outer runs > this ⇒ genuinely complex → whole-feature residual
MIN_ARC_FRAC = 0.05      # an arc shorter than this fraction of the perimeter → curve-label
AXIS_BBOX_FRAC = 1.8     # arc-ellipse axis cap, × the feature bbox's larger side (anti-balloon)
ACCEPT_SPILL = 0.12      # carved convex-arc construction over-reach |spill| / |construction| ≤ this (else → junctions)
GAP_MIN_IOU = 0.45       # a remainder-cover primitive must cover its component ≥ this
ADEQUATE_COV = 0.90      # construction IoU ≥ this ⇒ a clean decomposition (else residual)
NOTCH_POLY_TOL = 4.0 * CONTOUR_EPS_FRAC  # notch approxPolyDP eps = 4× D1's contour eps (one source of truth; →0.04)
REFINE_MAX_ITERS = 4     # residual-refinement passes after the first construction (bounds part count)
MIN_REFINE_GAIN = 0.005  # stop refining when a pass adds less than this much IoU
STRUCT_MIN_AREA = 0.75   # an L2-geometry annulus must cover ≥ this (area) to be preferred over the trace
STRUCT_TOL = 0.08        # …and be within this of the best min-score — then prefer the clean few-part ring
GAP_MAX_WIDTH_FRAC = 0.30  # gapped-annulus slot: max slot width / band (outer−inner) — keeps a fat blade/ideal-
#                            annulus mismatch from carving a wedge instead of a thin radial slot


@dataclass(frozen=True)
class Junction:
    """A curve/corner the decomposer labels rather than fits — the renderer overrepresents it."""
    point: tuple[float, float]
    kind: Literal["curve"]


@dataclass
class Decomposition:
    """One feature's contour-driven decomposition (in PADDED-canvas coords; full-image =
    offset by (bbox.x − pad, bbox.y − pad))."""
    parts: list[Part]                    # signed primitives: adds OR'd, then subtracts carved
    junctions: list[Junction]
    residual: np.ndarray | None          # uncovered mask (padded), or None
    coverage: float                      # area IoU — the residual/adequacy gate (familiar scale)
    pad: int
    shape: tuple[int, int]               # padded (H, W)
    n_add: int = field(default=0)
    n_sub: int = field(default=0)
    boundary_cov: float = field(default=0.0)  # boundary IoU (Cheng); arbitration ranks min(coverage, boundary_cov)
    structured: bool = field(default=False)   # True only for the L2-geometry annulus — earns the STRUCT bonus in _arbitrate


# --------------------------------------------------------------------------- #
# geometry helpers (the WRITE-NEW primitives, per the H11 audit)
# --------------------------------------------------------------------------- #
def _fit_arc_ellipse(points: np.ndarray, bbox: tuple[int, int, int, int]) -> RotatedEllipse | None:
    """Least-squares ellipse through an arc run's dense points, CONSTRAINED to the feature's own bbox
    so a partial, ill-conditioned arc can't balloon. None if too short, degenerate, or a runaway: an
    axis exceeding AXIS_BBOX_FRAC × the larger bbox side, or a centre fleeing >1 bbox-span outside the
    feature box. `bbox` is cv2.boundingRect(target); cv2.fitEllipse returns FULL axes → halve to semi."""
    if len(points) < 5:
        return None
    try:
        (cx, cy), (a_full, b_full), angle = cv2.fitEllipse(points.reshape(-1, 1, 2).astype(np.int32))
    except cv2.error:
        return None
    bx, by, bw, bh = bbox
    if a_full < 1.0 or b_full < 1.0 or max(a_full, b_full) > AXIS_BBOX_FRAC * max(bw, bh):
        return None
    if not (bx - bw <= cx <= bx + 2 * bw and by - bh <= cy <= by + 2 * bh):
        return None
    return RotatedEllipse(cx=float(cx), cy=float(cy), rx=float(a_full / 2.0),
                          ry=float(b_full / 2.0), angle=float(angle))


def _best_remainder_prim(region: np.ndarray, h: int, w: int):
    """Single best-IoU primitive covering an uncovered component (triangle for angular, rect, ellipse,
    inscribed circle). Returns the Primitive or None."""
    contour = _largest_contour(region)
    if contour is None:
        return None
    cands = [
        _fit_triangle(contour, h, w, 0, 0), _fit_rotated_rect(contour, h, w),
        _fit_rectangle(contour, h, w, 0, 0), _fit_rotated_ellipse(contour, h, w),
        _fit_ellipse(contour, h, w, 0, 0), _cand_inscribed_circle(region, h, w),
    ]
    best, best_iou = None, GAP_MIN_IOU
    for cand in cands:
        if cand is None:
            continue
        prim, fit, _area = cand
        iou = _iou_bool(fit, region)
        if iou > best_iou:
            best_iou, best = iou, prim
    return best


def _best_subtract_prim(region: np.ndarray, h: int, w: int) -> Triangle | None:
    """Bounded primitive INSCRIBED in a spill (over-coverage) component, to subtract a concave
    straight notch (e.g. the Gmail M's V). approxPolyDP reduces the component's own contour to its
    dominant vertices: 3 -> a Triangle through those points, so it lies on/inside the notch and
    cannot bulge into the shape (an enclosing fit would). Not 3 vertices -> None (only V-notches
    for now; a rectangular 4-vertex notch is a clean extension)."""
    contour = _largest_contour(region)
    if contour is None:
        return None
    approx = cv2.approxPolyDP(contour, NOTCH_POLY_TOL * cv2.arcLength(contour, True), True)
    if len(approx) != 3:
        return None
    p3 = approx.reshape(3, 2)
    return Triangle(v1=tuple(map(float, p3[0])),
                    v2=tuple(map(float, p3[1])),
                    v3=tuple(map(float, p3[2])))


def _best_overcover_prim(region: np.ndarray, h: int, w: int):
    """Best primitive to SUBTRACT over an over-coverage component — generalizes the notch-triangle to
    any over-region (round buttons, slivers). Picks the higher min(area IoU, boundary IoU) of
    {inscribed notch-triangle, best-fit primitive}; the boundary term makes a straight subtractor beat
    a curved ellipse on a FLAT edge (Adobe's flat-topped 'A'), where pure area IoU favours the ellipse.
    The caller's eats-guard rejects any that bulge into S."""
    cands = [pr for pr in (_best_subtract_prim(region, h, w), _best_remainder_prim(region, h, w))
             if pr is not None]
    if not cands:
        return None
    d = _band_width(region)
    return max(cands, key=lambda pr: min(_iou_bool(_raster_part(pr, h, w), region),
                                         _boundary_iou(_raster_part(pr, h, w), region, d)))


# --------------------------------------------------------------------------- #
# the decomposer (D2 trace + D3 junction/residual)
# --------------------------------------------------------------------------- #
def _spill_carves(adds: list[Part], target: np.ndarray, runs, h: int, w: int) -> list[Part]:
    """Carve the construction's over-reach (adds outside S) with half-planes from the straight runs,
    each oriented to put S on the inside. Candidates are accepted in deterministic descending order of
    spill removed (so a half-disc's baseline run carves first and can't be starved by a side run that
    eats shape), keeping a carve only when it removes more spill than it eats. No spill floor — whether
    carving is worthwhile at all is decided globally by the caller."""
    base = _compound_mask(adds, h, w).astype(bool)
    over = base & ~target.astype(bool)
    if not over.any():
        return []
    cen = _centroid(target)
    if cen is None:
        return []
    cen = np.asarray(cen, float)
    tgt = target.astype(bool)
    cands: list[tuple[int, float, float, RotatedRectangle, np.ndarray]] = []
    for seg in runs:
        if seg.kind != "straight":
            continue
        mid = (np.asarray(seg.start, float) + np.asarray(seg.end, float)) / 2.0
        rect = _half_plane_rect(seg.start, seg.end, mid - cen, h, w)
        cm = _raster_part(rect, h, w).astype(bool)
        cands.append((int(np.count_nonzero(over & cm)), float(rect.cx), float(rect.cy), rect, cm))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))      # most spill removed first; coord tie-break (H8)
    carves: list[Part] = []
    for _r0, _cx, _cy, rect, cm in cands:
        removed = int(np.count_nonzero(over & cm))
        eats = int(np.count_nonzero(tgt & cm))
        if removed > 0 and removed > eats:
            carves.append(Part(rect, "subtract"))
            over = over & ~cm
    return carves


def _arc_ellipse_construction(
    adds: list[Part], subs: list[Part], runs, target: np.ndarray, h: int, w: int
) -> tuple[list[Part], float]:
    """Carve the convex-arc adds with half-planes from the straight runs, then measure how much the
    carved construction (adds + those carves + the concave-arc subtracts) over-reaches S. Returns
    (carves, spill) with spill = |construction \\ S| / |construction|. Judging spill AFTER the carve is
    what separates a true half-disc (carves to ≈0 residual spill) from a runaway ellipse that engulfs
    background (high spill, un-carvable). Coverage is NOT gated here — the arcs cover only PART of S;
    the straight remainder fills the rest and the final ADEQUATE_COV gate judges the whole."""
    carves = _spill_carves(adds, target, runs, h, w)
    recon = _compound_mask(adds + carves + subs, h, w).astype(bool)
    body = int(np.count_nonzero(recon))
    spill = int(np.count_nonzero(recon & ~target.astype(bool))) / body if body else 1.0
    return carves, float(spill)


_SINGLE = frozenset(
    {"circle", "ellipse", "rectangle", "triangle", "rotated_rect", "rotated_ellipse"}
)
_ANNULAR = frozenset({"annular_band", "annular_sector"})
_HOLED = frozenset({"holed"})


def _single_fit(target: np.ndarray, hint: str, h: int, w: int) -> list[Part]:
    """A single-primitive L2 hint → that one primitive, fit directly (no edge-tracing)."""
    c = _largest_contour(target)
    if c is None:
        return []
    fit = None
    if hint == "circle":
        fit = _fit_circle(c, h, w, 0, 0)
    elif hint == "ellipse":
        fit = _fit_ellipse(c, h, w, 0, 0)
    elif hint == "rectangle":
        fit = _fit_rectangle(c, h, w, 0, 0)
    elif hint == "triangle":
        fit = _fit_triangle(c, h, w, 0, 0)
    elif hint == "rotated_rect":
        # rounded vs sharp: keep whichever scores higher by min(area, boundary) IoU — slack's pills win
        # as rounded; android's sharp rects stay sharp (rounding would shave the corners wrong).
        d = _band_width(target)
        fit = max((f for f in (_fit_rounded_rect(c, h, w), _fit_rotated_rect(c, h, w)) if f is not None),
                  key=lambda f: min(_iou_bool(f[1], target), _boundary_iou(f[1], target, d)), default=None)
    elif hint == "rotated_ellipse":
        fit = _fit_rotated_ellipse(c, h, w)
    return [Part(fit[0], "add")] if fit is not None else []


def _fit_annular(ring: RingGeom, hint: str, h: int, w: int,
                 bbox_xy: tuple[int, int], pad: int) -> list[Part]:
    """Annular hint → direct CSG from L2 ring geometry: Circle(outer) ADD − Circle(inner) SUBTRACT
    [− two radial half-plane SUBTRACTs for a sector]. The disc center is threaded from L2 (full-image)
    because a sector's center lies OUTSIDE its wedge mask; map it to the padded canvas."""
    ox, oy = bbox_xy
    cx, cy = ring.cx - ox + pad, ring.cy - oy + pad         # full-image → padded canvas
    parts = [Part(Circle(cx=cx, cy=cy, r=ring.outer_r), "add")]
    if ring.inner_r >= 1.0:
        parts.append(Part(Circle(cx=cx, cy=cy, r=ring.inner_r), "subtract"))
    if hint == "annular_sector":
        parts += _sector_cuts(cx, cy, ring.theta0, ring.theta1, h, w)
    return parts


def _sector_cuts(cx: float, cy: float, theta0: float, theta1: float,
                 h: int, w: int) -> list[Part]:
    """Two radial half-plane SUBTRACTs bounding a CONVEX wedge [theta0, theta1] (warpPolar y-down
    angles). Each cut line passes THROUGH the centre along a boundary ray, anchored AT c (c±ray) so the
    half-plane rect stays centred over the canvas even when the disc centre is near a corner. Δθ ≥ 180°
    → ring only. (A multi-blade aperture like Chrome routes through `gapped_annulus`, not here.)"""
    if (theta1 - theta0) >= math.pi - 1e-3:
        return []
    c = np.array([cx, cy], float)
    tm = 0.5 * (theta0 + theta1)
    m = np.array([math.cos(tm), math.sin(tm)])          # unit vector INTO the wedge (y-down)
    cuts: list[Part] = []
    for tb in (theta0, theta1):
        ray = np.array([math.cos(tb), math.sin(tb)])    # boundary-ray direction from the centre
        outside = -(m - float(np.dot(m, ray)) * ray)    # away from interior, ⟂ the ray
        cuts.append(Part(_half_plane_rect(c - ray, c + ray, outside, h, w), "subtract"))
    return cuts


def _fit_holed(target: np.ndarray, min_feat: int, h: int, w: int) -> list[Part]:
    """Concentric-hole construction: ADD the best primitive fit to the HOLE-FILLED body, then
    SUBTRACT a fitted primitive per detected interior hole — `Circle(outer) − Circle(inner)` for a
    disc-with-hole (YouTube Music's ring). The `holed` edge-trace alone can leave the outer arc
    demoted (adds=[] → coverage 0.0); this gives decompose_shape a base-ADD candidate to arbitrate.
    Returns [] when the filled body has no usable fit (then the trace/residual path takes over)."""
    filled = _fill_holes(target)
    outer = _best_remainder_prim(filled, h, w)
    if outer is None:
        return []
    parts = [Part(outer, "add")]
    for hole in _detect_holes(target):
        sub = _fit_hole(hole, h, w, min_feat)
        if sub is not None:
            parts.append(Part(sub, "subtract"))
    return parts


def _fit_gapped_annular(target: np.ndarray, ring: RingGeom, h: int, w: int,
                        bbox_xy: tuple[int, int], pad: int, min_feat: int) -> list[Part]:
    """Gapped-ring construction (chrome aperture): Circle(outer) ADD − Circle(inner) SUBTRACT − one
    RotatedRectangle per gap SUBTRACT. The gaps are the ideal annulus minus the actual mask (the thin
    slots between blades); each gap's minAreaRect IS the slot, naturally tangent to the inner circle
    (the gap starts there) — the user's redline. Edge-mismatch slivers are dropped by an area floor."""
    ox, oy = bbox_xy
    cx, cy = ring.cx - ox + pad, ring.cy - oy + pad      # full-image → padded canvas
    parts = [Part(Circle(cx=cx, cy=cy, r=ring.outer_r), "add")]
    if ring.inner_r >= 1.0:
        parts.append(Part(Circle(cx=cx, cy=cy, r=ring.inner_r), "subtract"))
    annulus = _compound_mask(parts, h, w)
    gaps = ((annulus.astype(bool) & ~target.astype(bool)).astype(np.uint8)) * 255
    n, lbl, stats, _cent = cv2.connectedComponentsWithStats(gaps, connectivity=8)
    band = max(1.0, ring.outer_r - ring.inner_r)
    cap = GAP_MAX_WIDTH_FRAC * band
    area_floor = max(min_feat * min_feat, int(0.004 * float(np.count_nonzero(annulus))))
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < area_floor:  # thin edge-mismatch sliver, not a real gap
            continue
        c = _largest_contour(((lbl == i).astype(np.uint8)) * 255)
        if c is None:
            continue
        # The gap's own minAreaRect (its true position + tilt + length), with the SHORT side clamped to
        # a thin slot — so a fat blade/ideal-annulus mismatch carves a thin tangent slot, not a wedge.
        (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(c)
        rw, rh = (min(rw, cap), rh) if rw <= rh else (rw, min(rh, cap))
        if min(rw, rh) < 1.0:
            continue
        parts.append(Part(RotatedRectangle(cx=float(rcx), cy=float(rcy), w=float(rw), h=float(rh),
                                           angle=float(ang)), "subtract"))
    return parts


def _decomp_from_parts(parts: list[Part], target: np.ndarray, min_feat: int,
                       pad: int, h: int, w: int, structured: bool = False) -> Decomposition:
    """Score a finished parts list into a Decomposition (cov, residual vs ADEQUATE_COV, counts).
    `structured=True` tags the L2-geometry annulus so _arbitrate can prefer it (chrome)."""
    a_iou, b_iou, _ = _compound_score(parts, target, h, w)
    recon = _compound_mask(parts, h, w)
    residual = ((target.astype(bool) & ~recon.astype(bool)).astype(np.uint8)) * 255
    res = residual if (a_iou < ADEQUATE_COV and _extent_ok(residual, min_feat)) else None
    n_add = sum(1 for p in parts if p.op == "add")
    return Decomposition(parts=parts, junctions=[], residual=res, coverage=float(a_iou),
                         pad=pad, shape=(h, w), n_add=n_add, n_sub=len(parts) - n_add,
                         boundary_cov=float(b_iou), structured=structured)


def _cover_components(source: np.ndarray, min_feat: int, h: int, w: int,
                      fit, op: Op, out: list[Part], guard=None) -> bool:
    """Segment `source` (uint8 {0,255}) into 8-connected components and, per min_feat-sized
    component, fit one primitive via `fit(comp, h, w)`, appending `Part(prim, op)` to `out` when it
    passes `guard` — a predicate on the candidate's rasterized bool mask (None ⇒ always append).
    Returns True if anything was appended. The shared body of the trace's four cover/carve passes
    (H5: steps 3, 3b, and the two refine passes were one block copied four times)."""
    changed = False
    n, lbl, _stats, _cent = cv2.connectedComponentsWithStats(source, connectivity=8)
    for i in range(1, n):
        comp = ((lbl == i).astype(np.uint8)) * 255
        if not _extent_ok(comp, min_feat):
            continue
        prim = fit(comp, h, w)
        if prim is None:
            continue
        if guard is None or guard(_raster_part(prim, h, w).astype(bool)):
            out.append(Part(prim, op))
            changed = True
    return changed


def _trace_decompose(target: np.ndarray, min_feat: int, pad: int, h: int, w: int,
                     use_arcs: bool = True) -> Decomposition:
    """Edge-trace-by-convexity (the D2/D3 default): arcs -> add/subtract ellipses, spill-carve,
    straight-remainder cover, concave-notch subtract, holes. Returns a Decomposition."""
    def _whole_residual(cov: float, junctions: list[Junction]) -> Decomposition:
        return Decomposition(parts=[], junctions=junctions, residual=target.copy(),
                             coverage=cov, pad=pad, shape=(h, w), boundary_cov=cov)

    parsed = parse_shape(target)
    if parsed is None:
        return _whole_residual(0.0, [])
    runs = parsed.outer.segments
    if len(runs) > HARD_RUN_CAP:                       # too many runs to arc-trace — remainder + refine
        use_arcs = False

    perim = sum(len(s.points) for s in runs) + 1e-9
    bbox = cv2.boundingRect(target)                    # the feature's own box — scales the axis cap
    adds: list[Part] = []
    subs: list[Part] = []
    junctions: list[Junction] = []
    carves: list[Part] = []

    if use_arcs:
        # 1. arcs → convex-add / concave-subtract ellipses (substantial + feature-constrained fit)
        arc_adds: list[Part] = []
        arc_segs: list = []
        for seg in runs:
            if seg.kind != "arc":
                continue
            if len(seg.points) / perim < MIN_ARC_FRAC:
                junctions.append(Junction(point=seg.start, kind="curve"))
                continue
            ell = _fit_arc_ellipse(seg.points, bbox)
            if ell is None:
                junctions.append(Junction(point=seg.start, kind="curve"))
                continue
            if seg.convex:
                arc_adds.append(Part(ell, "add"))
                arc_segs.append(seg)
            else:
                subs.append(Part(ell, "subtract"))

        # 2. accept the convex-arc adds AS A GROUP iff their carved construction beats the straight-only
        #    baseline (the no-arc trace competes in decompose_shape); else demote them to junctions.
        if arc_adds:
            arc_carves, arc_spill = _arc_ellipse_construction(arc_adds, subs, runs, target, h, w)
            if arc_spill <= ACCEPT_SPILL:               # contained (after carve) → keep; balloon → drop
                adds.extend(arc_adds)
                carves = arc_carves
            else:
                junctions.extend(Junction(point=s.start, kind="curve") for s in arc_segs)

    # 3. cover the straight-edged remainder (per uncovered component → best single primitive)
    recon = _compound_mask(adds + carves, h, w)
    under = ((target.astype(bool) & ~recon.astype(bool)).astype(np.uint8)) * 255
    _cover_components(under, min_feat, h, w, _best_remainder_prim, "add", adds)

    # 3b. carve concave straight notches: subtract a bounded primitive per spill (over-cover) comp
    recon = _compound_mask(adds + carves, h, w)
    spill = ((recon.astype(bool) & ~target.astype(bool)).astype(np.uint8)) * 255
    tgt = target.astype(bool)
    spill_b = spill.astype(bool)
    _cover_components(
        spill, min_feat, h, w, _best_subtract_prim, "subtract", subs,
        guard=lambda cm: int(np.count_nonzero(spill_b & cm)) > 0
        and int(np.count_nonzero(spill_b & cm)) > int(np.count_nonzero(tgt & cm)),
    )

    # 4. holes → subtract a fitted primitive
    for hole in parsed.holes:
        hm = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(hm, [np.vstack([s.points for s in hole.segments]).reshape(-1, 1, 2).astype(np.int32)], 255)
        sub = _fit_hole(hm, h, w, min_feat)
        if sub is not None:
            subs.append(Part(sub, "subtract"))

    # 5. refine: iterate contour-fit add/subtract over the residual until it converges
    tgt = target.astype(bool)
    for _ in range(REFINE_MAX_ITERS):
        recon = _compound_mask(adds + carves + subs, h, w)
        cov0 = _iou_bool(recon, target)
        under = ((tgt & ~recon.astype(bool)).astype(np.uint8)) * 255
        over = ((recon.astype(bool) & ~tgt).astype(np.uint8)) * 255
        ub = under.astype(bool)
        ob = over.astype(bool)
        # under-coverage → best-fit ADD (cover > spill); over-coverage → eats-guarded SUBTRACT.
        # Two statements (not `c1 or c2` in one) so BOTH passes always run — no short-circuit (H8).
        c1 = _cover_components(under, min_feat, h, w, _best_remainder_prim, "add", adds,
                               guard=lambda pm: int(np.count_nonzero(ub & pm)) > int(np.count_nonzero(~tgt & pm)))
        c2 = _cover_components(over, min_feat, h, w, _best_overcover_prim, "subtract", subs,
                               guard=lambda cm: int(np.count_nonzero(ob & cm)) > int(np.count_nonzero(tgt & cm)))
        if not (c1 or c2):
            break
        if _iou_bool(_compound_mask(adds + carves + subs, h, w), target) - cov0 < MIN_REFINE_GAIN:
            break

    parts = adds + carves + subs
    if not adds:
        return _whole_residual(_compound_score(parts, target, h, w)[0] if parts else 0.0, junctions)

    a_iou, b_iou, _ = _compound_score(parts, target, h, w)
    recon2 = _compound_mask(parts, h, w)
    residual = ((target.astype(bool) & ~recon2.astype(bool)).astype(np.uint8)) * 255
    res = residual if (a_iou < ADEQUATE_COV and _extent_ok(residual, min_feat)) else None
    return Decomposition(parts=parts, junctions=junctions, residual=res, coverage=float(a_iou),
                         pad=pad, shape=(h, w), n_add=len(adds), n_sub=len(carves) + len(subs),
                         boundary_cov=float(b_iou))


def _arbitrate(candidates: list[Decomposition]) -> Decomposition:
    """Pick the winning construction by min(area, boundary) IoU, with the structured-annulus
    preference folded into one key (was: max(min-score) + a post-hoc chrome override). A `structured`
    candidate — only the L2-geometry annulus ever sets the flag — that covers ≥ STRUCT_MIN_AREA earns
    a +STRUCT_TOL score bonus and wins ties, so the clean few-part ring beats a higher-fidelity
    many-part trace it is within STRUCT_TOL of. Provably identical to the old two-step because at most
    one candidate is structured and the bonus encodes `s(annulus) ≥ s(best) − STRUCT_TOL`; non-bonus
    ties fall to list order (H8), as the old `max` did."""
    def key(d: Decomposition) -> tuple[float, bool]:
        s = min(d.coverage, d.boundary_cov)
        bonus = d.structured and d.coverage >= STRUCT_MIN_AREA
        return (s + STRUCT_TOL, True) if bonus else (s, False)
    return max(candidates, key=key)


def decompose_shape(mask: np.ndarray, min_feat: int, hint: str | None = None,
                    ring: "RingGeom | None" = None,
                    bbox_xy: tuple[int, int] | None = None) -> Decomposition:
    """Decompose one feature mask into signed primitives + junctions + residual. A single-primitive
    hint is fit directly; an annular hint fits a ring/sector from L2 geometry and COMPETES with the
    edge-trace path (higher coverage wins); else edge-traced by convexity."""
    th, tw = mask.shape[:2]
    pad = max(min_feat, max(th, tw) // 2)
    h, w = th + 2 * pad, tw + 2 * pad
    target = np.zeros((h, w), dtype=np.uint8)
    target[pad:pad + th, pad:pad + tw] = mask

    if hint in _SINGLE:                                # single-primitive hint -> direct fit (preempts)
        parts = _single_fit(target, hint, h, w)
        if parts:
            return _decomp_from_parts(parts, target, min_feat, pad, h, w)

    cand_annular: Decomposition | None = None
    cand_holed: Decomposition | None = None
    if hint in _ANNULAR and ring is not None:          # ring/sector candidate from L2 geometry
        parts = _fit_annular(ring, hint, h, w, bbox_xy or (0, 0), pad)
        if parts:
            cand_annular = _decomp_from_parts(parts, target, min_feat, pad, h, w, structured=True)
    if hint in _HOLED or hint in _ANNULAR:             # concentric-hole ring: fill-fit ADD − holes SUB.
        parts = _fit_holed(target, min_feat, h, w)     # Competes for annular hints too, so a poor ring
        if parts:                                      # geometry can't beat a good fill-fit (instagram).
            cand_holed = _decomp_from_parts(parts, target, min_feat, pad, h, w)
    cand_gapped: Decomposition | None = None
    if hint == "gapped_annulus" and ring is not None:  # multi-blade aperture (chrome) → annulus − gap rects
        parts = _fit_gapped_annular(target, ring, h, w, bbox_xy or (0, 0), pad, min_feat)
        if parts:
            cand_gapped = _decomp_from_parts(parts, target, min_feat, pad, h, w, structured=True)

    traced = _trace_decompose(target, min_feat, pad, h, w)
    candidates = [traced]
    for c in (cand_annular, cand_holed, cand_gapped):
        if c is not None:
            candidates.append(c)
    if traced.coverage < ADEQUATE_COV:                         # sub-adequate → arbitrate alternatives
        candidates.append(_trace_decompose(target, min_feat, pad, h, w, use_arcs=False))  # straight-only
        prim = _best_remainder_prim(target, h, w)              # whole-shape single primitive
        if prim is not None:
            candidates.append(_decomp_from_parts([Part(prim, "add")], target, min_feat, pad, h, w))
    # contour-faithful arbitration: rank by min(area, boundary) — a high-area but boundary-wrong
    # construction (curved fit on a flat edge; local ellipse over a global ring) loses to the
    # boundary-aligned one — with the structured-annulus preference folded into the key (see
    # _arbitrate). coverage (area) still drives the residual/adequacy gates above.
    return _arbitrate(candidates)


# --------------------------------------------------------------------------- #
# overlay (testbed / debug)
# --------------------------------------------------------------------------- #
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0x3C, 0xB4, 0x4B), (0x43, 0x63, 0xD8), (0xF5, 0x82, 0x31),
    (0x91, 0x1E, 0xB4), (0x46, 0x99, 0x90), (0x9A, 0x63, 0x24),
)
_RESIDUAL_COL = (0xE6, 0x19, 0x4B)       # red — residual (low-res tier)
_JUNCTION_COL = (0x42, 0xD4, 0xF4)       # cyan — curve-junction label


def draw_decomposition_overlay(aligned: np.ndarray, decomps) -> np.ndarray:
    """Render each feature's decomposition over the aligned image: the reconstruction filled per
    feature, each ADD primitive's boundary outlined in a palette colour, residual in red, junctions
    as cyan dots. `decomps` is a list of (Decomposition, (bx, by)) — feature bbox in full-image."""
    out = aligned.copy()
    H, W = aligned.shape[:2]
    ci = 0
    for d, (bx, by) in decomps:
        ph, pw = d.shape
        ox, oy = bx - d.pad, by - d.pad                # padded-crop → full-image
        if d.parts:
            recon = _compound_mask(d.parts, ph, pw)
            ys, xs = np.where(recon > 0)
            fy, fx = ys + oy, xs + ox
            keep = (fy >= 0) & (fy < H) & (fx >= 0) & (fx < W)
            blended = (0.45 * out[fy[keep], fx[keep]] + 0.55 * np.array(_PALETTE[ci % len(_PALETTE)]))
            out[fy[keep], fx[keep]] = blended.astype(np.uint8)
            for p in d.parts:
                if p.op != "add":
                    continue
                pm = _raster_part(p.primitive, ph, pw)
                cnts, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    cv2.polylines(out, [c + np.array([ox, oy])], True, _PALETTE[ci % len(_PALETTE)], 2)
        if d.residual is not None:
            ys, xs = np.where(d.residual > 0)
            fy, fx = ys + oy, xs + ox
            keep = (fy >= 0) & (fy < H) & (fx >= 0) & (fx < W)
            out[fy[keep], fx[keep]] = _RESIDUAL_COL
        for j in d.junctions:
            cv2.circle(out, (int(round(j.point[0])) + ox, int(round(j.point[1])) + oy), 5, _JUNCTION_COL, -1)
        ci += 1
    return out
