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

import hashlib
import math
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from iconograph.application.compress.common import (
    Circle,
    Ellipse,
    Op,
    ParsedShape,
    Part,
    Primitive,
    RingGeom,
    ROUNDRECT_MAX_RADIUS_FRAC,
    RotatedEllipse,
    RotatedRectangle,
    RoundedRectangle,
    Triangle,
    _band_width,
    _boundary_iou,
    _cand_inscribed_circle,
    _cand_inscribed_rectangle,
    _centroid,
    _compound_mask,
    _compound_score,
    _detect_holes,
    _extent_ok,
    _fill_holes,
    _fit_arc_circle,
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
    _offset_part,
    _post_prune,
    _raster_part,
)
from iconograph.application.compress.contour_parse import CONTOUR_EPS_FRAC, parse_shape
from scipy.ndimage import convolve
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from skimage.morphology import medial_axis

# ── knobs ───────────────────────────────────────────────────────────────
HARD_RUN_CAP = 16        # outer runs > this ⇒ genuinely complex → whole-feature residual
MIN_ARC_FRAC = 0.02      # relative arc floor (was 0.05 — a cliff that silently deleted letterform
#                          counter arcs at 0.024-0.049 and crown scallops; probe-calibrated)
ARC_MIN_RUN_MULT = 2.0   # absolute arc floor: run pixel-length ≥ this × min_feat (renderability —
#                          an arc too short to seed a min-feature primitive is labeled, not fit)
AXIS_BBOX_FRAC = 1.8     # arc-ellipse axis cap, × the feature bbox's larger side (anti-balloon)
ARC_CIRCLE_RMSE_PX = 1.5    # Kasa circle acceptance: absolute floor (raster quantization + D1 endpoint slop)
ARC_CIRCLE_RMSE_FRAC = 0.02  # …or this × r — D1's eps (1% of perimeter) scales endpoint error with shape size
ACCEPT_SPILL = 0.12      # carved convex-arc construction over-reach |spill| / |construction| ≤ this (else → junctions)
GAP_MIN_IOU = 0.45       # a remainder-cover primitive must cover its component ≥ this
ADEQUATE_COV = 0.90      # construction IoU ≥ this ⇒ a clean decomposition (else residual)
HOLE_SINGLE_KEEP = 0.80  # single-prim hole fit ≥ this → keep the cheap path (simple holes byte-identical)
RECURSE_HOLE_MIN_AREA = 0.85  # recursive hole carve acceptance: area IoU on the hole crop (gates are
RECURSE_HOLE_MIN_BND = 0.65   # area-driven per the boundary-scoring pivot) + a boundary floor — a thin
#                               letterform band runs structurally lower on boundary IoU than a body does
RECURSE_HOLE_MAX_PARTS = 12  # parsimony cap on one hole's construction (an oversized winner is
#                              _post_prune'd first — the cap rejects only genuinely irreducible sprawl)
TRACE_PRUNE_PARTS = 12   # a trace construction larger than this gets the hard _post_prune cleanup
#                          (the refine loop can pile up redundant triangle tails — homebrew hit 40)
STROKE_ANG_TOL_DEG = 12.0   # parallel-run stroke pairing: max angle difference between the two edges
STROKE_MAX_WIDTH_FRAC = 0.45  # stroke width ≤ this × the feature bbox's larger side (else not a stroke)
STROKE_FILL_MIN = 0.75      # the band between paired runs must be ≥ this fraction foreground
STROKE_MIN_OVERLAP = 0.35   # paired runs' projections must overlap ≥ this × the shorter run's length
STROKE_MAX = 8              # stroke rects per construction (greedy, longest-overlap first)
DIFF_FIRE_THRESH = 0.88  # best deterministic min(area, boundary) below this → fire the diff_fit refiner
DIFF_TOP_K = 3           # warm-start seeds taken from the ranked candidate pool
DIFF_MAX_SEED_PARTS = 24  # a seed larger than this is not worth refining (matches diff_fit cap)
NOTCH_POLY_TOL = 4.0 * CONTOUR_EPS_FRAC  # notch approxPolyDP eps = 4× D1's contour eps (one source of truth; →0.04)
REFINE_MAX_ITERS = 4     # residual-refinement passes after the first construction (bounds part count)
MIN_REFINE_GAIN = 0.005  # stop refining when a pass adds less than this much IoU
STRUCT_MIN_AREA = 0.75   # an L2-geometry annulus must cover ≥ this (area) to be preferred over the trace
STRUCT_TOL = 0.08        # …and be within this of the best min-score — then prefer the clean few-part ring
PARSIMONY_TOL = 0.03     # arbitration parsimony: a clean (fewer-part, ≥2) construction may win if its
#                          min(area,boundary) is within this of the best — generalizes the annulus-only
#                          STRUCT bonus by part count (firefox#0 collapses 9→1 primitive at 0.05, so 0.03
#                          + the parts≥2 guard keep it safe; the structured key itself is unchanged)
GAP_MAX_WIDTH_FRAC = 0.30  # gapped-annulus slot: max slot width / band (outer−inner) — keeps a fat blade/ideal-
#                            annulus mismatch from carving a wedge instead of a thin radial slot
INNER_CONTACT_FRAC = 0.15  # gapped-annulus: a malformed gap re-anchors its slot at the hub via pixels within inner_r + this×band
UNION_MAX_PARTS = 12       # skeleton-union discovery: max branch primitives kept (largest-area first)
UNION_SPILL_FRAC = 0.20    # …each branch primitive may spill ≤ this fraction of its own area outside the shape
UNION_MIN_COVER_FRAC = 0.005  # …drop a branch whose primitive adds < this fraction of the shape (the tiny tail)
UNION_PEAK_DIST_MULT = 2.0  # circle seeds at full-shape DT peaks ≥ this × min_feat apart (one per lobe, not per ridge px)
UNION_ENCLOSE_EPS = 0.003  # boundary-aware union pass: stop when no candidate (incl. enclosing rect/tri/…)
#                            raises construction min(area,boundary) by this — the union returns the better of
#                            the area-greedy and boundary-aware constructions (slack's corner → enclosing rect)
SKEL_JUNCTION_NC = 4       # medial-axis pixel with ≥ this 3×3 neighbour-count (self+≥3) is a junction → cut here
SKEL_MIN_BRANCH_PX = 3     # ignore skeleton fragments shorter than this (spurs / single-pixel endpoints)
SKEL_DOT_ASPECT = 1.2      # branch with length/(2·radius) below this is a round DOT (Circle), else a capsule
SKEL_MERGE_COS = 0.906     # two branches whose unit directions align to within ~25° (|cos| ≥ this) may merge
SKEL_MERGE_GAP_FRAC = 0.05  # …if their nearest endpoints are within this × (h+w) (a junction-cut seam) — rejoins a
#                             collinear stroke (the top bar's halves) that junction-cutting split, so the pill is whole
PILL_RAD_TOL = 0.25        # circle-run→pill: grouped circles' radii must match to within this fraction of the larger
PILL_OVERLAP = 0.90        # …and consecutive centres lie within this × (r_i + r_j) — i.e. genuinely overlap, so a
#                            row of overlapping equal circles (figma's top bar) fuses to one capsule but spaced dots stay
PILL_LINE_TOL = 0.60       # …and every centre lies within this × median-r of the run's fitted axis (else not a pill)
PILL_MIN_RAD_FRAC = 0.45   # …and only circles ≥ this × the union's largest circle fuse — a real bar (figma's r≈240
#                            top run), never the small corner-fill circles that would form a junk side-capsule
PILL_KEEP_TOL = 0.04       # accept the pill-merged union if its min-IoU is within this of the unmerged — the clean
#                            pill structure is worth a small fidelity give (consistent renders); larger drops reject
ORBIT_SYM_ORDERS = range(2, 7)  # candidate n-fold rotational orders the orbit detector tests (react = 6)
ORBIT_SYM_MIN = 0.55       # min rfft harmonic-energy score to ATTEMPT an orbit construction (react ~0.67; controls ≤0.46).
#                            Only a compute gate — min(area,boundary) arbitration + the contiguity guard are the real safety.
ORBIT_R_BINS = 256         # warpPolar radial bins for the orbit symmetry profile (= L2's R_BINS)
ORBIT_THETA_BINS = 720     # warpPolar angular bins (= L2's THETA_BINS; 0.5°/row)
SPUR_MIN_ORDER = 6         # radial-spur detector: reject order < this — not a radial burst
SPUR_MAX_ORDER = 24        # …reject order > this (texture / noise); also sets find_peaks `distance`
SPUR_PEAK_PROM_FRAC = 0.25  # spike-peak prominence as × the outer-radius profile's full swing
SPUR_GAP_CV_MAX = 0.40     # spacing-CV gate: claude 0.24 fires, firefox 0.87 → [] (the regression guard;
#                            both are hint=sunburst, so uniformity is what tells a real burst apart)
SPUR_INNER_FRAC = 0.55     # blade inner end as × the hub inradius (reaches into the hub → contiguous union)
#                            (the warpPolar resolution is reused from ORBIT_R_BINS / ORBIT_THETA_BINS — H5)


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


def _fit_arc_prim(points: np.ndarray, bbox: tuple[int, int, int, int]) -> Primitive | None:
    """Circle-FIRST arc model: a Kasa least-squares circle when the run is genuinely
    circular (rmse ≤ max(ARC_CIRCLE_RMSE_PX, ARC_CIRCLE_RMSE_FRAC·r), diameter under
    the same AXIS_BBOX_FRAC anti-balloon cap, centre within the same escape box the
    ellipse fit uses); else the bbox-constrained ellipse. Why circle-first:
    cv2.fitEllipse is ill-conditioned on PARTIAL arcs (a half-circle arch run balloons
    to an eccentric runaway that the spill gate then rejects, collapsing the trace),
    while the 3-dof circle is well-conditioned on the same points — and a contour-
    anchored circle is the primitive the render tier idealizes best. Genuinely
    elliptical arcs (docker's hull, electron's orbits) fail the RMSE gate and keep
    their ellipse."""
    bx, by, bw, bh = bbox
    fit = _fit_arc_circle(points)
    if fit is not None:
        circ, rmse = fit
        if (rmse <= max(ARC_CIRCLE_RMSE_PX, ARC_CIRCLE_RMSE_FRAC * circ.r)
                and 2.0 * circ.r <= AXIS_BBOX_FRAC * max(bw, bh)
                and bx - bw <= circ.cx <= bx + 2 * bw
                and by - bh <= circ.cy <= by + 2 * bh):
            return circ
    return _fit_arc_ellipse(points, bbox)


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


def _concave_notch_triangles(parsed: ParsedShape) -> list[Triangle]:
    """Exact subtract-triangles for the outer contour's concave STRAIGHT V-notches, read from D1.
    A concave CORNER flanked by two straight runs is a V (Gmail's flap + the two bottom-leg notches);
    its carve triangle is (left-wall far end, valley, right-wall far end) = (prev.start, cur.start,
    cur.end). Exact and deterministic — no approxPolyDP guesswork (the reactive _best_subtract_prim
    mis-locates the same V). Corners are POINT-matched to the shared vertex (index alignment breaks
    once a smooth vertex exists). The caller eats-guards each triangle, so over-emitting is harmless."""
    segs = parsed.outer.segments
    cors = parsed.outer.corners
    n = len(segs)
    if n < 2:
        return []
    tris: list[Triangle] = []
    for i in range(n):
        cur = segs[i]
        prev = segs[(i - 1) % n]
        if cur.kind != "straight" or prev.kind != "straight":
            continue
        v = cur.start                                    # shared vertex (== prev.end) = candidate valley
        if not any(c.kind == "corner" and not c.convex
                   and abs(c.point[0] - v[0]) <= 2.0 and abs(c.point[1] - v[1]) <= 2.0
                   for c in cors):                       # concave sharp corner here? (point-match, not index)
            continue
        tris.append(Triangle(v1=prev.start, v2=v, v3=cur.end))   # left-wall top, valley, right-wall top
    return tris


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
    adds: list[Part], subs: list[Part], runs, target: np.ndarray, min_feat: int,
    h: int, w: int
) -> tuple[list[Part], float]:
    """Carve the convex-arc adds with half-planes from the straight runs, then carve the REMAINING
    interior spill with eats-guarded primitive subtracts (the refine loop's over-coverage machinery —
    an arch circle's lower half dips into a concave leg gap that no straight-run half-plane can reach
    without eating shape; without this the whole arc group is thrown away for spill the pipeline
    could trivially shave), then measure how much the carved construction (adds + carves + the
    concave-arc subtracts) still over-reaches S. Returns (carves, spill) with
    spill = |construction \\ S| / |construction|. Judging spill AFTER the carves is what separates a
    true half-disc / arch (carves to ≈0 residual spill) from a runaway that engulfs background.
    Coverage is NOT gated here — the arcs cover only PART of S; the straight remainder fills the
    rest and the final ADEQUATE_COV gate judges the whole."""
    carves = _spill_carves(adds, target, runs, h, w)
    tgt = target.astype(bool)
    recon_b = _compound_mask(adds + carves + subs, h, w).astype(bool)
    over_b = recon_b & ~tgt
    over = over_b.astype(np.uint8) * 255
    extra: list[Part] = []
    _cover_components(over, min_feat, h, w, _best_overcover_prim, "subtract", extra,
                      guard=lambda cm: int(np.count_nonzero(over_b & cm))
                      > int(np.count_nonzero(tgt & cm)))
    carves = carves + extra
    recon = _compound_mask(adds + carves + subs, h, w).astype(bool)
    body = int(np.count_nonzero(recon))
    spill = int(np.count_nonzero(recon & ~tgt)) / body if body else 1.0
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
        # rounded vs sharp vs FULL capsule: keep whichever scores higher by min(area, boundary) IoU.
        # _fit_rounded_rect's own corner estimate under-rounds true capsules (android's antennae:
        # cr 11.7 vs the capsule's 14.3 → boundary 0.34 vs 0.65), so the forced capsule —
        # corner_radius = min(w, h)/2 on the sharp fit's geometry — is probed as a third variant.
        d = _band_width(target)
        cands = [_fit_rounded_rect(c, h, w), _fit_rotated_rect(c, h, w)]
        rot = cands[1]
        if rot is not None:
            p = rot[0]
            cap = RoundedRectangle(cx=p.cx, cy=p.cy, w=p.w, h=p.h, angle=p.angle,
                                   corner_radius=min(p.w, p.h) / 2.0)
            cm = _raster_part(cap, h, w)
            cands.append((cap, cm, float(np.count_nonzero(cm))))
        fit = max((f for f in cands if f is not None),
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


def _fit_rr_annular(target: np.ndarray, h: int, w: int, min_feat: int) -> list[Part]:
    """Rounded-rect annulus: RoundedRectangle(outer) ADD − RoundedRectangle(inner) SUBTRACT, fit
    straight from the mask's filled body + its largest interior hole — the squircle ring the circle
    `_fit_annular` rounds off (Instagram). BOTH shells must fit as near-square rounded rects
    (min_aspect=1.0); it returns [] on a circular ring (a disc's degenerate corner_radius ≥
    ROUNDRECT_MAX_RADIUS_FRAC → let the Circle annulus own it), a sector (no interior hole / the
    concave wedge fails the rounded-rect solidity gate), or any non-squircle. Tagged structured like
    `cand_annular` and competes by min(area, boundary) — the circle still wins where it is the true
    shell (it is appended first, so it takes exact ties)."""
    filled = _fill_holes(target)
    oc = _largest_contour(filled)
    if oc is None:
        return []
    outer = _fit_rounded_rect(oc, h, w, min_aspect=1.0)             # near-square allowed (squircle ≈ 1:1)
    if outer is None:                                              # not a (near-convex) rounded rect
        return []
    o = outer[0]
    if o.corner_radius >= ROUNDRECT_MAX_RADIUS_FRAC * min(o.w, o.h):  # ≈ a circle → circle annulus owns it
        return []
    holes = _detect_holes(target)
    if not holes:                                                # solid (e.g. a sector wedge) → not a ring
        return []
    ic = _largest_contour(max(holes, key=lambda m: int(np.count_nonzero(m))))
    if ic is None:
        return []
    inner = _fit_rounded_rect(ic, h, w, min_aspect=1.0)
    if inner is None:                                            # inner boundary isn't a rounded rect → defer
        return []
    return [Part(o, "add"), Part(inner[0], "subtract")]


def _orbit_radial(target: np.ndarray, cx: float, cy: float, theta: float,
                  r_max: float) -> tuple[float, float, float]:
    """Along the ray from (cx, cy) at angle theta (image coords, y-down — the same frame as
    RotatedEllipse.angle): the OUTERMOST contiguous foreground run's (inner_r, outer_r) and the central
    run radius (a solid nucleus). (0, 0, 0) when the ray hits no foreground."""
    dx, dy = math.cos(theta), math.sin(theta)
    th, tw = target.shape
    on = np.zeros(int(r_max) + 1, dtype=bool)
    for t in range(on.size):
        x, y = int(round(cx + t * dx)), int(round(cy + t * dy))
        if 0 <= x < tw and 0 <= y < th and target[y, x]:
            on[t] = True
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return 0.0, 0.0, 0.0
    a_out = int(idx.max())
    a_in = a_out
    while a_in - 1 >= 0 and on[a_in - 1]:
        a_in -= 1
    r_dot = 0
    if on[0]:
        while r_dot + 1 < on.size and on[r_dot + 1]:
            r_dot += 1
    return float(a_in), float(a_out), float(r_dot)


def _orbit_envelope_b(a: float, r_side: float) -> float:
    """Semi-minor of an ellipse from its semi-major `a` and the envelope radius `r_side` 30° off the
    major axis (the 6-fold valley where two rings cross): an ellipse's polar radius
    r(α) = a·b/√(b²cos²α + a²sin²α), so at α=30° ⇒ b = 0.5·r·a/√(a² − 0.75·r²). Robust where a short-arc
    cv2.fitEllipse is not. Degenerate read → a (≈ circle)."""
    denom = a * a - 0.75 * r_side * r_side
    if r_side <= 0.0 or denom <= 1.0:
        return a
    return 0.5 * r_side * a / math.sqrt(denom)


def _fit_rot_orbits(target: np.ndarray, min_feat: int, h: int, w: int) -> list[Part]:
    """n-fold rotationally-symmetric orbit mark (react's 3 woven elliptical rings) → one rotated-ellipse
    annulus replicated at each symmetry orientation, **each ring in its own CSG group** so `_compound_mask`
    keeps the woven crossings (a single global subtract carves the rings to a bare rim). Returns [] unless
    strong rotational symmetry is detected — a no-op on every other shape. The order comes from the DFT of
    the outer-radius angular profile (n-fold → energy at harmonic n); each ring's ellipse is solved from
    the radius ENVELOPE (semi-major = max radius along the axis; semi-minor from the 30°-off valley in
    closed form), never a fragile short-arc fitEllipse. Deterministic (H8): warpPolar + rfft + ray-cast +
    closed form, no RNG. Competes by honest min(area, boundary); the symmetry score only gates compute."""
    # centre + outer radius from the CONTOUR (the enclosing circle is fixed by boundary points; feeding
    # minEnclosingCircle every fg pixel is O(N) on ~10^5 pts → seconds, the contour is ~10^2 pts → instant)
    c = _largest_contour(target)
    if c is None or len(c) < 5:
        return []
    (cx, cy), outer_r = cv2.minEnclosingCircle(c)
    if outer_r <= 0.0:
        return []
    # rotational order from the outer-radius angular profile's DFT — n-fold ⇒ energy at harmonic n
    fwd = cv2.WARP_POLAR_LINEAR + cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS
    polar = cv2.warpPolar(target, (ORBIT_R_BINS, ORBIT_THETA_BINS), (cx, cy), outer_r + 2.0, fwd)
    r_out = np.where(polar > 0, np.arange(ORBIT_R_BINS)[None, :], -1).max(axis=1).astype(float)
    r_out[r_out < 0] = 0.0
    mag = np.abs(np.fft.rfft(r_out - r_out.mean()))
    denom = float(mag[1:].sum()) + 1e-9
    order = max(ORBIT_SYM_ORDERS, key=lambda n: mag[n] / denom)
    if mag[order] / denom < ORBIT_SYM_MIN:                    # the no-op gate (compute only; arbitration is the real guard)
        return []
    # axis orientation θ0 = the farthest contour point (a ring's major-axis tip), in image angle
    pts = c.reshape(-1, 2).astype(float)
    j = int(np.argmax(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    theta0 = math.atan2(pts[j, 1] - cy, pts[j, 0] - cx)
    # outer + inner ellipse shells from the radius envelope at θ0 and ±(half-period) off it
    side = math.pi / order
    a_in, a_out, r_dot = _orbit_radial(target, cx, cy, theta0, outer_r + 2.0)
    if a_out <= 0.0:
        return []
    sp_in, sp_out, _ = _orbit_radial(target, cx, cy, theta0 + side, outer_r + 2.0)
    sm_in, sm_out, _ = _orbit_radial(target, cx, cy, theta0 - side, outer_r + 2.0)
    b_out = _orbit_envelope_b(a_out, 0.5 * (sp_out + sm_out))
    b_in = _orbit_envelope_b(a_in, 0.5 * (sp_in + sm_in))
    # replicate the annulus at each DISTINCT orientation (dedup mod 180°), one CSG group per ring
    step = 360.0 / order
    seen: set[int] = set()
    parts: list[Part] = []
    g = 0
    for i in range(order):
        a_deg = (math.degrees(theta0) + i * step) % 180.0
        key = round(a_deg / 5.0) % 36
        if key in seen:
            continue
        seen.add(key)
        g += 1
        parts.append(Part(RotatedEllipse(cx=cx, cy=cy, rx=a_out, ry=b_out, angle=a_deg), "add", group=g))
        if a_in > 2.0 and b_in > 2.0:
            parts.append(Part(RotatedEllipse(cx=cx, cy=cy, rx=a_in, ry=b_in, angle=a_deg), "subtract", group=g))
    if len(parts) < 2:                                        # need ≥1 real ring
        return []
    if r_dot >= min_feat:                                     # solid nucleus → its own group 0 (no subtract carves it)
        parts.append(Part(Circle(cx=cx, cy=cy, r=r_dot), "add", group=0))
    # contiguity guard (the feedback-contour-over-beam invariant): never a disjoint construction
    tgt_cc = cv2.connectedComponents((target > 0).astype(np.uint8))[0] - 1
    con_cc = cv2.connectedComponents((_compound_mask(parts, h, w) > 0).astype(np.uint8))[0] - 1
    if con_cc > tgt_cc:
        return []
    return parts


def _angular_run(col: np.ndarray, peak: int) -> int:
    """Length (in bins) of the contiguous foreground run straddling `peak` in a CIRCULAR angular
    column (a warpPolar row at one radius). 0 when `peak` itself sits on background."""
    n = col.size
    if col[peak] <= 0:
        return 0
    left = 0
    while left < n and col[(peak - left - 1) % n] > 0:
        left += 1
    right = 0
    while right < n and col[(peak + right + 1) % n] > 0:
        right += 1
    return left + right + 1


def _fit_rot_spurs(target: np.ndarray, min_feat: int, h: int, w: int) -> list[Part]:
    """n-fold radial sunburst (claude's spark) → N RoundedRectangle blades + a hub Circle, one blade
    per detected spike, all op='add' group 0 (a FLAT union — spurs meet at the hub, they do not weave).
    Returns [] unless a UNIFORM radial burst is detected — a no-op on every other shape. The order
    comes from PEAK-COUNTING the outer-radius angular profile (a spike train is not sinusoidal, so the
    orbit detector's rfft single-harmonic gate never fires; counting peaks recovers the exact order +
    the per-spike angles, tolerant of imperfect spacing). Each blade is a capsule from the hub edge to
    that spike's OWN tip, its width read at the blade centre. The spacing-CV gate is the regression
    guard for the other sunburst-hinted feature (firefox — non-uniform spikes → []). Deterministic (H8):
    warpPolar + find_peaks + DT + fillPoly, no RNG. Competes by honest min(area, boundary) — no
    structured bonus; arbitration + the contiguity guard are the real safety."""
    # centre + outer radius from the CONTOUR (minEnclosingCircle on every fg pixel is O(N) → seconds;
    # the circle is fixed by boundary points, so feed the contour, ~10^2 pts → instant, same circle)
    c = _largest_contour(target)
    if c is None or len(c) < 5:
        return []
    (cx, cy), outer_r = cv2.minEnclosingCircle(c)
    if outer_r <= 0.0:
        return []
    # outer-radius angular profile r_out[θ] = max foreground radius per angle (a warpPolar row max)
    fwd = cv2.WARP_POLAR_LINEAR + cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS
    polar = cv2.warpPolar(target, (ORBIT_R_BINS, ORBIT_THETA_BINS), (cx, cy), outer_r + 2.0, fwd)
    r_out = np.where(polar > 0, np.arange(ORBIT_R_BINS)[None, :], -1).max(axis=1).astype(float)
    r_out[r_out < 0] = 0.0
    r_scale = (outer_r + 2.0) / ORBIT_R_BINS                  # radius-bin → pixel
    # order + spike angles by peak-counting the CIRCULAR profile: triple-tile so a peak at the 0°/360°
    # seam keeps full prominence context, then keep the middle tile. The order gate also guards an
    # empty peak set — never divide by len(peaks) before it.
    ext = np.concatenate([r_out, r_out, r_out])
    prom = (r_out.max() - r_out.min()) * SPUR_PEAK_PROM_FRAC
    dist = max(1, ORBIT_THETA_BINS // SPUR_MAX_ORDER)
    peaks, _ = find_peaks(ext, prominence=prom, distance=dist)
    mid = np.sort(peaks[(peaks >= ORBIT_THETA_BINS) & (peaks < 2 * ORBIT_THETA_BINS)] - ORBIT_THETA_BINS)
    order = int(mid.size)
    if not (SPUR_MIN_ORDER <= order <= SPUR_MAX_ORDER):
        return []
    # uniformity gate: a true burst's spikes are evenly spaced (claude gap-CV ~0.24); the OTHER
    # sunburst-hinted feature (firefox ~0.87) is not → [] → byte-identical. The lone regression guard.
    gaps = np.diff(np.concatenate([mid, [mid[0] + ORBIT_THETA_BINS]])).astype(float)
    if gaps.mean() <= 0.0 or gaps.std() / gaps.mean() > SPUR_GAP_CV_MAX:
        return []
    # hub = largest inscribed circle (distance-transform peak); blades reach into it for contiguity
    insc = _cand_inscribed_circle(target, h, w)
    inrad = float(insc[0].rx) if insc is not None else 0.0
    parts: list[Part] = []
    if inrad >= min_feat:
        parts.append(Part(Circle(cx=cx, cy=cy, r=inrad), "add", group=0))
    r_in = SPUR_INNER_FRAC * inrad if inrad >= min_feat else 2.0
    # one rounded-rect blade per spike: a capsule from the hub edge to that spike's own tip
    for pk in mid:
        pk = int(pk)
        theta = pk * (2.0 * math.pi / ORBIT_THETA_BINS)      # image angle (y-down) = warpPolar row frame
        r_tip = float(r_out[pk]) * r_scale                   # this spike's OWN tip radius (non-uniform ok)
        r_cen = 0.5 * (r_in + r_tip)
        half_len = 0.5 * (r_tip - r_in)
        if half_len < 0.5 * min_feat:                        # peak tip inside the hub band → not a real
            continue                                         # blade; skip (else a negative w bow-ties the rrect)
        col = polar[:, int(np.clip(round(r_cen / r_scale), 1, ORBIT_R_BINS - 1))]
        alpha = 0.5 * _angular_run(col, pk) * (2.0 * math.pi / ORBIT_THETA_BINS)
        half_w = max(0.5 * min_feat, r_cen * math.sin(alpha))   # floor so the blade rasters ≥ min_feat wide
        ex, ey = cx + r_cen * math.cos(theta), cy + r_cen * math.sin(theta)
        parts.append(Part(RoundedRectangle(cx=ex, cy=ey, w=2.0 * half_len, h=2.0 * half_w,
                                           angle=math.degrees(theta), corner_radius=half_w),
                          "add", group=0))
    # contiguity guard (the feedback-contour-over-beam invariant): never a disjoint construction
    tgt_cc = cv2.connectedComponents((target > 0).astype(np.uint8))[0] - 1
    con_cc = cv2.connectedComponents((_compound_mask(parts, h, w) > 0).astype(np.uint8))[0] - 1
    if con_cc > tgt_cc:
        return []
    return parts


def _fit_holed(target: np.ndarray, min_feat: int, h: int, w: int,
               allow_recurse: bool = True) -> list[Part]:
    """Concentric-hole construction: ADD the best primitive fit to the HOLE-FILLED body, then
    SUBTRACT a fitted construction per detected interior hole (single primitive, or a recursive
    multi-part carve via _fit_hole_parts) — `Circle(outer) − Circle(inner)` for a disc-with-hole
    (YouTube Music's ring). The `holed` edge-trace alone can leave the outer arc demoted
    (adds=[] → coverage 0.0); this gives decompose_shape a base-ADD candidate to arbitrate.
    Returns [] when the filled body has no usable fit (then the trace/residual path takes over)."""
    filled = _fill_holes(target)
    outer = _best_remainder_prim(filled, h, w)
    if outer is None:
        return []
    parts = [Part(outer, "add")]
    for gi, hole in enumerate(_detect_holes(target)):
        parts.extend(_fit_hole_parts(hole, min_feat, h, w, neg_group=-(gi + 1),
                                     allow_recurse=allow_recurse))
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
    mid_r = 0.5 * (ring.inner_r + ring.outer_r)
    area_floor = max(min_feat * min_feat, int(0.004 * float(np.count_nonzero(annulus))))
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < area_floor:  # thin edge-mismatch sliver, not a real gap
            continue
        c = _largest_contour(((lbl == i).astype(np.uint8)) * 255)
        if c is None:
            continue
        (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(c)
        if min(rw, rh) <= cap:
            # clean thin slot → its own minAreaRect IS the slot (correct position + tilt)
            slot = RotatedRectangle(cx=float(rcx), cy=float(rcy), w=float(rw), h=float(rh), angle=float(ang))
        else:
            # malformed: the thin slot merged with an OUTER CRESCENT (a blade's outer edge falls inside
            # the ideal outer_r), so minAreaRect is a fat off-gap bar. Discard it and build a RADIAL slot
            # anchored at the inner-circle HUB CONTACT (the gap's true angle is crescent-free at the hub),
            # spanning inner→outer so it merges with the inner circle and sits on the gap, not the blade.
            ys, xs = np.where(lbl == i)
            hub = np.hypot(xs - cx, ys - cy) <= ring.inner_r + INNER_CONTACT_FRAC * band
            if int(np.count_nonzero(hub)) < min_feat:     # pure crescent, no hub contact → not a real gap
                continue
            ah = np.arctan2(ys[hub] - cy, xs[hub] - cx)
            th = math.atan2(float(np.mean(np.sin(ah))), float(np.mean(np.cos(ah))))
            dev = np.arctan2(np.sin(ah - th), np.cos(ah - th))    # wrap-safe deviation from θ
            width = min(cap, max(float(min_feat), float(dev.max() - dev.min()) * ring.inner_r))
            slot = RotatedRectangle(cx=cx + math.cos(th) * mid_r, cy=cy + math.sin(th) * mid_r,
                                    w=band, h=width, angle=math.degrees(th))
        if min(slot.w, slot.h) < 1.0:
            continue
        parts.append(Part(slot, "subtract"))
    return parts


def _union_greedy(target: np.ndarray, dt: np.ndarray, peaks: np.ndarray, min_feat: int,
                  h: int, w: int, boundary_aware: bool) -> list[Part]:
    """One greedy union pass over OVERLAPPING primitives seeded at full-shape DT peaks. Shared by both
    rankings: DT-peak lobe seeds, the deepest-residual circle, a rounded-rect / inscribed-rect of the
    largest residual component, the UNION_SPILL_FRAC spill gate, the UNION_MAX_PARTS cap.

      - boundary_aware=False → the ORIGINAL area-greedy union: each pass keeps the candidate adding the
        most NEW area ≥ cover_floor. Best for fans of equal lobes (figma dots, apple-rainbow petals).
      - boundary_aware=True → ALSO offers ENCLOSING fits (rect / rotated-rect / triangle / ellipse /
        rotated-ellipse of the residual contour) and ranks each pass by the construction's
        min(area,boundary) gain ≥ UNION_ENCLOSE_EPS — so a rounded square's corner becomes one enclosing
        rectangle, not a patch circle (area-only ranking would pick a huge low-boundary bbox first).

    Deterministic (H8): peaks ordered by DT desc, fixed candidate order, integer/float-stable keys, no RNG."""
    tb = target.astype(bool)
    cover_floor = max(min_feat * min_feat, int(UNION_MIN_COVER_FRAC * float(np.count_nonzero(tb))))
    bw = _band_width(target)

    def score(ps: list[Part]) -> float:
        m = _compound_mask(ps, h, w)
        return min(_iou_bool(m, target), _boundary_iou(m, target, bw))

    peak_circ = [Ellipse(cx=float(px), cy=float(py), rx=float(dt[py, px]), ry=float(dt[py, px]))
                 for py, px in peaks]                                        # det.: peaks ordered by DT desc (H8)
    used = [False] * len(peak_circ)
    parts: list[Part] = []
    for _ in range(UNION_MAX_PARTS):
        resid = tb & ~_compound_mask(parts, h, w).astype(bool)
        cands: list[tuple[Primitive, int]] = []
        cands += [(peak_circ[i], i) for i in range(len(peak_circ)) if not used[i]]  # full-radius lobe seeds
        dtr = np.where(resid, dt, 0.0)
        r = float(dtr.max())
        if r >= 1.0:                                        # fallback: deepest residual inscribed circle
            py, px = divmod(int(np.argmax(dtr)), w)
            cands.append((Ellipse(cx=float(px), cy=float(py), rx=r, ry=r), -1))
        rm = (resid.astype(np.uint8)) * 255                 # elongated parts: fit the largest residual comp
        nn, lbl, st, _c = cv2.connectedComponentsWithStats(rm, connectivity=8)
        if nn > 1:
            comp = ((lbl == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)) * 255
            contour = _largest_contour(comp)
            if contour is not None:
                rr = _fit_rounded_rect(contour, h, w)
                if rr is not None:
                    cands.append((rr[0], -1))
                if boundary_aware:                          # enclosing fits — only the boundary ranking guards them
                    for fit in (_fit_rectangle(contour, h, w, 0, 0), _fit_rotated_rect(contour, h, w),
                                _fit_triangle(contour, h, w, 0, 0), _fit_ellipse(contour, h, w, 0, 0),
                                _fit_rotated_ellipse(contour, h, w)):
                        if fit is not None:
                            cands.append((fit[0], -1))
            ir = _cand_inscribed_rectangle(comp, h, w)
            if ir is not None:
                cands.append((ir[0], -1))
        best, best_i = None, -1
        if boundary_aware:                                  # rank by construction min(area,boundary) gain
            best_s = (score(parts) if parts else 0.0) + UNION_ENCLOSE_EPS
            for prim, pi in cands:
                pm = _raster_part(prim, h, w).astype(bool)
                spill = int(np.count_nonzero(pm & ~tb))
                if spill <= UNION_SPILL_FRAC * max(1, int(np.count_nonzero(pm))):
                    s = score(parts + [Part(prim, "add")])
                    if s > best_s:
                        best_s, best, best_i = s, prim, pi
        else:                                               # rank by max NEW area (the original)
            best_new = cover_floor
            for prim, pi in cands:
                pm = _raster_part(prim, h, w).astype(bool)
                spill = int(np.count_nonzero(pm & ~tb))
                new = int(np.count_nonzero(pm & resid))
                if spill <= UNION_SPILL_FRAC * max(1, int(np.count_nonzero(pm))) and new > best_new:
                    best_new, best, best_i = new, prim, pi
        if best is None:
            break
        parts.append(Part(best, "add"))
        if best_i >= 0:
            used[best_i] = True
    return parts


def _fit_primitive_union(target: np.ndarray, min_feat: int, h: int, w: int) -> list[Part]:
    """Greedy union of OVERLAPPING primitives — for a shape that IS a union of overlapping convex
    primitives (figma's dots, apple-rainbow petals, obsidian facets) the contour-trace only polygon-
    approximates. Each lobe is seeded as a circle at a full-shape distance-transform PEAK. Runs BOTH
    rankings (`_union_greedy`): the area-greedy original AND a boundary-aware pass that also tries
    enclosing rect/triangle/ellipse fits, and returns whichever construction scores higher
    min(area,boundary) — so slack's rounded-square corner becomes an enclosing rectangle while figma's
    dot-fan keeps its circles (better-of-both ⇒ no fixture regresses). One contiguous piece (NOT the
    abandoned disjoint IoU-beam); competes with the edge-trace and the skeleton union."""
    dt = cv2.distanceTransform(target, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)   # full-shape inscribed radius / px
    peaks = peak_local_max(dt, min_distance=max(1, int(UNION_PEAK_DIST_MULT * min_feat)),  # one seed per lobe
                           threshold_abs=float(min_feat), exclude_border=False)            # (skip sub-feature peaks)
    bw = _band_width(target)

    def post(parts: list[Part]) -> list[Part]:
        merged = _merge_collinear_circles(parts)                      # fuse collinear circle-runs into pills…
        if merged is not parts:                                       # …keeping the clean pill for a small fidelity
            om = _compound_mask(parts, h, w)                          #    give (PILL_KEEP_TOL) — consistent renders
            mm = _compound_mask(merged, h, w)
            if min(_iou_bool(mm, target), _boundary_iou(mm, target, bw)) \
                    >= min(_iou_bool(om, target), _boundary_iou(om, target, bw)) - PILL_KEEP_TOL:
                parts = merged                                        # figma: pill lifts min-IoU; petal-fan drops it
        return _refine_union_parts(parts, target, min_feat, h, w)     # 2nd pass: drop redundant / collapse bunches

    pa = post(_union_greedy(target, dt, peaks, min_feat, h, w, boundary_aware=False))
    pb = post(_union_greedy(target, dt, peaks, min_feat, h, w, boundary_aware=True))
    if not pb:
        return pa
    if not pa:
        return pb
    mpa, mpb = _compound_mask(pa, h, w), _compound_mask(pb, h, w)
    a_pa, a_pb = _iou_bool(mpa, target), _iou_bool(mpb, target)
    min_pa = min(a_pa, _boundary_iou(mpa, target, bw))
    min_pb = min(a_pb, _boundary_iou(mpb, target, bw))
    # better-of-both, but the boundary-aware pass wins only when it ALSO holds area-coverage — area is the
    # fidelity gate, so a boundary lift that trades area away (signal#15/16) keeps the area-greedy result.
    return pb if (min_pb > min_pa and a_pb >= a_pa) else pa           # ties keep area-greedy (H8)


def _merge_collinear_circles(parts: list[Part]) -> list[Part]:
    """Fuse a row of OVERLAPPING equal circles into one capsule (RoundedRectangle) — the contour-fit pill
    the union spreads as discrete circles (figma's top bar = 3 collinear r≈240 circles). Circles (ADD
    Ellipses with rx≈ry) are union-found into runs by matched radius (PILL_RAD_TOL) + genuine overlap
    (PILL_OVERLAP); a run of ≥2 whose centres are collinear (max perpendicular offset ≤ PILL_LINE_TOL·r)
    becomes a capsule spanning the run with h=2r, corner_radius=r — a Slack-style pill. A stadium contains
    every circle on its axis, so coverage only grows; spaced dots (centres ≥ 2r apart) never fuse and stay
    circles. Non-circle parts and lone circles pass through unchanged. Deterministic (H8): index order."""
    circ = [(j, p.primitive) for j, p in enumerate(parts)
            if p.op == "add" and isinstance(p.primitive, Ellipse)
            and abs(p.primitive.rx - p.primitive.ry) <= PILL_RAD_TOL * max(p.primitive.rx, p.primitive.ry)]
    if len(circ) < 2:
        return parts
    rad_floor = PILL_MIN_RAD_FRAC * max((c.rx + c.ry) / 2.0 for _, c in circ)  # only major circles form a bar…
    circ = [(j, c) for j, c in circ if (c.rx + c.ry) / 2.0 >= rad_floor]       # …drop small corner-fill circles
    if len(circ) < 2:
        return parts
    rad = [(c.rx + c.ry) / 2.0 for _, c in circ]
    cen = [(c.cx, c.cy) for _, c in circ]
    n = len(circ)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):                                                # union overlapping equal-radius circles
        for b in range(a + 1, n):
            if abs(rad[a] - rad[b]) > PILL_RAD_TOL * max(rad[a], rad[b]):
                continue
            d = math.hypot(cen[a][0] - cen[b][0], cen[a][1] - cen[b][1])
            if d <= PILL_OVERLAP * (rad[a] + rad[b]):
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged_src: set[int] = set()                                      # source part indices consumed by a capsule
    capsules: list[Part] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        cxs = np.array([cen[m][0] for m in members], np.float32)
        cys = np.array([cen[m][1] for m in members], np.float32)
        r = float(np.median([rad[m] for m in members]))
        pts = np.column_stack((cxs, cys)).reshape(-1, 1, 2)
        vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        perp = np.abs(-(cxs - x0) * vy + (cys - y0) * vx)            # |centre offset| from the fitted axis
        if float(perp.max()) > PILL_LINE_TOL * r:                    # not collinear → not a pill; keep circles
            continue
        proj = (cxs - x0) * vx + (cys - y0) * vy
        pmin, pmax = float(proj.min()), float(proj.max())
        cx, cy = x0 + vx * (pmin + pmax) / 2.0, y0 + vy * (pmin + pmax) / 2.0
        capsules.append(Part(RoundedRectangle(cx=cx, cy=cy, w=(pmax - pmin) + 2.0 * r, h=2.0 * r,
                                              angle=math.degrees(math.atan2(vy, vx)), corner_radius=r), "add"))
        merged_src.update(circ[m][0] for m in members)
    if not capsules:
        return parts
    return [p for j, p in enumerate(parts) if j not in merged_src] + capsules


def _refine_union_parts(parts: list[Part], target: np.ndarray, min_feat: int, h: int, w: int) -> list[Part]:
    """Artifact-cleanup 2nd pass over a union construction — the greedy + pill-merge can leave a part made
    REDUNDANT by later adds / the pill it was folded into. Drop any ADD whose removal does NOT lower the
    construction's min(area, boundary) IoU — i.e. it is fully explained by the other parts. STRICTLY
    fidelity-gated: it never lowers the score, so it never regresses arbitration (a part that still carries
    boundary — figma's corner circles, apple-rainbow's petals — is load-bearing and kept). Reduces part
    count for cleaner renders only where it is free. Deterministic (H8): fixed index order, no RNG.

    figma's 3 small top-left circles are the motivating case: they only exist because the pill spans the bar-
    circle CENTRES, stopping short of the bar's left edge. So this pass first EXTENDS each pill along its axis
    to the target's full extent within the pill's band (a stadium that reaches the real bar ends), spill- and
    fidelity-gated; that genuinely covers the corner, after which the small circles fall to the redundancy drop
    — a clean pill instead of a pill + 3 stray circles, with no fidelity loss and nothing spilling outside."""
    tb = target.astype(bool)
    bw = _band_width(target)
    ys, xs = np.where(tb)
    parts = list(parts)

    def score(ps: list[Part]) -> float:
        m = _compound_mask(ps, h, w)
        return min(_iou_bool(m, target), _boundary_iou(m, target, bw))

    # (a) extend each pill along its axis to span the bar's real extent (absorbs corner gap-fillers)
    for i, p in enumerate(parts):
        if not isinstance(p.primitive, RoundedRectangle):
            continue
        rr = p.primitive
        th = math.radians(rr.angle)
        vx, vy = math.cos(th), math.sin(th)
        perp = np.abs(-(xs - rr.cx) * vy + (ys - rr.cy) * vx)         # target px within the pill's half-height band
        band = perp <= rr.h / 2.0
        if not band.any():
            continue
        proj = (xs[band] - rr.cx) * vx + (ys[band] - rr.cy) * vy
        pmin, pmax = float(proj.min()), float(proj.max())
        if (pmax - pmin) <= rr.w + 1.0:                              # already spans the band → nothing to extend
            continue
        ncx, ncy = rr.cx + vx * (pmin + pmax) / 2.0, rr.cy + vy * (pmin + pmax) / 2.0
        ext = RoundedRectangle(cx=ncx, cy=ncy, w=pmax - pmin, h=rr.h, angle=rr.angle, corner_radius=rr.corner_radius)
        em = _raster_part(ext, h, w).astype(bool)
        spill = int(np.count_nonzero(em & ~tb)) / max(1, int(np.count_nonzero(em)))
        cand = parts[:i] + [Part(ext, "add")] + parts[i + 1:]
        if spill <= UNION_SPILL_FRAC and score(cand) >= score(parts):
            parts = cand

    # (b) drop any ADD whose removal does NOT lower min(area, boundary) IoU — now-redundant gap-fillers
    changed = True
    while changed and len(parts) > 1:
        changed = False
        cur = score(parts)
        for i, p in enumerate(parts):
            if p.op != "add":
                continue
            rest = parts[:i] + parts[i + 1:]
            if score(rest) >= cur:                                    # part adds no min-IoU → fully redundant → drop
                parts = rest
                changed = True
                break
    return parts


def _skeleton_strokes(tb: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Decompose a shape's medial axis into maximal collinear STROKES.
    Returns [(xs, ys, r), …] — each stroke's axis pixels (concatenated) and its max inscribed radius.
    The axis is cut at junctions (nc ≥ SKEL_JUNCTION_NC) into branches, then branches split at a common
    junction seam are RE-JOINED when their fitted directions stay collinear (|cos| ≥ SKEL_MERGE_COS) —
    so a stroke that a junction split (the top bar's two halves) becomes one elongated stroke again,
    while a perpendicular dot/stem leaf stays its own stroke. Deterministic (H8): no RNG, index order."""
    skel, dist = medial_axis(tb, return_distance=True, rng=0)          # axis + per-pixel inscribed radius
    sk = skel.astype(np.uint8)
    nc = convolve(sk, np.ones((3, 3), np.uint8), mode="constant", cval=0) * sk   # 3×3 neighbour-count on axis
    pruned = sk.copy()
    pruned[nc >= SKEL_JUNCTION_NC] = 0                                 # cut at junctions → isolated branches
    nb, lbl, st, _c = cv2.connectedComponentsWithStats(pruned, connectivity=8)
    branches: list[dict] = []                                         # per branch: pixels, direction, endpoints
    for i in range(1, nb):
        ys, xs = np.where(lbl == i)
        if ys.size < SKEL_MIN_BRANCH_PX:
            continue
        pts = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
        vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        proj = (xs - x0) * vx + (ys - y0) * vy
        e0 = (x0 + vx * float(proj.min()), y0 + vy * float(proj.min()))
        e1 = (x0 + vx * float(proj.max()), y0 + vy * float(proj.max()))
        own_r = float(dist[ys, xs].max())
        own_len = float(proj.max() - proj.min())
        elong = own_len >= SKEL_DOT_ASPECT * 2.0 * own_r              # only elongated branches have a real axis
        branches.append({"xs": xs, "ys": ys, "dir": (vx, vy), "ends": (e0, e1), "elong": elong})
    n = len(branches)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    gap = SKEL_MERGE_GAP_FRAC * float(tb.shape[0] + tb.shape[1])
    for a in range(n):                                                # union-find merge of collinear seams
        for b in range(a + 1, n):
            if not (branches[a]["elong"] and branches[b]["elong"]):   # a round dot has no axis → never merge
                continue                                              # (its fitLine direction is arbitrary)
            (ax, ay), (bx, by) = branches[a]["dir"], branches[b]["dir"]
            if abs(ax * bx + ay * by) < SKEL_MERGE_COS:               # not collinear → different stroke
                continue
            near = min(math.hypot(p[0] - q[0], p[1] - q[1])
                       for p in branches[a]["ends"] for q in branches[b]["ends"])
            if near <= gap:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    strokes: list[tuple[np.ndarray, np.ndarray, float]] = []
    for members in groups.values():
        xs = np.concatenate([branches[m]["xs"] for m in members])
        ys = np.concatenate([branches[m]["ys"] for m in members])
        r = float(dist[ys, xs].max())
        if r >= 1.0:
            strokes.append((xs, ys, r))
    return strokes


def _fit_skeleton_union(target: np.ndarray, min_feat: int, h: int, w: int) -> list[Part]:
    """Union of CONTOUR-FIT primitives recovered from the medial-axis skeleton — for a shape that IS a
    union of overlapping convex parts (figma: a top pill + circular dots + a stem) the contour-trace only
    polygon-approximates. The medial axis is decomposed into maximal collinear STROKES (_skeleton_strokes);
    each stroke is fit to the CONTOUR (not inscribed in the residual gap, which tiled): its skeleton-distance
    gives the half-thickness r and cv2.fitLine its axis, so a round stroke (length/2r < SKEL_DOT_ASPECT)
    becomes a Circle of that radius and an elongated stroke a capsule (RoundedRectangle, h=2r,
    corner_radius=r — a Slack-style pill) sized to span the stroke. Strokes union as ADDs (overlapping at the
    junctions → one contiguous piece, NOT the abandoned disjoint IoU-beam); kept largest-coverage first up to
    UNION_MAX_PARTS, each gated by spill ≤ UNION_SPILL_FRAC and new coverage ≥ cover_floor. Competes with
    the edge-trace and the primitive union, winning only where its boundary IoU is higher — consistent
    renders across sizes."""
    tb = target.astype(bool)
    cover_floor = max(min_feat * min_feat, int(UNION_MIN_COVER_FRAC * float(np.count_nonzero(tb))))
    prims: list[Primitive] = []
    for xs, ys, r in _skeleton_strokes(tb):
        pts = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
        vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        proj = (xs - x0) * vx + (ys - y0) * vy
        pmin, pmax = float(proj.min()), float(proj.max())
        length = pmax - pmin
        cx, cy = x0 + vx * (pmin + pmax) / 2.0, y0 + vy * (pmin + pmax) / 2.0
        if length < SKEL_DOT_ASPECT * 2.0 * r:                        # round stroke → contour-fit Circle
            cx, cy = float(xs.mean()), float(ys.mean())               # centroid = the compact dot's centre
            prims.append(Circle(cx=cx, cy=cy, r=r))
        else:                                                          # elongated → Slack-style capsule
            prims.append(RoundedRectangle(cx=cx, cy=cy, w=length + 2.0 * r, h=2.0 * r,
                                          angle=math.degrees(math.atan2(vy, vx)), corner_radius=r))
    # keep the highest-coverage strokes first, gated by spill + new-coverage floor
    parts: list[Part] = []
    order = sorted(prims, key=lambda p: -int(np.count_nonzero(_raster_part(p, h, w))))
    for prim in order:
        if len(parts) >= UNION_MAX_PARTS:
            break
        pm = _raster_part(prim, h, w).astype(bool)
        spill = int(np.count_nonzero(pm & ~tb))
        new = int(np.count_nonzero(pm & tb & ~_compound_mask(parts, h, w).astype(bool)))
        if spill <= UNION_SPILL_FRAC * max(1, int(np.count_nonzero(pm))) and new >= cover_floor:
            parts.append(Part(prim, "add"))
    return _refine_union_parts(parts, target, min_feat, h, w)         # 2nd pass: drop redundant / collapse bunches


def _fit_block_carve(target: np.ndarray, min_feat: int, h: int, w: int) -> list[Part]:
    """Subtractive 'block-carve' candidate: ADD one block primitive fit to the hole-filled body, then
    SUBTRACT the exact D1 concave-notch triangles and carve the block's remaining over-reach (the
    rounded corners the sharp block can't match). Each subtract is kept only if it removes more
    block-not-shape than shape (eats-guard) AND holds/raises min(area, boundary) IoU — so the result is
    one contiguous piece (block minus notches), never disjoint. Competes in _arbitrate by
    min(area, boundary); appended LAST, it wins only on a strict margin, so it cannot regress any
    fixture (Gmail's M: 16-part trace min 0.637 → 9-part block-carve min 0.844). Deterministic (H8):
    cv2 fitters + numpy, area-sorted carve order, no RNG. Returns [] when no block fits."""
    filled = _fill_holes(target)
    c = _largest_contour(filled)
    if c is None:
        return []
    bw = _band_width(target)
    cands = [f for f in (_fit_rounded_rect(c, h, w), _fit_rotated_rect(c, h, w)) if f is not None]
    if cands:                                            # rounded vs sharp, keep higher min(area, boundary)
        block = max(cands, key=lambda f: min(_iou_bool(f[1], filled), _boundary_iou(f[1], filled, bw)))[0]
    else:                                                # near-square body (gmail) → both None → Rectangle
        block = _best_remainder_prim(filled, h, w)
    if block is None:
        return []
    parts = [Part(block, "add")]
    tb = target.astype(bool)

    def score(ps: list[Part]) -> float:
        m = _compound_mask(ps, h, w)
        return min(_iou_bool(m, target), _boundary_iou(m, target, bw))

    # exact D1 notch carves (the big V + bottom notches do the heavy lifting), eats-guarded
    for tri in _concave_notch_triangles(parse_shape(target)):
        sm = _raster_part(tri, h, w).astype(bool)
        bm = _compound_mask(parts, h, w).astype(bool)
        if int(np.count_nonzero(sm & bm & ~tb)) > int(np.count_nonzero(sm & tb)):
            parts.append(Part(tri, "subtract"))

    # carve the remaining over-reach (rounded corners), eats-guarded + min-IoU gated; re-segment each pass
    for _ in range(REFINE_MAX_ITERS):
        cur = score(parts)
        bm = _compound_mask(parts, h, w).astype(bool)
        over = ((bm & ~tb).astype(np.uint8)) * 255
        ncomp, lbl, st, _cent = cv2.connectedComponentsWithStats(over, connectivity=8)
        changed = False
        for i in sorted(range(1, ncomp), key=lambda j: -int(st[j, cv2.CC_STAT_AREA])):  # area desc (H8)
            comp = ((lbl == i).astype(np.uint8)) * 255
            if not _extent_ok(comp, min_feat):
                continue
            sub = _best_overcover_prim(comp, h, w)
            if sub is None:
                continue
            sm = _raster_part(sub, h, w).astype(bool)
            if int(np.count_nonzero(sm & bm & ~tb)) <= int(np.count_nonzero(sm & tb)):  # eats-guard
                continue
            cand = parts + [Part(sub, "subtract")]
            if score(cand) >= cur - 1e-9:                # accept only on hold/gain → regression-proof
                parts = cand
                cur = score(cand)
                bm = _compound_mask(parts, h, w).astype(bool)
                changed = True
        if not changed:
            break
    return parts


def _fit_local_features(target: np.ndarray, min_feat: int, h: int, w: int,
                        allow_recurse: bool = True) -> list[Part]:
    """Local-feature construction: the primitive union on the hole-FILLED body (a dominant body + its
    satellite lobes) plus a fitted SUBTRACT per detected interior hole — a heterogeneous shape rebuilt
    from its real parts. Competes HONESTLY by min(area, boundary) (no structured/parsimony nudge), so it
    wins only where it is strictly better — react's nested squares, slack/signal's non-elliptical dots,
    whatsapp, mastodon. Unlike the hint=None `_fit_primitive_union`, this runs on the FILLED body and
    carves holes, so it serves holed/hinted shapes (where that candidate is not built). A contiguity
    guard rejects any fit with MORE components than the target — never a disjoint construction (the
    feedback-contour-over-beam invariant; drops claude#0 / whatsapp#1). Deterministic (H8): reuses the
    deterministic union + hole fitters, no RNG. [] when the union is empty or the fit fragments."""
    filled = _fill_holes(target)
    parts = _fit_primitive_union(filled, min_feat, h, w)
    if not parts:
        return []
    for gi, hole in enumerate(_detect_holes(target)):
        parts = parts + _fit_hole_parts(hole, min_feat, h, w, neg_group=-(gi + 1),
                                        allow_recurse=allow_recurse)
    tgt_cc = cv2.connectedComponents((target > 0).astype(np.uint8))[0] - 1
    con_cc = cv2.connectedComponents((_compound_mask(parts, h, w) > 0).astype(np.uint8))[0] - 1
    if con_cc > tgt_cc:                                   # disjoint construction → reject (contiguity)
        return []
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
                     use_arcs: bool = True, _depth: int = 0) -> Decomposition:
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
            if (len(seg.points) / perim < MIN_ARC_FRAC
                    or len(seg.points) < ARC_MIN_RUN_MULT * min_feat):
                junctions.append(Junction(point=seg.start, kind="curve"))
                continue
            prim = _fit_arc_prim(seg.points, bbox)      # circle-first; ellipse fallback
            if prim is None:
                junctions.append(Junction(point=seg.start, kind="curve"))
                continue
            if seg.convex:
                arc_adds.append(Part(prim, "add"))
                arc_segs.append(seg)
            else:
                subs.append(Part(prim, "subtract"))     # concave arc → shave-off negate

        # 2. accept the convex-arc adds AS A GROUP iff their carved construction beats the straight-only
        #    baseline (the no-arc trace competes in decompose_shape); else demote them to junctions.
        if arc_adds:
            arc_carves, arc_spill = _arc_ellipse_construction(arc_adds, subs, runs, target, min_feat, h, w)
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

    # 4. holes → subtract a fitted primitive, or (depth 0 only) a recursive multi-part carve
    for gi, hole in enumerate(parsed.holes):
        hm = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(hm, [np.vstack([s.points for s in hole.segments]).reshape(-1, 1, 2).astype(np.int32)], 255)
        subs.extend(_fit_hole_parts(hm, min_feat, h, w, neg_group=-(gi + 1),
                                    allow_recurse=(_depth == 0)))

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
    if len(parts) > TRACE_PRUNE_PARTS:                  # refine-loop pileup → structure-first cleanup
        parts = _post_prune(parts, target, h, w, target_n=TRACE_PRUNE_PARTS)

    a_iou, b_iou, _ = _compound_score(parts, target, h, w)
    recon2 = _compound_mask(parts, h, w)
    residual = ((target.astype(bool) & ~recon2.astype(bool)).astype(np.uint8)) * 255
    res = residual if (a_iou < ADEQUATE_COV and _extent_ok(residual, min_feat)) else None
    n_add = sum(1 for p in parts if p.op == "add")
    return Decomposition(parts=parts, junctions=junctions, residual=res, coverage=float(a_iou),
                         pad=pad, shape=(h, w), n_add=n_add, n_sub=len(parts) - n_add,
                         boundary_cov=float(b_iou))


def _fit_stroke_rects(target: np.ndarray, min_feat: int, h: int, w: int,
                      allow_recurse: bool = True) -> list[Part]:
    """Contour-anchored STROKE construction for straight-edged glyphs (vscode's X +
    bracket, letterform bars): pair near-parallel straight D1 runs (outer AND hole
    contours — a bar's inner edge often borders a counter) at consistent width with
    overlapping projections and a mostly-foreground band between them, and emit one
    bar per pair whose LONG EDGES COINCIDE with the two runs — a RotatedRectangle, or
    a RoundedRectangle capsule when rounded end-caps fit better (both probed, best
    net coverage wins, deterministically). The bars are then finished by the shared
    machinery: under-coverage fill, hole carves (recursive — the counter triangle),
    eats-guarded spill subtracts. Competes honestly in arbitration; returns [] when
    fewer than two bars pair up. Deterministic (H8): pairs sorted by overlap/index,
    no RNG."""
    parsed = parse_shape(target)
    if parsed is None:
        return []
    pool = [s for s in parsed.outer.segments if s.kind == "straight"]
    for hole in parsed.holes:
        pool.extend(s for s in hole.segments if s.kind == "straight")
    pool = [s for s in pool if len(s.points) >= 2 * min_feat]
    if len(pool) < 2:
        return []
    bx, by, bw, bh = cv2.boundingRect(target)
    tgt = target.astype(bool)
    tol = math.radians(STROKE_ANG_TOL_DEG)

    info = []
    for s in pool:
        p0 = np.asarray(s.start, float)
        p1 = np.asarray(s.end, float)
        v = p1 - p0
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        info.append((p0, p1, v / n, n))

    pairs: list[tuple[float, int, int]] = []
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            ui, uj = info[i][2], info[j][2]
            d = abs(float(np.dot(ui, uj)))
            if d < math.cos(tol):                       # not parallel (mod 180°)
                continue
            u = ui if float(np.dot(ui, uj)) >= 0 else ui  # common dir = run i's
            nrm = np.array([-u[1], u[0]])
            width = abs(float(np.dot((info[j][0] + info[j][1]) / 2.0
                                     - (info[i][0] + info[i][1]) / 2.0, nrm)))
            if not (min_feat <= width <= STROKE_MAX_WIDTH_FRAC * max(bw, bh)):
                continue
            t_i = sorted([float(np.dot(info[i][0], u)), float(np.dot(info[i][1], u))])
            t_j = sorted([float(np.dot(info[j][0], u)), float(np.dot(info[j][1], u))])
            overlap = min(t_i[1], t_j[1]) - max(t_i[0], t_j[0])
            if overlap < max(2.0 * min_feat, STROKE_MIN_OVERLAP * min(info[i][3], info[j][3])):
                continue
            pairs.append((overlap, i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    used: set[int] = set()
    strokes: list[Part] = []
    for overlap, i, j in pairs:
        if len(strokes) >= STROKE_MAX or i in used or j in used:
            continue
        u = info[i][2]
        nrm = np.array([-u[1], u[0]])
        mid_i = (info[i][0] + info[i][1]) / 2.0
        off = float(np.dot((info[j][0] + info[j][1]) / 2.0 - mid_i, nrm))
        width = abs(off)
        t_i = sorted([float(np.dot(info[i][0], u)), float(np.dot(info[i][1], u))])
        t_j = sorted([float(np.dot(info[j][0], u)), float(np.dot(info[j][1], u))])
        t_lo, t_hi = min(t_i[0], t_j[0]), max(t_i[1], t_j[1])   # union extent along the bar
        base = float(np.dot(mid_i, u))
        center = mid_i + u * ((t_lo + t_hi) / 2.0 - base) + nrm * (off / 2.0)
        angle = math.degrees(math.atan2(u[1], u[0]))
        length = t_hi - t_lo
        cands: list[Primitive] = [
            RotatedRectangle(cx=float(center[0]), cy=float(center[1]),
                             w=length, h=width, angle=angle),
            RoundedRectangle(cx=float(center[0]), cy=float(center[1]),
                             w=length + width, h=width, angle=angle,
                             corner_radius=width / 2.0),     # capsule: rounded caps
        ]
        best, best_net = None, 0
        for prim in cands:
            cm = _raster_part(prim, h, w).astype(bool)
            net = int(np.count_nonzero(cm & tgt)) - int(np.count_nonzero(cm & ~tgt))
            inside = int(np.count_nonzero(cm & tgt)) / max(1, int(np.count_nonzero(cm)))
            if inside >= STROKE_FILL_MIN and net > best_net:
                best, best_net = prim, net
        if best is None:
            continue
        strokes.append(Part(best, "add"))
        used.update((i, j))
    if len(strokes) < 2:
        return []

    parts = list(strokes)
    # finish with the shared machinery: fill what the bars miss, carve the counters,
    # shave eats-guarded spill — same passes the trace uses (H5).
    recon = _compound_mask(parts, h, w)
    under = ((tgt & ~recon.astype(bool)).astype(np.uint8)) * 255
    _cover_components(under, min_feat, h, w, _best_remainder_prim, "add", parts)
    for gi, hole in enumerate(_detect_holes(target)):
        parts.extend(_fit_hole_parts(hole, min_feat, h, w, neg_group=-(gi + 1),
                                     allow_recurse=allow_recurse))
    recon_b = _compound_mask(parts, h, w).astype(bool)
    over_b = recon_b & ~tgt
    over = over_b.astype(np.uint8) * 255
    _cover_components(over, min_feat, h, w, _best_overcover_prim, "subtract", parts,
                      guard=lambda cm: int(np.count_nonzero(over_b & cm))
                      > int(np.count_nonzero(tgt & cm)))
    return parts


_HOLE_RECURSE_MEMO: dict[tuple, "Decomposition"] = {}    # process-local; key = (crop sha, shape, min_feat)


def _fit_hole_parts(hole_mask: np.ndarray, min_feat: int, h: int, w: int,
                    group: int = 0, neg_group: int = -1,
                    allow_recurse: bool = True) -> list[Part]:
    """Subtract-parts carving one interior hole. The single-prim _fit_hole when it fits
    >= HOLE_SINGLE_KEEP (the cheap path — simple holes stay byte-identical); else a
    depth-1 recursive decompose_shape on the hole crop, accepted only when its
    min(area, boundary) >= RECURSE_HOLE_MIN with <= RECURSE_HOLE_MAX_PARTS parts.
    An ALL-ADD construction flips flat (exact De Morgan: body − hole ==
    body − (∪ hole-adds), each add → a subtract in `group`); a construction that
    needs its own subtracts (a letterform arch = ring − inner − half-plane) is
    emitted VERBATIM into the NEGATIVE CSG group `neg_group`, which _compound_mask
    carves from the positive groups' union. [] when nothing carves cleanly.
    hole_mask: (h, w) uint8 {0,255}, one filled hole."""
    prim = _fit_hole(hole_mask, h, w, min_feat)
    single = [Part(prim, "subtract", group)] if prim is not None else []
    if prim is not None and _iou_bool(_raster_part(prim, h, w), hole_mask) >= HOLE_SINGLE_KEEP:
        return single
    if not allow_recurse or not _extent_ok(hole_mask, min_feat):
        return single
    x, y, cw, ch = cv2.boundingRect(hole_mask)              # uint8 only; (x, y) = (col, row)
    crop = hole_mask[y:y + ch, x:x + cw]
    # The same counter is reached via the trace, holed, and local-feature paths —
    # memoize the (expensive: depth-1 refiner) recursion per process. Deterministic:
    # pure function of the crop bytes + min_feat.
    key = (hashlib.sha256(crop.tobytes()).hexdigest(), crop.shape, min_feat)
    sub = _HOLE_RECURSE_MEMO.get(key)
    if sub is None:
        sub = decompose_shape(crop, min_feat, hint=None, _depth=1)
        _HOLE_RECURSE_MEMO[key] = sub
    parts_r, cov, bnd = sub.parts, sub.coverage, sub.boundary_cov
    if parts_r and len(parts_r) > RECURSE_HOLE_MAX_PARTS:   # oversized winner → prune before the cap
        sh, sw = sub.shape
        sub_tgt = np.zeros((sh, sw), dtype=np.uint8)
        sub_tgt[sub.pad:sub.pad + ch, sub.pad:sub.pad + cw] = crop
        parts_r = _post_prune(parts_r, sub_tgt, sh, sw, target_n=RECURSE_HOLE_MAX_PARTS)
        cov, bnd, _ = _compound_score(parts_r, sub_tgt, sh, sw)
    if (not parts_r or len(parts_r) > RECURSE_HOLE_MAX_PARTS
            or cov < RECURSE_HOLE_MIN_AREA
            or bnd < RECURSE_HOLE_MIN_BND):
        return single
    ox, oy = x - sub.pad, y - sub.pad                       # recursive parts live on sub's padded canvas
    if all(pt.op == "add" for pt in parts_r):
        return [Part(_offset_part(pt, ox, oy).primitive, "subtract", group) for pt in parts_r]
    return [Part(_offset_part(pt, ox, oy).primitive, pt.op, neg_group) for pt in parts_r]


def _arbitrate(candidates: list[Decomposition]) -> Decomposition:
    """Pick the winning construction: best min(area, boundary) IoU (+ the structured-annulus bonus),
    then a PARSIMONY preference — among candidates within PARSIMONY_TOL of the best score, return the
    fewest-part one (cleaner structure → consistent renders). The best-score key is UNCHANGED: a
    `structured` candidate (only the L2-geometry annulus sets the flag) covering ≥ STRUCT_MIN_AREA still
    earns +STRUCT_TOL in the score. Parsimony GENERALIZES that bonus to any candidate by part count, so
    it CAN override a structured best_d when a fewer-part candidate is genuinely ≥ STRUCT_TOL −
    PARSIMONY_TOL (=0.05) better on min-IoU — by design, a cleaner construction that good should win.
    Chrome is unaffected: its annulus leads the next candidate by ~0.38, far outside the 0.03 window.
    The part-count preference goes only down to 2 parts: a single primitive
    never wins this way (else a complex shape collapses onto one crude primitive — firefox 9→1 at 0.05).
    When no multi-part candidate is within tolerance, the plain best-score winner is returned (so a
    genuine single-primitive shape still wins, as the best-score candidate). Deterministic (H8): the
    best-score key matches the old tuple key (bonus breaks exact float ties); parsimony uses integer
    part-count + float score, stable min → remaining ties fall to list order."""
    def sval(d: Decomposition) -> float:
        base = min(d.coverage, d.boundary_cov)
        return base + STRUCT_TOL if (d.structured and d.coverage >= STRUCT_MIN_AREA) else base
    best_d = max(candidates, key=lambda d: (sval(d), d.structured and d.coverage >= STRUCT_MIN_AREA))
    near = [d for d in candidates if sval(d) >= sval(best_d) - PARSIMONY_TOL and len(d.parts) >= 2]
    if not near:
        return best_d
    return min(near, key=lambda d: (len(d.parts), -sval(d)))


def decompose_shape(mask: np.ndarray, min_feat: int, hint: str | None = None,
                    ring: "RingGeom | None" = None,
                    bbox_xy: tuple[int, int] | None = None,
                    _depth: int = 0) -> Decomposition:
    """Decompose one feature mask into signed primitives + junctions + residual. A single-primitive
    hint is fit directly; an annular hint fits a ring/sector from L2 geometry and COMPETES with the
    edge-trace path (higher coverage wins); else edge-traced by convexity. _depth=1 marks a
    recursive hole call: it disables further hole recursion. The diff refiner still fires at
    depth 1 — a complex counter's own decomposition (mastodon's m) is exactly the second-layer
    case that needs it; depth-2+ never exists (depth-1 calls cannot recurse further)."""
    th, tw = mask.shape[:2]
    pad = max(min_feat, max(th, tw) // 2)
    h, w = th + 2 * pad, tw + 2 * pad
    target = np.zeros((h, w), dtype=np.uint8)
    target[pad:pad + th, pad:pad + tw] = mask

    single_cands: list[Decomposition] = []
    if hint in _SINGLE:                                # single-primitive hint -> direct fit, but the
        parts = _single_fit(target, hint, h, w)        # local-feature union may beat it (slack/signal dots
        if parts:                                      # the hint calls a single ellipse but 2 parts fit better)
            cand_single = _decomp_from_parts(parts, target, min_feat, pad, h, w)
            single_cands.append(cand_single)
            lparts = _fit_local_features(target, min_feat, h, w, allow_recurse=(_depth == 0))
            if lparts:
                single_cands.append(_decomp_from_parts(lparts, target, min_feat, pad, h, w))
            best_single = _arbitrate(single_cands) if len(single_cands) > 1 else cand_single
            if min(best_single.coverage, best_single.boundary_cov) >= DIFF_FIRE_THRESH:
                return best_single                     # adequate single → the cheap path, unchanged
            # sub-adequate single (android's round-capped antennae at 0.47) → NO early exit:
            # fall through so the trace, the discovery candidates, and the diff refiner all
            # compete — the early return used to bypass the very fire gate built to rescue
            # exactly these features. The single candidates stay in the pool.

    cand_annular: Decomposition | None = None
    cand_rr_annular: Decomposition | None = None
    cand_holed: Decomposition | None = None
    if hint in _ANNULAR and ring is not None:          # ring/sector candidate from L2 geometry
        parts = _fit_annular(ring, hint, h, w, bbox_xy or (0, 0), pad)
        if parts:
            cand_annular = _decomp_from_parts(parts, target, min_feat, pad, h, w, structured=True)
        rr = _fit_rr_annular(target, h, w, min_feat)   # squircle ring (instagram): rrect − rrect shells
        if rr:
            cand_rr_annular = _decomp_from_parts(rr, target, min_feat, pad, h, w, structured=True)
    if hint in _HOLED or hint in _ANNULAR:             # concentric-hole ring: fill-fit ADD − holes SUB.
        parts = _fit_holed(target, min_feat, h, w, allow_recurse=(_depth == 0))  # Competes for annular hints too, so a poor ring
        if parts:                                      # geometry can't beat a good fill-fit (instagram).
            cand_holed = _decomp_from_parts(parts, target, min_feat, pad, h, w)
    cand_gapped: Decomposition | None = None
    if hint == "gapped_annulus" and ring is not None:  # multi-blade aperture (chrome) → annulus − gap rects
        parts = _fit_gapped_annular(target, ring, h, w, bbox_xy or (0, 0), pad, min_feat)
        if parts:
            cand_gapped = _decomp_from_parts(parts, target, min_feat, pad, h, w, structured=True)
    cand_union: Decomposition | None = None
    cand_skeleton: Decomposition | None = None
    cand_block: Decomposition | None = None
    if hint is None:                                   # union-of-primitives discovery (overlapping convex parts)
        uparts = _fit_primitive_union(target, min_feat, h, w)
        if uparts:
            cand_union = _decomp_from_parts(uparts, target, min_feat, pad, h, w)
        sparts = _fit_skeleton_union(target, min_feat, h, w)   # skeleton strokes (figma: pill + circles)
        if sparts:
            cand_skeleton = _decomp_from_parts(sparts, target, min_feat, pad, h, w)
        bparts = _fit_block_carve(target, min_feat, h, w)      # block − exact D1 notches − corner carves (gmail M)
        if bparts:
            cand_block = _decomp_from_parts(bparts, target, min_feat, pad, h, w)
    cand_local: Decomposition | None = None
    if not single_cands and (hint is not None or _detect_holes(target)):  # heterogeneous body+lobes − holes
        lparts = _fit_local_features(target, min_feat, h, w, allow_recurse=(_depth == 0))  # (react/slack/whatsapp);
        if lparts:                                     # single_cands already carries this candidate — don't rebuild
            cand_local = _decomp_from_parts(lparts, target, min_feat, pad, h, w)
    cand_strokes: Decomposition | None = None
    stparts = _fit_stroke_rects(target, min_feat, h, w, allow_recurse=(_depth == 0))
    if stparts:                                        # parallel-run bars (vscode X/bracket, letterform bars)
        cand_strokes = _decomp_from_parts(stparts, target, min_feat, pad, h, w)
    cand_orbits: Decomposition | None = None
    if hint in _HOLED or (hint is None and _detect_holes(target)):  # n-fold woven rings via per-group CSG (react atom)
        oparts = _fit_rot_orbits(target, min_feat, h, w)           # [] unless strong rotational symmetry; honest min-key
        if oparts:
            cand_orbits = _decomp_from_parts(oparts, target, min_feat, pad, h, w)
    cand_spurs: Decomposition | None = None
    if hint == "sunburst":                             # n-fold radial blades (claude spark) — narrow gate → []
        pparts = _fit_rot_spurs(target, min_feat, h, w)  # for every non-sunburst feature; honest min-key, no bonus
        if pparts:
            cand_spurs = _decomp_from_parts(pparts, target, min_feat, pad, h, w)

    traced = _trace_decompose(target, min_feat, pad, h, w, _depth=_depth)
    candidates = [traced] + single_cands              # sub-adequate singles stay in the pool
    for c in (cand_annular, cand_rr_annular, cand_holed, cand_gapped, cand_union, cand_skeleton, cand_local, cand_strokes, cand_orbits, cand_spurs):
        if c is not None:
            candidates.append(c)
    # the block-carve is a clean subtractive construction, but a single-block seed UNDER-covers a non-block
    # shape; admit it only when it ALSO has the best area-coverage, so it can never win by sniping boundary
    # while trading area away (signal's round sub-features) — area stays the fidelity gate (boundary-pivot).
    if cand_block is not None and cand_block.coverage >= max(c.coverage for c in candidates):
        candidates.append(cand_block)
    if traced.coverage < ADEQUATE_COV:                         # sub-adequate → arbitrate alternatives
        candidates.append(_trace_decompose(target, min_feat, pad, h, w, use_arcs=False, _depth=_depth))  # straight-only
        prim = _best_remainder_prim(target, h, w)              # whole-shape single primitive
        if prim is not None:
            candidates.append(_decomp_from_parts([Part(prim, "add")], target, min_feat, pad, h, w))
    # sub-adequate after every deterministic candidate → fire the diff_fit refiner: top-k seeds,
    # jointly optimized, decoded back to exact Parts and re-scored hard — additional candidates
    # only, competing through the same arbitration (zero-regression is structural). Lazy import:
    # torch loads only when a feature actually fires. Fires at depth 1 too — a complex hole's
    # recursive decomposition (mastodon's m) is exactly the second-layer case that needs it.
    best_min = max(min(c.coverage, c.boundary_cov) for c in candidates)
    if best_min < DIFF_FIRE_THRESH and _depth <= 1:
        from iconograph.application.compress.diff_fit import refine_candidates
        ranked = sorted(candidates, key=lambda d: (-min(d.coverage, d.boundary_cov), len(d.parts)))
        seeds: list[list[Part]] = []
        seen: set[tuple] = set()
        for d in ranked:
            if not 1 <= len(d.parts) <= DIFF_MAX_SEED_PARTS:
                continue
            key = tuple((type(pt.primitive).__name__, pt.op, pt.group) for pt in d.parts)
            if key in seen:                                # structurally identical seed → skip
                continue
            seen.add(key)
            seeds.append(d.parts)
            if len(seeds) == DIFF_TOP_K:
                break
        for parts in refine_candidates(seeds, target, min_feat, h, w):
            candidates.append(_decomp_from_parts(parts, target, min_feat, pad, h, w))
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
