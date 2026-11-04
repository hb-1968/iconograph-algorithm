"""Render constructions on small grids with label and silhouette rules.

Simple round adds and concentric subtracts become bounding-box squares and
square rings. Compound silhouettes use ordered pixel edits to reconnect and
straighten them under topology and anti-globbing constraints. The production
renderer selects this path for target sizes <=24.
"""

from dataclasses import replace

import cv2
import numpy as np
from skimage.morphology import medial_axis

from iconograph.application.compress.common import (
    Circle,
    Ellipse,
    Part,
    Polygon,
    Rectangle,
    RotatedEllipse,
    RotatedRectangle,
    RoundedRectangle,
    Triangle,
    Wedge,
    _compound_mask,
    _iou_bool,
    _mirror_point,
    _nbcount,
    _orbit,
    _outer_bg,
    _raster_part,
)
from iconograph.application.compress.render import (
    POCKET_RETAIN,
    _best_flip_axis,
    _flip_mask,
    _pocket_masks,
    _radial_family,
    _sym_prep,
    _topology,
    jag_score,
    naive_mask,
    reference_coverage,
    reference_mask,
)
from iconograph.application.compress.straighten import (
    prim_center,
    scale_about_center,
    snap_polygon_run2,
    snap_triangle_run2,
    translate_prim,
)

_ROUND = (Circle, Ellipse, RotatedEllipse)
_RECTY = (Rectangle, RotatedRectangle, RoundedRectangle)   # rect-family ring members —
#          an RRect−RRect annulus (instagram's body) is as much a square ring as a round
#          one; _label_render's old _ROUND-only gate was a missing case, not a doctrine

TOOTH_MAX = 3        # fg pixel with <= this many fg 8-neighbours is a tooth/spur (clear)
NOTCH_MIN = 5        # bg pixel with >= this many fg 8-neighbours is a concave dent (fill)
MAX_SWEEPS = 256     # hard convergence bound; the search breaks at the true fixpoint first
W_ACC = 20.0         # accuracy weight: 1 − area IoU vs the reference (high enough that a
#                      reconnection move's fid gain beats the transient jag bump)
W_SOFT = 0.02        # soft-coverage L1 tie term (deterministic tie-break toward truth)
EPS = 1e-9
W_AXIS = 1.5         # axis-blocky weight per ISOLATED diagonal boundary step (a staircase
#                      tooth approximating a CURVE) at T=24. jag is DP-vertex-based and a smooth
#                      staircase reads as low jag, so without this the render drifts curvy.
W_AXIS_T16 = 0.5     # the SAME weight at T=16, ramping linearly to W_AXIS at T=24. NOT zero
#                      (the old ramp's "T=16 is already a perfect cube" assumption was wrong —
#                      gmail's compound M rounded its corners/feet at 16 without any pressure)
#                      and NOT the full 1.5 (that over-carves discord's ~3px eye holes into a
#                      notch at 16); 0.5 fully de-curves gmail's M while keeping discord clean
#                      (W_AXIS sweep on gmail+discord@16).

EXT_NOTCH_MIN_FRAC = 0.01  # external hull∖solid CC area ≥ this × solid area — vue's V
#                            0.041 ✓; vue's second sliver 0.0021 ✗; adobe slivers ≤ 0.0009 ✗
EXT_NOTCH_VEE_MAX = 1.0    # mouth-chord / depth < this = a V (narrow + deep). vue 0.68;
#                            adobe's wide-shallow between-legs gap ≈ 2.9 — its coarse-triangle
#                            erasure is the RATIFIED form and must keep rendering solid
EXT_NOTCH_SUB_MIN = 0.5    # construction testimony: ≥ this fraction of the CC overlapped by
#                            the construction's OWN subtract rasters (vue 1.00, adobe 0.00) —
#                            only notches the D2 decomposition itself carved are identity
TRI_EPS_FRAC = 0.04  # approxPolyDP epsilon (× hull perimeter): coarse enough to collapse
#                      a triangle/flat-top silhouette to 3–4 vertices (D1 uses 0.01 for fine)
TRI_AGREE = 0.90     # min IoU(coarse hull-polygon, solid silhouette) to call it a triangle —
#                      a deep bottom notch (the polygon would over-fill it) fails the gate

# neck divots: cubify-tier OUTLINE SEPARATION (figma's stacked lobes)
# Mirrors render.py:_outline_gaps (the GEOMETRY-tier T>24 doctrine) at the cubify tier
# (T<=24, the shipping feed res), but detects necks from PART-PAIR GEOMETRY (robust where
# the union outline is convex <1px) instead of silhouette convexity defects. A "deliberate"
# 1px punch — the necks are sub-pixel at feed res and never appear naturally, so the render
# idealizes them in (same spirit as cube round-identity). User: "a 1px divot ... to show
# their separate geometry, not a SEAM ... keep the proportions consistent."
DIVOT_MIN_LOBES = 3       # a run of >= this many stacked round lobes = "overrepresented" ⇒
#                           separate them (figma's pill + 2 circles + bottom = 4); a 2-lobe
#                           head+chin or discord's contained eyes never qualifies.
DIVOT_MAX_ASPECT = 3.0    # intrinsic (rotation-invariant) max aspect of a lobe — circles 1.0,
#                           figma pill 2.0 pass; claude's thin radial blades (~5) excluded, so
#                           a radial fan can't masquerade as a stack (no radial machinery needed).
DIVOT_NECK_GAP_FRAC = 0.04   # max canonical gap (× max(h,w)) between consecutive lobe boundaries
#                              that still reads as a single neck (adjacency filter; figma's
#                              biggest neck gap is 8px ≪ 69px, pill↔bottom's 488px is excluded).
DIVOT_OVERLAP_MAX = 0.25  # lobes may overlap up to this × the smaller lobe's stacking extent and
#                           still neck; deeper overlap (contained eyes, side-by-side circles) = no neck.
DIVOT_P_OVERLAP_MIN = 0.5  # min perpendicular-footprint overlap (× the smaller lobe's perp HALF-
#                            extent) — two lobes must actually sit over each other to share a neck.
DIVOT_DEPTH = 1           # px cleared per side at a neck row (the "punch"); knob for visibility.
DIVOT_MAX_T = 16          # T<=16 only (user ruling). At T=24 the neck column is ~4px wide, so a
#                           1px/side punch pinches it to a prominent 2px bridge — choppier than
#                           the "subtle, not a SEAM" intent, and figma's @24 glob is "much less
#                           bad" anyway; the clean mild-glob baseline wins there. T=16 (the worst
#                           glob) keeps the separation. Raise to 24 to re-enable both-size divots.


# path A: label-driven cubic forms (simple round labels)


def _bbox_square(prim, canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """The axis-aligned BOUNDING-BOX square of a primitive at target scale (a circle ->
    its bounding square). Read from the rasterized primitive's bbox so rotation is handled."""
    h, w = canvas_hw
    x, y, bw, bh = cv2.boundingRect(_raster_part(prim, h, w))
    f = target / max(h, w)
    sq = np.zeros((target, target), np.uint8)
    sq[int(round(y * f)):int(round((y + bh) * f)),
       int(round(x * f)):int(round((x + bw) * f))] = 255
    return sq


def label_ring_spec(parts: list[Part], canvas_hw: tuple[int, int],
                    target: int) -> tuple[tuple[int, int, int, int], int] | None:
    """((y0, x0, h, w) mapped OUTER box, ODD wall px) for an annular label — one round-
    or-rect-family ADD with one round-or-rect-family SUBTRACT strictly inside — or None.
    PUBLIC: render_debug's nested-ring coordination consumes the same spec `_label_render`
    renders, so detection and render can never drift (one source of truth).

    The wall is an EXPLICIT ODD stroke width, not the emergent difference of two
    independently-rounded bbox squares (which gave instagram's lens 2px walls @24 while
    the SAME ring mapped to 1px @16 — no doctrine chose 2, rounding did): mean canonical
    per-side wall x (target/S), rounded, floored to odd, clamped to {1, 3} — the stroke
    doctrine's sanctioned widths. Box mapping = `_bbox_square`'s exact rounding (the
    byte-compatibility anchor: youtube-music's already-1px ring re-renders identically).
    None when the sub isn't strictly inside or the box can't host the wall + interior."""
    if not _annular_pair(parts):
        return None
    add, sub = parts
    fam = _ROUND + _RECTY
    if not isinstance(add.primitive, fam) or not isinstance(sub.primitive, fam):
        return None
    h, w = canvas_hw
    ro = _raster_part(add.primitive, h, w)
    ri = _raster_part(sub.primitive, h, w)
    if not int(np.count_nonzero(ro)) or not int(np.count_nonzero(ri)):
        return None
    xo, yo, bwo, bho = cv2.boundingRect(ro)
    xi, yi, bwi, bhi = cv2.boundingRect(ri)
    walls = (yi - yo, xi - xo, (yo + bho) - (yi + bhi), (xo + bwo) - (xi + bwi))
    if min(walls) <= 0:
        return None                        # subtract not strictly inside the add
    f = target / max(h, w)
    y0, y1 = int(round(yo * f)), int(round((yo + bho) * f))
    x0, x1 = int(round(xo * f)), int(round((xo + bwo) * f))
    bh_t, bw_t = y1 - y0, x1 - x0
    wall = max(1, int(round((sum(walls) / 4.0) * f)))
    if wall % 2 == 0:
        wall -= 1                          # floor-to-odd: 2 -> 1 (a 2px wall is the blot
    wall = min(wall, 3)                    # doctrine's complaint class), >= 3.5 -> 3
    if min(bh_t, bw_t) < 2 * wall + 1:
        return None
    return (y0, x0, bh_t, bw_t), wall


def ring_mask(box: tuple[int, int, int, int], wall: int, target: int) -> np.ndarray:
    """ONE-STROKE square ring: the filled (y0, x0, h, w) box minus its wall-inset
    interior, by numpy slicing (cv2.rectangle's `thickness` stroke straddles the box
    boundary and bleeds outward — probe-confirmed; slicing keeps the wall exactly inside).
    The single ring rasterizer shared by `_label_render` and render_debug's nested-ring
    coordination (H5 — the same form must never be built in two places)."""
    y0, x0, bh, bw = box
    m = np.zeros((target, target), np.uint8)
    m[y0:y0 + bh, x0:x0 + bw] = 255
    m[y0 + wall:y0 + bh - wall, x0 + wall:x0 + bw - wall] = 0
    return m


def _label_render(parts: list[Part], canvas_hw: tuple[int, int],
                  target: int) -> np.ndarray | None:
    """Render a SIMPLE LABEL cubically, or None if the construction is compound.
    Eligible: an annular pair (round OR rect-family add + subtract strictly inside —
    `label_ring_spec`) -> a one-stroke square ring with an explicit ODD wall; a single
    round ADD -> its bbox square; a round add + round sub that is NOT a clean concentric
    ring (spec None: touching walls, degenerate box) keeps the legacy two-bbox-square
    carve verbatim (zero blast radius on the non-concentric class). A square ring is
    accepted only when it is a valid closed ring (topology (1, 1))."""
    spec = label_ring_spec(parts, canvas_hw, target)
    if spec is not None:
        sq = ring_mask(spec[0], spec[1], target)
        return sq if _topology(sq) == (1, 1) else None
    adds = [p for p in parts if p.op == "add"]
    subs = [p for p in parts if p.op == "subtract"]
    if (len(adds) != 1 or len(subs) > 1
            or not isinstance(adds[0].primitive, _ROUND)
            or (subs and not isinstance(subs[0].primitive, _ROUND))):
        return None
    outer = _bbox_square(adds[0].primitive, canvas_hw, target)
    if not subs:
        return outer
    sq = outer & ~_bbox_square(subs[0].primitive, canvas_hw, target)
    return sq if _topology(sq) == (1, 1) else None


def _triangle_label(parts: list[Part], canvas_hw: tuple[int, int],
                    target: int) -> np.ndarray | None:
    """Render a TRIANGLE-DOMINANT compound silhouette (adobe's 'A') as a COARSE
    DIGITAL triangle / flat-top triangle, bypassing the pixel staircase the
    silhouette path puts on diagonals (user: 'a triangle with the flat top … at
    most 4 straight pixel segments, not the 4-1-4-1 jank').

    Detect at canonical resolution: the solid (holes-filled) outer silhouette's
    convex hull must approxPolyDP to 3–4 vertices AND agree with the silhouette at
    IoU >= TRI_AGREE — a deep real concavity (a bottom notch the hull would over-fill)
    fails the gate and falls through to the honest silhouette path. Render the hull
    polygon scaled to target with every edge snapped to the run-≥2 slope grid (≤ ~4
    straight segments per diagonal), carve subtract-testified external V-notches
    (vue's bold V — EXT_NOTCH_* gates; adobe's wide-shallow leg gap stays erased),
    then carve the counter (the canonical hole) when it survives at target as a
    closed ring. None when not triangle-confident (H8)."""
    h, w = canvas_hw
    canon = _compound_mask(parts, h, w)
    if not int(np.count_nonzero(canon)):
        return None
    cnts, _ = cv2.findContours(canon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    solid = np.zeros((h, w), np.uint8)
    cv2.fillPoly(solid, [cnt], 255)
    hull = cv2.convexHull(cnt)
    approx = cv2.approxPolyDP(hull, TRI_EPS_FRAC * cv2.arcLength(hull, True), True)
    if not (3 <= len(approx) <= 4):
        return None
    hull_poly = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hull_poly, [approx], 255)
    if _iou_bool(hull_poly, solid) < TRI_AGREE:
        return None
    f = target / max(h, w)

    def _coarse(poly) -> tuple | None:                 # int contour poly → snapped target pts
        verts = tuple((float(p[0][0]) * f, float(p[0][1]) * f) for p in poly)
        return (snap_triangle_run2(Triangle(*verts)) if len(verts) == 3
                else snap_polygon_run2(verts))

    snapped = _coarse(approx)
    if snapped is None:
        return None
    outer = _raster_part(Polygon(snapped), target, target)
    if _topology(outer)[0] != 1:
        return None
    # External V-notch carve (vue's bold V): `inner_canon` below only recovers
    # INTERIOR holes — a concavity OPEN at the hull edge renders solid. Carve
    # hull∖solid components that are substantial (EXT_NOTCH_MIN_FRAC), V-shaped
    # (mouth chord < depth), and testified by the construction's own subtracts.
    # Geometric re-raster (approxPolyDP + the quarter-px fillPoly idiom): the
    # ≥0.5-coverage resize of the interior path leaves 1px @16 on a tapering V.
    ext = ((hull_poly > 0) & (solid == 0)).astype(np.uint8)
    if int(np.count_nonzero(ext)):
        solid_area = float(np.count_nonzero(solid))
        sub_u = np.zeros((h, w), np.uint8)
        for p in parts:
            if p.op == "subtract":
                sub_u |= _raster_part(p.primitive, h, w)
        k3 = np.ones((3, 3), np.uint8)
        hull_line = cv2.dilate(((hull_poly > 0)
                                & (cv2.erode(hull_poly, k3) == 0)).astype(np.uint8), k3)
        ncc, nlbl, nstats, _c = cv2.connectedComponentsWithStats(ext, connectivity=8)
        for ci in range(1, ncc):
            area_cc = int(nstats[ci, cv2.CC_STAT_AREA])
            if area_cc < EXT_NOTCH_MIN_FRAC * solid_area:
                continue
            cc = nlbl == ci
            if int(np.count_nonzero(cc & (sub_u > 0))) < EXT_NOTCH_SUB_MIN * area_cc:
                continue
            mouth = cc & (hull_line > 0)
            if not np.any(mouth):
                continue                      # fully interior — the hole path's job
            mys, mxs = np.nonzero(mouth)
            chord = float(np.hypot(int(mys.max()) - int(mys.min()),
                                   int(mxs.max()) - int(mxs.min()))) + 1.0
            seed = np.full((h, w), 255, np.uint8)
            seed[mouth] = 0
            depth = float(cv2.distanceTransform(
                seed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[cc].max())
            if depth <= 0.0 or chord / depth >= EXT_NOTCH_VEE_MAX:
                continue
            cc_cnts, _h2 = cv2.findContours(cc.astype(np.uint8), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            if not cc_cnts:
                continue
            ncnt = max(cc_cnts, key=cv2.contourArea)
            napx = cv2.approxPolyDP(ncnt, TRI_EPS_FRAC * cv2.arcLength(ncnt, True), True)
            if len(napx) < 3:
                continue
            m = _raster_part(Polygon(tuple((float(p0[0][0]) * f, float(p0[0][1]) * f)
                                           for p0 in napx)), target, target)
            carved = outer.copy()
            carved[m > 0] = 0
            # accept only an OPEN notch: 1 component, hole count unchanged
            if (_topology(carved)[0] == 1
                    and _topology(carved)[1] == _topology(outer)[1]):
                outer = carved
    # Carve the counter: downscale the canonical hole the way the reference does
    # (INTER_AREA coverage >= 0.5), clamp it strictly inside the outer triangle (erode
    # by 1 px so the hole never touches the boundary), and keep it only when it forms a
    # clean ring. A sub-representable counter is honestly dropped — the solid triangle is
    # then the represented label.
    inner_canon = solid & ~canon
    if int(np.count_nonzero(inner_canon)):
        cov = cv2.resize(inner_canon, (target, target), interpolation=cv2.INTER_AREA)
        hole = (cov >= 128) & (cv2.erode(outer, np.ones((3, 3), np.uint8)) > 0)
        carved = outer.copy()
        carved[hole] = 0
        if _topology(carved) == (1, 1):
            return carved
    return outer


# path B: silhouette pixel-cubify (compound constructions)


def _candidates(mask: np.ndarray, nb: np.ndarray,
                ref: np.ndarray) -> list[tuple[int, int, int]]:
    """1px-brush moves (y, x, new_value), order (y, x, clear<fill). A fg pixel CLEARS if a
    tooth (<= TOOTH_MAX fg nb) OR bg in the reference; a bg pixel FILLS if a notch
    (>= NOTCH_MIN fg nb) OR fg in the reference and touching fg (ref-guided reconnect)."""
    fg = mask > 0
    rf = ref > 0
    clear = fg & ((nb <= TOOTH_MAX) | (~rf & (nb < 8)))
    fill = (~fg) & ((nb >= NOTCH_MIN) | (rf & (nb >= 1)))
    out: list[tuple[int, int, int]] = []
    ys, xs = np.nonzero(clear)
    out.extend((int(y), int(x), 0) for y, x in zip(ys, xs))
    ys, xs = np.nonzero(fill)
    out.extend((int(y), int(x), 255) for y, x in zip(ys, xs))
    out.sort(key=lambda c: (c[0], c[1], 0 if c[2] == 0 else 1))
    return out


def _topo_ok(trial: tuple[int, int], cur: tuple[int, int],
             ref: tuple[int, int], canon: tuple[int, int]) -> bool:
    """A flip's topology is acceptable iff it does not move the COMPONENT count further
    from the nearer of ref/canon (admits reconnection, blocks fragmentation) AND does not
    drop the HOLE count below min(current, canon) (a canonical hole is never destroyed).

    NOTE: no UPPER bound on holes. A hole-cap (holes <= max(ref, canon)) was tried to stop
    the W_AXIS_T16 blockification from splitting nested_counter's counter into a spurious 3rd
    hole at T=16 — but cubify LEGITIMATELY recovers counters the downscaled reference lost
    (mastodon@16's 'm' forms 2 counters where ref/canon have 1), and the two are
    indistinguishable by hole-count-vs-ref. The cap dropped mastodon's real counter, so it
    was reverted: a synthetic shape gaining one hole is preferable to a real fixture losing a
    counter. The detail-retention rework closed the remaining gap SPATIALLY instead: the R4
    hole-WITNESS guard (_hole_ok) blocks a NEW enclosed hole that contains no ref-bg /
    low-coverage pixel — mastodon's recovered counters are canon-bg-witnessed and pass,
    medium-m@24's unwitnessed blockify pinch is blocked; a count cap could never make that
    distinction."""
    ct, ht = trial
    cm, hm = cur
    comp_t = min(abs(ct - ref[0]), abs(ct - canon[0]))
    comp_c = min(abs(cm - ref[0]), abs(cm - canon[0]))
    if comp_t > comp_c:
        return False
    return ht >= min(hm, canon[1])


def _accuracy(mask: np.ndarray, ref: np.ndarray, cov: np.ndarray) -> float:
    """PHASE-A objective (lower = better): negative area IoU (monotonic for a ref-fg fill,
    so reconnection always registers a gain — climbs out of a fragmented naive)."""
    return -_iou_bool(mask, ref)


def _staircase_teeth(mask: np.ndarray) -> int:
    """Count ISOLATED diagonal boundary steps — a diagonal chain-move flanked on BOTH sides
    by axis moves. These are the staircase teeth a smooth CURVE decomposes into at the pixel
    level (the curviness jag_score's DP-vertex count is blind to). A straight diagonal RUN
    (an intentional angular edge — vscode's bracket, telegram's plane, a triangle leg) is
    consecutive diagonals, so its interior steps are NOT isolated and are not counted; only
    round-silhouette staircases pay. Boundary via CHAIN_APPROX_NONE (every pixel), wrap-
    aware. Deterministic (H8)."""
    total = 0
    for c in cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]:
        p = c[:, 0, :]
        if len(p) < 3:
            continue
        mv = np.diff(np.vstack([p, p[:1]]), axis=0)        # consecutive moves (wrapped)
        diag = (mv[:, 0] != 0) & (mv[:, 1] != 0)
        total += int(np.sum(diag & ~np.roll(diag, 1) & ~np.roll(diag, -1)))
    return total


def _cubic(mask: np.ndarray, ref: np.ndarray, cov: np.ndarray) -> float:
    """PHASE-B objective (lower = better): cubification + accuracy tradeoff + soft tie +
    axis-blocky. The axis term is ACTIVE at every feed size T<=24 (ramps W_AXIS_T16 -> W_AXIS
    over T=16..24): T=16 is NOT 'already a perfect cube' — gmail's compound M rounds its
    corners/feet without this pressure (jag_score is DP-vertex-based, so a smooth boundary
    reads as LOW jag and the greedy sheds vertices into curves). The T=16 weight is gentler
    than T=24's so it doesn't over-carve discord's tiny eye holes. Same blocky goal at both."""
    T = mask.shape[0]
    w_axis = W_AXIS_T16 + (W_AXIS - W_AXIS_T16) * max(0.0, (T - 16) / 8.0)
    jag = jag_score(mask)
    soft = float(np.abs((mask > 0).astype(np.float32) - cov).sum())
    return (jag + W_ACC * (1.0 - _iou_bool(mask, ref)) + W_SOFT * soft
            + w_axis * _staircase_teeth(mask))


def _cubify_axis(parts: list[Part], canvas_hw: tuple[int, int],
                 target: int) -> tuple[bool, int] | None:
    """Global mirror axis (vertical, a2) where index i reflects to a2 - i, or None. H5
    reuse of _sym_prep (canonical detection, target snap)."""
    h, w = canvas_hw
    solos = [_raster_part(p.primitive, h, w) for p in parts]
    prep = _sym_prep(parts, solos, canvas_hw, target)
    if prep is None:
        return None
    vertical, _a_t, a2, _delta, _leader_of, _self_paired = prep
    return vertical, a2


def _twin(y: int, x: int, axis: tuple[bool, int]) -> tuple[int, int]:
    """Mirror of (y, x) about the axis: x reflects if vertical, else y (a2 - i).
    Thin wrapper over the shared reflect formula (compress.common._mirror_point, H5)."""
    vertical, a2 = axis
    return _mirror_point(y, x, vertical, a2)


def _glob_ok(trial: np.ndarray, pockets: list[np.ndarray]) -> bool:
    """True if `trial` keeps >= POCKET_RETAIN of EVERY anti-glob pocket open (no glob). The one
    anti-glob test shared by _greedy / _close_to_canon / _reconcile_structure (H5)."""
    for pk in pockets:
        pn = int(np.count_nonzero(pk))
        if pn and int(np.count_nonzero((pk > 0) & (trial == 0))) < POCKET_RETAIN * pn:
            return False
    return True


DUAL_AXIS_IOU = 0.95     # BOTH-axis ref flip-IoU bar for R6 dual-axis enforcement
#                          (instagram's body 0.955/0.955; nothing between 0.955 and the
#                          perfect-1.0 label-path squares in the calibration scan)


def _annular_pair(parts: list[Part]) -> bool:
    """One add + one subtract — the annular construction class (instagram's ring body
    RRect−RRect, a single woven ring). The R6 dual-axis enforcement scope: label-path
    constructions never reach here, multi-part compounds keep single-axis behavior."""
    return (len(parts) == 2 and parts[0].op == "add"
            and parts[1].op == "subtract")


def _hole_ok(trial: np.ndarray, trial_topo: tuple[int, int], cur: np.ndarray,
             cur_topo: tuple[int, int], witness: np.ndarray | None) -> bool:
    """R4 — spatial hole-WITNESS guard: a move may only CREATE a new enclosed hole
    where the evidence says background belongs (`witness` = ref-bg after the seam
    carve OR canonical coverage < 0.5). medium-m@24's blockify pinched a 3px hole
    with ZERO witness px — blocked; mastodon@16's recovered counters and
    nested_counter's tolerated extra hole are canon-bg-witnessed — unchanged. This
    succeeds where the REVERTED hole-count cap (see _topo_ok's NOTE) could not: a
    count cap cannot tell a legitimate recovered counter from a spurious pinch, a
    spatial witness can. Cheap: full check runs only when the hole count RISES.
    A hole that merely grows/moves (overlaps a current enclosed region) is not
    'new'. Deterministic (H8)."""
    if witness is None or trial_topo[1] <= cur_topo[1]:
        return True
    t_enc = (trial == 0) & ~_outer_bg(trial)
    if not t_enc.any():
        return True
    c_enc = (cur == 0) & ~_outer_bg(cur)
    n, lbl = cv2.connectedComponents(t_enc.astype(np.uint8), connectivity=4)
    for q in range(1, n):
        cc = lbl == q
        if bool(np.any(cc & c_enc)):
            continue                                   # grown/moved existing hole
        if not bool(np.any(cc & witness)):
            return False                               # unwitnessed pinch
    return True


def _greedy(mask, score_fn, ref, cov, pockets, topo_ref, topo_canon, axis,
            witness=None, axis2=None):
    """Deterministic best-improvement hill-climb to a fixpoint under score_fn (lower =
    better). Each accepted flip clears the anti-glob + topology guards (+ the R4
    hole-witness guard when `witness` is given); with `axis` set, every flip is a
    MIRROR PAIR so a symmetric input stays exactly symmetric — and with `axis2` set
    (R6 dual-axis: instagram's ring body), every flip applies to the FULL 4-orbit
    (the shared _orbit, both-or-neither). H8: no RNG."""
    T = mask.shape[0]
    s_cur = score_fn(mask, ref, cov)
    cur_topo = _topology(mask)
    for _ in range(MAX_SWEEPS):
        nb = _nbcount(mask)
        best_gain, best_mask, best_s = EPS, None, None
        for y, x, val in _candidates(mask, nb, ref):
            trial = mask.copy()
            if axis2 is not None:
                orb = _orbit(y, x, [axis, axis2], (T, T))
                if orb is None:
                    continue
                for p in orb:
                    trial[p] = val
            else:
                trial[y, x] = val
                if axis is not None:
                    ty, tx = _twin(y, x, axis)
                    if not (0 <= ty < T and 0 <= tx < T):
                        continue
                    trial[ty, tx] = val
            if not _glob_ok(trial, pockets):
                continue
            tt = _topology(trial)
            if not _topo_ok(tt, cur_topo, topo_ref, topo_canon):
                continue
            if not _hole_ok(trial, tt, mask, cur_topo, witness):
                continue
            s2 = score_fn(trial, ref, cov)
            gain = s_cur - s2
            if gain > best_gain:
                best_gain, best_mask, best_s = gain, trial, s2
        if best_mask is None:
            break
        mask, s_cur = best_mask, best_s
        cur_topo = _topology(mask)
    return mask


def _close_to_canon(mask: np.ndarray, topo_canon: tuple[int, int],
                    pockets: list[np.ndarray],
                    witness: np.ndarray | None = None) -> np.ndarray:
    """Idealize toward the canonical topology: greedily fill the bg-adjacent pixel that
    most reduces (component + hole) distance to canon, bridging a fragmented loop closed.
    Pocket-guarded (+ R4 hole-witness guarded). Deterministic (H8)."""
    def td(t: tuple[int, int]) -> int:
        return abs(t[0] - topo_canon[0]) + abs(t[1] - topo_canon[1])

    for _ in range(MAX_SWEEPS):
        cur_topo = _topology(mask)
        d_cur = td(cur_topo)
        if d_cur == 0:
            break
        nb = _nbcount(mask)
        ys, xs = np.nonzero((mask == 0) & (nb >= 1))
        best_d, best_mask = d_cur, None
        for y, x in sorted((int(a), int(b)) for a, b in zip(ys, xs)):
            trial = mask.copy()
            trial[y, x] = 255
            if not _glob_ok(trial, pockets):
                continue
            tt = _topology(trial)
            if not _hole_ok(trial, tt, mask, cur_topo, witness):
                continue
            d2 = td(tt)
            if d2 < best_d:
                best_d, best_mask = d2, trial
        if best_mask is None:
            break
        mask = best_mask
    return mask


# neck divots (cubify-tier outline separation)

_LOBE = (Circle, Ellipse, RotatedEllipse, RoundedRectangle)


def _lobe_aspect(prim) -> float:
    """Intrinsic (rotation-invariant) aspect of a round lobe — its OWN long:short ratio,
    not the axis-aligned bbox (a thin blade at 45° has a square bbox). A circle is 1.0."""
    if isinstance(prim, Circle):
        return 1.0
    a, b = (prim.rx, prim.ry) if isinstance(prim, (Ellipse, RotatedEllipse)) else (prim.w, prim.h)
    lo = min(abs(a), abs(b))
    return abs(max(abs(a), abs(b)) / lo) if lo > EPS else 999.0


def _stacked_lobes(parts: list[Part], canvas_hw: tuple[int, int]
                   ) -> tuple[bool, list[tuple[float, float, float]]]:
    """Detect a run of >= DIVOT_MIN_LOBES round ADD lobes stacked along one axis with
    concave necks between consecutive members. Returns (vertical, necks) where each neck is
    (s_coord, p_lo, p_hi) — the canonical stacking-axis line of the waist plus the
    perpendicular-overlap interval of the two lobes that form it, so the carve pinches THAT
    column's waist (not unrelated outer edges of the row). Empty when the construction isn't
    an overrepresented stack. Pure geometry (H8)."""
    h, w = canvas_hw
    prims = [p.primitive for p in parts
             if p.op == "add" and isinstance(p.primitive, _LOBE)
             and _lobe_aspect(p.primitive) <= DIVOT_MAX_ASPECT]
    if len(prims) < DIVOT_MIN_LOBES:                       # cheap reject before any raster
        return True, []
    boxes = [cv2.boundingRect(_raster_part(pr, h, w)) for pr in prims]   # (x, y, bw, bh)
    cxs = [x + bw / 2.0 for x, _y, bw, _bh in boxes]
    cys = [y + bh / 2.0 for _x, y, _bw, bh in boxes]
    vertical = (max(cys) - min(cys)) >= (max(cxs) - min(cxs))
    # (s_center, s_lo, s_hi, p_lo, p_hi): S = stacking axis, P = perpendicular.
    info = []
    for (x, y, bw, bh), cx, cy in zip(boxes, cxs, cys):
        if vertical:
            info.append((cy, y, y + bh, x, x + bw))
        else:
            info.append((cx, x, x + bw, y, y + bh))
    gap_max = DIVOT_NECK_GAP_FRAC * max(h, w)
    necks: list[tuple[float, float, float]] = []
    involved: set[int] = set()
    for i, a in enumerate(info):
        for j, b in enumerate(info):
            if a[0] >= b[0]:                               # a must sit lower along S than b
                continue
            a_ext, b_ext = a[2] - a[1], b[2] - b[1]
            gap = b[1] - a[2]                              # b.lo − a.hi (≤0 = overlap)
            if not (-DIVOT_OVERLAP_MAX * min(a_ext, b_ext) <= gap <= gap_max):
                continue
            p_lo, p_hi = max(a[3], b[3]), min(a[4], b[4])  # perpendicular footprint overlap
            if p_hi - p_lo < DIVOT_P_OVERLAP_MIN * (min(a[4] - a[3], b[4] - b[3]) / 2.0):
                continue
            necks.append(((a[2] + b[1]) / 2.0, p_lo, p_hi))
            involved.add(i)
            involved.add(j)
    if len(involved) < DIVOT_MIN_LOBES:
        return vertical, []
    return vertical, sorted(necks)


def _carve_necks(mask: np.ndarray, vertical: bool,
                 necks: list[tuple[float, float, float]],
                 canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """Punch a DIVOT_DEPTH-px divot into each side of the WAIST at every neck — an outline
    indentation that signals the lobe division WITHOUT a seam. The waist is scoped to the
    necking pair's perpendicular-overlap window (so the bowl/spine's free edges aren't
    nicked). Each carve is kept only if it doesn't raise the component count (the spine keeps
    the lobes joined); a waist too thin to punch without meeting in the middle is skipped
    (honest partial). Deterministic (H8)."""
    if not necks:
        return mask
    h, w = canvas_hw
    f = target / max(h, w)
    base = _topology(mask)
    for s, p_lo, p_hi in necks:
        r = int(round(s * f))
        if not (0 <= r < target):
            continue
        pl, pr = int(round(p_lo * f)), int(round(p_hi * f))
        line = (mask[r] if vertical else mask[:, r]) > 0
        idx = np.nonzero(line)[0]
        idx = idx[(idx >= pl) & (idx <= pr)]               # fg within the necking window only
        if idx.size < 2 * DIVOT_DEPTH + 1:                 # too thin to punch without a seam
            continue
        lo, hi = int(idx[0]), int(idx[-1])
        if lo + DIVOT_DEPTH > hi - DIVOT_DEPTH:            # sides would meet → would be a seam
            continue
        trial = mask.copy()
        if vertical:
            trial[r, lo:lo + DIVOT_DEPTH] = 0
            trial[r, hi - DIVOT_DEPTH + 1:hi + 1] = 0
        else:
            trial[lo:lo + DIVOT_DEPTH, r] = 0
            trial[hi - DIVOT_DEPTH + 1:hi + 1, r] = 0
        if _topology(trial)[0] <= base[0]:
            mask = trial
            base = _topology(mask)
    return mask


# widen narrow OPEN negative-space gaps (gmail M-valley)
# A narrow OPEN concavity (gmail's M central V — negative space that opens to the background)
# renders ~1px at feed res and blots: the gap nearly closes, the outline rims merge, and
# refine's disjoint repair eats what's left. No existing pass WIDENS (pockets/_outline_gaps/
# _carve_necks only PRESERVE a ≤1px gap). Here we GROW the carving subtract part(s) so the gap
# reaches >=2px in the reference, BEFORE the greedy — the cube greedy then renders the wider gap
# (it matches the widened ref + its pocket guard). SCOPED to OPEN gaps + a ≤1px PINCH carve,
# committed only on a strict net-pinch DECREASE (no new slivers) under topology + fidelity
# guards. ENCLOSED counters are deliberately excluded: the widen over-enlarges discord's eyes
# (jag LOSS @24) and can't clear vscode's triangle-tip notch — enclosed-counter widening needs a
# different lever (open item). A no-open-narrow-gap construction is returned byte-identical.

WIDEN_F_STEP = 0.1       # subtract upscale step in the stepped widening search
WIDEN_F_MAX = 2.0        # cap — beyond this a subtract is over-carving, not widening a gap
WIDEN_FID_FLOOR = 0.85   # min area-IoU of the widened silhouette vs the original (bounds the trade)


def _zshift(a: np.ndarray, d: int, ax: int) -> np.ndarray:
    """Shift `a` by d px along axis ax, ZERO-filling the vacated border (NOT np.roll, which
    wraps — a border-opening gap would wrap into a false pinch)."""
    out = np.zeros_like(a)
    if d > 0:
        dst = (slice(d, None), slice(None)) if ax == 0 else (slice(None), slice(d, None))
        src = (slice(None, -d), slice(None)) if ax == 0 else (slice(None), slice(None, -d))
    else:
        d = -d
        dst = (slice(None, -d), slice(None)) if ax == 0 else (slice(None), slice(None, -d))
        src = (slice(d, None), slice(None)) if ax == 0 else (slice(None), slice(d, None))
    out[dst] = a[src]
    return out


def _pinch(mask: np.ndarray) -> np.ndarray:
    """Bool mask of background pixels squeezed to <=1px between two foreground walls (axis-
    aligned, d=1) — the witness for a ≤1px gap. d=1 EMPTY in a region ⇒ that gap is >=2px."""
    fg = mask > 0
    bg = ~fg
    return ((bg & _zshift(fg, 1, 1) & _zshift(fg, -1, 1))
            | (bg & _zshift(fg, 1, 0) & _zshift(fg, -1, 0)))


def _is_open(mask: np.ndarray, region: np.ndarray) -> bool:
    """True if `region` (background pixels of `mask`) touches the OUTER (border-connected)
    background — an OPEN concavity (gmail's M-valley), not an enclosed counter (discord's eyes,
    vscode's notch). Scopes the widen to OPEN gaps: enclosed counters are features the widen
    must not enlarge, and an enclosed notch like vscode's needs a different lever. Delegates
    to the shared compress.common._outer_bg (H5)."""
    return bool(np.any(region & _outer_bg(mask)))


SEAM_MAX_W_T = 1.5   # median carve-CC width at target <= this = a fused SEAM (chrome's
#                      seams read 0.16-0.26px; its donut hole 1.99@16/2.98@24 — a real gap)
SEAM_MAX_COMPS = 4   # canonical comps above this = a shattered decomposition, not seams
#                      (adobe-illustrator canon = 66 comps)


def _seam_curtains(parts: list[Part], canvas_hw: tuple[int, int],
                   target: int) -> np.ndarray | None:
    """R2 — intra-construction NEGATIVE-SEAM preservation (chrome's three swoosh gaps):
    the canonical geometry separates the mark into multiple components via sub-pixel-thin
    subtract seams that reference_mask pre-fuses and path-B then reconnects toward.
    Derive each seam from the CANONICAL subtracts (per-subtract carve = compound WITH vs
    WITHOUT it — the _widen_gaps isolation idiom, H5), qualify a carve-CC as a seam when
    it SEPARATES (filling it back merges canonical components) and is THIN (median
    2*dist*f at the medial-axis RIDGE <= SEAM_MAX_W_T — ridge-sampled, a plain per-px
    median biases low; scout-verified), and rasterize the union via the exact
    floor-coordinate map (_map_coords floor mode — resize sampling drops 0.2px seams).
    Fast None when canonical comps == 1 (youtube-music's reconnect stays untouched) or
    > SEAM_MAX_COMPS. The CALLER applies the union jointly (commit-or-revert: component
    count must strictly rise toward canon — docker's whale reverts byte-identical).
    Deterministic (H8: rng=0, sorted subtract order)."""
    canon = _compound_mask(parts, *canvas_hw)
    canon_comps = _topology(canon)[0]
    if canon_comps <= 1 or canon_comps > SEAM_MAX_COMPS:
        return None
    subs = [i for i, p in enumerate(parts) if p.op == "subtract"]
    if not subs:
        return None
    S = canvas_hw[0]
    union = np.zeros((target, target), bool)
    found = False
    for i in subs:
        no_s = _compound_mask([p for j, p in enumerate(parts) if j != i], *canvas_hw)
        carve = (no_s > 0) & ~(canon > 0)              # what this subtract removes, S-scale
        if not carve.any():
            continue
        ncc, lcc = cv2.connectedComponents(carve.astype(np.uint8), connectivity=8)
        for q in range(1, ncc):
            cc = lcc == q
            fill_back = canon.copy()
            fill_back[cc] = 255
            if _topology(fill_back)[0] >= canon_comps:
                continue                               # not separating
            skel, dist = medial_axis(cc, return_distance=True, rng=0)
            if not bool(skel.any()):
                continue
            med_w = float(np.median(2.0 * dist[skel])) * (target / S)
            if med_w > SEAM_MAX_W_T:
                continue                               # a real hole/gap, not a fused seam
            ys, xs = np.nonzero(cc)
            tys, txs = _map_coords(ys, xs, S, target, floor_map=True)
            union[tys, txs] = True
            found = True
    return union if found else None


GLYPH_MIN_FRAC = 0.10    # enclosed canonical region >= this fraction of canonical fg
#                          (mastodon's m channel 0.232; react's lens pockets 0.012)
GLYPH_SUBREP_MIN = 0.80  # medial-axis subrep fraction at the T=16 metric (mastodon 0.92;
#                          vscode 0.50 — its A/B REGRESSED, this gate is load-bearing)
GLYPH_FUSED_MIN = 3      # ref-fg curtain px >= this (mastodon 11; discord 1; yt-music 0)
CURTAIN_FG_MAX = 0.7     # drop curtain px whose canonical coverage exceeds this — the
#                          wall guard (an @16 skeleton spur through the bubble wall split
#                          mastodon's head without it)


def glyph_channel(parts: list[Part], canvas_hw: tuple[int, int]
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """(region, skel) of the glyph-class enclosed channel at CANONICAL res, or
    None — the ref/cov-INDEPENDENT detection core shared by `_glyph_curtain`
    (render tier) and refine's `glyph_ink_lines` (the m-as-ink-lines product).
    PUBLIC per the label_ring_spec precedent: one detector, two consumers, no
    drift (H5). Gates: enclosed region >= GLYPH_MIN_FRAC x canonical fg;
    medial-axis subrep fraction at the T=16 metric (2*dist*f16 < 1.5,
    ridge-sampled) >= GLYPH_SUBREP_MIN. Qualifying regions union. Callers pass
    parts at their own `_widen_gaps` state (cubify widens before the curtain;
    refine reproduces that for parity). Deterministic (H8: medial_axis rng=0)."""
    canon = _compound_mask(parts, *canvas_hw)
    fg_area = int(np.count_nonzero(canon))
    if fg_area == 0:
        return None
    enclosed = (canon == 0) & ~_outer_bg(canon)
    if not enclosed.any():
        return None
    f16 = 16.0 / canvas_hw[0]
    n, lbl, st, _c = cv2.connectedComponentsWithStats(enclosed.astype(np.uint8),
                                                      connectivity=4)
    region_u = np.zeros(canon.shape, bool)
    skel_u = np.zeros(canon.shape, bool)
    found = False
    for q in range(1, n):
        if int(st[q, cv2.CC_STAT_AREA]) < GLYPH_MIN_FRAC * fg_area:
            continue
        region = lbl == q
        skel, dist = medial_axis(region, return_distance=True, rng=0)
        n_skel = int(np.count_nonzero(skel))
        if n_skel == 0:
            continue
        widths16 = 2.0 * dist[skel] * f16
        if float(np.count_nonzero(widths16 < 1.5)) / n_skel < GLYPH_SUBREP_MIN:
            continue                                   # channel representable -> not this class
        region_u |= region
        skel_u |= skel
        found = True
    return (region_u, skel_u) if found else None


def _glyph_curtain(parts: list[Part], canvas_hw: tuple[int, int], target: int,
                   ref: np.ndarray, cov: np.ndarray) -> np.ndarray | None:
    """R3 — enclosed-GLYPH channel curtain (mastodon's white m): a large canonical
    enclosed region whose strokes are sub-pixel at feed res gets inked shut by notch-fill
    + blockify (the topology guard protects only the region's EXISTENCE, not its shape).
    Detection = `glyph_channel` (area + subrep gates; fires on mastodon ALONE — vscode's
    measured A/B regression is what the subrep gate excludes). The FUSED-EVIDENCE gate
    (>= GLYPH_FUSED_MIN curtain px already inked by the reference) is evaluated AT THE
    T=16 METRIC, target-independently: at 24 the reference keeps the channel white
    (fused=0 vs 3 at 16 — probe-verified), so the target-res ref is the wrong defect
    oracle there while the blockify teeth are the same disease; the f16 evidence matches
    the subrep gate's existing f16 convention. At target==16 the PASSED ref/cov are
    reused (byte-parity with the pre-lift behavior, incl. the seam-carved-ref
    interaction); at other targets fresh 16-res oracles are rastered (reached only after
    the detection gates — mastodon alone, ~one extra raster per run). Curtain =
    floor-map of the channel's medial axis (_map_coords floor mode) minus wall px
    (canonical coverage > CURTAIN_FG_MAX). The caller carves it from the start mask AND
    the reference (the witness map then keeps the channel R4-protected). (H8)."""
    gc = glyph_channel(parts, canvas_hw)
    if gc is None:
        return None
    _region_u, skel_u = gc
    S = canvas_hw[0]
    ys, xs = np.nonzero(skel_u)
    if target == 16:
        ref16, cov16 = ref, cov
    else:
        ref16 = reference_mask(parts, canvas_hw, 16)
        cov16 = reference_coverage(parts, canvas_hw, 16)
    t16y, t16x = _map_coords(ys, xs, S, 16, floor_map=True)
    cur16 = np.zeros((16, 16), bool)
    cur16[t16y, t16x] = True
    cur16 &= ~(cov16 > CURTAIN_FG_MAX)
    if int(np.count_nonzero(cur16 & (ref16 > 0))) < GLYPH_FUSED_MIN:
        return None                                    # no f16 fused evidence -> no defect
    tys, txs = _map_coords(ys, xs, S, target, floor_map=True)
    cur = np.zeros((target, target), bool)
    cur[tys, txs] = True
    cur &= ~(cov > CURTAIN_FG_MAX)                     # wall guard at the render target
    return cur if cur.any() else None


GLYPH_BAR_MIN_N = 4        # >= this many thin bars = a stroke-glyph (medium 4; vscode 3
#                            in-band, adobe 2 — scan-calibrated to fire on medium-m ALONE)
GLYPH_BAR_ASPECT = 2.5     # minAreaRect long/short >= this = an elongated bar
GLYPH_BAR_BAND16 = (1.0, 3.5)  # bar short-dim at the T=16 metric (px): below = sliver
#                            accents (adobe 0.17/0.24); at/above 3.5 = already legible
GLYPH_BAR_W_T = 3          # the floored stroke width at feed res — the user ruling:
#                            "a 3 wide with 1px rails should be legible" (hollow-3px
#                            doctrine: the perimeter outline then reads 1px rails + 1px
#                            white middle, and normalize/deburr have nothing to churn)


def _set_short_dim(pr, s_can: float, grow_only: bool = False):
    """Primitive with its SHORT dimension set (or grow-only floored) to `s_can`
    canonical px, about its center. Rectangle re-centers integer edges; the
    rotated/rounded kinds set the smaller of (w, h)."""
    if isinstance(pr, Rectangle):
        if pr.w <= pr.h:
            if grow_only and pr.w >= s_can:
                return pr
            cx = pr.x + pr.w / 2.0
            nw = max(1, int(round(s_can)))
            return Rectangle(int(round(cx - nw / 2.0)), pr.y, nw, pr.h)
        if grow_only and pr.h >= s_can:
            return pr
        cy = pr.y + pr.h / 2.0
        nh = max(1, int(round(s_can)))
        return Rectangle(pr.x, int(round(cy - nh / 2.0)), pr.w, nh)
    if isinstance(pr, (RotatedRectangle, RoundedRectangle)):
        if pr.w <= pr.h:
            if grow_only and pr.w >= s_can:
                return pr
            return replace(pr, w=float(s_can))
        if grow_only and pr.h >= s_can:
            return pr
        return replace(pr, h=float(s_can))
    return pr


def _scale_about(pr, f: float, cx: float, cy: float):
    """Primitive scaled by `f` about the COMMON point (cx, cy): scale about its
    own center (straighten.scale_about_center — per-kind exact), then translate
    the center onto C + f*(pc - C)."""
    pc_x, pc_y = prim_center(pr)
    pr2 = scale_about_center(pr, f)
    nx, ny = cx + f * (pc_x - cx), cy + f * (pc_y - cy)
    p2x, p2y = prim_center(pr2)
    return translate_prim(pr2, nx - p2x, ny - p2y)


def _glyph_stroke_relayout(parts: list[Part], canvas_hw: tuple[int, int],
                           target: int) -> list[Part]:
    """Stroke-glyph legibility rework (medium's M; user ruling: "a 3 wide with
    1px rails should be legible, it just requires widening and upscaling the M a
    bit") — implemented as exactly that: (1) UPSCALE the whole construction
    about its compound center to the canvas margin; (2) width-FLOOR every bar
    stroke (grow-only, in place — slant and composition preserved) to
    GLYPH_BAR_W_T px at feed res; (3) re-carve each inter-bar VALLEY as an
    explicit midline SUBTRACT strip (>= ~1.2px at target), traced along the
    ORIGINAL geometry's background between each adjacent bar pair and clamped
    to the rows where the original compound is genuinely open — the valleys
    keep the M's slanted identity and the topology cannot change (they re-open
    exactly what was open). Probes REFUTED the alternatives: upscale-only
    leaves 81% of strokes < 3px; in-place flooring alone seals the valleys; a
    packing/x-warp re-layout and its verticalized variant both mangle the
    fitted add/subtract interactions (implemented, measured, discarded).
    Identity for non-glyph constructions — callers key on `pp is not parts`.
    Detection: >= GLYPH_BAR_MIN_N rect-family adds with minAreaRect aspect >=
    GLYPH_BAR_ASPECT and short-dim in GLYPH_BAR_BAND16 at the f16 metric,
    non-radial (claude's blades). Deterministic (H8): sorted orders, fixed
    longest-run selection, no RNG."""
    S = max(canvas_hw)
    cand: list[tuple[int, float, bool]] = []    # in-BAND bars (the flooring set +
    #                                             the >= GLYPH_BAR_MIN_N fire count)
    vall: list[tuple[int, float]] = []          # ALL vertical strokes (valley pairs
    #                                             need the legible ones too — medium's
    #                                             right stem at 3.70px is out of band
    #                                             yet bounds the rightmost valley)
    for i, p in enumerate(parts):
        if p.op != "add" or not isinstance(p.primitive, _RECTY):
            continue
        nz = cv2.findNonZero(_raster_part(p.primitive, *canvas_hw))
        if nz is None:
            continue
        (rcx, _rcy), (rw, rh), rang = cv2.minAreaRect(nz)
        long_d, short_d = max(rw, rh), min(rw, rh)
        lang = rang if rw >= rh else rang + 90.0
        if short_d <= 0 or long_d < GLYPH_BAR_ASPECT * short_d:
            continue
        s16 = short_d * 16.0 / S
        if s16 < GLYPH_BAR_BAND16[0]:
            continue
        vert = 45.0 <= (lang % 180.0) < 135.0
        if vert:
            vall.append((i, float(rcx)))
        if s16 < GLYPH_BAR_BAND16[1]:
            cand.append((i, float(rcx), vert))
    if len(cand) < GLYPH_BAR_MIN_N:
        return parts
    if _radial_family(parts, [i for i, p in enumerate(parts) if p.op == "add"]) is not None:
        return parts                       # radial fan (claude) — comb pitch owns those
    vbars = sorted(vall, key=lambda c: (c[1], c[0]))
    if len(vbars) < 2:
        return parts
    # 1) upscale about the compound center to the canvas margin (probe: medium
    #    occupies 0.822 of the canvas; f_up = 1.217 -> pitch ~4.4px @16, room
    #    for 3px strokes + 1px valleys)
    comp0 = _compound_mask(parts, *canvas_hw)
    ys, xs = np.nonzero(comp0)
    if xs.size == 0:
        return parts
    ccy, ccx = float(ys.mean()), float(xs.mean())
    half_span = max(ccy - ys.min(), ys.max() - ccy,
                    ccx - xs.min(), xs.max() - ccx)
    if half_span <= 0:
        return parts
    f_up = min((S / 2.0 - 1.0) / half_span,
               min(ccy, ccx, S - 1 - ccy, S - 1 - ccx) / half_span)
    f_up = max(1.0, f_up)
    up = [replace(p, primitive=_scale_about(p.primitive, f_up, ccx, ccy))
          for p in parts]
    # 2) width-floor the bars in place (grow-only; slant preserved) + serif
    #    y-floor for the horizontal bars
    w_can = GLYPH_BAR_W_T * S / target
    floored = list(up)
    for i, _cx, vert in cand:
        floored[i] = replace(up[i],
                             primitive=_set_short_dim(up[i].primitive, w_can,
                                                      grow_only=True))
    # 3) midline valley subtracts between adjacent vertical bars, traced on the
    #    PRE-FLOOR upscaled geometry and clamped to its true-open rows
    comp_up = _compound_mask(up, *canvas_hw)
    rasters = {i: _raster_part(up[i].primitive, *canvas_hw) for i, _cx in vbars}
    order = [i for i, _cx in vbars]
    hw_min = 0.6 * S / target              # >= ~1.2px carve at target
    valleys: list[Part] = []
    for a, b in zip(order, order[1:]):
        ra, rb = rasters[a], rasters[b]
        rows = np.nonzero(ra.any(axis=1) & rb.any(axis=1))[0]
        run: list[tuple[int, float, float]] = []
        best_run: list[tuple[int, float, float]] = []
        for y in rows:
            xa = np.nonzero(ra[y])[0]
            xb = np.nonzero(rb[y])[0]
            ok = xa.size and xb.size
            if ok:
                right_a, left_b = int(xa.max()), int(xb.min())
                mid = (right_a + left_b) / 2.0
                ok = (right_a < left_b
                      and 0 <= int(round(mid)) < S
                      and not comp_up[y, int(round(mid))])
            if ok:
                run.append((int(y), mid, max((left_b - right_a) / 2.0, hw_min)))
                if len(run) > len(best_run):
                    best_run = run
            else:
                run = []
        if len(best_run) < 3:
            continue
        top = [(m - hw, float(y)) for y, m, hw in best_run]
        bot = [(m + hw, float(y)) for y, m, hw in reversed(best_run)]
        valleys.append(Part(Polygon(tuple(top + bot)), "subtract", group=0))
    if not valleys:
        return parts                       # nothing to separate — floor alone
    #                                        would fuse; stay honest, no fire
    return floored + valleys


def _widen_gaps(parts: list[Part], canvas_hw: tuple[int, int], target: int) -> list[Part]:
    """Grow subtract parts that carve a too-narrow OPEN valley (gmail's M central V) so the gap
    is >=2px at feed res. Per subtract: isolate its carve at target (ref WITH vs WITHOUT it),
    qualify it as carving a ≤1px OPEN gap (a pinch inside the carve, border-connected), then
    scale it up about its centre (stepped, capped at WIDEN_F_MAX) until the gap clears —
    committing ONLY if the target gap reaches >=2px AND TOTAL pinch strictly DECREASES (so
    isotropic growth that carves NEW narrow slivers elsewhere — gmail@24 — is rejected) AND
    topology is unchanged AND area-IoU stays >= WIDEN_FID_FLOOR. ENCLOSED counters are skipped
    (the widen over-enlarges discord's eyes / can't clear vscode's triangle-tip notch — those
    need a different lever). Subtracts processed in order on the running silhouette. Returns a
    new parts list (unchanged when nothing qualifies). Deterministic (H8)."""
    subs = [i for i, p in enumerate(parts) if p.op == "subtract"]
    if not subs:
        return parts
    orig = reference_mask(parts, canvas_hw, target)
    base_topo = _topology(orig)
    out = list(parts)
    for i in subs:
        s = out[i]
        cur = reference_mask(out, canvas_hw, target)
        no_s = reference_mask([p for j, p in enumerate(out) if j != i], canvas_hw, target)
        carve = (no_s > 0) & ~(cur > 0)                       # what this subtract removes at target
        if not carve.any() or not bool((_pinch(cur) & carve).any()):
            continue                                          # no ≤1px gap carved here → nothing to widen
        if not _is_open(cur, carve):                          # enclosed counter (eyes/notch) → skip
            continue
        cur_pinch = int(_pinch(cur).sum())
        f = 1.0 + WIDEN_F_STEP
        committed = None
        while f <= WIDEN_F_MAX + 1e-9:
            cand = list(out)
            cand[i] = Part(primitive=scale_about_center(s.primitive, f), op=s.op, group=s.group)
            m = reference_mask(cand, canvas_hw, target)
            if _topology(m) != base_topo or _iou_bool(m, orig) < WIDEN_FID_FLOOR:
                break                                        # over-carve (fragment / too much area)
            if not bool((_pinch(m) & carve).any()):          # target gap now >=2px
                if int(_pinch(m).sum()) < cur_pinch:         # AND no NEW narrow gaps → net win
                    committed = cand
                break                                        # target cleared: commit or reject, then stop
            f += WIDEN_F_STEP
        if committed is not None:
            out = committed
    return out


# dispatch


# cross-resolution structure constraint (@24 → @16)
# Each feed size cubifies INDEPENDENTLY, so @16 can scatter a structure that @24 renders
# coherently (figma's spine: @24 a solid connected bar; @16 thin-bridged scattered blocks —
# 1-connected but visually incoherent). The larger sibling (@24) is the structural authority:
# extract its connectivity BACKBONE (medial-axis skeleton + half-width), map it to the target
# grid, and reconcile @16's native render onto it (FILL the missing backbone, guarded), so the
# spine reads as a connected bar. Idempotent → no-op where @16 already covers the backbone, so
# fixtures @24 already renders coherently at 16 are byte-identical. Deterministic (H8).

BACKBONE_CAP = 0   # max disk radius (px) re-stroking the sibling backbone onto the target grid:
#                    0 = the 1-px medial-axis CENTERLINE only. The centerline loops AROUND holes
#                    (so the spine reconnects but @24's counters/gaps are preserved); a fatter cap
#                    (>=1) merges disks ACROSS a narrow @24 gap and over-fills it (figma's bowl).


def _map_coords(ys: np.ndarray, xs: np.ndarray, src: int, target: int,
                floor_map: bool) -> tuple[np.ndarray, np.ndarray]:
    """Project canonical/sibling px coordinates onto the target grid — THE shared
    forward-projection (H5): ROUND mode is the backbone convention (_structure_core);
    FLOOR mode ((i*target)//src) is the curtain convention — floor keeps a connected
    source set 8-connected at target scale and never drops sub-pixel-thin structures
    the way resize SAMPLING does (NN misses between sample points; coverage averaging
    thresholds a <0.5-coverage seam away entirely — scout-verified pinhole hazard).
    Clamped to target-1. Deterministic (H8: integer floor-div / half-even rint)."""
    if floor_map:
        ty = np.minimum((ys * target) // src, target - 1)
        tx = np.minimum((xs * target) // src, target - 1)
    else:
        ratio = target / src
        ty = np.minimum(np.rint(ys * ratio).astype(np.int64), target - 1)
        tx = np.minimum(np.rint(xs * ratio).astype(np.int64), target - 1)
    return ty, tx


def _structure_core(structure_ref: np.ndarray, target: int) -> np.ndarray:
    """The larger-sibling (@24) render's connectivity BACKBONE mapped to the target grid: the
    medial-axis skeleton (spine + branches) re-stroked as a thin solid stroke. structure_ref:
    the sibling cubify mask (src×src). Returns (target,target) uint8 — the form @16 must stay
    connected along. Deterministic (H8): medial_axis rng=0, exact ratio (24→16 = 2/3, no .5
    ties), row-major. At BACKBONE_CAP=0 the re-stroke is the 1-px centerline (radius 0 = 1px);
    a 2/3 DOWNSCALE cannot disconnect a connected centerline (no bridging needed), and the
    centerline loops around holes so @24's counters are preserved (a fatter cap would merge
    disks across a narrow counter and over-fill it). Projection via the shared _map_coords
    (round mode)."""
    src = int(structure_ref.shape[0])
    skel, dist = medial_axis(structure_ref > 0, return_distance=True, rng=0)
    ratio = target / src
    core = np.zeros((target, target), np.uint8)
    ys, xs = np.nonzero(skel)
    tys, txs = _map_coords(ys, xs, src, target, floor_map=False)
    for ty, tx, y, x in zip(tys.tolist(), txs.tolist(), ys.tolist(), xs.tolist()):
        r = min(BACKBONE_CAP, int(round(float(dist[y, x]) * ratio)))
        cv2.circle(core, (int(tx), int(ty)), max(0, r), 255, -1)
    return core


def _reconcile_structure(mask: np.ndarray, core: np.ndarray, pockets: list[np.ndarray],
                         topo_ref: tuple[int, int], topo_canon: tuple[int, int],
                         axis: tuple[bool, int] | None,
                         witness: np.ndarray | None = None) -> np.ndarray:
    """FILL @16 toward the sibling backbone `core`: greedily set each bg pixel that lies ON the
    backbone AND is 8-adjacent to fg, accepting it only if it clears the anti-glob pocket guard
    (_glob_ok), _topo_ok (no fragmentation / no destroyed canonical hole) and the R4
    hole-witness guard. With `axis` set, every fill is applied as a MIRROR PAIR (like _greedy)
    so a symmetric input stays symmetric. Re-solidifies a scattered spine into a connected bar;
    iterates so a 2px gap bridges in two passes. Idempotent → byte-identical no-op when mask
    already covers core. Modeled on _close_to_canon; deterministic (H8): row-major, no RNG."""
    T = mask.shape[0]
    cur_topo = _topology(mask)
    for _ in range(MAX_SWEEPS):
        nb = _nbcount(mask)
        ys, xs = np.nonzero((mask == 0) & (core > 0) & (nb >= 1))
        if len(ys) == 0:
            break
        changed = False
        for y, x in sorted((int(a), int(b)) for a, b in zip(ys, xs)):
            if mask[y, x]:                       # already filled earlier this sweep
                continue
            trial = mask.copy()
            trial[y, x] = 255
            if axis is not None:                 # mirror-pair the fill (preserve symmetry)
                ty, tx = _twin(y, x, axis)
                if not (0 <= ty < T and 0 <= tx < T):
                    continue
                trial[ty, tx] = 255
            if not _glob_ok(trial, pockets):
                continue
            tt = _topology(trial)
            if not _topo_ok(tt, cur_topo, topo_ref, topo_canon):
                continue
            if not _hole_ok(trial, tt, mask, cur_topo, witness):
                continue
            mask, cur_topo, changed = trial, tt, True
        if not changed:
            break
    return mask


def cubify(parts: list[Part], canvas_hw: tuple[int, int], target: int,
           structure_ref: np.ndarray | None = None) -> np.ndarray:
    """Render the construction at the feed resolution. Simple round labels render their
    cubic form (square / square ring); compound constructions are silhouette-cubified.
    `structure_ref` (the larger feed-res sibling's cubify mask, e.g. @24 when rendering @16)
    constrains the render to that backbone's connected form (supersedes the neck-divot pass).
    (target, target) uint8 {0,255}. Deterministic (H8)."""
    label = _label_render(parts, canvas_hw, target)
    if label is not None:
        return label
    tri = _triangle_label(parts, canvas_hw, target)
    if tri is not None:
        return tri

    parts = _widen_gaps(parts, canvas_hw, target)        # widen narrow OPEN gaps before rendering
    h, w = canvas_hw
    ref = reference_mask(parts, canvas_hw, target)
    naive = naive_mask(parts, canvas_hw, target)
    cov = reference_coverage(parts, canvas_hw, target)
    nm = (naive > 0).astype(np.uint8) * 255
    topo_canon = _topology(_compound_mask(parts, h, w))
    # R2 — seam curtains: carve canonical negative seams (chrome) out of the start mask
    # AND the reference, JOINT commit-or-revert (component count must strictly rise
    # toward canon; docker's whale reverts byte-identical). The carved ref makes
    # _topo_ok's ref term LOCK the separation for the rest of the greedy.
    curtain = _seam_curtains(parts, canvas_hw, target)
    if curtain is not None:
        nm_c = nm.copy()
        nm_c[curtain] = 0
        c_before, c_after = _topology(nm)[0], _topology(nm_c)[0]
        if c_before < c_after <= topo_canon[0]:
            nm = nm_c
            ref = ref.copy()
            ref[curtain] = 0
        else:
            curtain = None
    # R3 — glyph-channel curtain (mastodon@16's m): carve the sub-representable
    # enclosed channel out of the start mask AND the reference (same wiring as R2;
    # the R4 witness below then keeps the channel protected).
    glyph = _glyph_curtain(parts, canvas_hw, target, ref, cov)
    if glyph is not None:
        nm = nm.copy()
        nm[glyph] = 0
        ref = ref.copy()
        ref[glyph] = 0
    carved = None
    if curtain is not None or glyph is not None:
        carved = np.zeros((target, target), bool)
        if curtain is not None:
            carved |= curtain
        if glyph is not None:
            carved |= glyph
    pockets = _pocket_masks(ref)
    topo_ref = _topology(ref)
    # R4 — hole-witness map: new enclosed holes may form only where the (curtain-carved)
    # reference reads background OR canonical coverage is below half.
    witness = (ref == 0) | (cov < 0.5)

    # Symmetry (mirror-pair moves + symmetric start) applies only to a CONNECTED
    # construction; on a fragmented naive reconnection takes priority and symmetry is off.
    fragmented = _topology(nm)[0] > topo_canon[0]
    axis = None if fragmented else _cubify_axis(parts, canvas_hw, target)
    # R6 — DUAL-axis enforcement (user-directed): an annular construction whose
    # reference is symmetric about BOTH axes (instagram's ring body — _cubify_axis
    # picked H alone, so its V wall widths rasterized 2px/1px and no downstream fold
    # could reconcile them) gets the full 4-orbit treatment. Scoped to _annular_pair +
    # the DUAL_AXIS_IOU bar; intentionally-asymmetric accents (the camera dot) are
    # separate constructions and stay untouched.
    axis2 = None
    if axis is not None and _annular_pair(parts):
        fa1 = _best_flip_axis(ref, axis[0])
        fa2 = _best_flip_axis(ref, not axis[0])
        if (fa1 is not None and fa1[1] >= DUAL_AXIS_IOU
                and fa2 is not None and fa2[1] >= DUAL_AXIS_IOU):
            axis2 = (not axis[0], fa2[0])
    if axis is not None:
        vertical, a2 = axis
        mask = nm | _flip_mask(nm, a2, vertical)
        if axis2 is not None:
            mask = mask | _flip_mask(mask, axis2[1], axis2[0])   # full 4-orbit start
        if carved is not None:
            mask[carved] = 0                             # re-carve after the flip-union
    else:
        mask = nm

    if fragmented:
        mask = _greedy(mask, _accuracy, ref, cov, pockets, topo_ref, topo_canon, axis,
                       witness=witness, axis2=axis2)
        mask = _close_to_canon(mask, topo_canon, pockets, witness=witness)
    mask = _greedy(mask, _cubic, ref, cov, pockets, topo_ref, topo_canon, axis,
                   witness=witness, axis2=axis2)

    # The stacked-lobe constructions (figma's pill+circles) are exactly where @16 fragments —
    # the lumpy lobe column + the divot separation scatter the structure that @24 keeps coherent.
    # Scope the cross-resolution work to that signal: with a sibling (@24) reference, reconcile @16
    # to @24's connectivity backbone (this OWNS the separation, superseding the divot pass);
    # without one, fall back to the sub-pixel neck divots. NON-lobe constructions (discord's
    # head+eyes, gmail, vscode, claude) get NEITHER — the @24-skeleton fill misaligns with their
    # already-coherent @16 form and would distort it (discord's eyes), so they stay byte-identical.
    vertical, necks = (_stacked_lobes(parts, canvas_hw)
                       if target <= DIVOT_MAX_T else (False, []))
    if necks:
        if structure_ref is not None:
            core = _structure_core(structure_ref, target)
            rec = _reconcile_structure(mask, core, pockets, topo_ref, topo_canon, axis,
                                       witness=witness)
            div = _carve_necks(mask, vertical, necks, canvas_hw, target)
            # Pick the lower-jag of the two separation strategies: the @24-backbone RECONCILE
            # (re-solidifies a scattered structure — figma's spine, signal) vs the sub-pixel neck
            # DIVOTS (the default). A genuine fragmentation reconciles to a far lower jag than its
            # divots; a FALSE-POSITIVE lobe hit (apple-rainbow — already coherent at 16) reconciles
            # WORSE than its divots, so it keeps the divots → byte-identical to the no-structure
            # render. jag is a valid discriminator for the FILL direction (connecting lowers jag);
            # the metric only penalizes the desired separation in the CARVE direction.
            mask = rec if jag_score(rec) <= jag_score(div) else div
        else:
            mask = _carve_necks(mask, vertical, necks, canvas_hw, target)
    return mask
