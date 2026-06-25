"""Stage-3 R1 cubify (WS3), the feed-resolution render at T <= 24. HYBRID, two paths:

A) LABEL-DRIVEN (simple round labels) — a construction that is one round ADD (circle /
   ellipse) with at most one round concentric SUBTRACT is rendered straight from its
   geometry: the add's axis-aligned BOUNDING-BOX square minus the subtract's bounding
   square. A circle -> a square; an annulus -> a SQUARE RING, robust at any size. This
   represents the LABEL (user ruling: "it's NOT about fitting the shapes, it's about
   representing their labels") — IoU vs the round raster is irrelevant (a square outline
   barely overlaps a circle outline).

B) SILHOUETTE-CUBIFY (compound constructions) — everything else (a head + eyes, an M, an
   X) is cubified at the pixel level from the naive raster: a deterministic two-phase
   greedy 1px-brush search that reconnects a fragmented naive (Phase A, area-IoU), then
   blockifies it (Phase B, jag + accuracy), under anti-glob + topology guards, with
   mirror-pair moves when a global axis is detected (exact symmetry). Per-primitive cubic
   forms shatter a compound silhouette (bbox-squares fragment, diagonals staircase), so
   compound LABELS are represented by cubifying their composed silhouette.

Deterministic (H8): no RNG; greedy best-improvement over a fixed total order. Dispatched
from render.render_debug for target <= CUBIFY_MAX_T; geometry render kept for T > 24.
"""

import cv2
import numpy as np

from iconograph.application.compress.common import (
    Circle,
    Ellipse,
    Part,
    Polygon,
    RotatedEllipse,
    RoundedRectangle,
    Triangle,
    _compound_mask,
    _iou_bool,
    _raster_part,
)
from iconograph.application.compress.render import (
    POCKET_RETAIN,
    _flip_mask,
    _pocket_masks,
    _sym_prep,
    _topology,
    jag_score,
    naive_mask,
    reference_coverage,
    reference_mask,
)
from iconograph.application.compress.straighten import (
    snap_polygon_run2,
    snap_triangle_run2,
)

_ROUND = (Circle, Ellipse, RotatedEllipse)

TOOTH_MAX = 3        # fg pixel with <= this many fg 8-neighbours is a tooth/spur (clear)
NOTCH_MIN = 5        # bg pixel with >= this many fg 8-neighbours is a concave dent (fill)
MAX_SWEEPS = 256     # hard convergence bound; the search breaks at the true fixpoint first
W_ACC = 20.0         # accuracy weight: 1 − area IoU vs the reference (high enough that a
#                      reconnection move's fid gain beats the transient jag bump)
W_SOFT = 0.02        # soft-coverage L1 tie term (deterministic tie-break toward truth)
EPS = 1e-9
W_AXIS = 1.5         # axis-blocky weight per ISOLATED diagonal boundary step (a staircase
#                      tooth approximating a CURVE). Ramped by (T-16)/8 so T=16 is a no-op
#                      (already the 'perfect' cube) and T=24 is pushed to the SAME blocky
#                      goal — jag is DP-vertex-based and a smooth staircase reads as low jag,
#                      so without this the 24px render drifts curvier than 16.

TRI_EPS_FRAC = 0.04  # approxPolyDP epsilon (× hull perimeter): coarse enough to collapse
#                      a triangle/flat-top silhouette to 3–4 vertices (D1 uses 0.01 for fine)
TRI_AGREE = 0.90     # min IoU(coarse hull-polygon, solid silhouette) to call it a triangle —
#                      a deep bottom notch (the polygon would over-fill it) fails the gate

# ── neck divots: cubify-tier OUTLINE SEPARATION (figma's stacked lobes) ───
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


# ── path A: label-driven cubic forms (simple round labels) ───────────────


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


def _label_render(parts: list[Part], canvas_hw: tuple[int, int],
                  target: int) -> np.ndarray | None:
    """Render a SIMPLE ROUND LABEL cubically, or None if the construction is compound.
    Eligible: exactly one round ADD, with at most one round SUBTRACT (a circle -> square;
    an annulus -> add-bbox-square minus subtract-bbox-square = square ring). A square ring
    is accepted only when it is a valid closed ring (topology (1, 1))."""
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
    straight segments per diagonal), then carve the counter (the canonical hole) when
    it survives at target as a closed ring. None when not triangle-confident (H8)."""
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


# ── path B: silhouette pixel-cubify (compound constructions) ─────────────


def _nbcount(mask: np.ndarray) -> np.ndarray:
    """Per-pixel count (int16, 0..8) of FOREGROUND 8-neighbours, center excluded.
    BORDER_CONSTANT pads with 0 (off-canvas = background); normalise to {0,1} first."""
    b = (mask > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return cv2.filter2D(b, cv2.CV_16S, k, borderType=cv2.BORDER_CONSTANT)


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
    drop the HOLE count below min(current, canon) (a canonical hole is never destroyed)."""
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
    axis-blocky (T>16). The axis term ramps from 0 at T=16 so the 'perfect' cube is
    byte-unchanged and the larger grids pursue the SAME blocky geometry (user ruling)."""
    T = mask.shape[0]
    w_axis = W_AXIS * max(0.0, (T - 16) / 8.0)
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
    """Mirror of (y, x) about the axis: x reflects if vertical, else y (a2 - i)."""
    vertical, a2 = axis
    return (y, a2 - x) if vertical else (a2 - y, x)


def _greedy(mask, score_fn, ref, cov, pockets, topo_ref, topo_canon, axis):
    """Deterministic best-improvement hill-climb to a fixpoint under score_fn (lower =
    better). Each accepted flip clears the anti-glob + topology guards; with `axis` set,
    every flip is a MIRROR PAIR so a symmetric input stays exactly symmetric. H8: no RNG."""
    T = mask.shape[0]
    s_cur = score_fn(mask, ref, cov)
    cur_topo = _topology(mask)
    for _ in range(MAX_SWEEPS):
        nb = _nbcount(mask)
        best_gain, best_mask, best_s = EPS, None, None
        for y, x, val in _candidates(mask, nb, ref):
            trial = mask.copy()
            trial[y, x] = val
            if axis is not None:
                ty, tx = _twin(y, x, axis)
                if not (0 <= ty < T and 0 <= tx < T):
                    continue
                trial[ty, tx] = val
            globbed = False
            for pk in pockets:
                pn = int(np.count_nonzero(pk))
                if pn and int(np.count_nonzero((pk > 0) & (trial == 0))) < POCKET_RETAIN * pn:
                    globbed = True
                    break
            if globbed:
                continue
            if not _topo_ok(_topology(trial), cur_topo, topo_ref, topo_canon):
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
                    pockets: list[np.ndarray]) -> np.ndarray:
    """Idealize toward the canonical topology: greedily fill the bg-adjacent pixel that
    most reduces (component + hole) distance to canon, bridging a fragmented loop closed.
    Pocket-guarded. Deterministic (H8)."""
    def td(t: tuple[int, int]) -> int:
        return abs(t[0] - topo_canon[0]) + abs(t[1] - topo_canon[1])

    for _ in range(MAX_SWEEPS):
        d_cur = td(_topology(mask))
        if d_cur == 0:
            break
        nb = _nbcount(mask)
        ys, xs = np.nonzero((mask == 0) & (nb >= 1))
        best_d, best_mask = d_cur, None
        for y, x in sorted((int(a), int(b)) for a, b in zip(ys, xs)):
            trial = mask.copy()
            trial[y, x] = 255
            globbed = False
            for pk in pockets:
                pn = int(np.count_nonzero(pk))
                if pn and int(np.count_nonzero((pk > 0) & (trial == 0))) < POCKET_RETAIN * pn:
                    globbed = True
                    break
            if globbed:
                continue
            d2 = td(_topology(trial))
            if d2 < best_d:
                best_d, best_mask = d2, trial
        if best_mask is None:
            break
        mask = best_mask
    return mask


# ── neck divots (cubify-tier outline separation) ─────────────────────────

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


# ── dispatch ─────────────────────────────────────────────────────────────


def cubify(parts: list[Part], canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """Render the construction at the feed resolution. Simple round labels render their
    cubic form (square / square ring); compound constructions are silhouette-cubified.
    (target, target) uint8 {0,255}. Deterministic (H8)."""
    label = _label_render(parts, canvas_hw, target)
    if label is not None:
        return label
    tri = _triangle_label(parts, canvas_hw, target)
    if tri is not None:
        return tri

    h, w = canvas_hw
    ref = reference_mask(parts, canvas_hw, target)
    naive = naive_mask(parts, canvas_hw, target)
    cov = reference_coverage(parts, canvas_hw, target)
    nm = (naive > 0).astype(np.uint8) * 255
    pockets = _pocket_masks(ref)
    topo_ref = _topology(ref)
    topo_canon = _topology(_compound_mask(parts, h, w))

    # Symmetry (mirror-pair moves + symmetric start) applies only to a CONNECTED
    # construction; on a fragmented naive reconnection takes priority and symmetry is off.
    fragmented = _topology(nm)[0] > topo_canon[0]
    axis = None if fragmented else _cubify_axis(parts, canvas_hw, target)
    if axis is not None:
        vertical, a2 = axis
        mask = nm | _flip_mask(nm, a2, vertical)
    else:
        mask = nm

    if fragmented:
        mask = _greedy(mask, _accuracy, ref, cov, pockets, topo_ref, topo_canon, axis)
        mask = _close_to_canon(mask, topo_canon, pockets)
    mask = _greedy(mask, _cubic, ref, cov, pockets, topo_ref, topo_canon, axis)

    # Outline-separate an overrepresented stack of round lobes (figma): punch a sub-pixel
    # neck divot between consecutive lobes so they read as separate, proportional geometry.
    if target <= DIVOT_MAX_T:
        vertical, necks = _stacked_lobes(parts, canvas_hw)
        mask = _carve_necks(mask, vertical, necks, canvas_hw, target)
    return mask
