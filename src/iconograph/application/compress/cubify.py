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
from skimage.morphology import medial_axis

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
    _nbcount,
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
    scale_about_center,
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
#                      tooth approximating a CURVE) at T=24. jag is DP-vertex-based and a smooth
#                      staircase reads as low jag, so without this the render drifts curvy.
W_AXIS_T16 = 0.5     # the SAME weight at T=16, ramping linearly to W_AXIS at T=24. NOT zero
#                      (the old ramp's "T=16 is already a perfect cube" assumption was wrong —
#                      gmail's compound M rounded its corners/feet at 16 without any pressure)
#                      and NOT the full 1.5 (that over-carves discord's ~3px eye holes into a
#                      notch at 16); 0.5 fully de-curves gmail's M while keeping discord clean
#                      (W_AXIS sweep on gmail+discord@16).

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
    counter."""
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
    """Mirror of (y, x) about the axis: x reflects if vertical, else y (a2 - i)."""
    vertical, a2 = axis
    return (y, a2 - x) if vertical else (a2 - y, x)


def _glob_ok(trial: np.ndarray, pockets: list[np.ndarray]) -> bool:
    """True if `trial` keeps >= POCKET_RETAIN of EVERY anti-glob pocket open (no glob). The one
    anti-glob test shared by _greedy / _close_to_canon / _reconcile_structure (H5)."""
    for pk in pockets:
        pn = int(np.count_nonzero(pk))
        if pn and int(np.count_nonzero((pk > 0) & (trial == 0))) < POCKET_RETAIN * pn:
            return False
    return True


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
            if not _glob_ok(trial, pockets):
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
            if not _glob_ok(trial, pockets):
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


# ── widen narrow OPEN negative-space gaps (gmail M-valley) ────────────────────────────
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
    vscode's notch). Border flood-fill. Scopes the widen to OPEN gaps: enclosed counters are
    features the widen must not enlarge (it over-fires on discord's eyes otherwise), and an
    enclosed notch like vscode's needs a different lever (subtract-scaling can't clear its tip)."""
    m = (mask > 0).astype(np.uint8)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    flooded = pad.copy()
    cv2.floodFill(flooded, ff, (0, 0), 2)
    outer = flooded[1:-1, 1:-1] == 2
    return bool(np.any(region & outer))


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


# ── dispatch ─────────────────────────────────────────────────────────────


# ── cross-resolution structure constraint (@24 → @16) ─────────────────────
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


def _structure_core(structure_ref: np.ndarray, target: int) -> np.ndarray:
    """The larger-sibling (@24) render's connectivity BACKBONE mapped to the target grid: the
    medial-axis skeleton (spine + branches) re-stroked as a thin solid stroke. structure_ref:
    the sibling cubify mask (src×src). Returns (target,target) uint8 — the form @16 must stay
    connected along. Deterministic (H8): medial_axis rng=0, exact ratio (24→16 = 2/3, no .5
    ties), row-major. At BACKBONE_CAP=0 the re-stroke is the 1-px centerline (radius 0 = 1px);
    a 2/3 DOWNSCALE cannot disconnect a connected centerline (no bridging needed), and the
    centerline loops around holes so @24's counters are preserved (a fatter cap would merge
    disks across a narrow counter and over-fill it)."""
    src = int(structure_ref.shape[0])
    skel, dist = medial_axis(structure_ref > 0, return_distance=True, rng=0)
    ratio = target / src
    core = np.zeros((target, target), np.uint8)
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys.tolist(), xs.tolist()):
        ty = min(int(round(y * ratio)), target - 1)
        tx = min(int(round(x * ratio)), target - 1)
        r = min(BACKBONE_CAP, int(round(float(dist[y, x]) * ratio)))
        cv2.circle(core, (tx, ty), max(0, r), 255, -1)
    return core


def _reconcile_structure(mask: np.ndarray, core: np.ndarray, pockets: list[np.ndarray],
                         topo_ref: tuple[int, int], topo_canon: tuple[int, int],
                         axis: tuple[bool, int] | None) -> np.ndarray:
    """FILL @16 toward the sibling backbone `core`: greedily set each bg pixel that lies ON the
    backbone AND is 8-adjacent to fg, accepting it only if it clears the anti-glob pocket guard
    (_glob_ok) and _topo_ok (no fragmentation / no destroyed canonical hole). With `axis` set,
    every fill is applied as a MIRROR PAIR (like _greedy) so a symmetric input stays symmetric.
    Re-solidifies a scattered spine into a connected bar; iterates so a 2px gap bridges in two
    passes. Idempotent → byte-identical no-op when mask already covers core. Modeled on
    _close_to_canon; deterministic (H8): row-major order, no RNG."""
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
            rec = _reconcile_structure(mask, core, pockets, topo_ref, topo_canon, axis)
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
