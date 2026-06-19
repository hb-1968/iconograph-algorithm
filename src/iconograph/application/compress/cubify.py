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
    RotatedEllipse,
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

_ROUND = (Circle, Ellipse, RotatedEllipse)

TOOTH_MAX = 3        # fg pixel with <= this many fg 8-neighbours is a tooth/spur (clear)
NOTCH_MIN = 5        # bg pixel with >= this many fg 8-neighbours is a concave dent (fill)
MAX_SWEEPS = 256     # hard convergence bound; the search breaks at the true fixpoint first
W_ACC = 20.0         # accuracy weight: 1 − area IoU vs the reference (high enough that a
#                      reconnection move's fid gain beats the transient jag bump)
W_SOFT = 0.02        # soft-coverage L1 tie term (deterministic tie-break toward truth)
EPS = 1e-9


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


def _cubic(mask: np.ndarray, ref: np.ndarray, cov: np.ndarray) -> float:
    """PHASE-B objective (lower = better): cubification + accuracy tradeoff + soft tie."""
    jag = jag_score(mask)
    soft = float(np.abs((mask > 0).astype(np.float32) - cov).sum())
    return jag + W_ACC * (1.0 - _iou_bool(mask, ref)) + W_SOFT * soft


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


# ── dispatch ─────────────────────────────────────────────────────────────


def cubify(parts: list[Part], canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """Render the construction at the feed resolution. Simple round labels render their
    cubic form (square / square ring); compound constructions are silhouette-cubified.
    (target, target) uint8 {0,255}. Deterministic (H8)."""
    label = _label_render(parts, canvas_hw, target)
    if label is not None:
        return label

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
    return mask
