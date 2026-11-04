"""Legacy beam-search baseline for signed primitive constructions.

Propose additions and subtractions, score whole-construction IoU, then prune
redundant parts and fill gaps. Fits below ADEQUATE_IOU become deferrals.
The production build uses contour-based decomposition instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from iconograph.application.compress.common import (
    Circle,
    Ellipse,
    Part,
    Primitive,
    Rectangle,
    RotatedEllipse,
    RotatedRectangle,
    RoundedRectangle,
    Triangle,
    _cand_inscribed_circle,
    _compound_iou,
    _compound_mask,
    _detect_holes,
    _extent_ok,
    _fill_holes,
    _fit_circle,
    _fit_ellipse,
    _fit_hole,
    _fit_rectangle,
    _fit_rotated_ellipse,
    _fit_rotated_rect,
    _fit_triangle,
    _iou_bool,
    _largest_contour,
    _largest_lobe,
    _min_feature_px,
    _offset_part,
    _raster_part,
)
from iconograph.application.compress.segment_prim_classify import Classified
from iconograph.application.compress.L2_feature_segment import (
    Feature,
    FeatureSegmentation,
    ShapeHint,
)

if TYPE_CHECKING:
    from iconograph.application.categorize import Categorization


# engine knobs
BEAM_K = 4                  # partial constructions kept per beam depth
BEAM_M = 4                  # candidate children expanded per state per step
MAX_PARTS = 6               # construction part budget (beam depth bound)
ACCEPT_IOU = 0.92           # early-exit: a construction this good stops the search
MIN_GAIN = 0.005            # a step must add ≥ this much IoU — kept low for fidelity (deferring a
#                             shape is worse than one extra part); the capsule template, the
#                             _rank_key tie-break, and pass-2 elimination carry the simplicity
ADEQUATE_IOU = 0.90         # below this final IoU → Deferral (no adequate fit found)
PRUNE_TOL = 0.01            # backward elimination: drop a part if IoU loss < this

# single-primitive hints → a 1-part seed straight from the matching fitter
_SINGLE: frozenset[str] = frozenset(
    {"circle", "ellipse", "rectangle", "triangle", "rotated_rect", "rotated_ellipse"}
)


@dataclass(frozen=True)
class FittedShape:
    bbox: tuple[int, int, int, int]      # (x, y, w, h) full-image
    color_rgb: tuple[int, int, int]
    parts: tuple[Part, ...]              # FULL-image coords; adds OR'd, then subs carved
    iou: float                           # reconstruction IoU vs the source feature mask
    shape_hint: ShapeHint                # provenance: the L2 hint that seeded the fit


@dataclass(frozen=True)
class Deferral:
    bbox: tuple[int, int, int, int]
    color_rgb: tuple[int, int, int]
    mask: np.ndarray                     # bbox-cropped {0,255} mask, forwarded to a later pass
    iou: float                           # best-attempt IoU (< ADEQUATE_IOU)
    shape_hint: ShapeHint                # the hint that was tried


@dataclass
class FittedDecomposition:
    image_shape: tuple[int, int, int]
    shapes: list[Classified | FittedShape | Deferral]


# #
# candidate generation (enclosing + inscribed fits over a residual lobe)
# #
def _fitter_prims(lobe: np.ndarray, h: int, w: int, min_feat: int) -> list[Primitive]:
    """Every enclosing / inscribed primitive fit to a residual lobe, in a fixed order
    (deterministic tie-break: ellipse-family first). Enclosing fits reach a curved
    boundary; the inscribed circle anchors a fat round core. Each gated by min-feature
    extent (drops sub-feature noise)."""
    c = _largest_contour(lobe)
    if c is None:
        return []
    cands: list[tuple[Primitive, np.ndarray, float] | None] = [
        _fit_rotated_ellipse(c, h, w),
        _fit_ellipse(c, h, w, 0, 0),
        _fit_rotated_rect(c, h, w),
        _fit_rectangle(c, h, w, 0, 0),
        _fit_triangle(c, h, w, 0, 0),
        _cand_inscribed_circle(lobe, h, w),
    ]
    prims: list[Primitive] = []
    for cand in cands:
        if cand is None:
            continue
        prim, fit, _area = cand
        if _extent_ok(fit, min_feat):
            prims.append(prim)
    return prims


def _add_candidates(under: np.ndarray, h: int, w: int, min_feat: int) -> list[Part]:
    """ADD primitives over the largest uncovered-target lobe."""
    lobe = _largest_lobe(under)
    if lobe is None:
        return []
    return [Part(p, "add") for p in _fitter_prims(lobe, h, w, min_feat)]


def _sub_candidates(over: np.ndarray, h: int, w: int, min_feat: int) -> list[Part]:
    """SUBTRACT primitives over the largest spilled (recon-outside-target) lobe."""
    if not _extent_ok(over, min_feat):
        return []
    lobe = _largest_lobe(over)
    if lobe is None:
        return []
    return [Part(p, "subtract") for p in _fitter_prims(lobe, h, w, min_feat)]


# #
# deterministic ordering keys
# #
def _prim_key(p: Primitive) -> tuple:
    """A total ordering key for a primitive: type name first (disambiguates types),
    then rounded numeric fields. Same structure within a type → comparable; the leading
    type name resolves any cross-type comparison before field shapes can differ."""
    t = type(p).__name__
    if isinstance(p, Circle):
        return (t, round(p.cx, 2), round(p.cy, 2), round(p.r, 2))
    if isinstance(p, Ellipse):
        return (t, round(p.cx, 2), round(p.cy, 2), round(p.rx, 2), round(p.ry, 2))
    if isinstance(p, RotatedEllipse):
        return (t, round(p.cx, 2), round(p.cy, 2), round(p.rx, 2), round(p.ry, 2), round(p.angle, 2))
    if isinstance(p, Rectangle):
        return (t, p.x, p.y, p.w, p.h)
    if isinstance(p, RotatedRectangle):
        return (t, round(p.cx, 2), round(p.cy, 2), round(p.w, 2), round(p.h, 2), round(p.angle, 2))
    if isinstance(p, RoundedRectangle):                                   # defensive: L3 never emits one today
        return (t, round(p.cx, 2), round(p.cy, 2), round(p.w, 2), round(p.h, 2),
                round(p.angle, 2), round(p.corner_radius, 2))
    return (t, *(round(v, 2) for xy in (p.v1, p.v2, p.v3) for v in xy))   # Triangle


def _part_key(part: Part) -> tuple:
    return (part.op, _prim_key(part.primitive))


def _parts_key(parts: list[Part] | tuple[Part, ...]) -> tuple:
    """Order-independent key for a construction (sorts its part keys)."""
    return tuple(sorted(_part_key(p) for p in parts))


def _rank_key(iou: float, parts: list[Part] | tuple[Part, ...]) -> tuple:
    """Selection key (lower = better): higher (exact) IoU first, then — only to break a genuine tie —
    fewer subtracts (additive-favoring), fewer parts, then the deterministic part key. Exact (not
    bucketed) IoU keeps the beam's greedy trajectory unperturbed; IoU-bucketing here regressed docker
    0.92→0.88 by diverting the search into a simpler-but-worse basin. The capsule template + pass-2
    elimination are the real simplicity levers; this is just a deterministic tie-break."""
    return (
        -iou,
        sum(1 for p in parts if p.op == "subtract"),
        len(parts),
        _parts_key(parts),
    )


# #
# pass 1 — greedy + back-stepping beam search
# #
def _expand(parts: list[Part], target: np.ndarray, h: int, w: int,
            min_feat: int) -> list[list[Part]]:
    """Up to BEAM_M child constructions from one state — best ADD over the uncovered
    residual and best SUBTRACT over the spilled residual compete in one ranked pool;
    only children clearing MIN_GAIN survive."""
    recon = _compound_mask(parts, h, w)
    tb = target.astype(bool)
    rb = recon.astype(bool)
    under = ((tb & ~rb).astype(np.uint8)) * 255
    over = ((rb & ~tb).astype(np.uint8)) * 255
    base = _iou_bool(recon, target)
    cands = _add_candidates(under, h, w, min_feat) + _sub_candidates(over, h, w, min_feat)

    scored: list[tuple[list[Part], float]] = []
    for cand in cands:
        child = parts + [cand]
        iou = _compound_iou(child, target, h, w)
        if iou - base < MIN_GAIN:
            continue
        scored.append((child, iou))
    scored.sort(key=lambda t: _rank_key(t[1], t[0]))
    return [child for child, _iou in scored[:BEAM_M]]


def _pass1_beam(target: np.ndarray, seed: list[Part], h: int, w: int,
                min_feat: int) -> list[Part]:
    """Beam search from a (possibly empty) hint seed. Keeping the top-K partials means a
    locally-second-best step survives to win globally — the back-stepping. Bounded:
    ≤ BEAM_K·MAX_PARTS states, each ≤ a handful of candidates. Returns the best parts."""
    best = (_compound_iou(seed, target, h, w), _parts_key(seed), list(seed))
    beam = [best]
    for _depth in range(MAX_PARTS - len(seed)):
        if beam[0][0] >= ACCEPT_IOU:
            break
        children: list[tuple[float, tuple, list[Part]]] = []
        seen: set[tuple] = set()
        for _iou, _key, parts in beam:
            for child in _expand(parts, target, h, w, min_feat):
                k = _parts_key(child)
                if k in seen:
                    continue
                seen.add(k)
                children.append((_compound_iou(child, target, h, w), k, child))
        if not children:
            break
        children.sort(key=lambda t: _rank_key(t[0], t[2]))
        beam = children[:BEAM_K]
        if _rank_key(beam[0][0], beam[0][2]) < _rank_key(best[0], best[2]):
            best = beam[0]
    return best[2]


# #
# pass 2 — redundancy cleanup
# #
def _eliminate(parts: list[Part], target: np.ndarray, h: int, w: int) -> list[Part]:
    """Backward elimination to a fixpoint: drop any part whose removal costs < PRUNE_TOL
    IoU. Fixed order (adds first, largest raster-area first, then subs; _part_key as the
    final tiebreak) makes the fixpoint deterministic. Removes overlapping redundancy —
    several primitives piled on one spot collapse to the few that actually cover it."""
    parts = list(parts)
    while True:
        base = _compound_iou(parts, target, h, w)
        order = sorted(
            range(len(parts)),
            key=lambda i: (
                parts[i].op != "add",                                       # adds before subs
                -int(np.count_nonzero(_raster_part(parts[i].primitive, h, w))),
                _part_key(parts[i]),
            ),
        )
        dropped = False
        for i in order:
            trial = parts[:i] + parts[i + 1:]
            if _compound_iou(trial, target, h, w) >= base - PRUNE_TOL:
                parts = trial
                dropped = True
                break
        if not dropped:
            return parts


def _gap_fill(parts: list[Part], target: np.ndarray, h: int, w: int,
              min_feat: int) -> list[Part]:
    """Cover any remaining uncovered-target component (≥ min_feat extent) with one ADD
    primitive — the no-disjoint / arc-coverage guarantee. Each add is kept only if it
    raises the overall IoU, so a spilling cover can never hurt the score."""
    parts = list(parts)
    recon = _compound_mask(parts, h, w)
    under = ((target.astype(bool) & ~recon.astype(bool)).astype(np.uint8)) * 255
    n, lbl, _stats, _cent = cv2.connectedComponentsWithStats(under, connectivity=8)
    for i in range(1, n):
        comp = ((lbl == i).astype(np.uint8)) * 255
        if not _extent_ok(comp, min_feat):
            continue
        best_p, best_iou = None, 0.0
        for p in _fitter_prims(comp, h, w, min_feat):
            iou = _iou_bool(_raster_part(p, h, w), comp)
            if iou > best_iou:
                best_iou, best_p = iou, p
        if best_p is None:
            continue
        cand = Part(best_p, "add")
        if _compound_iou(parts + [cand], target, h, w) > _compound_iou(parts, target, h, w):
            parts.append(cand)
    return parts


def _pass2_cleanup(parts: list[Part], target: np.ndarray, h: int, w: int,
                   min_feat: int) -> list[Part]:
    """Eliminate redundancy, fill any gap, then eliminate again (gap-fill may expose a
    now-redundant earlier part)."""
    parts = _eliminate(parts, target, h, w)
    parts = _gap_fill(parts, target, h, w, min_feat)
    parts = _eliminate(parts, target, h, w)
    return parts


# #
# hint routing — the L2 shape_hint seeds the engine
# #
CAPSULE_MIN_ASPECT = 1.8        # elongated enough to be a capsule (rect + cap) vs a blob


def _fit_capsule(mask: np.ndarray, h: int, w: int) -> list[Part]:
    """An elongated rounded feature → a rotated rectangle ∪ a circular cap (the user's
    'rectangle capped with an ellipse'). From minAreaRect, try {bare, +end cap, −end cap, both
    caps} and keep the highest-IoU — the trial auto-detects which end is rounded, sidestepping the
    minAreaRect angle-sign convention. Returns [] when the feature isn't elongated (the caller
    falls back to a bare-rect seed)."""
    c = _largest_contour(mask)
    if c is None:
        return []
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) < CAPSULE_MIN_ASPECT:
        return []
    L, S = max(rw, rh), min(rw, rh)
    r = S / 2.0
    if rw >= rh:                                     # long axis = width direction
        ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    else:                                            # long axis = height direction (⊥ the width)
        ux, uy = -math.sin(math.radians(ang)), math.cos(math.radians(ang))

    def _body(shorten: int, shift: float) -> Part:
        blen = max(1.0, L - shorten * r)
        bcx, bcy = cx + shift * ux, cy + shift * uy
        size = (blen, S) if rw >= rh else (S, blen)
        return Part(RotatedRectangle(cx=float(bcx), cy=float(bcy),
                                     w=float(size[0]), h=float(size[1]), angle=float(ang)), "add")

    def _cap(sign: float) -> Part:
        ex, ey = cx + sign * (L / 2.0 - r) * ux, cy + sign * (L / 2.0 - r) * uy
        return Part(Circle(cx=float(ex), cy=float(ey), r=float(r)), "add")

    candidates = [
        [_body(0, 0.0)],                             # bare rect
        [_body(1, -r / 2.0), _cap(+1.0)],            # cap the +end (body shifts toward −end)
        [_body(1, +r / 2.0), _cap(-1.0)],            # cap the −end
        [_body(2, 0.0), _cap(+1.0), _cap(-1.0)],     # cap both ends (full stadium)
    ]
    best, best_iou = candidates[0], -1.0
    for parts in candidates:
        iou = _compound_iou(parts, mask, h, w)
        if iou > best_iou:
            best_iou, best = iou, parts
    return best


def _seed_single(mask: np.ndarray, hint: ShapeHint, h: int, w: int) -> list[Part]:
    """One-part seed from a single-primitive hint (the matching fitter)."""
    c = _largest_contour(mask)
    if c is None:
        return []
    fit: tuple[Primitive, np.ndarray, float] | None
    if hint == "circle":
        fit = _fit_circle(c, h, w, 0, 0)
    elif hint == "ellipse":
        fit = _fit_ellipse(c, h, w, 0, 0)
    elif hint == "rectangle":
        fit = _fit_rectangle(c, h, w, 0, 0)
    elif hint == "triangle":
        fit = _fit_triangle(c, h, w, 0, 0)
    elif hint == "rotated_rect":
        fit = _fit_rotated_rect(c, h, w)
    elif hint == "rotated_ellipse":
        fit = _fit_rotated_ellipse(c, h, w)
    else:
        fit = None
    return [Part(fit[0], "add")] if fit is not None else []


def _seed_half_circle(mask: np.ndarray, h: int, w: int) -> list[Part]:
    """Dome: the min-enclosing circle (covers the arc) MINUS the best axis-aligned half-plane
    rectangle (carves the empty half + arc halo). Tries all four sides + the bare circle and keeps
    the highest IoU — robust to which way the dome faces (L0 has axis-aligned it)."""
    c = _largest_contour(mask)
    if c is None:
        return []
    (cx, cy), r = cv2.minEnclosingCircle(c)
    if r < 1.0:
        return []
    circle = Part(Circle(cx=float(cx), cy=float(cy), r=float(r)), "add")
    icx, icy = int(round(cx)), int(round(cy))
    halves = (
        Rectangle(x=0, y=0, w=w, h=max(1, icy)),                # carve above the centre
        Rectangle(x=0, y=icy, w=w, h=max(1, h - icy)),          # carve below
        Rectangle(x=0, y=0, w=max(1, icx), h=h),                # carve left
        Rectangle(x=icx, y=0, w=max(1, w - icx), h=h),          # carve right
    )
    best, best_iou = [circle], _compound_iou([circle], mask, h, w)
    for rect in halves:
        parts = [circle, Part(rect, "subtract")]
        iou = _compound_iou(parts, mask, h, w)
        if iou > best_iou:
            best_iou, best = iou, parts
    return best


def _best_cover(region: np.ndarray, h: int, w: int) -> Primitive | None:
    """Single best-IoU primitive covering a solid region (a template's outer `add`)."""
    c = _largest_contour(region)
    if c is None:
        return None
    cands = [
        _fit_rotated_ellipse(c, h, w), _fit_ellipse(c, h, w, 0, 0),
        _fit_rotated_rect(c, h, w), _fit_rectangle(c, h, w, 0, 0),
        _fit_triangle(c, h, w, 0, 0), _cand_inscribed_circle(region, h, w),
    ]
    best, best_iou = None, -1.0
    for cand in cands:
        if cand is None:
            continue
        prim, fit, _a = cand
        iou = _iou_bool(fit, region)
        if iou > best_iou:
            best_iou, best = iou, prim
    return best


def _seed_holed(mask: np.ndarray, h: int, w: int, min_feat: int) -> list[Part]:
    """Holed shape: cover the hole-filled outer solid, then subtract a fitted primitive per
    interior hole (negation). The beam refines the outer if one primitive isn't enough."""
    outer = _best_cover(_fill_holes(mask), h, w)
    if outer is None:
        return []
    parts = [Part(outer, "add")]
    for hole in _detect_holes(mask):
        sub = _fit_hole(hole, h, w, min_feat)
        if sub is not None:
            parts.append(Part(sub, "subtract"))
    return parts


def _seed_annular(mask: np.ndarray, h: int, w: int, min_feat: int) -> list[Part]:
    """Ring / near-complete band: outer ellipse MINUS the fitted inner hole. When the band is
    open (no closed hole) the seed is just the outer ellipse — the beam carves the empty centre."""
    c = _largest_contour(mask)
    if c is None:
        return []
    outer, _f, _a = _fit_ellipse(c, h, w, 0, 0)
    parts = [Part(outer, "add")]
    holes = _detect_holes(mask)
    if holes:
        inner = _fit_hole(max(holes, key=lambda hm: int(np.count_nonzero(hm))), h, w, min_feat)
        if inner is not None:
            parts.append(Part(inner, "subtract"))
    return parts


def _seed_sunburst(mask: np.ndarray, h: int, w: int, min_feat: int) -> list[Part]:
    """Radial burst: an inscribed central disc + a min-enclosing triangle per spoke. Spokes =
    mask − disc, eroded 1px to sever them at the core (then dilated back to fit); the largest few
    within the part budget."""
    disc = _cand_inscribed_circle(mask, h, w)
    if disc is None:
        return []
    disc_prim, disc_fit, _a = disc
    parts = [Part(disc_prim, "add")]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spokes = cv2.erode(cv2.subtract(mask, disc_fit), se)
    n, lbl, stats, _cent = cv2.connectedComponentsWithStats(spokes, connectivity=8)
    for i in sorted(range(1, n), key=lambda j: -int(stats[j, cv2.CC_STAT_AREA]))[: MAX_PARTS - 1]:
        comp = cv2.dilate(((lbl == i).astype(np.uint8)) * 255, se)
        if not _extent_ok(comp, min_feat):
            continue
        c = _largest_contour(comp)
        if c is None:
            continue
        tri, _ft, _ar = _fit_triangle(c, h, w, 0, 0)
        parts.append(Part(tri, "add"))
    return parts


def _seed_for(mask: np.ndarray, hint: ShapeHint, h: int, w: int, min_feat: int) -> list[Part]:
    """Hint → seed construction (the simplicity-biased beam then refines / confirms it):
    elongated rect-likes → capsule; other single primitives → 1-part fit; half_circle → circle −
    half-plane; holed → fill − holes; ring/annular_band → outer − inner ellipse; sunburst → disc +
    spoke triangles; annular_sector and None/defer → cold start (the beam builds from scratch)."""
    if hint in ("rotated_rect", "rectangle"):
        return _fit_capsule(mask, h, w) or _seed_single(mask, hint, h, w)
    if hint in _SINGLE:
        return _seed_single(mask, hint, h, w)
    if hint == "half_circle":
        return _seed_half_circle(mask, h, w)
    if hint == "holed":
        return _seed_holed(mask, h, w, min_feat)
    if hint in ("ring", "annular_band"):
        return _seed_annular(mask, h, w, min_feat)
    if hint == "sunburst":
        return _seed_sunburst(mask, h, w, min_feat)
    return []


# #
# per-feature fit (both passes) + public orchestrator
# #
def _pad_for(th: int, tw: int, min_feat: int) -> int:
    """Padding margin around the feature so an enclosing primitive's overshoot lands in
    background (and is carvable by a subtract) instead of being silently clipped to the
    tight bbox — without it the negation mechanism is defeated (a clipped circle reads as a
    perfect half-disc and never gets carved)."""
    return max(min_feat, max(th, tw) // 2)


def _fit_one(feature: Feature, min_feat: int) -> tuple[list[Part], list[Part], float]:
    """Fit one feature on a PADDED canvas, returning (pass-1 parts, pass-2 parts, pass-2 IoU)
    with parts already in FULL-IMAGE coords. Padding lets enclosing fits overshoot into
    background so a subtract can carve them; the IoU is measured on the padded canvas, so it
    honestly reflects spill (a clipped tight-bbox IoU would not). A single-primitive seed
    that already clears ACCEPT_IOU skips the beam; everything else runs the full beam."""
    tight = feature.mask
    th, tw = tight.shape[:2]
    x, y = feature.bbox[0], feature.bbox[1]
    pad = _pad_for(th, tw, min_feat)
    h, w = th + 2 * pad, tw + 2 * pad
    target = np.zeros((h, w), dtype=np.uint8)
    target[pad:pad + th, pad:pad + tw] = tight

    seed = _seed_for(target, feature.shape_hint, h, w, min_feat)
    if seed and feature.shape_hint in _SINGLE and _compound_iou(seed, target, h, w) >= ADEQUATE_IOU:
        pass1 = seed                                 # a single-primitive / capsule template that is
        #                                              already a FIT — trust the clean canonical form
    else:
        # Cold-start beam is the floor. A composite/half-circle template only COMPETES with it — a
        # template seed must never trap the search into a worse basin (it regressed firefox 0.86→0.54
        # and vscode 0.93→0.88 when used as the sole seed). Run both, keep the rank-better.
        pass1 = _pass1_beam(target, [], h, w, min_feat)
        if seed:
            templ = _pass1_beam(target, seed, h, w, min_feat)
            if _rank_key(_compound_iou(templ, target, h, w), templ) < \
                    _rank_key(_compound_iou(pass1, target, h, w), pass1):
                pass1 = templ
    pass2 = _pass2_cleanup(pass1, target, h, w, min_feat)
    iou = _compound_iou(pass2, target, h, w)

    dx, dy = x - pad, y - pad                        # padded-crop → full-image coords
    return (
        [_offset_part(p, dx, dy) for p in pass1],
        [_offset_part(p, dx, dy) for p in pass2],
        float(iou),
    )


def fit_features(
    fs: FeatureSegmentation, cat: "Categorization | None" = None
) -> FittedDecomposition:
    """L3 — fit every miss-derived `Feature` into an overlapping-CSG construction; pass
    `Classified` (already L1-fit) through unchanged. A feature whose cleaned construction
    clears ADEQUATE_IOU becomes a `FittedShape`; otherwise a `Deferral` (no adequate fit).
    `cat` is accepted for forward-compat (e.g. a future burst relaxation) but unused here."""
    min_feat = _min_feature_px(int(fs.image_shape[0]))
    shapes: list[Classified | FittedShape | Deferral] = []
    for seg in fs.segments:
        if isinstance(seg, Classified):
            shapes.append(seg)
            continue
        feat = seg                                   # a Feature
        _pass1, parts, iou = _fit_one(feat, min_feat)
        color = tuple(int(c) for c in feat.color_rgb)
        if parts and iou >= ADEQUATE_IOU:
            shapes.append(FittedShape(
                bbox=feat.bbox, color_rgb=color, parts=tuple(parts),
                iou=iou, shape_hint=feat.shape_hint,
            ))
        else:
            shapes.append(Deferral(
                bbox=feat.bbox, color_rgb=color, mask=feat.mask.copy(),
                iou=iou, shape_hint=feat.shape_hint,
            ))
    return FittedDecomposition(image_shape=fs.image_shape, shapes=shapes)


# #
# overlay renderer (testbed / H7 strip + l3_debug share this)
# #
_FIT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0x3C, 0xB4, 0x4B), (0x43, 0x63, 0xD8), (0xF5, 0x82, 0x31),
    (0x91, 0x1E, 0xB4), (0x46, 0x99, 0x90), (0x9A, 0x63, 0x24),
)
_DEFER_COLOR = (0xE6, 0x19, 0x4B)                    # red — a deferral must visually pop


def draw_fit_overlay(aligned: np.ndarray, decomp: FittedDecomposition) -> np.ndarray:
    """Render L3 for review: each FittedShape's reconstructed mask in a palette colour,
    each Deferral's raw mask in red, blended over the aligned image. Classified
    pass-throughs are shown in the upstream panels, not here.

    aligned: (H, W, 3) uint8 RGB. Returns: (H, W, 3) uint8 RGB."""
    H, W = aligned.shape[:2]
    overlay = np.full_like(aligned, 255)
    ci = 0
    for shape in decomp.shapes:
        if isinstance(shape, FittedShape):
            m = _compound_mask(shape.parts, H, W)    # parts are full-image coords
            overlay[m > 0] = _FIT_PALETTE[ci % len(_FIT_PALETTE)]
            ci += 1
        elif isinstance(shape, Deferral):
            x, y, w, h = shape.bbox
            sub = overlay[y:y + h, x:x + w]
            sub[shape.mask > 0] = _DEFER_COLOR
    return (0.40 * aligned + 0.60 * overlay).astype(np.uint8)
