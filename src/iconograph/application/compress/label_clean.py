"""Stage-3 label-cleaner (WS2): clean a feature's decomposition before render.

Runs once per feature, AFTER decompose_shape and BEFORE the R1 render, on the
padded canonical canvas. Five conservative, no-regression cleanups on the
Decomposition's Parts:

  Job 1a  prune redundant parts        (_post_prune FREE pass)
  Job 1b  merge near-duplicate adds    (same group/type, IoU ≥ MERGE_IOU)
  Job 1c  re-fit loose primitives      (re-fit to the source the part covers)
  Job 2   drop no-op (zero-bite) subtracts  (label normalization)
  Job 3   re-select variant by sub-case (burst → rot-spurs, gated)

Every mutating step is gated on min(area, boundary) IoU vs the source (the score
the whole pipeline arbitrates on): a change is accepted only if it does NOT lower
that number (H1). A hard fallback returns the ORIGINAL decomposition if the net
result ever drops below base. Pure function, no RNG → byte-identical reruns (H8).

Roles (dominant/lobe/accent/shave/counter) are NOT attached here — render recomputes
them via infer_roles; Job 2 only removes the dead subtracts that would mislabel.
All primitives are H5 reuse of compress/common.py + compress/decompose.py.
"""

import numpy as np

from iconograph.application.compress.common import (
    PRUNE_TOL,
    Circle,
    Ellipse,
    Part,
    Rectangle,
    Triangle,
    _compound_score,
    _fit_circle,
    _fit_ellipse,
    _fit_rectangle,
    _fit_triangle,
    _iou_bool,
    _largest_contour,
    _post_prune,
    _raster_part,
)
from iconograph.application.compress.decompose import (
    Decomposition,
    _arbitrate,
    _decomp_from_parts,
    _fit_rot_spurs,
)

# Same-family fitters keyed by primitive type — the simple families that re-fit/merge
# cleanly from a contour. Rotated/rounded/wedge/polygon are left untouched (no entry).
_FIT = {Circle: _fit_circle, Ellipse: _fit_ellipse,
        Rectangle: _fit_rectangle, Triangle: _fit_triangle}

MERGE_IOU = 0.85    # two same-type adds whose solos overlap this much are one feature


def _score(parts: list[Part], target: np.ndarray, h: int, w: int) -> float:
    """min(area, boundary) IoU of a construction vs target — the gate metric."""
    return _compound_score(parts, target, h, w)[2]


def _accept(new: list[Part], cur: list[Part], target: np.ndarray,
            h: int, w: int, eps: float = 0.0) -> bool:
    """A swap is accepted only if it does not lower the gate metric (H1)."""
    return _score(new, target, h, w) >= _score(cur, target, h, w) - eps


def _merge_dupes(parts: list[Part], target: np.ndarray, h: int, w: int) -> list[Part]:
    """Job 1b: collapse near-duplicate same-type adds in the SAME positive group into
    one primitive re-fit on their union. Gated; restarts on each accepted merge so
    shifting indices stay valid. Never crosses CSG groups or touches negative groups."""
    changed = True
    while changed:
        changed = False
        pairs = []
        for i, pi in enumerate(parts):
            if pi.op != "add" or pi.group < 0 or type(pi.primitive) not in _FIT:
                continue
            for j in range(i + 1, len(parts)):
                pj = parts[j]
                if (pj.op == "add" and pj.group == pi.group
                        and type(pj.primitive) is type(pi.primitive)):
                    pairs.append((pi.group, i, j))
        pairs.sort()       # deterministic order (H8)
        for _g, i, j in pairs:
            solo_i = _raster_part(parts[i].primitive, h, w)
            solo_j = _raster_part(parts[j].primitive, h, w)
            if _iou_bool(solo_i, solo_j) < MERGE_IOU:
                continue
            contour = _largest_contour(solo_i | solo_j)
            if contour is None:
                continue
            prim = _FIT[type(parts[i].primitive)](contour, h, w, 0, 0)[0]
            trial = [p for k, p in enumerate(parts) if k not in (i, j)]
            trial.insert(i, Part(prim, "add", parts[i].group))
            if _accept(trial, parts, target, h, w):
                parts = trial
                changed = True
                break      # indices shifted — recompute pairs from scratch
    return parts


def _refit_loose(parts: list[Part], target: np.ndarray, h: int, w: int) -> list[Part]:
    """Job 1c: re-fit each simple-family add to the slice of the source it actually
    covers (tightens a circle/rect the trace left loose). Gated per swap."""
    out = list(parts)
    for i, p in enumerate(out):
        if p.op != "add" or p.group < 0 or type(p.primitive) not in _FIT:
            continue
        solo = _raster_part(p.primitive, h, w) & target
        contour = _largest_contour(solo)
        if contour is None:
            continue
        prim2 = _FIT[type(p.primitive)](contour, h, w, 0, 0)[0]
        trial = out[:i] + [Part(prim2, p.op, p.group)] + out[i + 1:]
        if _accept(trial, out, target, h, w):
            out = trial
    return out


def _drop_dead_subtracts(parts: list[Part], h: int, w: int) -> list[Part]:
    """Job 2: drop positive-group subtracts that bite zero of their group's add-union
    (byte-identical to the compound mask, so always safe) — they'd only make infer_roles
    label a dead subtract a 'shave'. Negative-group carves are opaque, never touched."""
    add_union: dict[int, np.ndarray] = {}
    for p in parts:
        if p.op == "add" and p.group >= 0:
            m = _raster_part(p.primitive, h, w)
            add_union[p.group] = m if p.group not in add_union else (add_union[p.group] | m)
    out = []
    for p in parts:
        if p.op == "subtract" and p.group >= 0:
            au = add_union.get(p.group)
            bite = 0 if au is None else int(np.count_nonzero(_raster_part(p.primitive, h, w) & au))
            if bite == 0:
                continue
        out.append(p)
    return out


def _reselect_variant(cur: Decomposition, target: np.ndarray, sub_case: str | None,
                      min_feat: int, pad: int, h: int, w: int) -> Decomposition:
    """Job 3: re-select a structurally-appropriate variant by sub-case. The decomposer
    already arbitrated every candidate it generated (incl. all ring-based annular forms,
    which need a RingGeom unavailable here), so re-running those is pure redundancy. The
    one reachable, value-adding signal is routing == 'burst': re-offer the radial-spur
    construction and re-arbitrate, admitting it only if it scores >= the incumbent so the
    pick can only hold or improve. All other routings are a no-op (decomposer's choice)."""
    if sub_case != "burst" or len(cur.parts) < 1:
        return cur
    spur_parts = _fit_rot_spurs(target, min_feat, h, w)
    if not spur_parts:
        return cur
    cand = _decomp_from_parts(spur_parts, target, min_feat, pad, h, w)
    if min(cand.coverage, cand.boundary_cov) < min(cur.coverage, cur.boundary_cov):
        return cur
    return _arbitrate([cur, cand])


def clean_labels(decomp: Decomposition, source_mask: np.ndarray,
                 sub_case: str | None, min_feat: int) -> Decomposition:
    """Clean a decomposition before render: prune/merge/re-fit/normalize/re-select,
    every step gated no-regression. See the module docstring for the five jobs.

    decomp: padded-canvas Decomposition from decompose_shape.
    source_mask: (H, W) uint8 {0,255}, the SAME padded target the decomposer scored
        against (shape == decomp.shape).
    sub_case: stage-1 routing ('standard'|'inverted'|'burst') or None.
    min_feat: min feature px for this canvas (== the decompose_shape call's).
    Returns: a cleaned Decomposition, parts in PADDED coords, min(area, boundary) IoU
        never below decomp's (H1). Deterministic (H8).
    """
    h, w = decomp.shape
    pad = decomp.pad
    parts = list(decomp.parts)
    base = _score(parts, source_mask, h, w)

    parts = _post_prune(parts, source_mask, h, w)        # Job 1a (FREE pass, no target_n)
    parts = _merge_dupes(parts, source_mask, h, w)       # Job 1b
    parts = _refit_loose(parts, source_mask, h, w)       # Job 1c
    parts = _drop_dead_subtracts(parts, h, w)            # Job 2

    cur = _decomp_from_parts(parts, source_mask, min_feat, pad, h, w,
                             structured=decomp.structured)
    cur = _reselect_variant(cur, source_mask, sub_case, min_feat, pad, h, w)  # Job 3

    # Hard fallback: never regress vs the original decomposition.
    if _score(cur.parts, source_mask, h, w) < base - PRUNE_TOL:
        return decomp
    return cur
