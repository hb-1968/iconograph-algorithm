"""Optional CPU PyTorch optimizer for a complete rendered scene.

Optimize proposed geometry at test time and evaluate it with hard-render
constraints. This helper is separate from the production perimeter chain.
"""

from __future__ import annotations

import numpy as np
import torch

from iconograph.application.compress._torch_common import _torch_determinism
from iconograph.application.compress.diff_fit import _Enc, _encode_part, _make_grid, _soft_mask

SEED = 0
SCENE_STEPS = 200         # fixed Adam budget (plateau stop allowed, final step always runs)
SCENE_LR = 0.1            # one group — every param is a target-px shift
TAU0, TAU1 = 1.5, 0.4     # sigmoid temperature anneal, target px (geometric)
POS_MAX = 4.0             # |Δposition| per construction bound, target px (tanh-scaled)
W_SYM = 1.0               # symmetry soft-IoU weight (the objective)
W_FID = 1.0               # fidelity-to-incumbent soft-IoU weight (don't drift)
W_PRIOR = 0.02            # pull each shift toward 0
GATE_FIXED = 6.0          # σ(6) ≈ 0.9975 — parts are present; the gate is never optimized here
PLATEAU_WIN = 40
PLATEAU_EPS = 1e-5
_EPS = 1e-9
_F64 = torch.float64


def _prob_or(softs: list[torch.Tensor]) -> torch.Tensor:
    """Probabilistic OR of per-construction occupancies (diff_fit's log1p idiom — keeps
    every construction's gradient alive where torch.prod would starve it)."""
    acc = torch.zeros_like(softs[0])
    for s in softs:
        acc = acc + torch.log1p(-s.clamp(max=1.0 - _EPS))
    return 1.0 - torch.exp(acc)


def _soft_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = (a * b).sum()
    union = (a + b - a * b).sum()
    return inter / (union + _EPS)


def refine_scene(constructions: list[tuple[list, tuple, str]],
                 rendered_masks: list[np.ndarray],
                 vertical: bool, a2: int,
                 canvas_hw: tuple[int, int], target: int) -> dict[int, tuple[float, float]]:
    """Decoded per-construction shifts {ci: (dx, dy)} in CANVAS px that improve composite
    mirror symmetry about the axis (vertical mirrors x; a2 is the DOUBLED target-px axis,
    so the mirror of x is a2 − x), or {} if nothing moved / encoding is unsupported.
    Deterministic (H8): optimizes on the target-res grid, no RNG."""
    h, w = canvas_hw
    s = target / float(max(h, w))
    _torch_determinism(SEED)

    # freeze each construction's geometry (detached params); only a shift DOF is learnable
    enc_groups: list[list[_Enc]] = []
    shifts: list[torch.Tensor] = []
    try:
        for parts, _c, _l in constructions:
            encs = [_encode_part(p, s, GATE_FIXED) for p in parts]
            for e in encs:
                e.params = e.params.detach()
                if e.ang is not None:
                    e.ang = e.ang.detach()
                e.gate = e.gate.detach()
            enc_groups.append(encs)
            shifts.append(torch.zeros(2, dtype=_F64, requires_grad=True))
    except (TypeError, ValueError):
        return {}                                  # an unencodable primitive → no-op
    if not shifts:
        return {}

    xy = _make_grid(target, target)
    xy_ref = xy.clone()
    if vertical:
        xy_ref[:, 0] = a2 - xy_ref[:, 0]
    else:
        xy_ref[:, 1] = a2 - xy_ref[:, 1]

    inc = np.zeros((target, target), np.float64)
    for m in rendered_masks:
        inc = np.maximum(inc, (m > 0).astype(np.float64))
    tgt = torch.tensor(inc.reshape(-1), dtype=_F64)

    def composite(grid: torch.Tensor, tau: float) -> torch.Tensor:
        softs = []
        for encs, sh in zip(enc_groups, shifts):
            dd = POS_MAX * torch.tanh(sh)
            softs.append(_soft_mask(encs, grid - dd, tau))
        return _prob_or(softs)

    opt = torch.optim.Adam(shifts, lr=SCENE_LR)
    hist: list[float] = []
    for step in range(SCENE_STEPS):
        tau = TAU0 * (TAU1 / TAU0) ** (step / max(1, SCENE_STEPS - 1))
        opt.zero_grad()
        comp = composite(xy, tau)
        comp_mir = composite(xy_ref, tau)
        loss = (W_SYM * (1.0 - _soft_iou(comp, comp_mir))
                + W_FID * (1.0 - _soft_iou(comp, tgt))
                + W_PRIOR * sum((POS_MAX * torch.tanh(sh)).pow(2).sum() for sh in shifts))
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
        if (len(hist) > PLATEAU_WIN and step < SCENE_STEPS - 1
                and hist[-PLATEAU_WIN - 1] - hist[-1] < PLATEAU_EPS):
            break

    out: dict[int, tuple[float, float]] = {}
    for ci, sh in enumerate(shifts):
        with torch.no_grad():
            dd = (POS_MAX * torch.tanh(sh)).numpy()
        dx = (round(2.0 * float(dd[0])) / 2.0) / s     # half-px snap (target), then to canvas px
        dy = (round(2.0 * float(dd[1])) / 2.0) / s
        if abs(dx) + abs(dy) > 1e-9:
            out[ci] = (dx, dy)
    return out
