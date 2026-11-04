"""Optimize render candidates with optional CPU PyTorch refinement.

Soft optimization proposes geometry; hard-render feasibility and scoring
decide whether it replaces the incumbent.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from iconograph.application.compress._torch_common import _torch_determinism
from iconograph.application.compress.common import Part, Polygon, _band_width

SEED = 0
TWEAK_STEPS = 150         # fixed Adam budget (plateau stop allowed, final step always runs)
TWEAK_LR = 0.15           # single group — every param is px-scale (no angles)
TAU0, TAU1 = 1.0, 0.3     # sigmoid temperature anneal, target px (geometric)
POS_MAX = 2.0             # |Δposition| bound, px (tanh-scaled)
LOG_SCALE_MAX = math.log(1.05)   # per-axis log-scale bound (tanh-scaled)
W_BAND = 5.0              # boundary-band weight in the soft IoU
W_SEP = 2.0               # anti-globbing: soft occupancy over pocket pixels
W_PRIOR = 0.01            # pull toward the deterministic incumbent
PLATEAU_WIN = 30
PLATEAU_EPS = 1e-5
_F64 = torch.float64


def _polygon_sdf(verts: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Signed distance from each xy point to the polygon: min point-segment
    distance over the (V,2) edge fan (diff_fit's triangle SDF generalized),
    sign via even-odd ray casting under no_grad — constant a.e., the gradient
    flows through the distance only."""
    a = verts
    b = torch.roll(verts, -1, 0)
    e = b - a
    w = xy[:, None, :] - a[None, :, :]
    t = ((w * e[None]).sum(-1) / (e * e).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
    proj = a[None] + t[..., None] * e[None]
    d = torch.linalg.vector_norm(xy[:, None, :] - proj, dim=-1).min(dim=1).values
    with torch.no_grad():
        x, y = xy[:, 0], xy[:, 1]
        ay, by = a[:, 1][None, :], b[:, 1][None, :]
        ax, bx = a[:, 0][None, :], b[:, 0][None, :]
        cond = (((ay <= y[:, None]) & (by > y[:, None]))
                | ((by <= y[:, None]) & (ay > y[:, None])))
        tc = (y[:, None] - ay) / (by - ay + 1e-12)
        xc = ax + tc * (bx - ax)
        inside = ((cond & (xc > x[:, None])).sum(dim=1) % 2) == 1
    return torch.where(inside, -d, d)


def refine_tweaks(leaders: list[tuple[int, Part]],
                  leader_of: dict[int, int],
                  pair_axis: dict[int, tuple[bool, int]],
                  coverage: np.ndarray,
                  pockets: list[np.ndarray],
                  target: int) -> dict[int, tuple[float, float, float, float]]:
    """Per-leader decoded deltas {index: (dx, dy, fx, fy)} — empty when nothing
    moved or no Polygon-form leaders exist (true-curve forms are skipped; the
    common case after straightening is all-Polygon). Deterministic (H8)."""
    poly_leaders = [(i, p) for i, p in leaders
                    if isinstance(p.primitive, Polygon)
                    and len(p.primitive.pts) >= 3]
    if not poly_leaders:
        return {}
    _torch_determinism(SEED)

    k = max(2, 64 // target)
    gt = k * target
    # grid in TARGET-px pixel-center coordinates, sub-sampled k×
    ax1 = (np.arange(gt, dtype=np.float64) + 0.5) / k - 0.5
    gx, gy = np.meshgrid(ax1, ax1)
    xy = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=_F64)

    cov = np.kron(coverage.astype(np.float64), np.ones((k, k)))
    tgt = torch.tensor(cov.ravel(), dtype=_F64)
    ref_bin = ((coverage >= 0.5).astype(np.uint8)) * 255
    band_px = _band_width(ref_bin)
    se = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * band_px + 1, 2 * band_px + 1))
    band = (cv2.dilate(ref_bin, se) & ~cv2.erode(ref_bin, se)) > 0
    wmap = torch.tensor(
        np.kron(np.where(band, W_BAND, 1.0), np.ones((k, k))).ravel(),
        dtype=_F64)
    gap = None
    if pockets:
        g = np.zeros_like(coverage, dtype=np.float64)
        for pk in pockets:
            g[pk > 0] = 1.0
        gap = torch.tensor(np.kron(g, np.ones((k, k))).ravel(), dtype=_F64)
        if float(gap.sum()) == 0.0:
            gap = None

    followers_of: dict[int, list[int]] = {}
    for j in sorted(leader_of):
        followers_of.setdefault(leader_of[j], []).append(j)

    encs = []
    for i, p in poly_leaders:
        verts = torch.tensor(np.asarray(p.primitive.pts, dtype=np.float64),
                             dtype=_F64)
        d = torch.zeros(2, dtype=_F64, requires_grad=True)
        ls = torch.zeros(2, dtype=_F64, requires_grad=True)
        encs.append({"i": i, "op": p.op, "group": p.group, "verts": verts,
                     "center": verts.mean(dim=0).detach(), "d": d, "ls": ls})

    params = [e["d"] for e in encs] + [e["ls"] for e in encs]
    opt = torch.optim.Adam(params, lr=TWEAK_LR)

    def occs(tau: float) -> list[tuple[str, int, torch.Tensor]]:
        out = []
        for e in encs:
            dd = POS_MAX * torch.tanh(e["d"])
            sc = torch.exp(LOG_SCALE_MAX * torch.tanh(e["ls"]))
            v = e["center"] + (e["verts"] - e["center"]) * sc[None, :] + dd[None, :]
            sdf = _polygon_sdf(v, xy)
            out.append((e["op"], e["group"], torch.sigmoid(-sdf / tau)))
            for j in followers_of.get(e["i"], []):
                vert, a2 = pair_axis[j]
                fxy = xy.clone()
                if vert:
                    fxy[:, 0] = a2 - fxy[:, 0]
                else:
                    fxy[:, 1] = a2 - fxy[:, 1]
                sdf_f = _polygon_sdf(v, fxy)
                out.append((e["op"], e["group"], torch.sigmoid(-sdf_f / tau)))
        return out

    def soft_csg(items: list[tuple[str, int, torch.Tensor]]) -> torch.Tensor:
        pos = None
        neg = None
        for g in sorted({gg for _o, gg, _t in items}):
            adds = [t for op, gg, t in items if gg == g and op == "add"]
            subs = [t for op, gg, t in items if gg == g and op == "subtract"]
            if adds:
                u = 1.0 - torch.exp(torch.stack(
                    [torch.log1p(-a.clamp(max=1.0 - 1e-7)) for a in adds]
                ).sum(dim=0))
            else:
                u = torch.zeros_like(xy[:, 0])
            for s_ in subs:
                u = u * (1.0 - s_)
            if g >= 0:
                pos = u if pos is None else 1.0 - (1.0 - pos) * (1.0 - u)
            else:
                neg = u if neg is None else 1.0 - (1.0 - neg) * (1.0 - u)
        out = pos if pos is not None else torch.zeros_like(xy[:, 0])
        if neg is not None:
            out = out * (1.0 - neg)
        return out

    hist: list[float] = []
    for step in range(TWEAK_STEPS):
        tau = TAU0 * (TAU1 / TAU0) ** (step / max(1, TWEAK_STEPS - 1))
        opt.zero_grad()
        occ = soft_csg(occs(tau))
        inter = (wmap * occ * tgt).sum()
        union = (wmap * (occ + tgt - occ * tgt)).sum()
        loss = 1.0 - inter / union.clamp_min(1e-9)
        if gap is not None:
            loss = loss + W_SEP * (occ * gap).sum() / gap.sum()
        prior = sum(((POS_MAX * torch.tanh(e["d"])) ** 2).sum()
                    + ((LOG_SCALE_MAX * torch.tanh(e["ls"])) ** 2).sum()
                    for e in encs)
        loss = loss + W_PRIOR * prior
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
        if (len(hist) > PLATEAU_WIN and step < TWEAK_STEPS - 1
                and hist[-PLATEAU_WIN - 1] - hist[-1] < PLATEAU_EPS):
            break

    out: dict[int, tuple[float, float, float, float]] = {}
    for e in encs:
        with torch.no_grad():
            dd = (POS_MAX * torch.tanh(e["d"])).numpy()
            sc = np.exp(LOG_SCALE_MAX * np.tanh(e["ls"].numpy()))
        dx = round(2.0 * float(dd[0])) / 2.0      # snap to the half-px grid
        dy = round(2.0 * float(dd[1])) / 2.0
        v = e["verts"].numpy()
        span = v.max(axis=0) - v.min(axis=0)
        fx, fy = float(sc[0]), float(sc[1])
        if span[0] > 1e-9:                        # snap scaled spans to ints
            fx = max(1.0, round(span[0] * fx)) / span[0]
        if span[1] > 1e-9:
            fy = max(1.0, round(span[1] * fy)) / span[1]
        if abs(dx) + abs(dy) + abs(fx - 1.0) + abs(fy - 1.0) > 1e-9:
            out[e["i"]] = (dx, dy, fx, fy)
    return out
