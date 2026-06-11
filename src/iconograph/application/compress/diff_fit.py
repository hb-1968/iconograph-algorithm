"""Stage 3 — diff_fit: test-time differentiable CSG refinement (torch CPU, NO training).

An OPTIONAL refiner inside D2's candidate pool: decompose_shape lazily imports
refine_candidates when its best deterministic candidate is sub-adequate, hands it the
top-k candidates' Part lists as warm starts, and re-enters every refined Part list
through _decomp_from_parts → _arbitrate. The refiner NEVER bypasses arbitration and
never reports its own soft score — decoded parts are re-scored by the hard cv2
rasterizer like every other candidate, so a bad decode simply loses.

Method: each primitive becomes a signed-distance field; occupancy o = σ(−sdf/τ)
(τ = blur width in px, annealed wide→crisp so parts feel the target from far away
early and sharpen late); gated parts (σ(gate) = soft existence switch, so the
optimizer can fade a useless part to zero instead of parking it); smooth per-group
CSG whose 0/1 limit equals _compound_mask (including NEGATIVE groups); one
boundary-band-weighted soft-IoU loss + a small gate-parsimony term; Adam descent;
alternating optimize/spawn rounds (spawns are deterministic: connected components of
the hard under/over-coverage); an optional mirror-symmetry seed that ties parameters
by evaluating each part on the grid AND the reflected grid.

Determinism (H8): no RNG anywhere — all initialization derives from the input mask
and the warm-start parts; the seed/thread/deterministic-algorithms trio is defense in
depth (verified bitwise-identical across processes on this build). torch is NOT a
core dependency: only this module imports it (pyproject extra `diff`); decompose.py
imports this module lazily inside its fire gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from iconograph.application.compress.common import (
    Circle,
    Ellipse,
    Part,
    Rectangle,
    RotatedEllipse,
    RotatedRectangle,
    RoundedRectangle,
    Triangle,
    Wedge,
    _band_width,
    _compound_mask,
    _extent_ok,
    _iou_bool,
    _offset_part,
    _post_prune,
    _raster_part,
)

# ── knobs ───────────────────────────────────────────────────────────────
SEED = 0
DIFF_RES = 160           # max side of the optimization grid (hard scoring at full res is upstream)
DIFF_STEPS = 250         # Adam budget per round (fixed → deterministic; plateau-stop allowed)
DIFF_LR_POS = 0.3        # Adam lr — px-scale params and gates
DIFF_LR_ANG = 0.05       # Adam lr — angle params (radians); 0.3 rad/step would be ~17°
TAU0, TAU1 = 1.5, 0.4    # sigmoid temperature anneal, px (geometric schedule)
BAND_BOOST = 5.0         # boundary-band pixel weight in the loss
W_GATE = 0.002           # parsimony pressure per live gate — 0.01 cascade-melted 14-part
#                          warm starts (each overlapping part is individually near-marginal)
GATE_INIT_WARM = 2.5     # warm-start existence logit (σ ≈ 0.92)
GATE_INIT_SPAWN = 0.5    # spawned-part logit (σ ≈ 0.62) — must earn its keep
GATE_KEEP = 0.5          # decode keeps a part iff σ(gate) ≥ this
DIFF_MAX_PARTS = 24
ANGLE_SNAP_DEG = 1.5     # decode: angle within this of 0 (mod 90°) → exact axis-aligned type
PLATEAU_WIN = 50         # plateau stop: improvement < PLATEAU_EPS over this many steps
PLATEAU_EPS = 1e-4
MAX_SPAWN = 6            # discrete adds per spawn round
SPAWN_ROUNDS = 2         # optimize → spawn → optimize
SPAWN_ASPECT = 2.0       # elongated residual component → capsule-ish RotatedRectangle
SYM_MIN_IOU = 0.90       # mask-vs-reflected-mask IoU to accept a mirror axis
SYM_PAIR_IOU = 0.50      # part-vs-reflected-partner raster IoU to deduplicate a mirrored pair

_EPS = 1e-9
_F64 = torch.float64


# ── differentiable geometry (pure functions) ────────────────────────────
def _make_grid(hs: int, ws: int) -> torch.Tensor:
    """(hs·ws, 2) float64 pixel-center coords; [:, 0] = x (col), [:, 1] = y (row)."""
    cols = torch.arange(ws, dtype=_F64) + 0.5
    rows = torch.arange(hs, dtype=_F64) + 0.5
    gx, gy = torch.meshgrid(cols, rows, indexing="xy")    # both (hs, ws); 'xy' = image convention
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


def _rot_into(xy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
              theta: torch.Tensor) -> torch.Tensor:
    """Grid → the primitive's local frame: q = R(−θ)(p − c). θ clockwise-positive in
    y-down px coords — the same convention _rounded_rect_poly / cv2 angles use, so
    encode/decode is a unit change only."""
    px = xy[:, 0] - cx
    py = xy[:, 1] - cy
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([px * c + py * s, -px * s + py * c], dim=-1)


def _part_sdf(kind: str, params: torch.Tensor, ang: torch.Tensor | None,
              xy: torch.Tensor) -> torch.Tensor:
    """Signed distance of every grid point to one primitive; (N,) float64, negative
    inside. Sizes go through abs() so a sign-flipped extent stays differentiable
    (clamp would zero the gradient and strand the parameter)."""
    if kind == "circle":
        d = xy - params[:2]
        return torch.linalg.vector_norm(d, dim=-1) - torch.abs(params[2])
    if kind == "rell":
        # Quilez near-field approximation — exact ellipse SDF is iterative; this is
        # smooth and sub-px accurate near the boundary, where σ(−sdf/τ) lives.
        q = _rot_into(xy, params[0], params[1], ang[0])
        r = torch.abs(params[2:4]) + 0.5
        k0 = torch.linalg.vector_norm(q / r, dim=-1)
        k1 = torch.linalg.vector_norm(q / (r * r), dim=-1)
        return k0 * (k0 - 1.0) / (k1 + _EPS)
    if kind in ("rrect", "rrect_round"):
        q = _rot_into(xy, params[0], params[1], ang[0])
        he = torch.abs(params[2:4]) / 2.0 + 0.25
        cr = None
        if kind == "rrect_round":
            cr = torch.minimum(torch.abs(params[4]), torch.minimum(he[0], he[1]) - 0.05)
            cr = torch.clamp(cr, min=0.0)
            he = he - cr
        d = torch.abs(q) - he
        sdf = (torch.linalg.vector_norm(torch.clamp(d, min=0.0), dim=-1)
               + torch.clamp(torch.maximum(d[:, 0], d[:, 1]), max=0.0))
        return sdf - cr if cr is not None else sdf
    if kind == "triangle":
        v = params.reshape(3, 2)
        dmin: torch.Tensor | None = None
        signs = []
        for i in range(3):
            a, b = v[i], v[(i + 1) % 3]
            e = b - a
            u = xy - a
            t = torch.clamp((u @ e) / (e @ e + _EPS), 0.0, 1.0)
            dist = torch.linalg.vector_norm(u - e.unsqueeze(0) * t.unsqueeze(1), dim=-1)
            dmin = dist if dmin is None else torch.minimum(dmin, dist)
            signs.append(e[0] * u[:, 1] - e[1] * u[:, 0])
        s = torch.stack(signs, dim=0)
        inside = (s >= 0).all(dim=0) | (s <= 0).all(dim=0)
        # sign() has no gradient — fine: the gradient flows through the distance.
        return torch.where(inside, -dmin, dmin)
    if kind == "wedge":
        # Quilez sdPie in a frame where the bisector maps to +y (rotate by θ_b − π/2).
        theta_b, alpha = ang[0], torch.clamp(torch.abs(ang[1]),
                                             min=math.radians(1.0), max=math.radians(179.0))
        q = _rot_into(xy, params[0], params[1], theta_b - math.pi / 2.0)
        qx = torch.abs(q[:, 0])
        qy = q[:, 1]
        r = torch.abs(params[2]) + 0.5
        sa, ca = torch.sin(alpha), torch.cos(alpha)
        ln = torch.sqrt(qx * qx + qy * qy + _EPS) - r
        proj = torch.clamp(qx * sa + qy * ca, min=torch.zeros((), dtype=_F64), max=r)
        mx = qx - sa * proj
        my = qy - ca * proj
        m = torch.sqrt(mx * mx + my * my + _EPS)
        return torch.maximum(ln, m * torch.sign(ca * qx - sa * qy))
    raise ValueError(f"unknown SDF kind {kind!r}")


@dataclass
class _Enc:
    """One part's differentiable encoding (downsampled-px units)."""
    kind: str
    op: str                      # "add" | "subtract" — never optimized
    group: int                   # structural (incl. negative groups) — never optimized
    params: torch.Tensor         # 1-D float64 px-scale params, requires_grad
    ang: torch.Tensor | None     # 1-D float64 radians, requires_grad
    gate: torch.Tensor           # scalar float64 logit, requires_grad


def _soft_mask(encs: list[_Enc], xy: torch.Tensor, tau: float,
               xy_ref: torch.Tensor | None = None) -> torch.Tensor:
    """Smooth gated per-group CSG in [0, 1]; the 0/1 limit equals _compound_mask
    (per group: OR adds, AND-NOT subtracts; positive groups OR'd, negative groups
    carved). Union is the probabilistic OR in log1p form — torch.prod starves every
    other part's gradient when one factor nears 0; log1p keeps each part's gradient
    ∝ 1/(1−o). With xy_ref, every part also contributes its mirror twin's occupancy
    (same tensors — exact parameter tying). The Python loop is over ≤24 parts, never
    pixels (torch vectorizes the grid)."""
    n = xy.shape[0]
    add_log: dict[int, torch.Tensor] = {}
    sub_log: dict[int, torch.Tensor] = {}
    grids = (xy,) if xy_ref is None else (xy, xy_ref)
    for e in encs:
        g = torch.sigmoid(e.gate)
        acc = add_log if e.op == "add" else sub_log
        for grid in grids:
            o = torch.sigmoid(-_part_sdf(e.kind, e.params, e.ang, grid) / tau) * g
            lg = torch.log1p(-o.clamp(max=1.0 - _EPS))
            acc[e.group] = lg if e.group not in acc else acc[e.group] + lg
    pos_logs: list[torch.Tensor] = []
    neg_logs: list[torch.Tensor] = []
    for gk in sorted(set(add_log) | set(sub_log)):
        a = 1.0 - torch.exp(add_log[gk]) if gk in add_log else torch.zeros(n, dtype=_F64)
        mg = a * torch.exp(sub_log[gk]) if gk in sub_log else a
        (pos_logs if gk >= 0 else neg_logs).append(torch.log1p(-mg.clamp(max=1.0 - _EPS)))
    if not pos_logs:
        return torch.zeros(n, dtype=_F64)
    pos = 1.0 - torch.exp(sum(pos_logs))
    if not neg_logs:
        return pos
    return pos * torch.exp(sum(neg_logs))                  # P · Π(1−N_g) = P AND-NOT (∪ neg groups)


def _diff_loss(soft: torch.Tensor, tgt: torch.Tensor, wmap: torch.Tensor,
               gates: list[torch.Tensor]) -> torch.Tensor:
    """1 − weighted soft-IoU + W_GATE · Σ σ(gate). One band-weighted IoU term is more
    stable than separate area + boundary losses."""
    inter = (wmap * soft * tgt).sum()
    union = (wmap * (soft + tgt - soft * tgt)).sum()
    gate_sum = sum(torch.sigmoid(g) for g in gates)
    return 1.0 - inter / (union + _EPS) + W_GATE * gate_sum


# ── encode / decode ──────────────────────────────────────────────────────
def _t(vals: list[float]) -> torch.Tensor:
    # torch.tensor with explicit dtype — torch's default is float32 and as_tensor
    # silently inherits it; float64 keeps the decode round-trip exact.
    return torch.tensor(np.asarray(vals, np.float64), dtype=_F64, requires_grad=True)


def _encode_part(p: Part, s: float, gate_init: float) -> _Enc:
    pr = p.primitive
    ang: list[float] | None = None
    if isinstance(pr, Circle):
        # encode as a free ellipse — a circle that needs to become a stem/capsule can
        # elongate (decode snaps back to Circle when rx ≈ ry, so true circles survive)
        kind, params, ang = "rell", [pr.cx * s, pr.cy * s, pr.r * s, pr.r * s], [0.0]
    elif isinstance(pr, Ellipse):
        kind, params, ang = "rell", [pr.cx * s, pr.cy * s, pr.rx * s, pr.ry * s], [0.0]
    elif isinstance(pr, RotatedEllipse):
        kind, params, ang = ("rell", [pr.cx * s, pr.cy * s, pr.rx * s, pr.ry * s],
                             [math.radians(pr.angle)])
    elif isinstance(pr, Rectangle):
        kind, params, ang = ("rrect", [(pr.x + pr.w / 2.0) * s, (pr.y + pr.h / 2.0) * s,
                                       pr.w * s, pr.h * s], [0.0])
    elif isinstance(pr, RotatedRectangle):
        kind, params, ang = ("rrect", [pr.cx * s, pr.cy * s, pr.w * s, pr.h * s],
                             [math.radians(pr.angle)])
    elif isinstance(pr, RoundedRectangle):
        kind, params, ang = ("rrect_round", [pr.cx * s, pr.cy * s, pr.w * s, pr.h * s,
                                             pr.corner_radius * s], [math.radians(pr.angle)])
    elif isinstance(pr, Wedge):
        kind, params = "wedge", [pr.cx * s, pr.cy * s, pr.r * s]
        ang = [math.radians((pr.a0 + pr.a1) / 2.0), math.radians((pr.a1 - pr.a0) / 2.0)]
    elif isinstance(pr, Triangle):
        kind = "triangle"
        params = [pr.v1[0] * s, pr.v1[1] * s, pr.v2[0] * s, pr.v2[1] * s,
                  pr.v3[0] * s, pr.v3[1] * s]
    else:
        raise TypeError(f"cannot encode primitive {type(pr).__name__}")
    gate = torch.tensor(float(gate_init), dtype=_F64, requires_grad=True)
    return _Enc(kind, p.op, p.group, _t(params), _t(ang) if ang is not None else None, gate)


def _snap(theta: float) -> tuple[bool, int]:
    """(within ANGLE_SNAP_DEG of a multiple of 90°?, that multiple's parity —
    1 means the 90° leg, i.e. swap the in-plane extents)."""
    per = math.pi / 2.0
    k = round(theta / per)
    return abs(theta - k * per) <= math.radians(ANGLE_SNAP_DEG), k % 2


def _decode_part(e: _Enc, s: float, h: int, w: int, min_feat: int) -> Part | None:
    """Exact full-res Part, or None when σ(gate) < GATE_KEEP (pruned) or the decoded
    raster fails _extent_ok (the min-size rule every generator obeys). Near-axis-
    aligned angles snap back to the exact axis-aligned types — stage 6's
    idealize/linearize tier must not lose axis alignment to a 0.4° gradient jiggle."""
    if float(torch.sigmoid(e.gate.detach())) < GATE_KEEP:
        return None
    v = (e.params.detach().cpu().numpy() / s).tolist()
    a = e.ang.detach().cpu().numpy().tolist() if e.ang is not None else None
    prim: object
    if e.kind == "circle":
        prim = Circle(cx=v[0], cy=v[1], r=max(1.0, abs(v[2])))
    elif e.kind == "rell":
        rx, ry = max(1.0, abs(v[2])), max(1.0, abs(v[3]))
        snapped, parity = _snap(a[0])
        if abs(rx - ry) <= 0.08 * max(rx, ry):             # still round → exact Circle
            prim = Circle(cx=v[0], cy=v[1], r=(rx + ry) / 2.0)
        elif snapped:
            if parity:
                rx, ry = ry, rx
            prim = Ellipse(cx=v[0], cy=v[1], rx=rx, ry=ry)
        else:
            prim = RotatedEllipse(cx=v[0], cy=v[1], rx=rx, ry=ry,
                                  angle=math.degrees(a[0]) % 180.0)
    elif e.kind == "rrect":
        wd, ht = max(1.0, abs(v[2])), max(1.0, abs(v[3]))
        snapped, parity = _snap(a[0])
        if snapped:
            if parity:
                wd, ht = ht, wd
            prim = Rectangle(x=int(round(v[0] - wd / 2.0)), y=int(round(v[1] - ht / 2.0)),
                             w=max(1, int(round(wd))), h=max(1, int(round(ht))))
        else:
            prim = RotatedRectangle(cx=v[0], cy=v[1], w=wd, h=ht,
                                    angle=math.degrees(a[0]) % 180.0)
    elif e.kind == "rrect_round":
        wd, ht = max(1.0, abs(v[2])), max(1.0, abs(v[3]))
        cr = min(abs(v[4]), min(wd, ht) / 2.0)
        snapped, parity = _snap(a[0])
        if snapped and parity:
            wd, ht = ht, wd
        ang_deg = 0.0 if snapped else math.degrees(a[0]) % 180.0
        prim = RoundedRectangle(cx=v[0], cy=v[1], w=wd, h=ht, angle=ang_deg, corner_radius=cr)
    elif e.kind == "wedge":
        alpha = min(max(abs(a[1]), math.radians(1.0)), math.radians(179.0))
        a0 = math.degrees(a[0] - alpha) % 360.0
        prim = Wedge(cx=v[0], cy=v[1], r=max(1.0, abs(v[2])),
                     a0=a0, a1=a0 + 2.0 * math.degrees(alpha))   # a1 > a0, may exceed 360
    else:
        prim = Triangle(v1=(v[0], v[1]), v2=(v[2], v[3]), v3=(v[4], v[5]))
    if not _extent_ok(_raster_part(prim, h, w), min_feat):
        return None
    return Part(prim, e.op, e.group)


# ── optimizer ────────────────────────────────────────────────────────────
def _optimize(encs: list[_Enc], tgt: torch.Tensor, wmap: torch.Tensor,
              xy: torch.Tensor, xy_ref: torch.Tensor | None = None) -> None:
    """In-place Adam over params/angles/gates: two param groups (angles in radians
    need ~6× finer steps than px-scale params), geometric τ anneal, plateau stop on
    loss floats only (deterministic)."""
    groups = [{"params": [e.params for e in encs] + [e.gate for e in encs], "lr": DIFF_LR_POS}]
    angs = [e.ang for e in encs if e.ang is not None]
    if angs:
        groups.append({"params": angs, "lr": DIFF_LR_ANG})
    opt = torch.optim.Adam(groups)
    hist: list[float] = []
    for step in range(DIFF_STEPS):
        tau = TAU0 * (TAU1 / TAU0) ** (step / DIFF_STEPS)
        loss = _diff_loss(_soft_mask(encs, xy, tau, xy_ref), tgt, wmap,
                          [e.gate for e in encs])
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if step >= PLATEAU_WIN and hist[-PLATEAU_WIN - 1] - hist[-1] < PLATEAU_EPS:
            break


# ── discrete spawn moves ─────────────────────────────────────────────────
def _spawn(parts: list[Part], target: np.ndarray, min_feat: int,
           h: int, w: int) -> list[Part]:
    """Deterministic topology moves the gradients can't make: one primitive per
    largest under-coverage component (ADD) / over-coverage component (SUBTRACT),
    elongated (aspect ≥ SPAWN_ASPECT) → minAreaRect capsule-ish RotatedRectangle,
    else a Circle at the distance-transform argmax. A subtract spawns into the
    lowest positive group whose mask covers its DT peak (per-group carve scoping).
    No RNG: components area-ordered with index tie-break, argmax is row-major."""
    recon = _compound_mask(parts, h, w)
    tgt = target.astype(bool)
    rec = recon.astype(bool)
    pos_groups = sorted({p.group for p in parts if p.group >= 0})
    group_masks = [(g, _compound_mask([p for p in parts if p.group == g], h, w))
                   for g in pos_groups]
    out: list[Part] = []
    for region, op in (((tgt & ~rec), "add"), ((rec & ~tgt), "subtract")):
        m8 = region.astype(np.uint8) * 255
        num, labels, stats, _c = cv2.connectedComponentsWithStats(m8, connectivity=8)
        comps = sorted(range(1, num), key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
        for ci in comps:
            if len(out) >= MAX_SPAWN:
                break
            cm = (labels == ci).astype(np.uint8) * 255
            if not _extent_ok(cm, min_feat):
                continue
            dist = cv2.distanceTransform(cm, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
            contours, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(c)
            if max(rw, rh) >= SPAWN_ASPECT * max(1.0, min(rw, rh)):
                prim: object = RotatedRectangle(cx=float(rcx), cy=float(rcy),
                                                w=max(1.0, float(rw)), h=max(1.0, float(rh)),
                                                angle=float(rang))
            else:
                prim = Circle(cx=float(ix), cy=float(iy), r=max(1.0, float(dist[iy, ix])))
            grp = 0
            if op == "subtract":
                for g, gm in group_masks:
                    if gm[iy, ix]:
                        grp = g
                        break
            out.append(Part(prim, op, grp))
    return out


# ── mirror symmetry (vscode-class glyphs) ───────────────────────────────
def _reflection_m(px: float, py: float, ux: float, uy: float) -> np.ndarray:
    """2×3 affine reflecting across the line through (px, py) with unit dir (ux, uy)
    (Householder R = 2uuᵀ − I; translation keeps the line invariant)."""
    r2 = 2.0 * np.array([[ux * ux, ux * uy], [ux * uy, uy * uy]], np.float64) - np.eye(2)
    p = np.array([px, py], np.float64)
    t = p - r2 @ p
    return np.hstack([r2, t.reshape(2, 1)])


def _mirror_axis(target: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Best of {vertical, horizontal, PCA major, PCA minor} axes through the centroid,
    accepted iff reflect-and-IoU ≥ SYM_MIN_IOU. Returns (point, unit_dir) or None."""
    m = cv2.moments(target, binaryImage=True)
    if m["m00"] == 0.0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    theta = 0.5 * math.atan2(2.0 * m["mu11"], m["mu20"] - m["mu02"])
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_iou = SYM_MIN_IOU
    for ux, uy in ((0.0, 1.0), (1.0, 0.0),
                   (math.cos(theta), math.sin(theta)),
                   (-math.sin(theta), math.cos(theta))):
        mat = _reflection_m(cx, cy, ux, uy)
        ref = cv2.warpAffine(target, mat, (target.shape[1], target.shape[0]),
                             flags=cv2.INTER_NEAREST, borderValue=0)
        iou = _iou_bool(target, ref)
        if iou > best_iou:
            best_iou = iou
            best = (np.array([cx, cy], np.float64), np.array([ux, uy], np.float64))
    return best


def _reflect_pt(x: float, y: float, p: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    r2 = 2.0 * np.outer(u, u) - np.eye(2)
    q = p + r2 @ (np.array([x, y], np.float64) - p)
    return float(q[0]), float(q[1])


def _reflect_part(part: Part, p: np.ndarray, u: np.ndarray) -> Part:
    """Analytic mirror of one Part across the axis (point p, unit dir u). A direction
    at angle φ reflects to 2β − φ (β = axis angle); reflected near-axis-aligned
    angles snap back to the exact axis-aligned types."""
    beta = math.degrees(math.atan2(u[1], u[0]))
    pr = part.primitive

    def _ref_angle(angle: float) -> float:
        return (2.0 * beta - angle) % 180.0

    if isinstance(pr, Circle):
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        out: object = Circle(cx=x, cy=y, r=pr.r)
    elif isinstance(pr, (Ellipse, RotatedEllipse)):
        ang0 = pr.angle if isinstance(pr, RotatedEllipse) else 0.0
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        na = _ref_angle(ang0)
        snapped, parity = _snap(math.radians(na))
        if snapped:
            rx, ry = (pr.ry, pr.rx) if parity else (pr.rx, pr.ry)
            out = Ellipse(cx=x, cy=y, rx=rx, ry=ry)
        else:
            out = RotatedEllipse(cx=x, cy=y, rx=pr.rx, ry=pr.ry, angle=na)
    elif isinstance(pr, Rectangle):
        ccx, ccy = pr.x + pr.w / 2.0, pr.y + pr.h / 2.0
        x, y = _reflect_pt(ccx, ccy, p, u)
        na = _ref_angle(0.0)
        snapped, parity = _snap(math.radians(na))
        if snapped:
            wd, ht = (pr.h, pr.w) if parity else (pr.w, pr.h)
            out = Rectangle(x=int(round(x - wd / 2.0)), y=int(round(y - ht / 2.0)),
                            w=int(round(wd)), h=int(round(ht)))
        else:
            out = RotatedRectangle(cx=x, cy=y, w=float(pr.w), h=float(pr.h), angle=na)
    elif isinstance(pr, RotatedRectangle):
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        out = RotatedRectangle(cx=x, cy=y, w=pr.w, h=pr.h, angle=_ref_angle(pr.angle))
    elif isinstance(pr, RoundedRectangle):
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        out = RoundedRectangle(cx=x, cy=y, w=pr.w, h=pr.h, angle=_ref_angle(pr.angle),
                               corner_radius=pr.corner_radius)
    elif isinstance(pr, Wedge):
        x, y = _reflect_pt(pr.cx, pr.cy, p, u)
        half = (pr.a1 - pr.a0) / 2.0
        nb = (2.0 * beta - (pr.a0 + pr.a1) / 2.0) % 360.0
        a0 = (nb - half) % 360.0
        out = Wedge(cx=x, cy=y, r=pr.r, a0=a0, a1=a0 + 2.0 * half)
    else:
        v1 = _reflect_pt(pr.v1[0], pr.v1[1], p, u)
        v2 = _reflect_pt(pr.v2[0], pr.v2[1], p, u)
        v3 = _reflect_pt(pr.v3[0], pr.v3[1], p, u)
        out = Triangle(v1=v1, v2=v2, v3=v3)
    return Part(out, part.op, part.group)


def _dedupe_mirror(parts: list[Part], p: np.ndarray, u: np.ndarray,
                   h: int, w: int) -> list[Part]:
    """Drop a part whose reflection duplicates an already-kept part — its twin is
    regenerated by the grid-reflection tying, so keeping both double-counts."""
    rasters = [_raster_part(pt.primitive, h, w) for pt in parts]
    keep: list[Part] = []
    kept_idx: list[int] = []
    for i, pt in enumerate(parts):
        refl = _raster_part(_reflect_part(pt, p, u).primitive, h, w)
        if any(_iou_bool(refl, rasters[j]) >= SYM_PAIR_IOU for j in kept_idx):
            continue
        keep.append(pt)
        kept_idx.append(i)
    return keep


# ── public entry ─────────────────────────────────────────────────────────
def refine_candidates(seeds: list[list[Part]], target: np.ndarray, min_feat: int,
                      h: int, w: int, symmetry: bool = True) -> list[list[Part]]:
    """One refined Part list per usable seed (+ one mirror-symmetric variant of the
    best seed when an axis is detected); may return []. target: the PADDED (h, w)
    uint8 canvas decompose_shape built. The caller wraps each result via
    _decomp_from_parts so it competes in _arbitrate on hard scores — soft scores are
    never compared or returned."""
    torch.manual_seed(SEED)                      # defense in depth — nothing below draws RNG
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    # Optimize on a bbox crop, not the (mostly padding) decompose canvas — the grid
    # budget goes to the shape, roughly doubling effective resolution. ox/oy map
    # crop coords ↔ full-canvas coords (encode shifts in, decode shifts out).
    bx, by, bw, bh = cv2.boundingRect(target)
    mrg = max(8, int(round(0.08 * max(bw, bh))))
    x0, y0 = max(0, bx - mrg), max(0, by - mrg)
    x1, y1 = min(w, bx + bw + mrg), min(h, by + bh + mrg)
    crop = target[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 4 or cw < 4:
        return []
    s = DIFF_RES / float(max(ch, cw))
    hs, ws = max(8, int(round(ch * s))), max(8, int(round(cw * s)))
    small = cv2.resize(crop, (ws, hs), interpolation=cv2.INTER_AREA)     # dsize = (W, H)
    t8 = (small >= 128).astype(np.uint8)
    d = max(2, int(round(_band_width(target) * s)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    band = cv2.dilate(t8, kern) & ~cv2.erode(t8, kern)
    tgt = torch.tensor(t8.astype(np.float64).reshape(-1), dtype=_F64)
    wmap = torch.tensor((1.0 + (BAND_BOOST - 1.0) * band.astype(np.float64)).reshape(-1),
                        dtype=_F64)
    xy = _make_grid(hs, ws)

    def _enc_all(parts: list[Part], gate_init: float) -> list[_Enc]:
        return [_encode_part(_offset_part(p, -x0, -y0), s, gate_init) for p in parts]

    def _dec_all(encs: list[_Enc]) -> list[Part]:
        out_p: list[Part] = []
        for e in encs:
            dp = _decode_part(e, s, h, w, min_feat)
            if dp is not None:
                out_p.append(_offset_part(dp, x0, y0))
        return out_p

    out: list[list[Part]] = []
    usable = [list(parts) for parts in seeds if 1 <= len(parts) <= DIFF_MAX_PARTS]

    for seed in usable:
        encs = _enc_all(seed, GATE_INIT_WARM)
        final: list[Part] | None = None
        for rnd in range(SPAWN_ROUNDS):
            _optimize(encs, tgt, wmap, xy)
            decoded = _dec_all(encs)
            if not decoded:
                break
            final = decoded
            if rnd == SPAWN_ROUNDS - 1:
                break
            new = _spawn(decoded, target, min_feat, h, w)
            new = new[:max(0, DIFF_MAX_PARTS - len(decoded))]
            if not new:
                break
            encs = _enc_all(decoded, GATE_INIT_WARM) + _enc_all(new, GATE_INIT_SPAWN)
        if final:
            out.append(_post_prune(final, target, h, w))

    if symmetry and usable:
        axis = _mirror_axis(target)
        if axis is not None:
            p_axis, u_axis = axis
            base = _dedupe_mirror(usable[0], p_axis, u_axis, h, w)
            if 1 <= len(base) <= DIFF_MAX_PARTS:
                p_s = (p_axis - np.array([x0, y0], np.float64)) * s
                r2 = 2.0 * np.outer(u_axis, u_axis) - np.eye(2)
                xy_ref = torch.tensor((xy.numpy() - p_s) @ r2.T + p_s, dtype=_F64)
                encs = _enc_all(base, GATE_INIT_WARM)
                _optimize(encs, tgt, wmap, xy, xy_ref=xy_ref)
                decoded = _dec_all(encs)
                if decoded:
                    full: list[Part] = []
                    for pt in decoded:
                        full.append(pt)
                        twin = _reflect_part(pt, p_axis, u_axis)
                        # a part sitting ON the axis is its own twin — emit once
                        if _iou_bool(_raster_part(twin.primitive, h, w),
                                     _raster_part(pt.primitive, h, w)) < 0.9:
                            full.append(twin)
                    out.append(_post_prune(full[:DIFF_MAX_PARTS], target, h, w))
    return out
