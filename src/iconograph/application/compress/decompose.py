"""Stage 3 D2+D3 — contour-driven decomposer: trace primitives + tag junctions/residual.

Pipeline position (the contour-driven decomposer that supersedes the monolithic L3 beam):
    D1 contour_parse (parse_shape) → D2+D3 decompose (THIS)

Given a feature mask, D2 traces the D1 boundary into a small set of signed primitives and D3 tags
the leftovers. The mechanism is **edge-tracing by convexity** (the locked design, twice
counterexample-stress-tested):

  - a CONVEX arc run → an ellipse that ADDS;  a CONCAVE arc run → an ellipse that SUBTRACTS
    (so Apple's bite / a ring's inner edge / a notch carve, generically — no hint needed);
  - an added ellipse that SPILLS outside the shape is carved by a half-plane from the straight run
    it crosses (the whale hull's flat top → ellipse − rect = half-disc);
  - the straight-edged remainder is covered by best-fit primitives (triangles for angular regions,
    rectangles, ellipses) — "triangles + ellipses, pill-overlapped";
  - interior HOLES subtract a fitted primitive;
  - a primitive is accepted iff its boundary aligns its run (≥ALIGN_FRAC of the run within ALIGN_TAU
    px) AND the construction stays ⊆ S; an un-traceable run becomes a **junction label** for the
    renderer to overrepresent; what no primitive explains (coverage < ADEQUATE_COV) is the small
    **residual** mask (low-res render tier).

Operates on a PADDED canvas so an enclosing ellipse can spill into background and be carved.
Deterministic (H8): cv2 fitters + numpy, fixed run order, no RNG. Output is a `Decomposition`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from iconograph.application.compress.common import (
    Part,
    RotatedEllipse,
    RotatedRectangle,
    _cand_inscribed_circle,
    _centroid,
    _compound_iou,
    _compound_mask,
    _extent_ok,
    _fit_circle,
    _fit_ellipse,
    _fit_hole,
    _fit_rectangle,
    _fit_rotated_ellipse,
    _fit_rotated_rect,
    _fit_triangle,
    _iou_bool,
    _largest_contour,
    _min_feature_px,
    _raster_part,
)
from iconograph.application.compress.contour_parse import parse_shape

# ── knobs ───────────────────────────────────────────────────────────────
HARD_RUN_CAP = 16        # outer runs > this ⇒ genuinely complex → whole-feature residual
MIN_ARC_FRAC = 0.05      # an arc shorter than this fraction of the perimeter → curve-label
ALIGN_TAU = 3.0          # px: a run point counts as "on" a primitive's boundary
ALIGN_FRAC = 0.6         # ≥ this fraction of a run's points aligned ⇒ accept the primitive
SPILL_FRAC = 0.04        # construction spill > this fraction of its area ⇒ try carving
GAP_MIN_IOU = 0.45       # a remainder-cover primitive must cover its component ≥ this
ADEQUATE_COV = 0.90      # construction IoU ≥ this ⇒ a clean decomposition (else residual)


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
    coverage: float                      # IoU(construction, feature)
    pad: int
    shape: tuple[int, int]               # padded (H, W)
    n_add: int = field(default=0)
    n_sub: int = field(default=0)


# --------------------------------------------------------------------------- #
# geometry helpers (the WRITE-NEW primitives, per the H11 audit)
# --------------------------------------------------------------------------- #
def _fit_arc_ellipse(points: np.ndarray, h: int, w: int) -> RotatedEllipse | None:
    """Least-squares ellipse through an arc run's dense points. None if too short or the fit is
    degenerate/runaway (a too-shallow arc under-determines the conic — the sagitta floor in D1 keeps
    runs ≥ ~60°, and the alignment check downstream rejects a bad extrapolation)."""
    if len(points) < 5:
        return None
    try:
        (cx, cy), (a_full, b_full), angle = cv2.fitEllipse(points.reshape(-1, 1, 2).astype(np.int32))
    except cv2.error:
        return None
    if a_full < 1.0 or b_full < 1.0 or max(a_full, b_full) > 3.0 * max(h, w):
        return None
    return RotatedEllipse(cx=float(cx), cy=float(cy), rx=float(a_full / 2.0),
                          ry=float(b_full / 2.0), angle=float(angle))


def _aligned(prim, run_pts: np.ndarray, h: int, w: int) -> bool:
    """Boundary-alignment acceptance: ≥ ALIGN_FRAC of the run's points lie within ALIGN_TAU px of the
    primitive's rasterized boundary (distance-transform of the 1-px boundary ring; sample at run pts)."""
    m = _raster_part(prim, h, w)
    boundary = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if not boundary.any():
        return False
    dist = cv2.distanceTransform(cv2.bitwise_not(boundary), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ys = np.clip(run_pts[:, 1], 0, h - 1)
    xs = np.clip(run_pts[:, 0], 0, w - 1)
    return float(np.mean(dist[ys, xs] <= ALIGN_TAU)) >= ALIGN_FRAC


def _half_plane_rect(a, b, outside_dir: np.ndarray, h: int, w: int) -> RotatedRectangle:
    """A huge rotated rectangle covering the half-plane on the `outside_dir` side of line a→b — the
    spill-carve primitive (subtract)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    u = b - a
    u = u / (np.linalg.norm(u) + 1e-9)
    nrm = np.array([-u[1], u[0]])
    if float(np.dot(nrm, outside_dir)) < 0:
        nrm = -nrm
    diag = float(np.hypot(h, w))
    mid = (a + b) / 2.0 + nrm * (diag / 2.0)
    angle = float(np.degrees(np.arctan2(u[1], u[0])))
    return RotatedRectangle(cx=float(mid[0]), cy=float(mid[1]), w=2.0 * diag, h=diag, angle=angle)


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


# --------------------------------------------------------------------------- #
# the decomposer (D2 trace + D3 junction/residual)
# --------------------------------------------------------------------------- #
def _spill_carves(adds: list[Part], target: np.ndarray, runs, h: int, w: int) -> list[Part]:
    """Greedily carve the construction's over-reach (adds outside S) with half-planes from the
    straight runs (each oriented to put S on the inside). Keep a carve only if it removes more spill
    than it eats of the shape."""
    base = _compound_mask(adds, h, w).astype(bool)
    over = base & ~target.astype(bool)
    if int(over.sum()) < SPILL_FRAC * int(base.sum()):
        return []
    cen = _centroid(target)
    if cen is None:
        return []
    cen = np.asarray(cen, float)
    carves: list[Part] = []
    for seg in runs:
        if seg.kind != "straight":
            continue
        mid = (np.asarray(seg.start, float) + np.asarray(seg.end, float)) / 2.0
        rect = _half_plane_rect(seg.start, seg.end, mid - cen, h, w)
        cm = _raster_part(rect, h, w).astype(bool)
        removed = int((over & cm).sum())
        eats = int((target.astype(bool) & cm).sum())
        if removed > eats and removed > 0:
            carves.append(Part(rect, "subtract"))
            over = over & ~cm
    return carves


_SINGLE = frozenset(
    {"circle", "ellipse", "rectangle", "triangle", "rotated_rect", "rotated_ellipse"}
)


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
        fit = _fit_rotated_rect(c, h, w)
    elif hint == "rotated_ellipse":
        fit = _fit_rotated_ellipse(c, h, w)
    return [Part(fit[0], "add")] if fit is not None else []


def decompose_shape(mask: np.ndarray, min_feat: int, hint: str | None = None) -> Decomposition:
    """Decompose one feature mask (uint8 {0,255}) into signed primitives + junctions + residual.
    A single-primitive L2 hint is fit directly; everything else is edge-traced by convexity."""
    th, tw = mask.shape[:2]
    pad = max(min_feat, max(th, tw) // 2)
    h, w = th + 2 * pad, tw + 2 * pad
    target = np.zeros((h, w), dtype=np.uint8)
    target[pad:pad + th, pad:pad + tw] = mask

    def _whole_residual(cov: float, junctions: list[Junction]) -> Decomposition:
        return Decomposition(parts=[], junctions=junctions, residual=target.copy(),
                             coverage=cov, pad=pad, shape=(h, w))

    if hint in _SINGLE:                                # single-primitive hint → direct fit
        parts = _single_fit(target, hint, h, w)
        if parts:
            cov = _compound_iou(parts, target, h, w)
            recon = _compound_mask(parts, h, w)
            residual = ((target.astype(bool) & ~recon.astype(bool)).astype(np.uint8)) * 255
            res = residual if (cov < ADEQUATE_COV and _extent_ok(residual, min_feat)) else None
            return Decomposition(parts=parts, junctions=[], residual=res, coverage=float(cov),
                                 pad=pad, shape=(h, w), n_add=len(parts))

    parsed = parse_shape(target)
    if parsed is None:
        return _whole_residual(0.0, [])
    runs = parsed.outer.segments
    if len(runs) > HARD_RUN_CAP:                       # genuinely complex (sunburst) → residual tier
        return _whole_residual(0.0, [])

    perim = sum(len(s.points) for s in runs) + 1e-9
    adds: list[Part] = []
    subs: list[Part] = []
    junctions: list[Junction] = []

    # 1. arcs → convex add / concave subtract ellipses (substantial + boundary-aligned)
    for seg in runs:
        if seg.kind != "arc":
            continue
        if len(seg.points) / perim < MIN_ARC_FRAC:
            junctions.append(Junction(point=seg.start, kind="curve"))
            continue
        ell = _fit_arc_ellipse(seg.points, h, w)
        if ell is None or not _aligned(ell, seg.points, h, w):
            junctions.append(Junction(point=seg.start, kind="curve"))
            continue
        (adds if seg.convex else subs).append(Part(ell, "add" if seg.convex else "subtract"))

    # 2. spill-carve the convex-arc adds against the straight runs
    carves = _spill_carves(adds, target, runs, h, w) if adds else []

    # 3. cover the straight-edged remainder (per uncovered component → best single primitive)
    recon = _compound_mask(adds + carves, h, w)
    under = ((target.astype(bool) & ~recon.astype(bool)).astype(np.uint8)) * 255
    n, lbl, stats, _cent = cv2.connectedComponentsWithStats(under, connectivity=8)
    for i in range(1, n):
        comp = ((lbl == i).astype(np.uint8)) * 255
        if not _extent_ok(comp, min_feat):
            continue
        prim = _best_remainder_prim(comp, h, w)
        if prim is not None:
            adds.append(Part(prim, "add"))

    # 4. holes → subtract a fitted primitive
    for hole in parsed.holes:
        hm = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(hm, [np.vstack([s.points for s in hole.segments]).reshape(-1, 1, 2).astype(np.int32)], 255)
        sub = _fit_hole(hm, h, w, min_feat)
        if sub is not None:
            subs.append(Part(sub, "subtract"))

    parts = adds + carves + subs
    if not adds:
        return _whole_residual(_compound_iou(parts, target, h, w) if parts else 0.0, junctions)

    cov = _compound_iou(parts, target, h, w)
    recon2 = _compound_mask(parts, h, w)
    residual = ((target.astype(bool) & ~recon2.astype(bool)).astype(np.uint8)) * 255
    res = residual if (cov < ADEQUATE_COV and _extent_ok(residual, min_feat)) else None
    return Decomposition(parts=parts, junctions=junctions, residual=res, coverage=float(cov),
                         pad=pad, shape=(h, w), n_add=len(adds), n_sub=len(carves) + len(subs))


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
