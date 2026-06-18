"""Stage-4 consistency cleanup — a deterministic post-render pass.

Two scene-level modes, each the complement of what per-construction render can
express (render.py is never modified):

(1) MIRROR-SYMMETRY reconciliation (mask-domain, layer-on-top) — when a logo's
CANONICAL silhouette is mirror-symmetric but its render drifted off-axis
(gmail@16), the produced mask is half-mirrored about the axis so it reads clean
and symmetric. The gate is the canonical axis (the "this logo IS symmetric"
signal), so a large render drift is still corrected; an already-symmetric render
(vscode) is left byte-identical, as is any logo with no global axis (figma).

(2) WHOLE-GLYPH ROTATION (Parts-domain, re-render) — angle-tension reduction. A
glyph whose strokes sit on diagonals composes from 1-px 45° staircases (high
jag). Rotating the WHOLE glyph onto the 45° grid can land those strokes on the
axis / run-≥2 slope members, so render expresses them as straight runs. The pass
rotates every construction's Parts by a candidate angle, refits the rotated glyph
to the incumbent's canvas footprint (fair scale), re-renders, and HARD-scores the
composite: a rotation ships only when it is feasible (fidelity ≥ floor + topology
preserved vs its OWN rotated reference) AND cuts jag-density by ≥ JAG_MARGIN_FRAC.
Identity wins ties, so an upright glyph is only displaced by a strict win — the
pass is byte-identical when nothing beats the margin (the no-op contract). Rotation
fires only on ROTATION-ELIGIBLE marks — those with no canonical "up" (multi-fold
symmetric, or a single thin stroke) — so a composed logo with an intended orientation
is never tilted. (Whole-glyph rotation of figma/vscode mangles them into tilted blobs
even as jag drops; the jag floor can't tell "cleaner same shape" from "same logo,
tilted", so the ELIGIBILITY gate, not the floor, is what prevents that.) The
monogram/letter class is a further eligibility signal, deferred to stage 5.

Per-PART consistency — equalizing same-kind components that render unequal, and
de-caving a convex part that rendered concave — was investigated and found
INFEASIBLE as a layer-on-top post-pass: the seeds decompose into a single
construction whose parts are FUSED in the silhouette, and a per-part solo render
is smaller/squarer than that part's joint footprint, so re-stamping it either
fragments the logo (figma@16/@24: 1→2 components) or square-ifies a circle
(figma@32), violating the represented-geometry heuristic. figma's three circles
are already geometrically equal (canonical radii 234.6 / 235.9 / 239.3, spread
~2%); their rendered inequality is
render's INDEPENDENT per-circle straightening, which only a render-level group
harmonization can fix. That conflicts with "layer on top" and stays a render-level
decision (see README §R1 round 10): the rotation mode sidesteps the fused-mask
limit by transforming the Parts and RE-RENDERING, not by re-stamping the mask.

Deterministic (H8): no RNG, fixed candidate-angle order, float64 rotation math,
integer mirror; render_debug is itself deterministic. When no gate fires the output
is value-identical to the input. The single public export is cleanup_scene.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

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
    _band_width,
    _boundary_iou,
    _compound_mask,
    _iou_bool,
    _mirror_axis,
    _scale,
    _snap,
)
from iconograph.application.compress.render import (
    SYM_AXIS_TOL_DEG,
    _flip_mask,
    _topology,
    floor_for,
    jag_norm,
    reference_mask,
    render_debug,
)
from iconograph.application.compress.straighten import translate_prim
from skimage.morphology import medial_axis

SYM_OK = 0.95                # render already this self-symmetric → leave it (vscode)
SYM_AXIS_SEARCH = 6          # ± half-px steps searched for the true flip axis


def _scene_axis(constructions: list[tuple[list[Part], tuple, str]],
                canvas_hw: tuple[int, int]) -> bool | None:
    """Detect a global mirror axis on the per-construction-OR canonical silhouette
    (built per construction to avoid the cross-construction CSG-group collision a
    flat pool would cause). Canonical detection is the 'this logo IS symmetric'
    signal — the render drift we fix can be large, so we do NOT gate on the target's
    own symmetry. Returns True (vertical, mirrors x), False (horizontal), or None
    (no axis / a diagonal an integer grid can't enforce)."""
    h, w = canvas_hw
    scene = np.zeros((h, w), np.uint8)
    for parts, _color, _label in constructions:
        scene |= _compound_mask(parts, h, w)
    ax = _mirror_axis(scene)
    if ax is None:
        return None
    _p, u = ax
    tol = math.sin(math.radians(SYM_AXIS_TOL_DEG))
    if abs(float(u[0])) <= tol:
        return True
    if abs(float(u[1])) <= tol:
        return False
    return None


def _best_flip_axis(u_mask: np.ndarray, vertical: bool) -> tuple[int, float] | None:
    """Search a ±SYM_AXIS_SEARCH half-px window (±3 px) around the silhouette's own
    centre for the integer flip axis a2 maximising IoU(U, flip(U)) — aligning the mirror
    with what was drawn. Returns (a2, best_iou) or None on an empty mask."""
    ys, xs = np.nonzero(u_mask)
    if xs.size == 0:
        return None
    centre = float(xs.mean()) if vertical else float(ys.mean())
    a2c = int(round(2.0 * centre))
    best: tuple[int, float] | None = None
    for a2 in range(a2c - SYM_AXIS_SEARCH, a2c + SYM_AXIS_SEARCH + 1):
        iou = _iou_bool(u_mask, _flip_mask(u_mask, a2, vertical))
        if best is None or iou > best[1]:
            best = (a2, iou)
    return best


def _half_mirror(m: np.ndarray, a2: int, vertical: bool) -> np.ndarray:
    """Make m exactly symmetric about a2/2 by mirroring the more-authoritative half
    onto the other (the half whose reflection is closest to m — least change). This
    is the right reconciliation for a real drift: AND would erode the asymmetric
    residue, OR would over-fill both sides; half-mirror reproduces the true form."""
    n = m.shape[1] if vertical else m.shape[0]
    idx = np.arange(n)
    lo = 2 * idx <= a2                      # the low-index side, inclusive of the axis
    hi = 2 * idx >= a2

    def fold(side: np.ndarray) -> np.ndarray:
        s = np.zeros_like(m)
        if vertical:
            s[:, side] = m[:, side]
        else:
            s[side, :] = m[side, :]
        return s | _flip_mask(s, a2, vertical)

    a, b = fold(lo), fold(hi)
    return a if _iou_bool(a, m) >= _iou_bool(b, m) else b


def _reconcile_symmetry(constructions: list[tuple[list[Part], tuple, str]],
                        rendered_masks: list[np.ndarray],
                        canvas_hw: tuple[int, int]) -> list[np.ndarray]:
    """Half-mirror the produced silhouette when the CANONICAL shape has a global
    mirror axis and the render is not already symmetric (≥ SYM_OK → leave it, e.g.
    vscode). Scoped to single-construction logos (the per-construction colour case);
    multi-construction symmetry (pairing mirror-partner segments) is deferred."""
    if len(constructions) != 1:
        return list(rendered_masks)
    vertical = _scene_axis(constructions, canvas_hw)
    if vertical is None:
        return list(rendered_masks)
    m = rendered_masks[0]
    fa = _best_flip_axis(m, vertical)
    if fa is None:
        return list(rendered_masks)
    a2, pre = fa
    if pre >= SYM_OK:
        return list(rendered_masks)                 # already symmetric → untouched
    sym = _half_mirror(m, a2, vertical)
    if _topology(sym)[0] > _topology(m)[0]:
        return list(rendered_masks)                 # never fragment the silhouette
    return [sym]


# ── whole-glyph rotation (mode 2: Parts-domain, re-render) ───────────────────
# Rotation is a SUPPRESSIBLE scene DOF (the wedge doctrine at glyph scale):
# offer the 45°-grid orientations, score, let the floor + jag margin arbitrate.
ROTATION_ANGLES: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0)
#   identity FIRST (H8 tie-break: an upright glyph is the default); 90°-family
#   before the 45°-family so a clean quarter-turn beats a diagonal turn on ties.
JAG_MARGIN_FRAC = 0.05      # a rotation must cut jag-density (Ĵ) by ≥ this to displace upright


def _rotate_pts(pts, cx: float, cy: float, theta_deg: float) -> np.ndarray:
    """(N,2)-like → (N,2) float64, rotated about (cx,cy). Positive theta_deg is
    CLOCKWISE on screen (cv2 y-down / minAreaRect convention) — same matrix the
    renderer's _rounded_rect_poly uses, so rotated angle fields stay consistent."""
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    centre = np.array([cx, cy], dtype=np.float64)
    return (rot @ (np.asarray(pts, dtype=np.float64) - centre).T).T + centre


def _rotate_primitive(pr, theta_deg: float, cx: float, cy: float):
    """Rigid rotation of one primitive about (cx,cy) by theta_deg. Modeled on
    common._reflect_part: rotate the centre/vertices, add theta to angle-bearing
    types, and (for the axis-aligned Ellipse/Rectangle) promote to the rotated
    type — or stay axis-aligned with swapped extents when theta lands on a 90°
    multiple (via _snap). Returns a Primitive."""
    def rc(px: float, py: float) -> tuple[float, float]:
        q = _rotate_pts([[px, py]], cx, cy, theta_deg)[0]
        return float(q[0]), float(q[1])

    if isinstance(pr, Circle):
        x, y = rc(pr.cx, pr.cy)
        return Circle(cx=x, cy=y, r=pr.r)
    if isinstance(pr, (Ellipse, RotatedEllipse)):
        ang0 = pr.angle if isinstance(pr, RotatedEllipse) else 0.0
        x, y = rc(pr.cx, pr.cy)
        na = (ang0 + theta_deg) % 180.0
        snapped, parity = _snap(math.radians(na))
        if snapped:
            rx, ry = (pr.ry, pr.rx) if parity else (pr.rx, pr.ry)
            return Ellipse(cx=x, cy=y, rx=rx, ry=ry)
        return RotatedEllipse(cx=x, cy=y, rx=pr.rx, ry=pr.ry, angle=na)
    if isinstance(pr, Rectangle):
        ccx, ccy = pr.x + pr.w / 2.0, pr.y + pr.h / 2.0
        x, y = rc(ccx, ccy)
        na = theta_deg % 180.0
        snapped, parity = _snap(math.radians(na))
        if snapped:
            wd, ht = (pr.h, pr.w) if parity else (pr.w, pr.h)
            return Rectangle(x=int(round(x - wd / 2.0)), y=int(round(y - ht / 2.0)),
                             w=int(round(wd)), h=int(round(ht)))
        return RotatedRectangle(cx=x, cy=y, w=float(pr.w), h=float(pr.h), angle=na)
    if isinstance(pr, RotatedRectangle):
        x, y = rc(pr.cx, pr.cy)
        return RotatedRectangle(cx=x, cy=y, w=pr.w, h=pr.h, angle=(pr.angle + theta_deg) % 180.0)
    if isinstance(pr, RoundedRectangle):
        x, y = rc(pr.cx, pr.cy)
        return RoundedRectangle(cx=x, cy=y, w=pr.w, h=pr.h,
                                angle=(pr.angle + theta_deg) % 180.0,
                                corner_radius=pr.corner_radius)
    if isinstance(pr, Wedge):
        x, y = rc(pr.cx, pr.cy)
        sweep = pr.a1 - pr.a0                 # rotation-invariant; shift both ends clockwise
        a0 = pr.a0 + theta_deg
        return Wedge(cx=x, cy=y, r=pr.r, a0=a0, a1=a0 + sweep)
    if isinstance(pr, Polygon):
        rp = _rotate_pts(list(pr.pts), cx, cy, theta_deg)
        return Polygon(pts=tuple((float(px), float(py)) for px, py in rp))
    rv = _rotate_pts([pr.v1, pr.v2, pr.v3], cx, cy, theta_deg)   # Triangle
    return Triangle(v1=(float(rv[0][0]), float(rv[0][1])),
                    v2=(float(rv[1][0]), float(rv[1][1])),
                    v3=(float(rv[2][0]), float(rv[2][1])))


def _rotate_part_lists(part_lists: list[list[Part]], theta_deg: float,
                       centre: tuple[float, float]) -> list[list[Part]]:
    """Rotate every Part of every construction by theta_deg about `centre`,
    preserving op + CSG group (a grouped construction survives the rotation)."""
    cx, cy = centre
    return [[Part(_rotate_primitive(p.primitive, theta_deg, cx, cy), p.op, p.group)
             for p in parts] for parts in part_lists]


def _silhouette_bbox(part_lists: list[list[Part]], frame: int,
                     shift: float) -> tuple[int, int, int, int] | None:
    """Tight bbox (x,y,w,h) in CANVAS coords of the union silhouette, measured in a
    `frame`×`frame` raster with all parts shifted by (+shift,+shift) so a rotated
    glyph spilling past [0,S] is not clipped before it is measured."""
    big = np.zeros((frame, frame), np.uint8)
    for parts in part_lists:
        sp = [Part(translate_prim(p.primitive, shift, shift), p.op, p.group) for p in parts]
        big |= _compound_mask(sp, frame, frame)
    ys, xs = np.nonzero(big)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return (int(x0 - shift), int(y0 - shift), x1 - x0 + 1, y1 - y0 + 1)


def _refit_part_lists(part_lists: list[list[Part]],
                      src_bbox: tuple[int, int, int, int],
                      dst_bbox: tuple[int, int, int, int]) -> list[list[Part]]:
    """Translate+uniform-scale every construction so the rotated glyph's bbox maps
    onto dst_bbox's centre and larger extent — keeping the rotated render at the
    SAME canvas footprint (hence target scale) as the incumbent, so the jag
    comparison is fair and the rotated glyph stays inside the canvas."""
    sx, sy, sw, sh = src_bbox
    dx, dy, dw, dh = dst_bbox
    src_cx, src_cy = sx + sw / 2.0, sy + sh / 2.0
    dst_cx, dst_cy = dx + dw / 2.0, dy + dh / 2.0
    smax = max(sw, sh)
    f = (max(dw, dh) / smax) if smax > 0 else 1.0
    out: list[list[Part]] = []
    for parts in part_lists:
        np_parts: list[Part] = []
        for p in parts:
            prim = translate_prim(p.primitive, -src_cx, -src_cy)
            prim = _scale(prim, f)
            prim = translate_prim(prim, dst_cx, dst_cy)
            np_parts.append(Part(prim, p.op, p.group))
        out.append(np_parts)
    return out


def _composite(masks: list[np.ndarray], hw: tuple[int, int]) -> np.ndarray:
    out = np.zeros(hw, np.uint8)
    for m in masks:
        out |= m
    return out


ROT_ELIGIBLE_SYM_IOU = 0.90    # silhouette ≈ its own 90° rotation ⇒ multi-fold symmetric (no canonical up)
ROT_ELIGIBLE_THINNESS = 0.35   # minAreaRect short/long ≤ this ⇒ a single thin stroke (orientation-free)


def _rotation_eligible(constructions: list[tuple[list[Part], tuple, str]],
                       canvas_hw: tuple[int, int]) -> bool:
    """A mark is rotation-eligible only if it has NO canonical 'up', so rotating it
    cannot destroy identity. Two structural signals (the jag metric cannot make this
    call — figma/vscode mangle with a LOWER jag): (a) multi-fold rotational symmetry
    — the canonical silhouette ≈ its own 90° rotation; (b) a single thin stroke — one
    component whose min-area-rect is elongated (the monogram/stroke case, the user's
    'M'). A composed logo (multiple distinct shapes, square-ish, no rot-symmetry) is
    neither, so it stays upright. The monogram/letter class is a stage-5 signal (TBD)."""
    h, w = canvas_hw
    sil = np.zeros((h, w), np.uint8)
    for parts, _c, _l in constructions:
        sil |= _compound_mask(parts, h, w)
    ys, xs = np.nonzero(sil)
    if xs.size == 0:
        return False
    mom = cv2.moments(sil, binaryImage=True)
    if mom["m00"] == 0.0:
        return False
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    rot90 = cv2.warpAffine(sil, cv2.getRotationMatrix2D((cx, cy), 90.0, 1.0),
                           (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if _iou_bool(sil, rot90) >= ROT_ELIGIBLE_SYM_IOU:        # (a) multi-fold symmetric
        return True
    n_cc, *_ = cv2.connectedComponentsWithStats(sil, connectivity=8)
    if n_cc - 1 == 1:                                         # (b) one component …
        (_c, (rw, rh), _ang) = cv2.minAreaRect(
            np.column_stack([xs, ys]).astype(np.float32))
        long_side = max(rw, rh)
        if long_side > 0.0 and min(rw, rh) / long_side <= ROT_ELIGIBLE_THINNESS:
            return True                                       # … and elongated → a stroke
    return False


def _rotation_search(constructions: list[tuple[list[Part], tuple, str]],
                     rendered_masks: list[np.ndarray],
                     canvas_hw: tuple[int, int], target: int) -> list[np.ndarray]:
    """Try each 45°-grid rotation of the WHOLE glyph; re-render and hard-score the
    composite. Ship the lowest-Ĵ feasible candidate iff it beats upright by
    JAG_MARGIN_FRAC, else return the incumbent unchanged (the no-op contract)."""
    if not constructions or not rendered_masks:
        return list(rendered_masks)
    if not _rotation_eligible(constructions, canvas_hw):
        return list(rendered_masks)         # composed logo with a canonical 'up' → never tilt
    h, w = canvas_hw
    centre = (w / 2.0, h / 2.0)
    inc_comp = _composite(rendered_masks, rendered_masks[0].shape)
    inc_jn = jag_norm(inc_comp)
    if not inc_comp.any() or inc_jn <= 0.0:
        return list(rendered_masks)
    part_lists = [parts for parts, _c, _l in constructions]
    inc_bbox = _silhouette_bbox(part_lists, h, 0.0)     # orig parts fit in the square canvas
    if inc_bbox is None:
        return list(rendered_masks)
    floor = floor_for(target)
    frame = 3 * h                                       # holds a 45° rotation about centre
    best: tuple[float, list[np.ndarray]] | None = None
    for theta in ROTATION_ANGLES:
        if theta == 0.0:
            continue
        rot_lists = _rotate_part_lists(part_lists, theta, centre)
        src_bbox = _silhouette_bbox(rot_lists, frame, float(h))
        if src_bbox is None:
            continue
        refit = _refit_part_lists(rot_lists, src_bbox, inc_bbox)
        rrs = [render_debug(rl, canvas_hw, target) for rl in refit]
        masks = [r.mask for r in rrs]
        comp = _composite(masks, (target, target))
        ref = _composite([r.ref for r in rrs], (target, target))
        if not comp.any() or not ref.any():
            continue
        fid = min(_iou_bool(comp, ref), _boundary_iou(comp, ref, _band_width(ref)))
        if fid < floor or _topology(comp) != _topology(ref):
            continue                                    # rotation must preserve the shape
        jn = jag_norm(comp)
        if best is None or jn < best[0]:                # first (lowest-index) angle wins ties
            best = (jn, masks)
    if best is not None and best[0] <= inc_jn * (1.0 - JAG_MARGIN_FRAC):
        return best[1]
    return list(rendered_masks)


# ── cross-construction symmetry (the multi-construction case _reconcile_symmetry defers) ──
SCENE_SYM_MARGIN = 0.02   # composite symmetry IoU (about the CANONICAL axis) must improve by ≥ this
SCENE_FID_KEEP = 0.85     # the shifted composite must stay this close to the incumbent (no drift)


def _symmetry_scene(constructions: list[tuple[list[Part], tuple, str]],
                    rendered_masks: list[np.ndarray],
                    canvas_hw: tuple[int, int], target: int) -> list[np.ndarray]:
    """Cross-construction mirror symmetry: a test-time optimizer (refine/scene_fit, lazy
    torch) nudges PER-CONSTRUCTION placement to symmetrize the composite about the shared
    CANONICAL axis (from the supersampled reference, not the drifted render). Decoded
    shifts are re-rendered and SHIP only if the composite symmetry about that axis
    improves while staying close to the incumbent and preserving topology — else a
    byte-identical no-op. Scoped to multi-construction marks; single-construction symmetry
    stays with _reconcile_symmetry. No torch installed (core path) ⇒ silent no-op."""
    if len(constructions) < 2 or len(rendered_masks) != len(constructions):
        return list(rendered_masks)
    vertical = _scene_axis(constructions, canvas_hw)
    if vertical is None:
        return list(rendered_masks)
    canon = _composite([reference_mask(parts, canvas_hw, target)
                        for parts, _c, _l in constructions], (target, target))
    fa = _best_flip_axis(canon, vertical)
    if fa is None:
        return list(rendered_masks)
    a2, _canon_sym = fa
    inc = _composite(rendered_masks, (target, target))
    if not inc.any():
        return list(rendered_masks)
    inc_sym = _iou_bool(inc, _flip_mask(inc, a2, vertical))
    if inc_sym >= SYM_OK:                       # render already symmetric about the axis
        return list(rendered_masks)
    try:
        from iconograph.application.refine.scene_fit import refine_scene
    except ImportError:
        return list(rendered_masks)             # torch is the optional `diff` extra
    shifts = refine_scene(constructions, rendered_masks, bool(vertical), a2,
                          canvas_hw, target)
    if not shifts:
        return list(rendered_masks)
    new_masks: list[np.ndarray] = []
    for ci, (parts, _c, _l) in enumerate(constructions):
        if ci in shifts:
            dx, dy = shifts[ci]
            sp = [Part(translate_prim(p.primitive, dx, dy), p.op, p.group) for p in parts]
            new_masks.append(render_debug(sp, canvas_hw, target).mask)
        else:
            new_masks.append(rendered_masks[ci])
    new_comp = _composite(new_masks, (target, target))
    if not new_comp.any():
        return list(rendered_masks)
    new_sym = _iou_bool(new_comp, _flip_mask(new_comp, a2, vertical))
    fid = _iou_bool(new_comp, inc)
    if (new_sym >= inc_sym + SCENE_SYM_MARGIN and fid >= SCENE_FID_KEEP
            and _topology(new_comp) == _topology(inc)):
        return new_masks
    return list(rendered_masks)


def cleanup_scene(constructions: list[tuple[list[Part], tuple, str]],
                  rendered_masks: list[np.ndarray],
                  canvas_hw: tuple[int, int], target: int) -> list[np.ndarray]:
    """Scene-level consistency cleanup over the per-construction renders.
    constructions: the (parts, color_rgb, label) list a fixture decomposes into
        (parts in FULL-canvas coords — the list render_debug builds).
    rendered_masks: one (target,target) uint8 {0,255} mask per construction.
    canvas_hw: the full-canvas (S, S). target: hicolor size.
    Returns a new list of corrected masks, same length/order. Deterministic (H8).

    Three modes, in order: (1) single-construction mirror-symmetry reconciliation
    (mask-domain), (2) cross-construction symmetry (per-construction placement
    optimizer, Parts re-render), (3) whole-glyph rotation (eligible marks, Parts
    re-render) when it strictly reduces angle tension. Each is gated to a
    byte-identical no-op when its win condition fails."""
    masks = _reconcile_symmetry(constructions, rendered_masks, canvas_hw)
    masks = _symmetry_scene(constructions, masks, canvas_hw, target)
    return _rotation_search(constructions, masks, canvas_hw, target)


# ── Phase 2: outline / connect / squeeze (pixel-level, on the fused composite) ──────
# Phase 1 corrected the per-construction renders; Phase 2 works on the COMPOSITE
# silhouette, reducing it to a 1-px centerline after bridging rasterization gaps and
# merging near-parallel strokes — guarded so intended topology (counters, genuinely
# separate components) survives. Library-first (H2): cv2.MORPH_CLOSE + skimage
# medial_axis. Deterministic (H8). Emits the skeleton (the user's "outline" product);
# downstream re-stroking from the medial-axis width map is a follow-on.


def _gap_tol_px(target: int) -> int:
    """Largest gap (target px) treated as a rasterization artifact to bridge/merge: 1 px in
    the tiny-T regime where a 1-px split/seam is unreadable, 0 at T ≥ 64 where a 1-2 px
    feature is intended detail (so Phase 2 tapers to a no-op, per the round-3+ tiny-T
    scoping doctrine)."""
    return 1 if target <= 32 else 0


def _connect_squeeze(u: np.ndarray, target: int, topo_canon: tuple[int, int]) -> np.ndarray:
    """One topology-guarded MORPH_CLOSE (RECT, ODD kernel) that BOTH bridges collinear
    rasterization gaps ("connect line components") and fuses near-parallel strokes
    ("squeeze thin features") — a closing does both directions at once. Descending-kernel
    back-off: take the largest gap-tolerance whose close lands on the CANONICAL topology;
    if none does, leave the fused mask untouched (never fill a counter or over-merge
    separate components). BORDER_CONSTANT 0 so a dilation never bridges a stroke against
    the canvas edge."""
    for tol in range(_gap_tol_px(target), 0, -1):
        k = 2 * tol + 1
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(u, cv2.MORPH_CLOSE, se, iterations=1,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if _topology(closed) == topo_canon:
            return closed
    return u


SKEL_SPUR_MAX = 3         # dead-end branches shorter than this many px are staircase barbs


def _prune_spurs(skel: np.ndarray, max_spur: int = SKEL_SPUR_MAX) -> np.ndarray:
    """Remove short dead-end barbs (medial-axis staircase spurs) from a 1-px skeleton: cut
    at junctions (≥3 neighbours), drop branch components shorter than max_spur that carry a
    true endpoint (degree-1, neighbour-count+self == 2), keep junctions and every through-
    or long branch. Topology-preserving (only dead-ends go, so components/holes are
    unchanged — a loop has no endpoint and survives). Deterministic (H8): the neighbour
    convolution + row-major component labels carry no RNG."""
    sk = (skel > 0).astype(np.uint8)
    if int(sk.sum()) == 0:
        return skel
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT) * sk      # neighbours + self
    junctions = (nb >= 4).astype(np.uint8)                      # ≥3 neighbours
    branches = sk & (1 - junctions)                             # cut the skeleton at junctions
    n, labels, stats, _c = cv2.connectedComponentsWithStats(branches, connectivity=8)
    out = sk.copy()
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= max_spur:
            continue
        comp = labels == i
        if bool((nb[comp] == 2).any()):                         # dead-end barb → drop
            out[comp] = 0
    return (out * 255).astype(np.uint8)


LCORNER_MIN_LEN = 4       # only straighten branches at least this many px long
LCORNER_STRAIGHT_DEV = 1.3  # max perpendicular deviation (px) to call a branch a STRAIGHT stroke
LCORNER_DIAG_LO = 20.0    # branch angle from the nearer axis (deg); [LO, HI] = a diagonal worth
LCORNER_DIAG_HI = 70.0    #   L-ifying (outside ⇒ already ~axis-aligned, leave it)
LCORNER_MAX_HALFWIDTH = 2.0  # only L-ify THIN strokes (≤ this half-width): a diagonal centerline of
#                              a thin stroke is a staircase artifact; a THICK diagonal stroke's
#                              diagonal IS its real shape (and L-ifying it would push the corner off
#                              the stroke, where re-stroking has no width) → leave it


def _lcorner_straighten(skel: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Replace each approximately-STRAIGHT THIN DIAGONAL skeleton branch (a 1-px staircase)
    with two AXIS-ALIGNED runs joined by ONE L corner — the pixel-level L-corner doctrine
    ([[project_lcorner_doctrine_round5]]): a diagonal centerline reduces to "single line
    features" with heavy 1-px stepping removed. THICK diagonal strokes (median half-width >
    LCORNER_MAX_HALFWIDTH — the diagonal is the real shape), CURVED branches (perp deviation
    above the straight tolerance), already-axis-aligned branches, loops, and short branches
    are left untouched; junctions are kept so the skeleton stays connected. The whole pass is
    reverted if it changes topology (a defensive guard). `dist` is the medial-axis half-width
    map (the thin/thick discriminator). Deterministic (H8): row-major component labels,
    dominant-run-first corner choice, no RNG."""
    sk = (skel > 0).astype(np.uint8)
    if int(sk.sum()) == 0:
        return skel
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT) * sk
    junctions = (nb >= 4).astype(np.uint8)
    branches = sk & (1 - junctions)
    n, labels, stats, _c = cv2.connectedComponentsWithStats(branches, connectivity=8)
    out = sk.copy()
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < LCORNER_MIN_LEN:
            continue
        ys, xs = np.nonzero(labels == i)
        if float(np.median(dist[ys, xs])) > LCORNER_MAX_HALFWIDTH:
            continue                                  # thick stroke → the diagonal is its real shape
        pts = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, x0, y0 = (float(v) for v in
                          cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        perp = np.abs(-(xs - x0) * vy + (ys - y0) * vx)
        if float(perp.max()) > LCORNER_STRAIGHT_DEV:
            continue                                  # curved → not a straight stroke
        proj = (xs - x0) * vx + (ys - y0) * vy
        i0, i1 = int(np.argmin(proj)), int(np.argmax(proj))
        ax, ay, bx, by = int(xs[i0]), int(ys[i0]), int(xs[i1]), int(ys[i1])
        dx, dy = abs(bx - ax), abs(by - ay)
        if dx == 0 or dy == 0:
            continue                                  # already axis-aligned
        ang = math.degrees(math.atan2(min(dx, dy), max(dx, dy)))
        if ang < LCORNER_DIAG_LO:
            continue                                  # near-axis stub → leave (an L is just the run)
        corner = (bx, ay) if dx >= dy else (ax, by)   # dominant (longer) axis run first
        out[labels == i] = 0                          # clear the staircase
        cv2.line(out, (ax, ay), corner, 1, 1, cv2.LINE_4)
        cv2.line(out, corner, (bx, by), 1, 1, cv2.LINE_4)
    if _topology(out * 255) != _topology(skel):
        return skel                                   # never let straightening alter topology
    return (out * 255).astype(np.uint8)


def _to_outline(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cleaned 1-px skeleton uint8 {0,255}, distance map) of a filled mask. The skeleton is
    the centerline with short staircase barbs pruned and straight diagonal runs reduced to
    axis-aligned L-corner single-line features (medial_axis is topology-preserving and
    1-px-wide; rng=0 for H8). The distance map is medial_axis's per-foreground-pixel
    inscribed radius (half-stroke-width) on `u` — the RE-STROKING source, valid even at moved
    L-corner / loop-fill pixels since it is defined over the whole filled shape."""
    skel_raw, dist = medial_axis((u > 0), return_distance=True, rng=0)
    skel = _lcorner_straighten(_prune_spurs(skel_raw.astype(np.uint8) * 255), dist)
    return skel, dist


SKEL_LOOP_PASSES = 12     # bounded loop-break passes (real skeletons carry only a few holes)
SKEL_LOOP_FILL_FRAC = 0.06  # a spurious skeleton hole below this fraction of the canvas is filled


def _break_spurious_loops(skel: np.ndarray, max_holes: int) -> np.ndarray:
    """Remove SPURIOUS skeleton loops (medial_axis discretization artifacts on fat regions —
    e.g. claude's radial hub, where the axis circles a solid center) until the skeleton's
    hole count is ≤ max_holes (the FILLED shape's hole count; a skeleton is homotopy-
    equivalent to its source, so any hole it does NOT share is an artifact). Each pass FILLS
    the smallest excess hole's interior (a loop bounded by blade-junction pixels can't be
    opened by removing one pixel without disconnecting a blade, so filling — making the tiny
    enclosed pocket solid, faithful to the solid source region — is the robust move). The
    fill cap (SKEL_LOOP_FILL_FRAC of the canvas) keeps it to tiny artifacts; genuine
    counters are larger AND already counted in max_holes (we only ever touch EXCESS holes,
    smallest first). Deterministic (H8): smallest-hole-first, bounded passes."""
    sk = (skel > 0).astype(np.uint8)
    fill_cap = max(6, int(round(SKEL_LOOP_FILL_FRAC * sk.size)))
    for _ in range(SKEL_LOOP_PASSES):
        if _topology(sk * 255)[1] <= max_holes:
            break
        nlab, lbl, st, _c = cv2.connectedComponentsWithStats((sk == 0).astype(np.uint8),
                                                             connectivity=4)
        border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
        ids = sorted((i for i in range(1, nlab) if i not in border),
                     key=lambda i: (int(st[i, cv2.CC_STAT_AREA]), i))   # smallest hole first
        if not ids or int(st[ids[0], cv2.CC_STAT_AREA]) > fill_cap:
            break                                        # remaining holes too large to be artifacts
        sk = sk.copy()
        sk[lbl == ids[0]] = 1                            # fill the tiny enclosed pocket
    return (sk * 255).astype(np.uint8)


def _restroke(skel: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """RE-STROKE: reconstruct a FILLED stroke mask from a 1-px skeleton by unioning, at each
    skeleton pixel, a filled disk of radius = the medial-axis half-width there (per-pixel
    faithful — preserves the source's local thickness; round caps/joins). `dist` is sampled
    on the FILLED source; valid at moved skeleton pixels because it's defined over the whole
    shape, and only THIN strokes (≤ LCORNER_MAX_HALFWIDTH) are ever L-cornered, so a corner
    sits at most ~1 px off the stroke where radius 1 is already the right thin width. The loop
    is over skeleton pixels (≤ a few hundred), never image pixels. Deterministic (H8):
    row-major order."""
    out = np.zeros(skel.shape, np.uint8)
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cv2.circle(out, (int(x), int(y)), max(1, int(round(float(dist[y, x])))), 255, -1)
    return out


def _phase2_outline(constructions: list[tuple[list[Part], tuple, str]],
                    rendered_masks: list[np.ndarray],
                    canvas_hw: tuple[int, int], target: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Phase-2 core: fuse the per-construction renders → topology-guarded connect/squeeze →
    cleaned medial-axis skeleton (prune + L-corner straighten + spurious-loop break) →
    (skeleton, distance map). The connect/squeeze guard is the CANONICAL topology
    (supersampled reference) so counters / genuine separations survive a gap-bridge; the loop
    break matches the skeleton's holes to the fused shape's. Deterministic (H8)."""
    if not rendered_masks:
        return np.zeros((target, target), np.uint8), np.zeros((target, target), np.float64)
    u = _composite(rendered_masks, (target, target))
    if not u.any():
        return u, np.zeros((target, target), np.float64)
    canon = _composite([reference_mask(parts, canvas_hw, target)
                        for parts, _c, _l in constructions], (target, target))
    u2 = _connect_squeeze(u, target, _topology(canon))
    skel, dist = _to_outline(u2)
    return _break_spurious_loops(skel, _topology(u2)[1]), dist


def outline_scene(constructions: list[tuple[list[Part], tuple, str]],
                  rendered_masks: list[np.ndarray],
                  canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """The cleaned 1-px composite skeleton (the centerline "outline" — Phase-2 internal /
    preview product). `refine_outline` re-strokes this into the shipped filled icon."""
    return _phase2_outline(constructions, rendered_masks, canvas_hw, target)[0]


def restroke_outline(constructions: list[tuple[list[Part], tuple, str]],
                     rendered_masks: list[np.ndarray],
                     canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """The RE-STROKED filled icon: the cleaned Phase-2 skeleton reconstructed back to a filled
    stroke mask (per-pixel medial-axis width). (target, target) uint8 {0,255}. Single mask —
    per-construction colour re-attribution is a deferred follow-on. Deterministic (H8)."""
    skel, dist = _phase2_outline(constructions, rendered_masks, canvas_hw, target)
    return _restroke(skel, dist)


def refine_outline(constructions: list[tuple[list[Part], tuple, str]],
                   rendered_masks: list[np.ndarray],
                   canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """Stage-4 refine SHIPPED output: Phase-1 scene cleanup of the per-construction renders
    (mirror reconcile → cross-construction symmetry → eligible rotation), then Phase-2 fuse →
    connect/squeeze → cleaned skeleton → RE-STROKE back to a filled stroke icon. The
    re-stroked filled mask IS refine's product (the skeleton, cleaned then given its thickness
    back, so the centerline cleanup shapes the final fill). Deterministic (H8)."""
    masks = cleanup_scene(constructions, rendered_masks, canvas_hw, target)
    return restroke_outline(constructions, masks, canvas_hw, target)
