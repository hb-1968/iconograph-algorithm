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
    _mirror_point,
    _nb4count,
    _orbit,
    _outer_bg,
    _scale,
    _snap,
)
from iconograph.application.compress.render import (
    SYM_AXIS_SEARCH,
    SYM_AXIS_TOL_DEG,
    _best_flip_axis,
    _flip_mask,
    _topology,
    floor_for,
    jag_norm,
    reference_mask,
    render_debug,
)
from iconograph.application.common import masks_8adjacent, mst_bridges, nearest_gap
from iconograph.application.compress.cubify import (
    _glyph_stroke_relayout,
    _map_coords,
    glyph_channel,
)
from iconograph.application.compress.straighten import translate_prim
from scipy.signal import find_peaks
from skimage.morphology import medial_axis

SYM_OK = 0.95                # render already this self-symmetric → leave it (vscode)
# SYM_AXIS_SEARCH + _best_flip_axis: LIFTED to compress/render.py (H5 — cubify's R6
# dual-axis detection shares them); imported above, same names and semantics.
# _orbit: LIFTED to compress/common.py alongside _mirror_point for the same reason.


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


def _self_rot_iou(mask: np.ndarray, angles: tuple[float, ...]) -> float:
    """Max IoU of `mask` vs its own rotation about its centroid over `angles` (degrees). The single
    source of truth for the rotational-symmetry / chirality primitive shared by _rotation_eligible
    (90°) and _label_axes (90/180/270°). INTER_NEAREST integer selection ⇒ H8-safe — NOT the
    thread-nondeterministic warpPolar/INTER_LINEAR class (feedback_cv2_warppolar_nondeterministic)."""
    mom = cv2.moments(mask, binaryImage=True)
    if mom["m00"] == 0.0:
        return 0.0
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    h, w = mask.shape[:2]
    return max(_iou_bool(mask, cv2.warpAffine(
        mask, cv2.getRotationMatrix2D((cx, cy), a, 1.0),
        (w, h), flags=cv2.INTER_NEAREST, borderValue=0)) for a in angles)


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
    if _self_rot_iou(sil, (90.0,)) >= ROT_ELIGIBLE_SYM_IOU:  # (a) multi-fold symmetric
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


# ── Perimeter outline + diagonal "disjoint" repair (Stage-4 v1, pixel-level) ─────────
# A DIFFERENT product from the skeleton path above: the user's "outline" here is the 1-px
# BOUNDARY PERIMETER of the silhouette (a closed loop hugging the edge), NOT the medial-
# axis centerline. Renders are monochromatic, so the outline is the perimeter of the
# UNION silhouette (outer edge + every hole/counter ring). A staircased perimeter touches
# itself only at CORNERS across a diagonal step (8-connected, not 4-connected); the repair
# ("disjointing") fills ONE pixel per such junction so the loop is FACE-connected. The fill
# goes on the INTERIOR side of the gap by default (no outward bulge), EXTERIOR only when
# there is no interior corner. Library-first (H2): cv2.erode for the ring, vectorized numpy
# 2×2 windows for the junctions (no per-pixel loop). Additive-only ⇒ never opens a seam;
# a no-diagonal outline is a byte-identical no-op. Deterministic (H8): no RNG.

PERIM_MAX_PASSES = 8       # connect_disjoint fixpoint cap (a fill can expose a new neighbour touch)


def silhouette_outline(silhouette: np.ndarray) -> np.ndarray:
    """1-px INNER boundary ring of a filled mask: every foreground pixel with at least one
    background 8-neighbour. silhouette: (H,W) uint8 {0,255}. Returns (H,W) uint8 {0,255}.

    `M & ~erode(M)` keeps the outermost layer only — stays inside M's footprint, unlike
    MORPH_GRADIENT (dilate−erode) which is 2-px and bulges outward. Hole/counter rings come
    for free (a hole's interior is background, so its rim erodes out). BORDER_CONSTANT 0 is
    load-bearing: cv2.erode defaults to BORDER_REFLECT_101, under which a pixel touching the
    canvas edge sees a mirrored foreground neighbour and does NOT erode out — so the
    silhouette's edge would be missing from the ring (the same convention as _connect_squeeze)."""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(silhouette, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return silhouette & ~eroded


def _diag_fill(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """One vectorized pass: for every 2×2 window of outline `b` that is a DIAGONAL-ONLY touch
    (two fg on a diagonal, two bg on the anti-diagonal), choose ONE empty corner to fill so
    the touch becomes face-connected. Prefer the INTERIOR corner (silhouette `s` == True — the
    side inside the shape, no outward bulge); when neither empty corner is interior, take a
    deterministic one (the only available, exterior, side). `b`, `s`: (H,W) bool. Returns the
    (H,W) bool fill mask (caller ORs it into the outline). No RNG (H8)."""
    tl, tr = b[:-1, :-1], b[:-1, 1:]
    bl, br = b[1:, :-1], b[1:, 1:]
    pat_a = tl & br & ~tr & ~bl      # fg on the ╲ diagonal; empty corners TR=(r,c+1), BL=(r+1,c)
    pat_b = tr & bl & ~tl & ~br      # fg on the ╱ diagonal; empty corners TL=(r,c),   BR=(r+1,c+1)
    s_tl, s_tr = s[:-1, :-1], s[:-1, 1:]
    s_bl, s_br = s[1:, :-1], s[1:, 1:]
    fill = np.zeros_like(b)
    # The corner DOCTRINE below (interior first, deterministic exterior fallback) has ONE
    # scalar source of truth: _diag_corners — the vectorized expressions here must stay
    # equivalent to it (the pocket-averse path's byte-identity rests on that equivalence).
    # pat_a → fill TR if interior (or both-exterior fallback), else BL (the interior one)
    fill[:-1, 1:] |= pat_a & (s_tr | ~s_bl)
    fill[1:, :-1] |= pat_a & ~s_tr & s_bl
    # pat_b → fill TL if interior (or both-exterior fallback), else BR
    fill[:-1, :-1] |= pat_b & (s_tl | ~s_br)
    fill[1:, 1:] |= pat_b & ~s_tl & s_br
    return fill


def _diag_touch_list(fg: np.ndarray) -> list[tuple[int, int, str]]:
    """Row-major sorted diagonal-only 2x2 touches of a bool mask, as (r, c, kind):
    kind 'a' = own ink on the ╲ diagonal (empty TR/BL), 'b' = on the ╱ (empty TL/BR).
    THE shared touch enumeration for _connect_averse and connect_disjoint_paired (H5)."""
    tl, tr = fg[:-1, :-1], fg[:-1, 1:]
    bl, br = fg[1:, :-1], fg[1:, 1:]
    return sorted(
        [(int(r), int(c), "a") for r, c in np.argwhere(tl & br & ~tr & ~bl)]
        + [(int(r), int(c), "b") for r, c in np.argwhere(tr & bl & ~tl & ~br)])


def _diag_corners(r: int, c: int, kind: str, sil: np.ndarray) -> tuple[tuple, tuple]:
    """Scalar form of _diag_fill's corner doctrine — the SINGLE source of the interior/
    exterior choice for a diagonal-only 2x2 touch anchored at (r, c): returns
    (d_int, d_ext) where d_int is the corner _diag_fill would fill (interior preferred,
    deterministic exterior fallback) and d_ext the other empty corner. kind 'a' = own ink
    on the ╲ diagonal (empty TR/BL); 'b' = own ink on the ╱ diagonal (empty TL/BR)."""
    if kind == "a":
        return (((r, c + 1), (r + 1, c)) if (sil[r, c + 1] or not sil[r + 1, c])
                else ((r + 1, c), (r, c + 1)))
    return (((r, c), (r + 1, c + 1)) if (sil[r, c] or not sil[r + 1, c + 1])
            else ((r + 1, c + 1), (r, c)))


def connect_disjoint(outline: np.ndarray, silhouette: np.ndarray, target: int,
                     max_passes: int = PERIM_MAX_PASSES,
                     pocket_averse: bool = False) -> np.ndarray:
    """Make the perimeter FACE-connected (4-connectivity): fill ONE pixel per diagonal-only
    corner touch, interior side preferred. Iterates to a fixpoint (a fill can create a new
    neighbouring touch) capped at max_passes. Additive only — never removes a pixel, so it
    can never open a seam; a no-diagonal outline is a byte-identical no-op (zero regression).
    outline / silhouette: (target,target) uint8 {0,255}. Returns (target,target) uint8.
    Deterministic (H8).

    pocket_averse=True switches to the POCKET-AVERSE variant (the tight-wireframe
    doctrine: corner connections on the OUTSIDE instead of the inside): per touch, when
    the default interior fill would shrink/seal a TIGHT pocket, the EXTERIOR empty corner
    is filled instead — accepted only when it is background, adds no 2x2 blot, gains no
    foreign 4-adjacency beyond the touch's own two pixels, and strictly reduces the
    diagonal-only touch count; otherwise the interior fill proceeds as today. The averse
    variant must ONLY ship through perimeter_ring's arbitration (un-arbitrated it can
    redirect reopen_channels' carving and CREATE tight pockets — mathematica@16)."""
    out = outline > 0
    sil = silhouette > 0
    if pocket_averse:
        return _connect_averse(out, sil, max_passes)
    for _ in range(max_passes):
        fill = _diag_fill(out, sil)
        if not fill.any():
            break
        out = out | fill
    return (out * 255).astype(np.uint8)


def _connect_averse(out: np.ndarray, sil: np.ndarray, max_passes: int) -> np.ndarray:
    """Scalarized pocket-averse body of connect_disjoint (see its docstring). Enumerates
    each pass's diagonal-only touches row-major from the pass-start snapshot and applies
    the IDENTICAL corner choice as _diag_fill unless the swap conditions hold, so cells
    with no swap are byte-identical to the vectorized path by construction. H8."""
    h, w = out.shape
    for _ in range(max_passes):
        touches = _diag_touch_list(out)
        if not touches:
            break
        any_fill = False
        for r, c, k in touches:
            own = ((r, c), (r + 1, c + 1)) if k == "a" else ((r, c + 1), (r + 1, c))
            d_int, d_ext = _diag_corners(r, c, k, sil)     # the shared corner doctrine
            if out[d_int]:                                 # already repaired this pass
                continue
            trial = out.copy()
            trial[d_int] = True
            seals_tight = any(
                _is_tight(area, ph, pw)
                and any(abs(d_int[0] - py) + abs(d_int[1] - px) == 1 for py, px in cells)
                for cells, area, ph, pw, _t, _l in _enclosed_bg(trial))
            if not seals_tight:                            # today's behaviour, verbatim
                out[d_int] = True
                any_fill = True
                continue
            ok = not out[d_ext]
            if ok:
                ext_trial = out.copy()
                ext_trial[d_ext] = True
                ok = (int(np.count_nonzero(_block_corners(ext_trial)))
                      <= int(np.count_nonzero(_block_corners(out))))
                if ok:
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        q = (d_ext[0] + dy, d_ext[1] + dx)
                        if 0 <= q[0] < h and 0 <= q[1] < w and out[q] and q not in own:
                            ok = False                     # foreign 4-adjacency
                            break
                if ok and _diag_touches(ext_trial) >= _diag_touches(out):
                    ok = False                             # must strictly reduce touches
            out[d_ext if ok else d_int] = True
            any_fill = True
        if not any_fill:
            break
    return (out * 255).astype(np.uint8)


# ── un-blot small features: counters that merge into the outline at tiny T ───────────
# At feed res a small interior counter (discord's eyes @16) can sit so close to another
# boundary that silhouette_outline merges its rim into the outline mass — the feature
# "blots" and stops reading as a distinct element (at 24 the eyes outline as separate
# rings; at 16 the face is too small for erode to open any gap). This pass restores a
# >= UNBLOT_GAP-px wall to the nearest other boundary by MOVING the hole 1px (toward open
# space) or, failing that, DOWNSIZING it — entirely in the MASK domain (layer-on-top: a
# solo part re-render is infeasible, the part's solo footprint != its joint footprint, per
# the project_cleanup_layer_on_top_scope memory). The refine-tier analog of the render-tier
# separation passes (render._outline_gaps / render_debug._separate_divots /
# cubify._carve_necks), in a different domain (interior holes of the rendered silhouette).
# Deterministic (H8), topology-guarded, byte-identical no-op when nothing blots.

UNBLOT_GAP = 3           # min wall thickness (fg px) from a counter to the nearest other
#                          boundary for it to read as un-blotted. A 1-px gap in the OUTLINE
#                          needs a 3-px wall (W≥3 leaves a non-outline fg pixel between the two
#                          rims; W=2 merges them): discord's eyes are wall-3/distinct @24,
#                          wall-2/blotted @16 — the threshold that splits those two (knob).
UNBLOT_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
_SE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def _min_feat_at(target: int) -> int:
    """Smallest counter worth keeping at target res — mirrors compress.common._min_feature_px
    (max(2, ceil(s / MIN_FEATURE_DIVISOR=48))) evaluated at the target grid."""
    return max(2, math.ceil(target / 48))


def _significant_holes(mask: np.ndarray, min_feat: int) -> list[np.ndarray]:
    """Each interior hole (enclosed-background component) of `mask` whose bbox spans >=
    min_feat in BOTH dims, as its own uint8 {0,255} region. Border flood-fill of the inverted
    mask isolates enclosed bg (mirrors render_debug._sig_topology's technique — reimplemented
    here, not imported, to avoid the render_debug → refine.common import cycle). H8."""
    m = (mask > 0).astype(np.uint8)
    pad = cv2.copyMakeBorder(m * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(pad)
    ffm = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(inv, ffm, (0, 0), 0)                  # border-connected bg → 0; enclosed bg remains
    n, lbl, st, _c = cv2.connectedComponentsWithStats(inv[1:-1, 1:-1], connectivity=8)
    return [((lbl == i).astype(np.uint8) * 255) for i in range(1, n)
            if min(int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])) >= min_feat]


# _outer_bg: LIFTED to compress/common.py (H5 — cubify._is_open and the render-tier
# witness/curtain machinery share it); imported above with the same name and
# semantics (the complement of _significant_holes' enclosed bg).


def _wall(fg: np.ndarray, hole: np.ndarray, outer_bg: np.ndarray) -> int:
    """Wall thickness = number of foreground px between `hole` and the nearest OUTER-silhouette
    boundary (`outer_bg`). Other interior holes are NOT boundaries here — two counters sitting
    close (discord's eyes) is not a blot; only a counter merging with the silhouette EDGE is.
    Grow the hole 1px/ring through the FOREGROUND ONLY (routing around other holes); the first
    ring touching the outer background gives the thickness. Cheap on feed-res masks. H8."""
    fgb = fg > 0
    grown = hole > 0
    for d in range(1, int(max(fg.shape)) + 2):
        rim = (cv2.dilate(grown.astype(np.uint8), _SE3) > 0) & ~grown
        if bool(np.any(rim & outer_bg)):
            return d - 1
        nxt = rim & fgb
        if not bool(np.any(nxt)):                        # sealed in by other holes; never reached edge
            return int(max(fg.shape))
        grown = grown | nxt
    return int(max(fg.shape))


CHANNEL_MAX_PASSES = 16   # tunnel/carve fixpoint cap (a carve can expose the next candidate)
TIGHT_MAX_AREA = 4        # enclosed-bg pockets at/under this area = TIGHT (wireframe too tight)
TIGHT_SLIT_LEN = 3        # 1xN enclosed runs with N >= this = hollow-3px-stroke middles (intended)
RELIEF_MAX_PASSES = 3     # a 1x1 pocket exits the tight class in <= 3 outward moves (1x1->2x3)


def _enclosed_bg(ring: np.ndarray) -> list[tuple]:
    """Non-border enclosed background components of a ring image under 4-connectivity,
    row-major by bbox: [(cells frozenset{(y,x)}, area, h, w, top, left), ...]. Border-
    label exclusion drops the outer background; 4-conn background cannot leak through a
    diagonal-only ink touch, so enclosure is stable across connect_disjoint passes. H8."""
    bg = (~(ring > 0)).astype(np.uint8)
    n, lbl, st, _c = cv2.connectedComponentsWithStats(bg, connectivity=4)
    border = set(np.unique(lbl[0, :])) | set(np.unique(lbl[-1, :])) \
        | set(np.unique(lbl[:, 0])) | set(np.unique(lbl[:, -1]))
    out = []
    for i in range(1, n):
        if i in border:
            continue
        out.append((frozenset(map(tuple, np.argwhere(lbl == i))),
                    int(st[i, cv2.CC_STAT_AREA]),
                    int(st[i, cv2.CC_STAT_HEIGHT]), int(st[i, cv2.CC_STAT_WIDTH]),
                    int(st[i, cv2.CC_STAT_TOP]), int(st[i, cv2.CC_STAT_LEFT])))
    return sorted(out, key=lambda c: (c[4], c[5]))


def _is_tight(area: int, h: int, w: int) -> bool:
    """TIGHT = a too-small enclosed white pocket. A 1xN run with N >= TIGHT_SLIT_LEN is
    the INTENDED middle of a hollow 3px stroke, never tight (a 1xN with N < TIGHT_SLIT_LEN
    is already area <= TIGHT_MAX_AREA, so no separate slit-area clause is needed)."""
    if min(h, w) == 1 and max(h, w) >= TIGHT_SLIT_LEN:
        return False
    return area <= TIGHT_MAX_AREA


def count_tight_pockets(ring: np.ndarray) -> int:
    """Number of TIGHT enclosed pockets in a ring — the tight-wireframe gate metric
    (0 = every enclosed white region is a legitimate counter or hollow-stroke middle)."""
    return sum(1 for _cells, area, h, w, _t, _l in _enclosed_bg(ring)
               if _is_tight(area, h, w))


def count_tight_ref(ring: np.ndarray, reference: np.ndarray,
                    protected: np.ndarray | None = None) -> int:
    """count_tight_pockets EXCLUDING pockets overlapping a (1px-dilated) REFERENCE hole:
    a counter the ideal geometry prescribes (discord's eyes) is a RESTORED FEATURE, not
    wireframe tightness — the raw tight count rises when such a counter comes back, which
    is exactly the metric artifact this ref-aware sibling exists to remove. Dilated 1px
    because unblot legally moves a counter one pixel off its reference position. The gate
    metric of the detail-retention rework; perimeter_ring's internal arbitration keeps
    the RAW count (a ref-blind tail must not chase reference alignment).

    `protected` (assemble_render's accent zone) additionally excludes pockets fully under
    its 1px dilation — the same exclusion shape as the ref-hole clause: instagram@24's
    3x3-OUTLINE dot carries a DELIBERATE 1x1 interior hole (user-ratified form) whose
    canonical is a solid circle, so no reference hole can ever back it; charging it as
    wireframe tightness would misread a designed accent as a defect."""
    rz0 = np.zeros(ring.shape, np.uint8)
    for cells, *_r in _enclosed_bg(reference):
        for c in cells:
            rz0[c] = 255
    rz = cv2.dilate(rz0, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if protected is not None and protected.any():
        pz = cv2.dilate((protected > 0).astype(np.uint8), _SE3,
                        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        rz = rz | pz
    return sum(1 for cells, area, h, w, _t, _l in _enclosed_bg(ring)
               if _is_tight(area, h, w) and not any(rz[c] for c in cells))


def _retention_violation(before: np.ndarray, after: np.ndarray,
                         reference: np.ndarray | None) -> np.ndarray:
    """Bool mask of REFERENCE-SUPPORTED ink of `before` whose 3x3-local ink `after`
    lost — whole-feature deletion scores here (android's antennae); a legit 2px->1px
    thin does not (the surviving offset is adjacent). THE single source of the
    retention formula (H5): _retention_loss / _retention_comp_max / the harness's
    ret column all reduce this mask."""
    if reference is None:
        return np.zeros(before.shape, bool)
    removed = (before > 0) & (reference > 0)
    has = cv2.dilate(after, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return removed & ~has


def _retention_loss(before: np.ndarray, after: np.ndarray,
                    reference: np.ndarray | None) -> int:
    """Count of retention-violation px (see _retention_violation)."""
    return int(np.count_nonzero(_retention_violation(before, after, reference)))


def _retention_comp_max(before: np.ndarray, after: np.ndarray,
                        reference: np.ndarray | None) -> int:
    """Largest 8-connected component of the retention violation — the accent-budget
    unit: several sub-budget minority-accent drops are dual-axis symmetrization's
    normal job; ONE feature-sized ref-supported loss is destruction."""
    viol = _retention_violation(before, after, reference).astype(np.uint8)
    if not viol.any():
        return 0
    n, _l, st, _c = cv2.connectedComponentsWithStats(viol, connectivity=8)
    return max(int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n))


def _diag_touches(fg: np.ndarray) -> int:
    """Count of diagonal-only 2x2 touches (the material connect_disjoint repairs) — a
    'must not increase' guard for the relief/averse moves."""
    tl, tr = fg[:-1, :-1], fg[:-1, 1:]
    bl, br = fg[1:, :-1], fg[1:, 1:]
    return int(np.count_nonzero(tl & br & ~tr & ~bl)
               + np.count_nonzero(tr & bl & ~tl & ~br))


def _clean_loop_counter(fg: np.ndarray, cells: frozenset) -> bool:
    """True when `cells` (an enclosed pocket) is the interior of an INTENDED outlined
    dot/ring: its surrounding ink is ONE component, that component is a pure degree-2
    loop (every px exactly two 4-neighbours), and the pocket is the loop's sole hole
    (adobe's i-dot, linkedin's box corner, figma's ring dot). Such pockets are glyph
    features — the relief pass must never inflate them."""
    h, w = fg.shape
    n, lbl = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    comp_ids = set()
    for (py, px) in cells:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            qy, qx = py + dy, px + dx
            if 0 <= qy < h and 0 <= qx < w and fg[qy, qx]:
                comp_ids.add(int(lbl[qy, qx]))
    if len(comp_ids) != 1:
        return False
    comp = lbl == comp_ids.pop()
    deg = _nb4count((comp * 255).astype(np.uint8))
    if not bool(np.all(deg[comp] == 2)):
        return False
    holes = _enclosed_bg((comp * 255).astype(np.uint8))
    return len(holes) == 1 and holes[0][0] == cells


def _block_corners(f: np.ndarray) -> np.ndarray:
    """(H-1, W-1) bool map of 2x2 all-foreground windows (top-left anchored) — the ONE
    canonical "blot" primitive: a 2px smudge in 1px line art is exactly such a window.
    Shared by _blot2x2 (count) and _block_spread (corner spread) per H5."""
    return f[:-1, :-1] & f[1:, :-1] & f[:-1, 1:] & f[1:, 1:]


def _block_spread(f: np.ndarray) -> np.ndarray:
    """True where a pixel participates in ANY 2x2 all-ink block — _block_corners spread
    back onto its four corners. Shared by reopen_channels and the mirror-paired carve
    variants (H5 — one copy, was an inline closure)."""
    blk = _block_corners(f)
    out = np.zeros_like(f)
    out[:-1, :-1] |= blk
    out[1:, :-1] |= blk
    out[:-1, 1:] |= blk
    out[1:, 1:] |= blk
    return out


def reopen_channels(ring: np.ndarray, target: int) -> np.ndarray:
    """NEGATIVE-SPACE continuity, ring tier: carve chain-blot ink out of the final
    face-connected outline. Where band bends / bunched strokes fuse the two rails
    (chrome@16's annulus, firefox@16's crescent bend), the ring contains 2x2 all-ink
    blocks that chop the white channel between rails into pockets. Greedily remove
    blot px (px inside a 2x2 block) that touch background (4-neighbour or canvas
    border — carving only ever EXTENDS existing negative space, so no new pinholes),
    guarded so the ink's face-connected component count never rises. Clean 1px
    linework has no 2x2 blocks, so counters drawn as clean loops (discord's eyes) are
    untouchable by construction. Fixpoint to CHANNEL_MAX_PASSES. Deterministic (H8:
    row-major carve order)."""
    if not ring.any():
        return ring
    base_c4 = cv2.connectedComponents(ring, connectivity=4)[0] - 1
    # Delegates to the shared _orbit_carve with EMPTY axes (one-point orbits) — the
    # identical fixpoint carve; carving is ink-monotonic, so the orbit loop's
    # per-candidate nb<4 re-check can never reject a pass-start candidate (H5: one
    # carve loop for the unpaired, paired and fold-repair passes).
    return _orbit_carve(
        ring.copy(), [],
        lambda trial: cv2.connectedComponents(trial, connectivity=4)[0] - 1 <= base_c4)


_RELIEF_DIRS = (("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1)))


def _relief_accept(trial: np.ndarray, base_c4: int, base_blot: int, base_diag: int,
                   n_encl: int, grown: frozenset,
                   grown2: frozenset | None = None) -> list | None:
    """The relief structural-accept GAUNTLET — THE single source for both relief twins
    (H5): ink c4-component count exactly unchanged, no new 2x2 blot, diagonal-only
    touches not increased, enclosed-pocket count unchanged, and the relieved pocket(s)
    == old + wall exactly (grow, never merge; the paired twin checks its partner via
    grown2). Returns the trial's enclosed-bg list on pass, None on any reject."""
    if cv2.connectedComponents(trial.astype(np.uint8), connectivity=4)[0] - 1 != base_c4:
        return None
    if int(np.count_nonzero(_block_corners(trial))) > base_blot:
        return None
    if _diag_touches(trial) > base_diag:
        return None
    tps = _enclosed_bg((trial * 255).astype(np.uint8))
    if len(tps) != n_encl:
        return None
    if not any(cc[0] == grown for cc in tps):
        return None
    if grown2 is not None and not any(cc[0] == grown2 for cc in tps):
        return None
    return tps


def _relief_cand(cur: np.ndarray, outer: np.ndarray, wall: list, di: int,
                 dy: int, dx: int, py0: int, px0: int, ph: int, pw: int,
                 scy: float, scx: float, trial: np.ndarray) -> tuple:
    """The relief RANKING tuple (grows-short-dim, outward clearance capped 3,
    mirror-invariant outwardness, fixed direction index, trial) — THE shared ranking
    construction for relieve_tight_pockets and its paired variant (H5)."""
    h, w = cur.shape
    clr = 3                                                # outward clearance, capped
    for (y, x) in wall:
        run = 0
        for d2 in range(2, 5):
            q = (y + dy * d2, x + dx * d2)
            if 0 <= q[0] < h and 0 <= q[1] < w and not cur[q] and outer[q]:
                run += 1
            else:
                break
        clr = min(clr, 1 + run)
    outw = dy * ((py0 + ph / 2.0) - scy) + dx * ((px0 + pw / 2.0) - scx)
    grows_short = 1 if ((ph < pw and dy != 0) or (pw < ph and dx != 0)
                        or ph == pw) else 0
    return (grows_short, clr, outw, -di, trial)


def _pocket_move(cur: np.ndarray, cells: frozenset, dname: str,
                 outer: np.ndarray) -> tuple[np.ndarray, frozenset, list] | None:
    """One directional wall-move OFFER for tight-pocket relief — THE single source of
    the relief geometry chain (flat contiguous wall -> free-exterior outward -> outer-
    position corner connectors -> 1px foreign-ink separation), shared by
    relieve_tight_pockets and its mirror-paired variant (H5: the structural guards
    c4/blot/diag/pocket-count stay at the CALL SITES — the paired variant re-checks
    them on the combined primary+partner trial). `cur` bool mask, `outer` = _outer_bg
    of it. Returns (trial bool mask, grown pocket cells, wall px) or None when blocked."""
    h, w = cur.shape
    dy, dx = dict(_RELIEF_DIRS)[dname]
    face = [(y, x) for (y, x) in cells if (y + dy, x + dx) not in cells]
    wall = [(y + dy, x + dx) for (y, x) in face]
    if len({(y if dy else x) for (y, x) in wall}) != 1:
        return None                                    # staircase face -> not a flat wall
    axis = 1 if dy else 0
    coords = sorted(p[axis] for p in wall)
    if coords != list(range(coords[0], coords[-1] + 1)):
        return None                                    # gap in the run
    outward = [(y + dy, x + dx) for (y, x) in wall]
    if not all(0 <= y < h and 0 <= x < w for y, x in outward):
        return None                                    # canvas edge
    if not all((not cur[y, x]) and outer[y, x] for y, x in outward):
        return None                                    # not free exterior
    wall_sorted = sorted(wall, key=lambda p: p[axis])
    step = (0, 1) if axis == 1 else (1, 0)
    ends = ((wall_sorted[0][0] - step[0], wall_sorted[0][1] - step[1]),
            (wall_sorted[-1][0] + step[0], wall_sorted[-1][1] + step[1]))
    anchors = [a for a in ends if 0 <= a[0] < h and 0 <= a[1] < w and cur[a]]
    conns = []
    for a in anchors:                                  # corner connection ON THE OUTER POSITION
        cpx = (a[0] + dy, a[1] + dx)
        if not (0 <= cpx[0] < h and 0 <= cpx[1] < w):
            return None
        if cur[cpx]:
            continue                                   # already ink -> nothing to add
        if not outer[cpx]:
            return None
        conns.append(cpx)
    new_ink = outward + conns
    seed = np.zeros((h, w), np.uint8)
    for p in wall + anchors:
        seed[p] = 1
    loc = seed.copy()                                  # 2-step geodesic ink neighbourhood
    for _g in range(2):
        loc = cv2.dilate(loc, _SE3, borderType=cv2.BORDER_CONSTANT,
                         borderValue=0) & cur.astype(np.uint8)
    loc |= seed
    if any(0 <= ny + dy8 < h and 0 <= nx + dx8 < w
           and cur[ny + dy8, nx + dx8] and not loc[ny + dy8, nx + dx8]
           for ny, nx in new_ink
           for dy8 in (-1, 0, 1) for dx8 in (-1, 0, 1)):
        return None                                    # 1px separation to FOREIGN ink
    trial = cur.copy()
    for p in wall:
        trial[p] = False
    for p in new_ink:
        trial[p] = True
    return trial, frozenset(cells | set(wall)), wall


def relieve_tight_pockets(ring: np.ndarray, target: int) -> np.ndarray:
    """Outward wall relief for TIGHT enclosed pockets — the user's hand-doctrine encoded:
    a too-tight wireframe section gains interior room by moving its 1px wall one step
    OUTWARD into free exterior, the reconnecting corner pixels placed on the OUTER
    positions — never by eating interior white. Per pass (<= RELIEF_MAX_PASSES), per
    still-tight pocket (row-major; clean-loop counters skipped — intended dot/ring
    interiors), each direction d offers: wall = the pocket's face+d (must be one flat
    contiguous run); outward = wall+d (all in-canvas, background AND border-connected
    exterior — never another pocket); anchors = the along-run ink neighbours at the
    wall's ends, reconnected via anchor+d corner pixels (ink already, or free exterior).
    Guards per candidate: 1px stroke separation (no new ink 8-adjacent to FOREIGN ink =
    ink outside the 2-step geodesic neighbourhood of wall+anchors — blocks slack's
    two-pill abutment), ink c4-component count EXACTLY unchanged, no new 2x2 blot,
    diagonal-only touches not increased, enclosed-pocket count unchanged AND the
    relieved pocket == old + wall exactly (grow, never merge). Rank: (grows-short-dim,
    outward clearance capped 3, outwardness = d . (pocket centroid - ink centroid) —
    mirror-invariant, fixed direction index). Interior-locked pockets (react's crossing
    lattice) and guard-blocked ones are honest partials. Byte-identical no-op when
    nothing qualifies. Deterministic (H8)."""
    cur = ring > 0
    h, w = cur.shape
    ys, xs = np.nonzero(cur)
    if xs.size == 0:
        return ring
    base_c4 = cv2.connectedComponents(cur.astype(np.uint8), connectivity=4)[0] - 1
    base_blot = int(np.count_nonzero(_block_corners(cur)))
    base_diag = _diag_touches(cur)
    scy, scx = float(ys.mean()), float(xs.mean())
    for _ in range(RELIEF_MAX_PASSES):
        changed = False
        pockets = _enclosed_bg((cur * 255).astype(np.uint8))
        n_encl = len(pockets)
        for cells, area, ph, pw, py0, px0 in pockets:
            if not _is_tight(area, ph, pw) or _clean_loop_counter(cur, cells):
                continue
            outer = _outer_bg((cur * 255).astype(np.uint8))
            cands = []
            for di, (dname, (dy, dx)) in enumerate(_RELIEF_DIRS):
                offer = _pocket_move(cur, cells, dname, outer)
                if offer is None:
                    continue
                trial, grown, wall = offer
                if _relief_accept(trial, base_c4, base_blot, base_diag,
                                  n_encl, grown) is None:
                    continue
                cands.append(_relief_cand(cur, outer, wall, di, dy, dx,
                                          py0, px0, ph, pw, scy, scx, trial))
            if cands:
                cands.sort(key=lambda cnd: cnd[:4], reverse=True)
                cur = cands[0][4]
                changed = True
        if not changed:
            break
    return (cur * 255).astype(np.uint8)


def _shift_region(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer (dy,dx) shift of a region, ZERO-filled at the vacated border (NOT np.roll, which
    wraps). Pure-numpy slicing — deterministic, no interpolation (H8)."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def _recarve(sil: np.ndarray, old_hole: np.ndarray, new_hole: np.ndarray) -> np.ndarray:
    """Replace `old_hole` with `new_hole` in `sil`: fill the old hole (→ fg), carve the new
    (→ bg). uint8 {0,255}."""
    out = sil.copy()
    out[old_hole > 0] = 255
    out[new_hole > 0] = 0
    return out


def unblot_features(silhouette: np.ndarray, target: int) -> np.ndarray:
    """Move or downsize blotting counters so each keeps a >= UNBLOT_GAP-px wall to the nearest
    other boundary. For each significant hole that blots (wall < UNBLOT_GAP): try the 8 single-
    px MOVES, keep those that reach the wall AND preserve topology, pick the one with the
    THICKEST resulting wall (tie-break toward the silhouette centroid — mirror-invariant, so a
    symmetric pair like discord's eyes stays symmetric); if no move works, DOWNSIZE the hole
    1px (erode, floored at min_feat). Honest partial (leave it) if neither helps. Mask-domain,
    topology-guarded, deterministic; a no-blot silhouette is returned byte-identical."""
    sil = silhouette.copy()
    if not sil.any():
        return sil
    min_feat = _min_feat_at(target)
    base = _topology(sil)
    outer = _outer_bg(sil)                              # outer boundary is invariant under interior moves
    ys, xs = np.nonzero(sil)
    cy, cx = float(ys.mean()), float(xs.mean())
    for hole in _significant_holes(sil, min_feat):
        if _wall(sil, hole, outer) >= UNBLOT_GAP:
            continue                                    # not blotting
        hy, hx = np.nonzero(hole)
        hcy, hcx = float(hy.mean()), float(hx.mean())
        best, best_key = None, None
        for dy, dx in UNBLOT_MOVES:
            nh = _shift_region(hole, dy, dx)
            if not nh.any():
                continue
            trial = _recarve(sil, hole, nh)
            if _topology(trial) != base:                # breached / merged / lost a hole
                continue
            w = _wall(trial, nh, outer)
            if w < UNBLOT_GAP:
                continue
            toward = dy * (cy - hcy) + dx * (cx - hcx)  # mirror-invariant tie-break
            key = (w, toward)
            if best_key is None or key > best_key:
                best, best_key = trial, key
        if best is None:                                # downsize fallback
            nh = cv2.erode(hole, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if nh.any():
                ny, nx = np.nonzero(nh)
                if (int(ny.max() - ny.min()) + 1 >= min_feat
                        and int(nx.max() - nx.min()) + 1 >= min_feat):
                    trial = _recarve(sil, hole, nh)
                    if _topology(trial) == base and _wall(trial, nh, outer) >= UNBLOT_GAP:
                        best = trial
        if best is not None:
            sil = best
    return sil


WIDEN_TGT_THRESH = 110   # @24 footprint coverage (downscaled grayscale 0-255) above which the
#                          @24 sibling "wants" foreground at a pixel — the continuous zone target.
WIDEN_MAX_PASS = 6


def _flanked(mask: np.ndarray) -> np.ndarray:
    """Bool: background pixels with foreground on BOTH horizontal OR both vertical 4-neighbours —
    an inner gap/notch being bridged (a concavity, e.g. gmail's M top-centre seam), NOT an outward
    protrusion tip. Vectorized, deterministic (H8)."""
    f = mask > 0
    h = np.zeros_like(f)
    v = np.zeros_like(f)
    h[:, 1:-1] = f[:, :-2] & f[:, 2:]
    v[1:-1, :] = f[:-2, :] & f[2:, :]
    return (h | v) & (mask == 0)


RADIAL_MIN_PEAKS = 6     # >= this many prominent angular-mass peaks (blades) = a radial/sunburst
#                          mark (claude/firefox), EXCLUDED from the continuity widen per the
#                          project's radial-family exclusion. Seeds: claude 11; gmail/figma 2,
#                          discord 1, vscode 4 — a wide margin.


def _is_radial(sil: np.ndarray) -> bool:
    """True for sunburst/radial-blade marks: the angular mass profile (foreground pixels binned by
    angle about the centroid) has >= RADIAL_MIN_PEAKS prominent peaks = many blades. The project's
    own sunburst signal (peak-COUNT, not rfft — a spike train smears harmonics). Deterministic (H8):
    a pure-numpy angular histogram, NOT cv2.warpPolar — that op races near the peak-count threshold
    under a warm multithreaded pool and flaked this gate (and every caller: widen, deburr)
    regardless of thread pinning; the histogram is thread-independent by construction."""
    s = cv2.resize(sil, (96, 96), interpolation=cv2.INTER_NEAREST)     # INTER_NEAREST: no interp, deterministic
    ys, xs = np.nonzero(s)
    if xs.size == 0:
        return False
    cx, cy = float(xs.mean()), float(ys.mean())
    ang = np.degrees(np.arctan2(ys.astype(np.float64) - cy, xs.astype(np.float64) - cx)) % 360.0
    prof = np.bincount(np.minimum(ang.astype(np.int64), 359), minlength=360).astype(np.float64)
    if prof.max() <= 0:
        return False
    k = np.ones(9) / 9.0                                  # circular box-smooth the per-degree histogram
    prof = np.convolve(np.concatenate([prof, prof, prof]), k, "same")[360:720]   # (~warpPolar's radial avg)
    prof = np.concatenate([prof, prof[:30]])              # wrap so peaks at 0deg aren't missed
    peaks, _ = find_peaks(prof, prominence=0.25 * float(prof.max()), distance=12)
    return int(np.count_nonzero(peaks < 360)) >= RADIAL_MIN_PEAKS


def widen_underrepresented(sil: np.ndarray, target: int,
                           structure_ref: np.ndarray | None = None,
                           protected_bg: np.ndarray | None = None) -> np.ndarray:
    """Cross-resolution inner-feature CONTINUITY widen. When the @24 sibling (`structure_ref`)
    renders an inner feature as a continuous filled zone but @16 pinches it (gmail's M-centre: a
    1px tip + an unclosed top-centre seam), fill @16 toward @24's footprint so it reads continuous.

    Mechanism: downscale @24 to the target grid as the footprint @24 "wants" (`WIDEN_TGT_THRESH`);
    fill only deficit pixels that sit at a PINCH (thin fg that vanishes under a 3x3 erosion) or at
    a FLANKED inner gap (_flanked) — never an outward protrusion tip; each fill applied as a MIRROR
    PAIR about the glyph's vertical axis (symmetric inputs stay symmetric → clean zone, not a blob)
    and accepted only if `_topology` (components, holes) is unchanged (so separate counters — e.g.
    discord's eyes — can never merge). No-op when structure_ref is None (the @24 render is
    authoritative and never widens). Deterministic (H8: no RNG, INTER_AREA, row-major).

    NOTE: this is a GENERAL continuity rule; geometrically it ALSO fires on any fixture whose inner
    features are thinner than its @24 (discord's body, etc.) — those are not distinguishable from
    gmail's by any geometric signal. The harness/caller scopes WHICH fixtures it runs on.

    Detail-retention rework: ALL enclosed holes of the input (any size — discord@16's eyes
    are 1x2, below the min-feat significance floor) are PROTECTED: the widen never fills a
    hole px, and a fill whose mirror TWIN lands in a hole is skipped (both-or-neither) —
    the old hole-COUNT topology guard let the widen legally shrink an eye to 1px, which
    symmetrize's spurious-hole fill then erased. `protected_bg` (optional): a
    caller-scoped zone (glyph channel / relayout valley) ORed into the enclosed-hole
    protection — the continuity widen never inks it."""
    if structure_ref is None or not sil.any():               # @24 is authoritative — never widen it
        return sil
    if _is_radial(structure_ref):                            # radial-blade family (claude/firefox)
        return sil                                           # excluded per the project convention
    protected = np.zeros(sil.shape, bool)                    # ALL enclosed holes of the input:
    for cells, *_r in _enclosed_bg(sil):                     # a continuity widen must never eat
        for c in cells:                                      # counter interiors (any size)
            protected[c] = True
    if protected_bg is not None:                             # glyph-channel / relayout-valley
        protected |= protected_bg > 0                        # zone (caller-scoped, W3 guard)
    tgt = cv2.resize(structure_ref, (target, target), interpolation=cv2.INTER_AREA) > WIDEN_TGT_THRESH
    xs = np.nonzero(sil)[1]
    axis = (int(xs.min()) + int(xs.max())) / 2.0          # vertical mirror axis (bbox centre)
    eroded = cv2.erode(sil, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    core = cv2.dilate(eroded, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    pinch = cv2.dilate(((sil > 0) & (core == 0)).astype(np.uint8) * 255, _SE3,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    base = _topology(sil)
    cur = sil.copy()
    for _ in range(WIDEN_MAX_PASS):
        allow = tgt & (pinch | _flanked(cur)) & ~protected   # @24 wants it, pinch/inner gap, no hole
        nb = cv2.dilate(cur, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        cand = (cur == 0) & allow & (nb > 0)
        if not cand.any():
            break
        changed = False
        for y, x in sorted(map(tuple, np.argwhere(cand))):
            if cur[y, x]:
                continue
            tx = int(round(2.0 * axis - x))               # mirror twin (keep symmetry)
            if 0 <= tx < target and protected[y, tx]:
                continue                                  # twin lands in a hole: both-or-neither
            trial = cur.copy()
            trial[y, x] = 255
            if 0 <= tx < target:
                trial[y, tx] = 255
            if _topology(trial) == base:
                cur, changed = trial, True
        if not changed:
            break
    return cur


SPECK_MAX = 2            # isolated micro-components at/under this area are despeck candidates
SPECK_FAMILY_MIN = 4     # >= this many sibling candidates = an intended PATTERN (signal's burst
#                          dashes), kept; 1-3 are noise (docker's mast/container fragments)
PEEL_N4 = 2              # outside-ref fringe peel touches px with at most this many ink 4-neighbours


def despeck_artifacts(silhouette: np.ndarray, target: int,
                      reference: np.ndarray | None) -> np.ndarray:
    """Artifact removal the n4==1-only deburr is structurally blind to (docker's floating
    specks are n4==0, its tail n4==2). Runs BEFORE widen_underrepresented — widen
    otherwise fattens a speck into a floating chimney.

    (a) ISOLATED micro-components: area <= SPECK_MAX whose 1px-dilate ring is all-bg
    (>= 2px of void to any other ink) — ref-BLIND, because docker's mast fragments ARE
    reference-supported yet read as dirt at icon scale. Two probed-in protections:
    keep when the candidate's ref overlap lies wholly in TINY separate reference
    components (the reference itself prescribes a floating feature there), and keep ALL
    candidates when >= SPECK_FAMILY_MIN qualify (an intended pattern — signal@24's burst
    dashes, whose reference is ONE fused component so ref-connectivity cannot protect
    them; the >= 4 family count is the discriminator that survived probing).
    Hole-count guarded (a speck removal drops a component, never a counter).

    (b) OUTSIDE-REFERENCE excess-width fringe peel, row-major to fixpoint: peel an ink
    px with reference==0 and n4 <= PEEL_N4 iff its row ink-run AND col ink-run are each
    LONGER than the longest intersecting reference run in that direction AND at least
    one direction has NO intersecting ref run at all (the px protrudes into ref-empty
    space). All three clauses are load-bearing: equal-width protects registration-
    shifted strokes (linkedin@16's +1-shifted stems), both-excess protects single-
    direction fat, ref-empty-direction protects staircase corner px of large bodies
    (instagram@24's ring corners, android@24's head edge — without it the peel broke
    their symmetrize folds). Exact-topology guarded per px. The scalar run walks are
    the H2 carve-out (<= 24-length extents from an anchor, 15-25x faster than any
    vectorized idiom at this scale — scout-benchmarked).

    No-op without a reference. Deterministic (H8: row-major, fixed passes)."""
    if reference is None or not silhouette.any():
        return silhouette
    ref = reference > 0
    cur = silhouette.copy()
    base = _topology(cur)
    rn, rlbl, rst, _rc = cv2.connectedComponentsWithStats(reference, connectivity=8)
    n, lbl, st, _c = cv2.connectedComponentsWithStats(cur, connectivity=8)
    cands = []
    for i in range(1, n):                    # (a) isolated micro-components
        comp = (lbl == i).astype(np.uint8) * 255
        if int(st[i, cv2.CC_STAT_AREA]) > SPECK_MAX:
            continue
        ring = (cv2.dilate(comp, _SE3, borderType=cv2.BORDER_CONSTANT,
                           borderValue=0) > 0) & (comp == 0)
        if bool(np.any(ring & (cur > 0))):
            continue                         # touches other ink -> not floating
        ref_ids = {int(r) for r in np.unique(rlbl[(comp > 0) | ring]) if r != 0}
        if ref_ids and all(int(rst[r, cv2.CC_STAT_AREA]) <= 2 * SPECK_MAX + 1
                           for r in ref_ids):
            continue                         # the REFERENCE prescribes a tiny floating
        #                                      feature here -> intended; a fragment of a
        #                                      BIG ref body (docker's mast) is render noise
        cands.append(comp)
    if len(cands) < SPECK_FAMILY_MIN:
        for comp in cands:
            trial = cur.copy()
            trial[comp > 0] = 0
            nt = _topology(trial)
            if nt[1] == base[1] and trial.any():
                cur, base = trial, nt

    def _run(mask: np.ndarray, y: int, x: int, horiz: bool) -> tuple[int, int]:
        """[lo, hi] extent of the ink run through (y, x) along the row/column."""
        line = mask[y, :] if horiz else mask[:, x]
        i = x if horiz else y
        lo = i
        while lo > 0 and line[lo - 1]:
            lo -= 1
        hi = i
        while hi < line.size - 1 and line[hi + 1]:
            hi += 1
        return lo, hi

    def _ref_run_max(y: int, x: int, lo: int, hi: int, horiz: bool) -> int:
        """Longest reference run on the same row/col intersecting [lo, hi]."""
        line = ref[y, :] if horiz else ref[:, x]
        best = i = 0
        while i < line.size:
            if line[i]:
                j = i
                while j < line.size - 1 and line[j + 1]:
                    j += 1
                if j >= lo and i <= hi:
                    best = max(best, j - i + 1)
                i = j + 1
            else:
                i += 1
        return best

    def _excess(mask: np.ndarray, y: int, x: int) -> bool:
        m = mask > 0
        lo, hi = _run(m, y, x, True)
        rr = _ref_run_max(y, x, lo, hi, True)
        if hi - lo + 1 <= rr:
            return False                     # registration-shift protected (equal width)
        lo, hi = _run(m, y, x, False)
        rc = _ref_run_max(y, x, lo, hi, False)
        if hi - lo + 1 <= rc:
            return False
        return rr == 0 or rc == 0            # must protrude into ref-EMPTY space

    for _ in range(target):                  # (b) outside-ref excess-width fringe peel
        n4 = _nb4count(cur)
        cand = (cur > 0) & (n4 <= PEEL_N4) & ~ref
        if not cand.any():
            break
        changed = False
        for y, x in sorted(map(tuple, np.argwhere(cand))):
            if not cur[y, x] or not _excess(cur, y, x):
                continue
            trial = cur.copy()
            trial[y, x] = 0
            if _topology(trial) == base and trial.any():
                cur, changed = trial, True
        if not changed:
            break
    return cur


def deburr_nibs(silhouette: np.ndarray, target: int,
                reference: np.ndarray | None = None,
                max_nib: int = 1) -> np.ndarray:
    """Remove barely-attached single-pixel "bumps" (cubify raster jaggies) from a composited
    silhouette. A bump is a FG pixel with EXACTLY ONE 4-connected (shared-edge) neighbour, where
    that neighbour is a SOLID body pixel (>= 3 shared edges) — a lone pixel protruding from a wide
    body. This catches jaggy edge bumps (figma's top row, gmail's arm edges, vscode's ears) while a
    thin-feature TIP is KEPT, because a tip's single neighbour is itself thin (< 3 shared edges — a
    chevron apex, a 1px antenna). Radial-blade marks are exempt only when BOTH the reference AND
    the silhouette read radial (DUAL gate) — a solid box+letters mark whose reference alone
    crosses the peak threshold (linkedin@24, ref peaks == RADIAL_MIN_PEAKS exactly) is NOT
    exempted. Topology-guarded (no component split, no lost counter), bounded to `max_nib`
    passes.

    ACCEPTED TRADEOFF (user ruling — "aggressive de-jag, all bumps"): the reference and symmetry
    gates are dropped, so a genuine sharp FILLED point (a triangle / wedge apex) — which also
    protrudes from a solid body — WILL be shaved. No geometric signal separates a rogue bump from
    an intended sharp point; over-fires are surfaced by the full-fixture audit, not prevented here.
    No-op (byte-identical) when reference is None, the mark is radial, or nothing qualifies.
    Deterministic (H8: no RNG, row-major order, thread-pinned _is_radial, fixed cap)."""
    if reference is None or not silhouette.any():          # no oracle -> never guess
        return silhouette
    if _is_radial(reference) and _is_radial(silhouette):   # radial-blade family — DUAL gate: a solid
        return silhouette                                  # box+letters mark whose REFERENCE alone
    #                                                        crosses the peak threshold (linkedin@24,
    #                                                        ref peaks == RADIAL_MIN_PEAKS exactly) no
    #                                                        longer disables deburr; true radials read
    #                                                        radial in BOTH masks. The knob itself
    #                                                        cannot move: spotify/yt-music sit AT 6 and
    #                                                        normalize's radial family depends on them.
    base = _topology(silhouette)                           # fixed original (components, holes)
    cur = silhouette.copy()
    _NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for _ in range(max_nib):
        n4 = _nb4count(cur)
        thin = (cur > 0) & (n4 == 1)                       # a bump: exactly one shared-edge neighbour
        if not thin.any():
            break
        changed = False
        for y, x in sorted(map(tuple, np.argwhere(thin))):
            if cur[y, x] == 0:                             # already peeled this pass
                continue
            q = next(((y + dy, x + dx) for dy, dx in _NB4              # its single 4-neighbour
                      if 0 <= y + dy < target and 0 <= x + dx < target and cur[y + dy, x + dx]), None)
            if q is None or n4[q] < 3:                     # neighbour is a thin stroke, not a solid body ->
                continue                                   # keep it (protects genuine thin-feature tips)
            trial = cur.copy()
            trial[y, x] = 0
            nt = _topology(trial)
            if nt[0] <= base[0] and nt[1] >= base[1]:      # no fragmentation, no lost counter
                cur, changed = trial, True
        if not changed:
            break
    return cur


STROKE_MIN_LEN = 2       # only normalize elongated strokes (a <2px-long speck is not a "line")


def _blot2x2(mask: np.ndarray) -> int:
    """Count of 2x2 all-foreground blocks (_block_corners) — the "blot" metric: a 2px
    smudge in 1px line art is exactly a 2x2 all-ink window, so 0 == clean line art.
    Pure numpy slice-AND: a cv2.erode with a 2x2 SE needs BORDER_CONSTANT 0 forced
    (the default border value reads ~foreground and falsely blots a lone corner
    pixel) and its even-kernel anchor is off-centre; the shifted views have no such
    traps."""
    return int(np.count_nonzero(_block_corners(mask > 0)))


def normalize_stroke_width(silhouette: np.ndarray, target: int, *,
                           reference: np.ndarray | None = None,
                           trace: list | None = None,
                           protected_bg: np.ndarray | None = None) -> np.ndarray:
    """Normalize <=2px strokes to an ODD width: per stroke, THIN-to-1px, HOLLOW-
    WIDEN-to-3px and KEEP compete, and the candidate leaving the least residual blot
    on the GATE metric (2x2 all-ink blocks in the final face-connected outline) wins
    — ties rank thin < keep < widen. A stroke is
    <=2px where a 3x3 opening removes it (a >=3px core survives); removals only ever
    touch that residue. The widen stamp is a 3x3 SQUARE sweep (cv2.dilate) so every
    centreline pixel is interior BY CONSTRUCTION and the final perimeter ring reads
    as two 1px rails + a 1px white middle at ANY angle — the round cv2.circle(r=1)
    plus-stamp left no interior along diagonal centrelines, so its "widen" still
    smudged (the old diagonal-blot failure). Every candidate is guarded against the
    input's (components, holes); a failing trial gets ONE rescue — _break_spurious_
    loops fills tiny spurious pockets (medial-axis discretization artifacts) and must
    then match exactly. Radial-blade sunbursts are NOT exempt (the old whole-mark
    _is_radial early-out also swallowed slack/react/spotify): the blade family is
    thinned UNIFORMLY — one union-thin move, all blades or none — so the sunburst
    stays even; widen is never offered there (it merges adjacent blades).

    Detail-retention rework additions: (1) PER-COMP SKELETON REPAIR — global spur
    pruning can orphan a short thin component's centreline (the pruned skeleton covers
    a fraction of the comp's own medial axis: vscode@16's chevron arms at 2/5), making
    thin-skel ship a scatter; when the comp-skel is DEGENERATE (2x its length < the
    comp's OWN medial-axis length) the centreline is re-derived per component, plus
    the body-attachment px so the thin stays connected. Gated by a zero-cost exact
    pre-check (own <= comp area, so 2*len >= area means the repair provably can't
    fire) — medial_axis costs ~50ms/call REGARDLESS of size, so the call is skipped
    unless repair is possible. (2) WIDEN ANTI-SPILL — the widen candidate is dropped
    when its stamp adds ink beyond the 1px-dilated reference (a hollow-widen bulking
    past ref+1px is fat, not normalization: vscode@16's arm wing, linkedin@16's
    row-15 bar). (3) PROTECTED-BG GUARD — the widen candidate is also dropped when
    its 3x3 stamp inks a `protected_bg` pixel (glyph channel / relayout inter-stroke
    valley; the widen CANDIDATE only — _break_spurious_loops' rescue path is
    untouched). NO retention test lives here — probed a no-op on the named cells
    and regressive elsewhere; retention is symmetrize's job.

    Deterministic (H8: rng=0, row-major components, fixed candidate order).
    Byte-identical when no <=2px stroke is present. trace: append-only harness
    records; None in production. protected_bg: optional caller-scoped negative-space
    zone the widen must never ink (see guard (3) above)."""
    if not silhouette.any():
        if trace is not None:
            trace.append({"radial": False, "thin_px": 0, "n_comps": 0})
        return silhouette
    opened = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, _SE3,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    thin_parts = ((silhouette > 0) & (opened == 0)).astype(np.uint8)   # the <=2px strokes/bands
    thin_px = int(np.count_nonzero(thin_parts))
    if not thin_parts.any():
        if trace is not None:
            trace.append({"radial": False, "thin_px": 0, "n_comps": 0})
        return silhouette
    base = _topology(silhouette)
    skel_raw = medial_axis(silhouette > 0, rng=0)
    skel = _break_spurious_loops(                          # drop spurious medial-axis loops up front
        _prune_spurs(skel_raw.astype(np.uint8) * 255), base[1]) > 0
    n, lbl, st, _c = cv2.connectedComponentsWithStats(thin_parts, connectivity=8)
    radial = _is_radial(silhouette)
    if trace is not None:
        trace.append({"radial": radial, "thin_px": thin_px, "n_comps": n - 1})
    def _accept(trial: np.ndarray) -> np.ndarray | None:
        """Topology guard with ONE loop-break rescue; a rescue that fills the wrong
        pocket fails the exact recheck and the candidate drops out."""
        if _topology(trial) == base:
            return trial
        rescued = _break_spurious_loops(trial, base[1])
        return rescued if _topology(rescued) == base else None

    def _score(mask: np.ndarray) -> int:
        """Candidate score = the GATE metric itself: 2x2-ink blot of the final
        face-connected outline. Scoring the raw ring instead misses the blocks
        connect_disjoint's corner fills complete on staircase rails (android's
        widened antennae looked clean pre-connect, blotted post-connect)."""
        return _blot2x2(connect_disjoint(silhouette_outline(mask), mask, target))

    cur = silhouette.copy()
    if radial:                                             # blade family: uniform thin or nothing
        fam = np.zeros(silhouette.shape, bool)
        for i in range(1, n):
            comp = lbl == i
            if int((skel & comp).sum()) >= STROKE_MIN_LEN:
                fam |= comp
        verdict = "radial-leave"
        if fam.any():
            id_score = _score(cur)                         # thin must not raise the gate metric
            for name, drop in (("radial-thin-skel", fam & ~skel),
                               ("radial-thin-other", fam & skel)):
                trial = cur.copy()
                trial[drop] = 0
                ok = _accept(trial)
                if ok is not None and _score(ok) <= id_score:
                    cur, verdict = ok, name
                    break
        if trace is not None:
            trace.append({"comp": 0, "area": int(np.count_nonzero(fam)),
                          "len": int((skel & fam).sum()), "frag": -1.0,
                          "verdict": verdict, "base": base})
        return cur

    refd = None if reference is None else (
        cv2.dilate(reference, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
    for i in range(1, n):
        comp = (lbl == i)
        cskel = skel & comp
        length = int(cskel.sum())
        area_i = int(st[i, cv2.CC_STAT_AREA])
        # PER-COMP SKELETON REPAIR (see docstring). Pre-gate is exact: the comp's own
        # medial axis is a subset of the comp, so 2*length >= area means the degeneracy
        # condition (2*len < own-len) provably cannot fire — skip the ~50ms call.
        if 2 * length < area_i:
            own = medial_axis(comp, rng=0)
            if 2 * length < int(own.sum()):                # comp-skel DEGENERATE only
                other = ((cur > 0) & ~comp).astype(np.uint8) * 255
                join = comp & (cv2.dilate(other, _SE3, borderType=cv2.BORDER_CONSTANT,
                                          borderValue=0) > 0)
                cskel = own | join                         # medial + body-attachment px:
                length = int(cskel.sum())                  # the repaired thin stays CONNECTED
        rec = {"comp": i, "area": area_i, "len": length,
               "frag": -1.0, "verdict": "leave", "base": base}
        if length < STROKE_MIN_LEN:                        # not an elongated line -> leave it
            rec["verdict"] = "skip-minlen"
            if trace is not None:
                trace.append(rec)
            continue
        skmask = cskel.astype(np.uint8) * 255
        rec["frag"] = round(                               # kept as a trace field only
            (cv2.connectedComponents(skmask, connectivity=4)[0] - 1) / length, 3)
        thin_skel = cur.copy(); thin_skel[comp & ~cskel] = 0           # keep the medial-axis offset
        thin_other = cur.copy(); thin_other[comp & cskel] = 0          # or the other offset
        widen = cur | cv2.dilate(skmask, _SE3,                         # 3x3 SQUARE sweep -> hollow
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
        # IDENTITY competes too: an action must BEAT the smudge it replaces on the
        # gate metric. Tie ranks: thin(0) < keep(1) < widen(2) — thin ties normalize
        # to odd width; a widen that doesn't reduce blot just bulks (android's antennae).
        cands = [(_score(cur), 1, 0, -1, "keep", cur)]
        for name, trial in (("thin-skel", thin_skel), ("thin-other", thin_other),
                            ("widen", widen)):
            if name == "widen" and refd is not None and bool(
                    np.any((trial > 0) & (cur == 0) & ~refd)):
                continue                                   # ANTI-SPILL: bulking past ref+1px
            if name == "widen" and protected_bg is not None and bool(
                    np.any((trial > 0) & (cur == 0) & (protected_bg > 0))):
                continue                                   # glyph-channel/valley guard (W3):
            #   hollow-widen must never ink a glyph's negative space — the widen
            #   CANDIDATE only; _break_spurious_loops' rescue path stays untouched
            #   (react@24's 1px rescue-fill has zero widen verdicts, scan-proven)
            ok = _accept(trial)
            if ok is None:
                continue
            cands.append((_score(ok),
                          0 if name.startswith("thin") else 2,
                          int(np.count_nonzero(ok != cur)),
                          len(cands), name, ok))
        *_key, name, cur = min(cands)                      # min-key: blot, tie-rank, fewest
        #                                                    changed px, append order — the unique
        #                                                    index means the ndarray never compares
        rec["verdict"] = name if len(cands) > 1 else "leave"   # keep=identity won; leave=no action valid
        if trace is not None:
            trace.append(rec)
    return cur


SYM_AXIS_IOU = 0.70      # accept a label mirror axis when min(outer-flip-IoU, hole-flip-IoU) >= this
#                          — hole-aware: a small feature (youtube's play triangle) can be body-
#                          symmetric yet hole-ASYMMETRIC, which correctly rejects the false axis.
SYM_ASYM_MAX = 0.032     # AND reject the axis if a single one-sided region larger than this fraction
#                          of the canonical fg has no mirror partner — a BOUNDARY feature the IoU
#                          gate misses (apple's bite, edge's swirl; folding would destroy it).
SYM_ROT_MARGIN = 0.10    # AND reject the axis if rotation-IoU exceeds V/H-flip-IoU by more than this
#                          — a CHIRAL mark (slack's pinwheel gap +0.218) or one whose true mirror axis
#                          is off the V/H grid (react's atom, off-axis at ~60°, gap +0.196) reads far
#                          more rotational than V/H-mirror; folding either about V/H merges its arms.
#                          True V/H folders sit at gap <= -0.011 (chrome); the empty band (-0.011, 0.196)
#                          is wide, so 0.10 has ~0.1 cushion on both sides. Measured over the n=30 set.
#                          Conservative (user ruling: err toward NOT folding); genuine symmetric
#                          marks' residual asymmetry is smaller/scattered (instagram 0.028, yt-music
#                          0.023) than a real asymmetric feature (apple 0.05, edge 0.04-0.10).


def _label_axes(constructions: list[tuple[list[Part], tuple, str]],
                canvas_hw: tuple[int, int]) -> list[bool]:
    """The mirror axes (True=vertical/reflect-x, False=horizontal/reflect-y) the LABEL is symmetric
    about — hole-aware so a body-symmetric-but-hole-asymmetric axis (youtube: V) is rejected while a
    genuine one (youtube/yt-music: H; instagram: both) is kept. Canonical geometry, not the render."""
    h, w = canvas_hw
    canon = np.zeros((h, w), np.uint8)
    for parts, _c, _l in constructions:
        canon |= _compound_mask(parts, h, w)
    holes = ((canon == 0) & ~(_outer_bg(canon) > 0)).astype(np.uint8) * 255
    rot_iou = _self_rot_iou(canon, (90.0, 180.0, 270.0))   # canonical's best rotation-IoU — the
    #                          CHIRALITY reference the per-axis guard below compares each flip-IoU to
    axes: list[bool] = []
    for vert in (True, False):
        fo = _best_flip_axis(canon, vert)
        if fo is None:
            continue
        hi = _iou_bool(holes, _flip_mask(holes, fo[0], vert)) if holes.any() else 1.0
        if min(fo[1], hi) < SYM_AXIS_IOU:
            continue
        if rot_iou - fo[1] > SYM_ROT_MARGIN:      # far more rotational than V/H-mirror -> chiral (slack)
            continue                              # or off-grid mirror axis (react) -> folding corrupts it
        asym = ((canon > 0) & ~(_flip_mask(canon, fo[0], vert) > 0)).astype(np.uint8)
        _n, _l, st, _c = cv2.connectedComponentsWithStats(asym, connectivity=8)
        largest = max((int(st[i, cv2.CC_STAT_AREA]) for i in range(1, _n)), default=0)
        if largest > SYM_ASYM_MAX * max(1, int((canon > 0).sum())):   # big one-sided feature (apple's
            continue                                                  # bite / edge's swirl) -> not symmetric
        axes.append(vert)
    return axes


def _ref_hole_zone(mask: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Current enclosed-bg px belonging to holes that overlap a 1px-DILATED reference
    hole (ANY size — dilated because unblot legally moves a counter 1px off its ref
    position): such a hole is a REF-PRESCRIBED counter (discord@16's 1x2 eyes, below
    the min-feat significance floor), never 'spurious'."""
    if reference is None:
        return np.zeros(mask.shape, bool)
    rz0 = np.zeros(mask.shape, np.uint8)
    for cells, *_r in _enclosed_bg(reference):
        for c in cells:
            rz0[c] = 255
    rz = cv2.dilate(rz0, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if not rz.any():
        return np.zeros(mask.shape, bool)
    out = np.zeros(mask.shape, bool)
    for cells, *_r in _enclosed_bg(mask):
        if any(rz[c] for c in cells):
            for c in cells:
                out[c] = True
    return out


def _exact_axes(sil: np.ndarray) -> list[tuple[bool, int]]:
    """Axes the mask is EXACTLY symmetric about: [(vertical, a2), ...] with flip-IoU == 1.
    (a2 is the DOUBLED axis coordinate — even a2 has a genuine fixed row/col, odd a2
    pairs every pixel with a distinct partner; _orbit handles both.)"""
    axes = []
    for vert in (True, False):
        fa = _best_flip_axis(sil, vert)
        if fa is not None and fa[1] == 1.0:
            axes.append((vert, fa[0]))
    return axes


# _orbit: LIFTED to compress/common.py (H5 — cubify's R6 dual-axis moves share it);
# imported above, same name and semantics.


def _orbit_carve(cur: np.ndarray, axes: list[tuple[bool, int]], accept) -> np.ndarray:
    """THE shared orbit-paired blot-px carve loop (H5): carve 2x2-block ink px with a
    bg 4-neighbour in whole mirror orbits (both-or-neither; one row-major representative
    per orbit), fixpoint to CHANNEL_MAX_PASSES. `accept(trial) -> bool` supplies the
    structural guard — _deblot_paired passes exact-topology equality, reopen_channels_
    paired the c4-component cap (a plain callable parameter, H4-clean). Deterministic."""
    for _ in range(CHANNEL_MAX_PASSES):
        blot = _block_spread(cur > 0)
        if not blot.any():
            break
        cand = blot & (_nb4count(cur) < 4)
        changed = False
        for y, x in sorted(map(tuple, np.argwhere(cand))):
            orb = _orbit(y, x, axes, cur.shape)
            if orb is None or (y, x) != orb[0]:
                continue                                   # one representative per orbit
            blocks_now = _block_spread(cur > 0)
            nb_now = _nb4count(cur)
            if not all(cur[p] and blocks_now[p] and nb_now[p] < 4 for p in orb):
                continue                                   # both-or-neither
            trial = cur.copy()
            for p in orb:
                trial[p] = 0
            if accept(trial):
                cur, changed = trial, True
        if not changed:
            break
    return cur


def _deblot_paired(sil: np.ndarray, axes: list[tuple[bool, int]],
                   target: int) -> np.ndarray:
    """Orbit-paired SILHOUETTE de-blot: the shared _orbit_carve under an exact-topology
    guard. An exactly-symmetric input stays EXACTLY symmetric — the fold-then-repair
    step (instagram@16's re-fattened lens ring thins to a clean 1px ring where a
    normalize re-run scatters it). Deterministic (H8)."""
    if not sil.any():
        return sil
    base = _topology(sil)
    return _orbit_carve(sil.copy(), axes, lambda trial: _topology(trial) == base)


def _fold_or_repair(folded: np.ndarray, before: np.ndarray, target: int,
                    reference: np.ndarray | None) -> np.ndarray:
    """FOLD-THEN-REPAIR — replaces the old _no_reblot VETO (which shipped instagram's
    massive asymmetry to protect a dozen blot px; blot must not outrank identity when
    the fold can be de-blotted). (1) A fold that destroys a FEATURE-SIZED ref-supported
    region (largest retention-violation component > min_feat^2 — yt-music@16's 9px
    flag) is rejected outright: several sub-budget minority-accent drops are dual-axis
    symmetrization's normal job, ONE feature-sized loss is destruction. (2) The fold is
    accepted when its pre-tail ring blot <= the input's. (3) Else the orbit-paired
    silhouette carve repairs the fold's re-fattened bands (exactly-symmetric stays
    exactly symmetric) and is accepted under the same retention cap + blot bar.
    (4) Else the input ships (the old veto). Deterministic (H8)."""
    acc_cap = _min_feat_at(target) ** 2
    if _retention_comp_max(before, folded, reference) > acc_cap:
        return before                                      # feature-scale destruction

    def _blot_ring(m: np.ndarray) -> int:
        return _blot2x2(connect_disjoint(silhouette_outline(m), m, target))

    bar = _blot_ring(before)
    if _blot_ring(folded) <= bar:
        return folded
    repaired = _deblot_paired(folded, _exact_axes(folded), target)
    if (_retention_comp_max(before, repaired, reference) <= acc_cap
            and _blot_ring(repaired) <= bar):
        return repaired
    return before


def symmetrize_render(silhouette: np.ndarray,
                      constructions: list[tuple[list[Part], tuple, str]],
                      canvas_hw: tuple[int, int], target: int,
                      reference: np.ndarray | None = None) -> np.ndarray:
    """Force the rendered silhouette to the LABEL's mirror symmetry at the pixel level. Detects the
    label's true mirror axes (V and/or H) hole-awarely (_label_axes). A DUAL-axis mark (both V and
    H, e.g. instagram) is symmetrized by a 4-orbit MAJORITY VOTE (a pixel is kept iff >= 2 of its
    {self, V-mirror, H-mirror, VH-mirror} are foreground) — exactly V+H symmetric, minority accents
    drop, and unlike a half-mirror it neither misaligns the two rings nor shatters a thinned 1px
    band at the mirror seam. A SINGLE-axis mark (e.g. yt-music's play triangle) DROPS its small
    asymmetric accents (components with no mirror partner, <= min-feature^2) then half-mirrors the
    authoritative (cleaner) half onto the other (_half_mirror — corrects a lopsided render), kept
    only if it does not fragment the silhouette; spurious small holes it opens are filled (kept
    holes = the label's real counters via _significant_holes; ref-hole-backed counters are NEVER
    spurious). No-op when the label has no mirror axis.

    Detail-retention rework: the accent-drop is RETENTION-GUARDED (a ref-supported feature —
    android's antenna stubs, signal's bubble tail — is never dropped; a ref-edge jag whose
    adjacent ink survives still is); the single-axis fold acceptance allows RESTORED
    ref-supported mirror components (android@24's mirrored antenna) without reading them as
    fragmentation; and every accepted fold passes _fold_or_repair — fold-then-repair replacing
    the old _no_reblot veto: a fold whose ring blot rises is REPAIRED by the orbit-paired
    silhouette carve (exactly-symmetric stays exactly symmetric) and only vetoed when the
    repair still can't reach the input's blot or the fold/repair destroys a feature-sized
    ref-supported region (> min_feat^2 in one component). Deterministic (H8)."""
    if not silhouette.any() or not constructions:
        return silhouette
    axes = _label_axes(constructions, canvas_hw)
    if not axes:
        return silhouette
    cur = silhouette.copy()
    if len(axes) >= 2:                     # DUAL axis (e.g. instagram): 4-orbit MAJORITY VOTE — a
        av = _best_flip_axis(cur, True)    # pixel is kept if >= 2 of {self, V-mirror, H-mirror,
        ah = _best_flip_axis(cur, False)   # VH-mirror} are fg. Exactly V+H symmetric, minority
        if av is None or ah is None:       # accents drop, and unlike half-mirror it neither
            return silhouette              # misaligns the two rings nor shatters a 1px band at the
        fg = (cur > 0).astype(np.int16)    # seam.
        fv = (_flip_mask(cur, av[0], True) > 0).astype(np.int16)
        fh = (_flip_mask(cur, ah[0], False) > 0).astype(np.int16)
        fvh = (_flip_mask(_flip_mask(cur, av[0], True), ah[0], False) > 0).astype(np.int16)
        voted = ((fg + fv + fh + fvh) >= 2).astype(np.uint8) * 255
        if _topology(voted)[1] < _topology(silhouette)[1]:               # keep counters
            return silhouette
        # RESOLVED RESIDUAL (R6 → nested-annuli round): the dual-axis vote used to drop
        # instagram@16's 1px camera dot, and a ref-supported-accent restore was built and
        # probed OUT — the dot's reference pixel is 8-connected INTO the lens ring's
        # composite-ref component at 16 (area 27), so no composite-reference gate can
        # separate an intentional accent from wall wobble (restoring by violation-size
        # alone un-folds the symmetrization, asym 0->56 at 24). The fix landed UPSTREAM:
        # refine_perimeter's `protected=` carve-out removes the deliberately-placed accent
        # before the chain and restores it on the final ring, so this vote never sees it.
        # The comment stays as the record of why a post-hoc restore HERE cannot work.
        return _fold_or_repair(voted, silhouette, target, reference)     # fold-then-repair
    min_feat = _min_feat_at(target)
    acc_max = min_feat * min_feat
    for vert in axes:
        fa = _best_flip_axis(cur, vert)
        if fa is None:
            continue
        a2, _iou = fa
        asym = (cur > 0) & ~(_flip_mask(cur, a2, vert) > 0)          # material with no mirror partner
        n, lbl, stats, _c = cv2.connectedComponentsWithStats(asym.astype(np.uint8), connectivity=8)
        trial = cur.copy()
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) <= acc_max:            # small asymmetric accent -> drop,
                t2 = trial.copy()                                     # UNLESS it is a ref-supported
                t2[lbl == i] = 0                                      # FEATURE (android's antenna stubs,
                if _retention_loss(trial, t2, reference) > 0:         # signal's bubble tail): a ref-edge
                    continue                                          # jag keeps adjacent ink and stays
                trial = t2                                            # droppable; a whole feature never
        sym = _half_mirror(trial, a2, vert)
        st_sym, st_cur = _topology(sym), _topology(cur)
        new_ok = 0
        if st_sym[0] > st_cur[0] and reference is not None:
            n2, lbl2 = cv2.connectedComponents(sym, connectivity=8)
            for j in range(1, n2):                                    # a RESTORED ref-supported mirror
                comp2 = lbl2 == j                                     # feature (android@24's mirrored
                if (not bool(np.any(comp2 & (cur > 0)))               # antenna) is not fragmentation —
                        and bool(np.any(comp2 & (reference > 0)))     # without this allowance the
                        and int(comp2.sum()) <= 2 * acc_max):         # never-fragment guard rejects the
                    new_ok += 1                                       # restoring fold
        if st_sym[0] - new_ok <= st_cur[0] and st_sym[1] >= st_cur[1]:   # no fragment AND no lost counter
            cur = sym
    outer = _outer_bg(cur)                                            # fill spurious (non-counter) holes
    significant = np.zeros(cur.shape, dtype=bool)
    for hole in _significant_holes(cur, min_feat):
        significant |= (hole > 0)
    spurious = (cur == 0) & (outer == 0) & ~significant
    spurious &= ~_ref_hole_zone(cur, reference)            # a REF-PRESCRIBED counter is never spurious
    if spurious.any():
        cur = cur.copy(); cur[spurious] = 255
    return _fold_or_repair(cur, silhouette, target, reference)           # fold-then-repair


def connect_disjoint_paired(outline: np.ndarray, silhouette: np.ndarray, target: int,
                            axes: list[tuple[bool, int]],
                            max_passes: int = PERIM_MAX_PASSES) -> np.ndarray:
    """Mirror-paired connect_disjoint. connect_disjoint is provably NOT H-mirror-
    equivariant: its both-exterior fallback always fills TOP corners (pat_a→TR /
    pat_b→TL), so a mirror-symmetric silhouette can emit an ASYMMETRIC ring
    (instagram@16's top notch took two exterior fills, the mirrored bottom notch one
    interior fill). Here each pass enumerates diagonal-only touches row-major from the
    pass-start snapshot; only each touch-ORBIT's row-major-first touch chooses a corner
    (the shared _diag_corners doctrine) and the fill lands together with its mirror
    images — the corner CHOICE itself becomes axis-consistent. Deterministic (H8)."""
    out = outline > 0
    sil = silhouette > 0
    h, w = out.shape
    for _ in range(max_passes):
        touches = _diag_touch_list(out)
        if not touches:
            break
        any_fill = False
        for r, c, k in touches:
            orbit = {(r, c, k)}                            # orbit of the TOUCH: mirror its two
            for vert, a2 in axes:                          # own-ink px, re-anchor the window
                for rr, cc, kk in list(orbit):
                    own = (((rr, cc), (rr + 1, cc + 1)) if kk == "a"
                           else ((rr, cc + 1), (rr + 1, cc)))
                    m1 = _mirror_point(*own[0], vert, a2)
                    m2 = _mirror_point(*own[1], vert, a2)
                    (ya, xa), (yb, xb) = sorted((m1, m2))
                    nr, nc = ya, min(xa, xb)
                    orbit.add((nr, nc, "a" if xa == nc else "b"))
            orbit = sorted(orbit)
            if (r, c, k) != orbit[0]:
                continue                                   # one representative per orbit
            d_int, _d_ext = _diag_corners(r, c, k, sil)    # the shared corner doctrine
            fills = {d_int}
            for vert, a2 in axes:
                for p in list(fills):
                    fills.add(_mirror_point(*p, vert, a2))
            if any(not (0 <= y < h and 0 <= x < w) for y, x in fills):
                continue
            for p in fills:
                if not out[p]:
                    out[p] = True
                    any_fill = True
        if not any_fill:
            break
    return (out * 255).astype(np.uint8)


def reopen_channels_paired(ring: np.ndarray, target: int,
                           axes: list[tuple[bool, int]]) -> np.ndarray:
    """Mirror-paired reopen_channels: the shared _orbit_carve under the c4-component
    cap (same bg-adjacency guard as the unpaired pass). axes=[] degenerates to the
    plain pass. H8."""
    if not axes:
        return reopen_channels(ring, target)
    if not ring.any():
        return ring
    base_c4 = cv2.connectedComponents(ring, connectivity=4)[0] - 1
    return _orbit_carve(
        ring.copy(), axes,
        lambda trial: cv2.connectedComponents(trial, connectivity=4)[0] - 1 <= base_c4)


def relieve_tight_pockets_paired(ring: np.ndarray, target: int,
                                 axes: list[tuple[bool, int]]) -> np.ndarray:
    """Mirror-paired tight-pocket relief: mirror-PARTNER pockets move together (the
    partner takes the axis-mirrored direction on the combined trial — both or neither),
    self-mirrored pockets offer only axis-consistent directions. The per-direction
    geometry/separation chain is the SHARED _pocket_move (H5 — one source of truth with
    the unpaired pass); the structural guards run on the COMBINED trial. axes=[]
    degenerates to the plain pass. Deterministic (H8)."""
    if not axes:
        return relieve_tight_pockets(ring, target)
    cur = ring > 0
    h, w = cur.shape
    ys, xs = np.nonzero(cur)
    if xs.size == 0:
        return ring
    base_c4 = cv2.connectedComponents(cur.astype(np.uint8), connectivity=4)[0] - 1
    base_blot = int(np.count_nonzero(_block_corners(cur)))
    base_diag = _diag_touches(cur)
    scy, scx = float(ys.mean()), float(xs.mean())

    def _twin_cells(cells: frozenset, vert: bool, a2: int) -> frozenset:
        return frozenset(_mirror_point(y, x, vert, a2) for y, x in cells)

    def _mirror_dir(dname: str, vert: bool) -> str:
        m = {True: {"left": "right", "right": "left"},
             False: {"up": "down", "down": "up"}}[vert]
        return m.get(dname, dname)

    for _ in range(RELIEF_MAX_PASSES):
        changed = False
        pockets = _enclosed_bg((cur * 255).astype(np.uint8))
        n_encl = len(pockets)
        pockmap = {cells for cells, *_r in pockets}
        done: set[frozenset] = set()
        for cells, area, ph, pw, py0, px0 in pockets:
            if (cells in done or not _is_tight(area, ph, pw)
                    or _clean_loop_counter(cur, cells)):
                continue
            partner, pvert, pa2 = None, None, None
            selfdirs = {d for d, _v in _RELIEF_DIRS}
            for vert, a2 in axes:
                tw = _twin_cells(cells, vert, a2)
                if tw == cells:                            # self-mirrored: only axis-consistent dirs
                    selfdirs -= ({"left", "right"} if vert else {"up", "down"})
                elif tw in pockmap:
                    partner, pvert, pa2 = tw, vert, a2
                    break
            outer = _outer_bg((cur * 255).astype(np.uint8))
            cands = []
            for di, (dname, (dy, dx)) in enumerate(_RELIEF_DIRS):
                if partner is None and dname not in selfdirs:
                    continue
                offer = _pocket_move(cur, cells, dname, outer)
                if offer is None:
                    continue
                trial, grown, wall = offer
                grown2 = None
                if partner is not None:                    # mirrored application on the partner
                    offer2 = _pocket_move(trial, partner, _mirror_dir(dname, pvert), outer)
                    if offer2 is None:
                        continue                           # both or neither
                    trial, grown2, _w2 = offer2
                if _relief_accept(trial, base_c4, base_blot, base_diag,
                                  n_encl, grown, grown2) is None:
                    continue
                cands.append(_relief_cand(cur, outer, wall, di, dy, dx,
                                          py0, px0, ph, pw, scy, scx, trial))
            if cands:
                cands.sort(key=lambda cnd: cnd[:4], reverse=True)
                cur = cands[0][4]
                changed = True
                if partner is not None:
                    done.add(partner)
            done.add(cells)
        if not changed:
            break
    return (cur * 255).astype(np.uint8)


def _ring_asym(ring: np.ndarray, axes: list[tuple[bool, int]]) -> int:
    """Total asymmetry px of a ring about the given exact axes (the paired-tail
    arbitration term)."""
    return sum(int(np.count_nonzero(
        (ring > 0) != (_flip_mask(ring, a2, vert) > 0))) for vert, a2 in axes)


def perimeter_ring(sil: np.ndarray, target: int) -> np.ndarray:
    """The shared stage-4 outline tail (ONE tail, two callers: refine_perimeter and the
    refine_debug harness — H5). plain and pocket-averse variants (+ a mirror-PAIRED
    variant when the input silhouette is EXACTLY symmetric about a V/H axis) each run
    the full connect → reopen → relieve chain and compete on
    (blot, tight, ring-asym-about-the-input-axes, fixed order plain < averse < paired).
    blot stays the ABSOLUTE gate — chrome@16's paired carve re-blots and loses; gmail@24's
    paired variant wins on asym at equal blot/tight (the unpaired tail carved its bottom-
    left only, shipping an asymmetric ring from a symmetric silhouette). The averse
    variant must never ship un-arbitrated (mathematica@16 0→3 tight un-arbitrated).
    axes==[] degenerates to the previous 2-candidate tight-only arbitration.
    Deterministic (H8)."""
    axes = _exact_axes(sil)
    ring = silhouette_outline(sil)
    cands = []
    for averse in (False, True):
        conn = connect_disjoint(ring, sil, target, pocket_averse=averse)
        cands.append(relieve_tight_pockets(reopen_channels(conn, target), target))
    if axes:
        connp = connect_disjoint_paired(ring, sil, target, axes)
        cands.append(relieve_tight_pockets_paired(
            reopen_channels_paired(connp, target, axes), target, axes))
    best = min(enumerate(cands),
               key=lambda iv: (_blot2x2(iv[1]), count_tight_pockets(iv[1]),
                               _ring_asym(iv[1], axes), iv[0]))
    return best[1]


def build_reference(constructions: list, canvas_hw: tuple[int, int],
                    target: int) -> tuple[np.ndarray, np.ndarray | None]:
    """(reference, relayout_valley_zone): the per-construction ideal-geometry
    reference union — WITH the W4 stroke-relayout substitution applied (the
    render idealized medium's M to its own re-laid-out reference; despeck's
    outside-ref peel and deburr would eat the widened strokes against the raw
    one). For each construction the relayout re-shaped, the zone of THIN
    reference-background valleys ((ref==0) that a 3x3 opening removes, scoped to
    the construction's reference bbox) is returned for the protected_bg guards —
    hollow-widen must never ink a 1px inter-stroke valley. ONE implementation
    for production refine_perimeter AND the refine_debug mirror (H5).
    Deterministic (H8)."""
    reference = np.zeros((target, target), np.uint8)
    zone = None
    for parts, _c, _l in constructions:
        pp = _glyph_stroke_relayout(parts, canvas_hw, target)
        rm = reference_mask(pp, canvas_hw, target)
        reference |= rm
        if pp is not parts:
            bg = ((rm == 0).astype(np.uint8)) * 255
            opened = cv2.morphologyEx(bg, cv2.MORPH_OPEN, _SE3,
                                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
            val = (bg > 0) & (opened == 0)
            ys, xs = np.nonzero(rm)
            if xs.size:
                box = np.zeros((target, target), bool)
                box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                val &= box
            zone = val if zone is None else (zone | val)
    return reference, zone


def glyph_ink_lines(constructions: list, canvas_hw: tuple[int, int],
                    target: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(Z, K) for the glyph-channel class (mastodon's white m), or None. Z = the
    channel ZONE at target (floor-map of the canonical enclosed glyph region);
    K = the ink-line skeleton at target (floor-map of the region's medial axis).
    Detection = compress.cubify.glyph_channel — the SAME detector the render
    curtain uses (one detector, two consumers; H5) — run on the RAW construction
    parts (the detection gates are robust to the render tier's _widen_gaps
    state: mastodon 0.232 area / 0.92 subrep, wide margins). USER RULING: the
    white m (negative space) ships as 1px interior INK lines inside the bubble
    perimeter ring — refine_perimeter fills the channel before the chain (the
    chain then works a solid body) and stamps K back inside the ring after.
    Deterministic (H8)."""
    S = canvas_hw[0]
    zone = None
    kskel = None
    for parts, _c, _l in constructions:
        gc = glyph_channel(parts, canvas_hw)
        if gc is None:
            continue
        region_u, skel_u = gc
        ys, xs = np.nonzero(region_u)
        ty, tx = _map_coords(ys, xs, S, target, floor_map=True)
        z = np.zeros((target, target), bool)
        z[ty, tx] = True
        ys, xs = np.nonzero(skel_u)
        ty, tx = _map_coords(ys, xs, S, target, floor_map=True)
        k = np.zeros((target, target), bool)
        k[ty, tx] = True
        zone = z if zone is None else (zone | z)
        kskel = k if kskel is None else (kskel | k)
    return None if zone is None else (zone, kskel)


def fill_glyph_channels(sil: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """FILL the glyph channel before the refine chain: every enclosed-bg
    component overlapping the zone fills to ink, plus leftover zone-white — the
    chain then processes a SOLID body and has nothing to churn (mastodon probe:
    +26px@16 / +66px@24 filled; nrm churn 34→8px@24). The m returns as ink
    lines via stamp_glyph_ink after perimeter_ring."""
    fill = np.zeros(sil.shape, bool)
    for cells, *_r in _enclosed_bg(sil):
        comp = np.zeros(sil.shape, bool)
        for c in cells:
            comp[c] = True
        if bool((comp & zone).any()):
            fill |= comp
    fill |= zone & (sil == 0)
    if not fill.any():
        return sil
    out = sil.copy()
    out[fill] = 255
    return out


def stamp_glyph_ink(ring: np.ndarray, kskel: np.ndarray, target: int) -> np.ndarray:
    """STAMP the glyph's 1px ink-line skeleton inside the outlined body: re-thin
    the floor-mapped canonical skeleton at target (medial_axis rng=0 — the
    library-scout verdict over skimage.thin: repo precedent + H8), prune
    staircase barbs (_prune_spurs), TRIM to 8-separation from the ring (the
    masks_8adjacent guarantee — ink lines never touch the wall), and if the trim
    splits the glyph, reconnect with mst_bridges 1px LINE_4 segments accepted
    only when the whole bridge stays off the ring's 1px dilation (deliberate
    junctions never weld to the wall). ONE implementation for production and
    the refine_debug mirror. Deterministic (H8)."""
    if not kskel.any():
        return ring
    kt = medial_axis(kskel, rng=0)
    kt = _prune_spurs(kt.astype(np.uint8) * 255) > 0
    ring_d = cv2.dilate(ring, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    k2 = kt & ~ring_d
    if not k2.any():
        return ring
    nk, klbl = cv2.connectedComponents(k2.astype(np.uint8), connectivity=8)
    if nk - 1 > 1:
        for pa, pb in mst_bridges([klbl == q for q in range(1, nk)]):
            seg = np.zeros(ring.shape, np.uint8)
            cv2.line(seg, (pa[1], pa[0]), (pb[1], pb[0]), 255, 1, cv2.LINE_4)   # (x,y)
            if not bool(np.any((seg > 0) & ring_d)):
                k2 |= seg > 0
    out = ring.copy()
    out[k2] = 255
    return out


def carve_accent(sil: np.ndarray, protected: np.ndarray | None
                 ) -> tuple[np.ndarray, np.ndarray | None, list[np.ndarray], list[bool]]:
    """Remove the DELIBERATELY-PLACED assembly accent zone from the silhouette
    before the refine chain (refine's ref-keyed oracles are blind to it — see
    refine_perimeter). Returns (sil_carved, accent, acc_ccs, acc_attached):
    accent = the accent ink verbatim (None when protected is None/disjoint);
    acc_ccs = its 8-connected components; acc_attached = whether each CC was
    8-adjacent to the REMAINING silhouette at carve time (signal's tail nub is
    ATTACHED to the bubble, unlike instagram's separated dot) — the restore's
    re-bridge eligibility. ONE implementation for production refine_perimeter
    AND the refine_debug harness mirror (the gate-predicate-drift taxonomy)."""
    if protected is None or not bool(np.any((sil > 0) & (protected > 0))):
        return sil, None, [], []
    accent = ((sil > 0) & (protected > 0)).astype(np.uint8) * 255
    carved = sil.copy()
    carved[protected > 0] = 0
    na, albl = cv2.connectedComponents((accent > 0).astype(np.uint8), connectivity=8)
    acc_ccs = [(albl == q).astype(np.uint8) * 255 for q in range(1, na)]
    acc_attached = [masks_8adjacent(c, carved) for c in acc_ccs]
    return carved, accent, acc_ccs, acc_attached


def restore_accent(ring: np.ndarray, accent: np.ndarray | None,
                   acc_ccs: list[np.ndarray], acc_attached: list[bool]) -> np.ndarray:
    """OR the carved accent back onto the final ring, then ATTACH-REPAIR: an
    accent CC that was 8-adjacent to the silhouette at carve time but is now
    8-disconnected from the ring (the chain may legally shift the parent rim
    ±1px) gets ONE nearest-gap 1px LINE_4 bridge to the nearest ring px.
    Instagram's dot was never attached → byte-identical no-op there."""
    if accent is None:
        return ring
    out = ring | accent
    for c, was in zip(acc_ccs, acc_attached):
        if not was or masks_8adjacent(c, ring):
            continue
        if not ring.any():
            continue
        pa, pb, _d = nearest_gap(np.argwhere(c > 0), np.argwhere(ring > 0))
        cv2.line(out, (pa[1], pa[0]), (pb[1], pb[0]), 255, 1, cv2.LINE_4)   # (x,y)
    return out


def refine_perimeter(constructions: list[tuple[list[Part], tuple, str]],
                     rendered_masks: list[np.ndarray],
                     canvas_hw: tuple[int, int], target: int,
                     structure_ref: np.ndarray | None = None,
                     protected: np.ndarray | None = None) -> np.ndarray:
    """Stage-4 v1 product: the FACE-CONNECTED 1-px boundary perimeter of the rendered
    silhouette. Unions the per-construction renders → un-blots small counters → continuity-widens
    toward the @24 sibling (`structure_ref`; self-gates to a no-op @24 and on the radial family —
    ACTIVE since the tight-wireframe rework: production and harness run the same chain) → deburrs
    single-pixel edge nibs that fall OUTSIDE the ideal reference (cubify union-seam raster
    artifacts, see deburr_nibs) → stroke-width normalize → symmetrize → perimeter_ring (the
    arbitrated outline tail: connect plain/pocket-averse → reopen → tight-pocket relief).
    Signature matches the (cons, masks, (S,S), target) seam (assemble_render's output);
    `canvas_hw` = (S,S) is the parts' coord frame, used to raster the per-construction
    reference. Returns (target,target) uint8 {0,255}. Deterministic (H8).

    `protected` (assemble_render's accent return): ink of a DELIBERATELY-PLACED assembly
    accent (instagram's separated camera dot) that refine's oracles are structurally blind
    to — its reference is empty at the placed position (the r≈0.8px circle never reaches
    the ≥0.5 downsample coverage at 16, and the nested-ring coordination legitimately moves
    it), so despeck would eat it as an isolated speck, symmetrize's vote/accent-drop/hole-
    fill has no retention backing for it, and the tail could deform its interior. CARVE-OUT
    + RESTORE: the accent zone is removed from the silhouette before the chain (which then
    processes the exactly-symmetric remainder — `_exact_axes` keeps seeing the clean two-
    ring mask, so the mirror-paired tail stays available) and OR-ed back verbatim onto the
    final ring. (A post-hoc ref-gated restore was built and probed OUT in symmetrize — the
    accent's ref px 8-connects into the ring's composite-ref component; see the KNOWN
    RESIDUAL note there. Carve-out sidesteps the composite-reference inseparability.)"""
    if not rendered_masks:
        return np.zeros((target, target), np.uint8)
    sil = _composite(rendered_masks, (target, target))
    if not sil.any():
        return sil
    sil, accent, acc_ccs, acc_attached = carve_accent(sil, protected)
    if accent is not None and not sil.any():
        return accent
    # W3 — mastodon's white m: FILL the glyph channel so the chain works a solid
    # body (nothing to churn), guard the zone against hollow-widen ink, and stamp
    # the m back as 1px interior ink lines after the outline (the user ruling).
    gz = glyph_ink_lines(constructions, canvas_hw, target)
    glyph_zone = gz[0] if gz is not None else None
    if gz is not None:
        sil = fill_glyph_channels(sil, glyph_zone)
    sil = unblot_features(sil, target)
    reference, valley_zone = build_reference(constructions, canvas_hw, target)
    if valley_zone is not None:                       # W4: relayout inter-stroke valleys join
        glyph_zone = (valley_zone if glyph_zone is None
                      else (glyph_zone | valley_zone))   # the protected_bg guard zone
    sil = despeck_artifacts(sil, target, reference)   # artifact removal — BEFORE widen (which
    #                                                   otherwise fattens a speck into a chimney)
    sil = widen_underrepresented(sil, target, structure_ref,
                                 protected_bg=glyph_zone)   # @16 continuity widen (self-gated)
    sil = deburr_nibs(sil, target, reference)   # drop cubify union-seam edge nibs (outside-ref)
    sil = normalize_stroke_width(sil, target, reference=reference,
                                 protected_bg=glyph_zone)   # 2px -> 1px / hollow 3px,
    #                                             blot-arbitrated; comp-skel repair + anti-spill
    sil = symmetrize_render(sil, constructions, canvas_hw, target, reference)   # label mirror fold
    #   (reflects features, so runs LAST); the chirality guard in _label_axes excludes the
    #   over-folding class, and _fold_or_repair de-blots re-fattening folds instead of vetoing them.
    ring = perimeter_ring(sil, target)
    if gz is not None:
        ring = stamp_glyph_ink(ring, gz[1], target)
    return restore_accent(ring, accent, acc_ccs, acc_attached)
