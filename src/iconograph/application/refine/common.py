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
    _nb4count,
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
from scipy.signal import find_peaks
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
    # pat_a → fill TR if interior (or both-exterior fallback), else BL (the interior one)
    fill[:-1, 1:] |= pat_a & (s_tr | ~s_bl)
    fill[1:, :-1] |= pat_a & ~s_tr & s_bl
    # pat_b → fill TL if interior (or both-exterior fallback), else BR
    fill[:-1, :-1] |= pat_b & (s_tl | ~s_br)
    fill[1:, 1:] |= pat_b & ~s_tl & s_br
    return fill


def connect_disjoint(outline: np.ndarray, silhouette: np.ndarray, target: int,
                     max_passes: int = PERIM_MAX_PASSES) -> np.ndarray:
    """Make the perimeter FACE-connected (4-connectivity): fill ONE pixel per diagonal-only
    corner touch, interior side preferred. Iterates to a fixpoint (a fill can create a new
    neighbouring touch) capped at max_passes. Additive only — never removes a pixel, so it
    can never open a seam; a no-diagonal outline is a byte-identical no-op (zero regression).
    outline / silhouette: (target,target) uint8 {0,255}. Returns (target,target) uint8.
    Deterministic (H8)."""
    out = outline > 0
    sil = silhouette > 0
    for _ in range(max_passes):
        fill = _diag_fill(out, sil)
        if not fill.any():
            break
        out = out | fill
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


def _outer_bg(mask: np.ndarray) -> np.ndarray:
    """Bool mask of the OUTER (border-connected) background — everything outside the silhouette,
    excluding interior counters. Border flood-fill (the complement of _significant_holes' bg)."""
    m = (mask > 0).astype(np.uint8)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    flooded = pad.copy()
    cv2.floodFill(flooded, ff, (0, 0), 2)               # border-connected bg → 2
    return flooded[1:-1, 1:-1] == 2


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
                           structure_ref: np.ndarray | None = None) -> np.ndarray:
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
    gmail's by any geometric signal. The harness/caller scopes WHICH fixtures it runs on."""
    if structure_ref is None or not sil.any():               # @24 is authoritative — never widen it
        return sil
    if _is_radial(structure_ref):                            # radial-blade family (claude/firefox)
        return sil                                           # excluded per the project convention
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
        allow = tgt & (pinch | _flanked(cur))             # @24 wants it, at a pinch or inner gap
        nb = cv2.dilate(cur, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        cand = (cur == 0) & allow & (nb > 0)
        if not cand.any():
            break
        changed = False
        for y, x in sorted(map(tuple, np.argwhere(cand))):
            if cur[y, x]:
                continue
            trial = cur.copy()
            trial[y, x] = 255
            tx = int(round(2.0 * axis - x))               # mirror twin (keep symmetry)
            if 0 <= tx < target:
                trial[y, tx] = 255
            if _topology(trial) == base:
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
    chevron apex, a 1px antenna). Radial-blade marks (claude/firefox) are exempt via _is_radial on
    the reference. Topology-guarded (no component split, no lost counter), bounded to `max_nib`
    passes.

    ACCEPTED TRADEOFF (user ruling — "aggressive de-jag, all bumps"): the reference and symmetry
    gates are dropped, so a genuine sharp FILLED point (a triangle / wedge apex) — which also
    protrudes from a solid body — WILL be shaved. No geometric signal separates a rogue bump from
    an intended sharp point; over-fires are surfaced by the full-fixture audit, not prevented here.
    No-op (byte-identical) when reference is None, the mark is radial, or nothing qualifies.
    Deterministic (H8: no RNG, row-major order, thread-pinned _is_radial, fixed cap)."""
    if reference is None or not silhouette.any():          # no oracle -> never guess
        return silhouette
    if _is_radial(reference):                              # radial-blade family (claude/firefox) — clean ref
        return silhouette
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


STROKE_DIAG_FRAC = 0.6   # a <=2px stroke whose 1px centreline is >= this fraction diagonal would
#                          staircase (8-connected but 4-BROKEN) if thinned -> WIDEN to 3px; a
#                          predominantly axis-aligned centreline THINS to 1px (vscode's diagonal
#                          tails widen; instagram's axis-aligned band thins). The user's rule:
#                          thin unless thinning "breaks the shape" = a 4-connectivity break.
STROKE_MIN_LEN = 2       # only normalize elongated strokes (a <2px-long speck is not a "line")


def normalize_stroke_width(silhouette: np.ndarray, target: int) -> np.ndarray:
    """Normalize 2px-thick strokes to an ODD width: thin to 1px by default, widen to 3px when
    thinning would break the shape. A stroke is <=2px where a 3x3 opening removes it (a >=3px core
    survives). Its half-width reads ~1 (medial_axis dist == ceil(w/2), so 1px and 2px both read 1 —
    thinning a 1px stroke is a no-op, harmless). The thin-vs-widen call is the stroke centreline's
    DIAGONALITY: a predominantly-diagonal 1px centreline 4-fragments (each px its own 4-component,
    _n4/len ~ 1.0) so thinning staircases it (the user's "break"); a predominantly axis-aligned
    centreline stays face-connected (_n4/len ~ 1/len). Above STROKE_DIAG_FRAC the stroke is WIDENED
    (radius-1 re-stroke = 3px); below it, THINNED to its 1px skeleton (offset-swept, topology-
    guarded — if thinning disconnects at every offset it is widened instead). Every change is
    topology-guarded. Deterministic (H8: rng=0, row-major, fixed). Byte-identical when no <=2px
    stroke is present. Radial-blade sunbursts (claude) are exempt — their thin blades are intended
    features, and normalizing them unevenly breaks the radial symmetry."""
    if not silhouette.any() or _is_radial(silhouette):
        return silhouette
    opened = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, _SE3,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    thin_parts = ((silhouette > 0) & (opened == 0)).astype(np.uint8)   # the <=2px strokes/bands
    if not thin_parts.any():
        return silhouette
    skel_raw, dist = medial_axis(silhouette > 0, return_distance=True, rng=0)
    skel = _prune_spurs(skel_raw.astype(np.uint8) * 255) > 0
    base = _topology(silhouette)
    n, lbl, _st, _c = cv2.connectedComponentsWithStats(thin_parts, connectivity=8)
    cur = silhouette.copy()
    for i in range(1, n):
        comp = (lbl == i)
        cskel = skel & comp
        length = int(cskel.sum())
        if length < STROKE_MIN_LEN:                        # not an elongated line -> leave it
            continue
        skmask = cskel.astype(np.uint8) * 255
        frag = (cv2.connectedComponents(skmask, connectivity=4)[0] - 1) / length
        if frag <= STROKE_DIAG_FRAC:                       # axis-aligned centreline -> THIN to 1px
            keep_skel = cur.copy(); keep_skel[comp & ~skel] = 0       # keep the medial-axis offset
            keep_other = cur.copy(); keep_other[comp & skel] = 0      # or the other offset
            if _topology(keep_skel) == base:
                cur = keep_skel
            elif _topology(keep_other) == base:
                cur = keep_other
            else:                                          # thinning disconnects at every offset -> widen
                trial = cur | _restroke(skmask, np.ones_like(dist))
                if _topology(trial) == base:
                    cur = trial
        else:                                              # diagonal centreline -> WIDEN to 3px
            trial = cur | _restroke(skmask, np.ones_like(dist))
            if _topology(trial) == base:
                cur = trial
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


def symmetrize_render(silhouette: np.ndarray,
                      constructions: list[tuple[list[Part], tuple, str]],
                      canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """Force the rendered silhouette to the LABEL's mirror symmetry at the pixel level. Detects the
    label's true mirror axes (V and/or H) hole-awarely (_label_axes). A DUAL-axis mark (both V and
    H, e.g. instagram) is symmetrized by a 4-orbit MAJORITY VOTE (a pixel is kept iff >= 2 of its
    {self, V-mirror, H-mirror, VH-mirror} are foreground) — exactly V+H symmetric, minority accents
    drop, and unlike a half-mirror it neither misaligns the two rings nor shatters a thinned 1px
    band at the mirror seam. A SINGLE-axis mark (e.g. yt-music's play triangle) DROPS its small
    asymmetric accents (components with no mirror partner, <= min-feature^2) then half-mirrors the
    authoritative (cleaner) half onto the other (_half_mirror — corrects a lopsided render), kept
    only if it does not fragment the silhouette; spurious small holes it opens are filled (kept
    holes = the label's real counters via _significant_holes). No-op when the label has no mirror
    axis. Deterministic (H8)."""
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
        return voted if _topology(voted)[1] >= _topology(silhouette)[1] else silhouette   # keep counters
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
            if int(stats[i, cv2.CC_STAT_AREA]) <= acc_max:            # small asymmetric accent -> drop
                trial[lbl == i] = 0
        sym = _half_mirror(trial, a2, vert)
        st_sym, st_cur = _topology(sym), _topology(cur)
        if st_sym[0] <= st_cur[0] and st_sym[1] >= st_cur[1]:        # no fragment AND no lost counter
            cur = sym
    outer = _outer_bg(cur)                                            # fill spurious (non-counter) holes
    significant = np.zeros(cur.shape, dtype=bool)
    for hole in _significant_holes(cur, min_feat):
        significant |= (hole > 0)
    spurious = (cur == 0) & (outer == 0) & ~significant
    if spurious.any():
        cur = cur.copy(); cur[spurious] = 255
    return cur


def refine_perimeter(constructions: list[tuple[list[Part], tuple, str]],
                     rendered_masks: list[np.ndarray],
                     canvas_hw: tuple[int, int], target: int,
                     structure_ref: np.ndarray | None = None) -> np.ndarray:
    """Stage-4 v1 product: the FACE-CONNECTED 1-px boundary perimeter of the rendered
    silhouette. Unions the per-construction renders → un-blots small counters → deburrs
    single-pixel edge nibs that fall OUTSIDE the ideal reference (cubify union-seam raster
    artifacts, see deburr_nibs) → silhouette → inner ring → diagonal disjoint repair. Signature
    matches the (cons, masks, (S,S), target) seam (assemble_render's output); `canvas_hw` = (S,S)
    is the parts' coord frame, used to raster the per-construction reference. `structure_ref` (the
    @24 sibling silhouette) is accepted for the cross-resolution seam, but the widen pass it would
    feed is DORMANT here pending a per-fixture scope decision (see widen_underrepresented).
    Returns (target,target) uint8 {0,255}. Deterministic (H8)."""
    if not rendered_masks:
        return np.zeros((target, target), np.uint8)
    sil = _composite(rendered_masks, (target, target))
    if not sil.any():
        return sil
    sil = unblot_features(sil, target)
    reference = _composite(
        [reference_mask(parts, canvas_hw, target) for parts, _c, _l in constructions],
        (target, target))
    sil = deburr_nibs(sil, target, reference)   # drop cubify union-seam edge nibs (outside-ref)
    sil = normalize_stroke_width(sil, target)   # 2px lines -> 1px (thin) or 3px (widen)
    sil = symmetrize_render(sil, constructions, canvas_hw, target)   # fold to the label's mirror axes
    #   (reflects features, so runs LAST); the chirality guard in _label_axes now excludes the
    #   over-folding class — chiral pinwheels (slack) and off-V/H-axis mirrors (react) no-op.
    ring = silhouette_outline(sil)
    return connect_disjoint(ring, sil, target)
