"""Render signed primitive groups and assemble small-grid masks.

At target sizes <=24, use label cleaning and cubify. Larger targets use
role-based straightening and ordered candidate search, constrained by fidelity,
topology and feature retention. Assembly coordinates sibling features, gaps
and protected accents before refinement.

build_constructions prepares the per-feature geometry. assemble_render returns
masks, diagnostics and a protected-accent mask. Keep reported drops and
feasibility failures: a rendered mask is not necessarily an accepted fit.
Earlier search designs and fixture results are in DEVELOPMENT.md.
"""

from __future__ import annotations

import math
import time
import os
from dataclasses import dataclass, field, replace
from typing import Literal

import cv2
import numpy as np

from iconograph.application.compress.common import (
    SYM_MIN_IOU,
    SYM_PAIR_IOU,
    Circle,
    Ellipse,
    Part,
    _offset_part,
    Polygon,
    Primitive,
    Rectangle,
    RotatedEllipse,
    RotatedRectangle,
    RoundedRectangle,
    Triangle,
    Wedge,
    _band_width,
    _boundary_iou,
    _cand_inscribed_circle,
    _centroid,
    _compound_from_masks,
    _compound_mask,
    _fit_ellipse,
    _fit_rectangle,
    _fit_rotated_ellipse,
    _fit_rotated_rect,
    _fit_rounded_rect,
    _fit_triangle,
    _iou_bool,
    _largest_contour,
    _min_feature_px,
    _mirror_axis,
    _raster_part,
    _reflection_m,
    _scale_part,
)
from iconograph.application.compress.straighten import (
    DEG_TRUE,
    K45,
    SLOPE_GRID_DEG,
    TRI_SNAP_TOL,
    _snap_angle,
    asterisk_polygon,
    prim_bbox,
    prim_center,
    scale_about_center,
    scale_about_center_xy,
    snap_slope,
    snap_slope_run2,
    snap_triangle_run2,
    star_polygon,
    straighten,
    stroke_axis_runs,
    translate_prim,
)
from iconograph.application.common import (
    masks_8adjacent,
    mst_bridges,
    nearest_gap,
)
from iconograph.application.compress.decompose import decompose_shape
from iconograph.application.compress.label_clean import clean_labels
from iconograph.application.compress.segment_prim_classify import Classified
from iconograph.application.compress.L2_feature_segment import Feature

Role = Literal["dominant", "lobe", "accent", "shave", "counter"]

SUPERSAMPLE = 4           # reference protocol: ×k raster → exact box mean → ≥0.5
LOBE_FRAC = 0.25          # add ≥ this × dominant area ⇒ lobe, else accent
COUNTER_INTERIOR_FRAC = 0.85  # bite ≥ this fraction interior ⇒ counter, else shave

# Base curvature ladder: part extent (TARGET px) → max degree before role offset.
BASE_DEGREE_BANDS = ((8, 0), (16, 1), (32, 2), (64, 3))
FORCE_TRUE_EXTENT = 64    # ≥ this px: true curve only — straightening is vandalism
ROLE_OFFSET: dict[Role, int] = {
    "dominant": -1, "lobe": -1, "accent": 0, "shave": +1, "counter": +1,
}

# Jag metric: DP vertices + off-grid-slope length penalty. DP(ε=1) absorbs the
# PERFECT 45° staircase (the optimal digital diagonal) into one segment; the
# length term catches digitally-straight-but-visually-stairy slopes (e.g. 30°).
JAG_EPS = 1.0
JAG_ANGLE_TOL = 3.0       # degrees around {0, 45, 90, 135} that count as on-grid
JAG_OFF_W = 0.25          # penalty weight per off-grid boundary px

# Composition constraints (the spec's size-compromise clauses; LOW floor).
INFLATE_FRAC = 0.35       # subordinate candidate \ original ≤ this × original
INFLATE_ABS = 2           # ... with a 2-px absolute allowance (tiny parts)
BITE_GROW_FRAC = 0.30     # shave: extra parent eaten ≤ this × original bite
BITE_ABS = 2
RETAIN_FRAC = 0.60        # every add keeps ≥ this × its reference visibility

# Round 2 — symmetrization + label collapse.
SYM_AXIS_TOL_DEG = 3.0    # accept only near-vertical/horizontal mirror axes
COLLAPSE_MIN_IOU = 0.80   # canonical fit gate — the "well-approximates" test
COLLAPSE_MAX_T = 32       # collapse is a tiny-grid degradation strategy
CUBIFY_MAX_T = 24         # feed resolution: T <= this renders via the pixel-cubify
#                           operator (cubify.py) instead of geometry straightening
RADIAL_MIN_BLADES = 6     # radial-family detection (claude has 12)
RADIAL_DIST_CV = 0.15     # blade center-distance CV about the hub
RADIAL_GAP_CV = 0.40      # angular-gap CV (mirrors _fit_rot_spurs' gate)

# Round 3 — digital-line doctrine, anti-globbing, pronouncedness.
W_STEP = 1.0              # J2: per 1-px tooth beyond the chamfer allowance
POCKET_MIN = 3            # px — significant negative-space pocket floor
POCKET_RETAIN = 0.5       # fraction of pocket pixels that must stay open
OUTLINE_MIN_DEPTH = 1.0   # round 9 — a silhouette convexity defect this deep
#                           (absolute px) flanked by two ROUND adds (domes) =
#                           "separate on the outline": the notch pixels get
#                           ABSOLUTE pocket protection (the newly-violated
#                           sep/pocket clauses are toothless once the baseline
#                           fuses the notch — figma's domes). Round 8 used a
#                           0.15×T frac (=2.4 px @16) which MISSED figma's
#                           lower-dome pinch (only 1.29 px deep). The
#                           TWO-ROUND-FLANK gate — not depth — is the real
#                           over-carve discriminator (vscode/discord/claude/
#                           gmail stay [] at a 1-px floor), so depth drops to 1.
OUTLINE_RETAIN = 0.5      # fraction of the dome-notch that must stay open
PRON_MIN_DEPTH = 1.5      # px — protrusion significance floor
PRON_FRAC = 0.6           # d_render ≥ this × d_ref
RETAIN_NARROW = 0.4       # retention floor for narrowed cands w/ pron. held

# round 4 — removal on globbing evidence + extrapolated symmetry
GAP_FUSE_FRAC = 0.5       # ring fused-fraction above which a gap is
#                           sub-representable at target
REMOVE_MAX_FRAC = 0.25    # accent-removal budget: max fraction of reference
#                           ink removable per run (family decimation exempt)
GLOB_ATTR_FRAC = 0.25     # pocket attribution: an accent owns a glob when its
#                           solo covers ≥ this of the pocket's filled pixels
REG_EXT_TOL = 0.06        # size group: max relative extent spread
REG_ALIGN_TOL = 0.02      # × S — center-alignment cluster tolerance
REG_PITCH_CV = 0.05       # uniform-pitch group: max spacing CV
REG_RENDER_TOL = 1.0      # px — alignment/pitch tolerance on the final render
POCKET_RETAIN_REF = 0.3   # aggregate pocket-openness bar for reference-
#                           substituted runs: min-feature-width renders of
#                           idealized thin-gap geometry cannot hold 0.5
SPILL_FRAC_REF = 0.20     # extrapolated runs know their ideal: a part may
#                           paint ≤ this fraction of itself outside it (open-
#                           corner negative space has no pocket to guard it —
#                           figma's silhouette step)
RADIAL_REG_TOL_DEG = 2.0  # max gap deviation from the comb that still counts
#                           as a REGULAR radial render (asterisk6 is exact)
ROUND_ID_MIN_EXTENT = 7   # px — round-identity floor applies from this extent

# round 6 — proportion honesty + claimed strays
PROP_SPILL_FRAC = 0.10    # a candidate may paint at most this fraction of
#                           itself outside the reference BEYOND what its
#                           same-degree f=1 twin already paints: proportion
#                           drift is priced, straightening quantization is
#                           free (circle@16's square spills corners at f=1
#                           too → extra 0). vscode@16's dominant bought −J
#                           with f=1.23 (+17-px spill over the f=1 twin's 1,
#                           cap 5.2); gmail@16's V counters shrank to 0.68
#                           (uncarved extra 5 over cap 2.3). Measured with
#                           the WORKING guard — do not retune against runs
#                           from before the Polygon-exemption bug fix.
STRAY_OUT_FRAC = 0.60     # a sub-min-feature final-mask component lying ≥
#                           this fraction outside the reference is a stray
#                           even when a live part claims it (gmail@24/32:
#                           the dominant's sliver between the V counter tips)


def floor_for(target: int) -> float:
    """The global proportionality floor — deliberately low for now (spec)."""
    return 0.55 if target <= 24 else (0.65 if target <= 48 else 0.75)


@dataclass(frozen=True)
class TweakRecord:
    """One part's chosen tweak (or drop) — the render report's unit."""
    index: int
    role: Role
    extent_px: float          # continuous min bbox side at target scale
    budget: int               # max allowed degree
    degree: int               # chosen degree (DEG_TRUE = true curve)
    size_fx: float            # per-axis extent scale applied (1.0 = none)
    size_fy: float
    dx: float                 # grid-snap translation applied
    dy: float
    dropped: str | None       # reason, or None if rendered
    violation: str | None     # set when the feasibility fallback fired
    mirrored_of: int | None = None   # follower → leader index (symmetrization)


@dataclass
class RenderResult:
    mask: np.ndarray                      # (T, T) uint8 {0,255} — the R1 render
    ref: np.ndarray                       # supersampled-threshold reference R_T
    naive: np.ndarray                     # INTER_NEAREST baseline (the score floor)
    records: list[TweakRecord]
    jag: float
    jag_naive: float
    jag_norm: float                       # jag / boundary px (cross-size comparable)
    jag_norm_naive: float
    area_iou: float                       # render vs ref
    boundary_iou: float
    floor: float
    violations: list[str] = field(default_factory=list)
    drops: list[str] = field(default_factory=list)
    topo_render: tuple[int, int] = (0, 0)  # (components, holes)
    topo_ref: tuple[int, int] = (0, 0)
    topo_canon: tuple[int, int] = (0, 0)   # canonical-res truth — the reference
    #                                        itself can degenerate at tiny T (a
    #                                        1-px moat closing); matching EITHER
    #                                        ref or canon passes the audit
    sym: float | None = None               # IoU(final, flip(final)) when a global
    #                                        mirror axis fired, else None
    axis: tuple[str, float] | None = None  # ("x"|"y", snapped target coordinate)
    collapsed: str | None = None           # winning collapse-variant summary
    pairs: int = 0                         # mirror pairs coordinated (round 3 audit)
    staircase: float = 0.0                 # J2's W_STEP term on the final mask
    fid_ref: float | None = None           # round 4: floor-substitute fid vs the
    #                                        winning variant's OWN reference
    #                                        (removal: survivors; extrapolate:
    #                                        idealized geometry); NEVER in the
    #                                        arbitration key
    ref_kind: str | None = None            # "removal" | "extrapolate"
    removed_ink: int = 0                   # reference px attributed to removals
    reg: str | None = None                 # detected regularity: "radial" |
    #                                        "group" | None (harness audit)


# reference protocol + baseline


def reference_coverage(parts: list[Part], canvas_hw: tuple[int, int],
                       target: int, k: int = SUPERSAMPLE) -> np.ndarray:
    """The PRE-THRESHOLD float coverage map (target, target) float32 in [0, 1]:
    parts rastered at k×target, exact numpy box-mean. The render_fit refiner's
    optimization target; reference_mask thresholds it."""
    h, w = canvas_hw
    f = (k * target) / max(h, w)
    hi = _compound_mask([_scale_part(p, f) for p in parts], k * target, k * target)
    return (hi.astype(np.float32) / 255.0).reshape(target, k, target, k).mean(axis=(1, 3))


def reference_mask(parts: list[Part], canvas_hw: tuple[int, int], target: int,
                   k: int = SUPERSAMPLE) -> np.ndarray:
    """R_T: coverage ≥ 0.5 binarization. NOT cv2.INTER_AREA — its uint8 cast
    truncates, so an exactly-half-covered block would round DOWN; the float
    mean thresholds the 0.5 case correctly."""
    return ((reference_coverage(parts, canvas_hw, target, k) >= 0.5)
            .astype(np.uint8)) * 255


def naive_mask(parts: list[Part], canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """The baseline the renderer must beat: nearest-neighbour downsample of the
    canonical-resolution compound mask (floor source-pixel selection, no averaging)."""
    h, w = canvas_hw
    canon = _compound_mask(parts, h, w)
    return cv2.resize(canon, (target, target), interpolation=cv2.INTER_NEAREST)


# jag metric (J2 — the digital-line doctrine, round 3)


def _staircase_penalty(contour: np.ndarray) -> float:
    """1-px teeth beyond the chamfer allowance, from the raw chain code.
    CHAIN_APPROX_NONE guarantees unit king-moves; diagonal = (dx≠0)&(dy≠0).
    A staircase REGION = D-runs merged across single-axis-move separators
    (pure 45° lines AND mixed 2-1-2-1 words); its n_dd = Σ(L−1) over D-runs =
    the count of run-length-1 steps; penalty = max(0, n_dd − (K45 − 1)) —
    a chamfer of ≤ K45 diagonal moves is a free corner cut."""
    pts = contour.reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return 0.0
    moves = np.vstack([np.diff(pts, axis=0), (pts[0] - pts[-1])[None, :]])
    diag = (moves[:, 0] != 0) & (moves[:, 1] != 0)
    if not diag.any():
        return 0.0
    if diag.all():
        return float(max(0, (len(diag) - 1) - (int(K45) - 1)))
    start = int(np.argmin(diag))          # rotate to start on an axis move so
    diag = np.roll(diag, -start)          # the cyclic wrap never splits a run
    change = np.flatnonzero(np.diff(diag.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [len(diag)]))
    pen = 0.0
    region_ndd = 0
    open_region = False
    for k in range(len(bounds) - 1):
        is_d = bool(diag[bounds[k]])
        run = int(bounds[k + 1] - bounds[k])
        if is_d:
            region_ndd += run - 1
            open_region = True
        elif run > 1 and open_region:     # axis run ≥ 2 closes the region;
            pen += max(0, region_ndd - (int(K45) - 1))   # a single axis step
            region_ndd = 0                # keeps a mixed staircase open
            open_region = False
    if open_region:
        pen += max(0, region_ndd - (int(K45) - 1))
    return float(pen)


def _staircase_total(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return sum(_staircase_penalty(c) for c in contours)


def _off_slope(theta_deg: float) -> bool:
    """True when a DP segment's direction is off the DIGITAL SLOPE GRID
    (runs ≥ 2 px or pure axis; 45° in-grid — its teeth are priced by the
    staircase term instead)."""
    a = theta_deg % 180.0
    for m in SLOPE_GRID_DEG:
        d = abs(a - m)
        if min(d, 180.0 - d) <= JAG_ANGLE_TOL:
            return False
    return True


def jag_score(mask: np.ndarray) -> float:
    """J2 = Σ contours [DP-vertex count + JAG_OFF_W·(off-slope-grid segment
    length) + W_STEP·(staircase teeth beyond the K45 chamfer allowance)].
    The metric's unit of badness is the RUN, not the chord: DP absorbs any
    digital line geometrically, so without the staircase term a 1-px 45°
    staircase — the worst-decomposing structure downstream — scores FREE
    (the round-2 metric's inverted assumption). RETR_CCOMP so holes count;
    CHAIN_APPROX_NONE because both DP and the chain pass need the full
    boundary sequence; one findContours call serves both terms."""
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    j = 0.0
    for c in contours:
        if len(c) < 3:
            j += float(len(c))
            continue
        ap = cv2.approxPolyDP(c, JAG_EPS, True).reshape(-1, 2).astype(np.float64)
        v = len(ap)
        j += float(v)
        for i in range(v):
            ax, ay = ap[i]
            bx, by = ap[(i + 1) % v]
            seg = math.hypot(bx - ax, by - ay)
            if seg < 1e-9:
                continue
            if _off_slope(math.degrees(math.atan2(by - ay, bx - ax))):
                j += JAG_OFF_W * seg
        j += W_STEP * _staircase_penalty(c)
    return j


def _boundary_px(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return sum(len(c) for c in contours)


def jag_norm(mask: np.ndarray) -> float:
    """Ĵ = J / boundary px — comparable across target sizes."""
    return jag_score(mask) / max(1, _boundary_px(mask))


def _topology(mask: np.ndarray) -> tuple[int, int]:
    """(foreground components, holes). RETR_CCOMP hierarchy for holes — immune to
    the edge-touching-background merge of the mask-inversion method."""
    n_cc, *_ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0 if hier is None else sum(
        1 for i in range(len(contours)) if hier[0][i][3] != -1)
    return n_cc - 1, holes


# structural role inference


def infer_roles(parts: list[Part], canvas_hw: tuple[int, int]) -> list[Role]:
    """Roles from CSG structure + mask algebra alone (canonical resolution).
    Counter-vs-shave: a subtract whose bite is ≥ COUNTER_INTERIOR_FRAC inside
    erode(group add-union, band) is interior — a hole/counter; a bite touching
    the parent's boundary band is a shave (bevel/notch). BORDER_CONSTANT erode:
    canvas-edge pixels count as background, conservatively shrinking 'interior'."""
    h, w = canvas_hw
    solos = [_raster_part(p.primitive, h, w) for p in parts]
    roles: list[Role | None] = [None] * len(parts)
    for g in sorted({p.group for p in parts}):
        idxs = [i for i, p in enumerate(parts) if p.group == g]
        if g < 0:
            for i in idxs:                # the whole negative group IS a counter
                roles[i] = "counter"
            continue
        adds = [i for i in idxs if parts[i].op == "add"]
        subs = [i for i in idxs if parts[i].op == "subtract"]
        union = np.zeros((h, w), np.uint8)
        for i in adds:
            union |= solos[i]
        areas = {i: int(np.count_nonzero(solos[i])) for i in adds}
        if adds:
            dom = sorted(adds, key=lambda i: (-areas[i], i))[0]
            for i in adds:
                if i == dom:
                    roles[i] = "dominant"
                else:
                    roles[i] = "lobe" if areas[i] >= LOBE_FRAC * areas[dom] else "accent"
        if subs:
            band = _band_width(union)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band + 1, 2 * band + 1))
            interior = cv2.erode(union, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            for i in subs:
                bite = solos[i] & union
                nb = int(np.count_nonzero(bite))
                if nb == 0:
                    roles[i] = "shave"    # no-op subtract; min-feature pass drops it
                    continue
                frac = int(np.count_nonzero(bite & interior)) / nb
                roles[i] = "counter" if frac >= COUNTER_INTERIOR_FRAC else "shave"
    return [r if r is not None else "accent" for r in roles]


def allowed_degree(role: Role, extent_px: float) -> int:
    """Curvature budget: base extent ladder + role offset, clamped [0, DEG_TRUE].
    No offset in the forced-true band."""
    if extent_px >= FORCE_TRUE_EXTENT:
        return DEG_TRUE
    base = DEG_TRUE
    for bound, d in BASE_DEGREE_BANDS:
        if extent_px < bound:
            base = d
            break
    return max(0, min(DEG_TRUE, base + ROLE_OFFSET[role]))


ELONGATED_ASPECT = 2.0    # local max/min dim ≥ this ⇒ the budget keys on LENGTH


def _budget_extent(p) -> float:
    """The extent that drives a part's curvature budget, in the part's LOCAL
    frame. For elongated parts (aspect ≥ ELONGATED_ASPECT) the LONG dimension
    governs: rotating a long thin blade moves far more ink per degree than
    rounding a corner, so a 3×20-px sunburst spur must keep its angle (D2+)
    rather than snap to the 90° grid and collapse into its siblings (claude@16).
    The canvas-axis bbox would hide a rotated blade's aspect entirely."""
    lo, hi = _local_dims(p)
    return hi if (lo > 0 and hi >= ELONGATED_ASPECT * lo) else lo


# candidate generation


def _build_candidate(part_t: Part, degree: int, fx: float, fy: float,
                     mode: int) -> tuple[Part, float, float]:
    """One tweak: scale about center → straighten → grid-snap translate.
    mode 0 = no snap, 1 = center to the half-px grid (integer-corner spans have
    centers on it), 2 = bbox min corner to the nearest INTEGER — the crisp case
    under fillPoly's outline-inclusive convention. Returns (part, dx, dy)."""
    base = scale_about_center_xy(part_t.primitive, fx, fy)
    form = straighten(base, degree)
    dx = dy = 0.0
    if mode == 1:
        cx, cy = prim_center(form)
        dx, dy = round(2.0 * cx) / 2.0 - cx, round(2.0 * cy) / 2.0 - cy
    elif mode == 2:
        x0, y0, _x1, _y1 = prim_bbox(form)
        dx = float(round(x0)) - x0
        dy = float(round(y0)) - y0
    if dx or dy:
        form = translate_prim(form, dx, dy)
    return Part(primitive=form, op=part_t.op, group=part_t.group), dx, dy


def _floatify(part: Part) -> Part:
    """Rectangle's int fields quantize at the canvas→target scale step (a 0.5-px
    loss the snap candidates can never recover — they snap to the NEAREST grid,
    they don't explore ±1). Convert to an exact float box before scaling."""
    p = part.primitive
    if isinstance(p, Rectangle):
        cx, cy = prim_center(p)
        return Part(RotatedRectangle(cx=cx, cy=cy, w=float(p.w), h=float(p.h),
                                     angle=0.0), part.op, part.group)
    return part


def _size_factors(extent: float, min_feat: int) -> list[float]:
    """Keep (f=1) + integer-px extent targets {round−1, round, round+1} ∪
    {floor, ceil}, clamped ≥ min_feat. round±1 matters even for integer extents:
    the reference's ≥0.5 coverage includes BOTH boundary cells when continuous
    edges land exactly on half-cells, so a 10-px-extent part can render 11 px in
    the reference — the +1 target must be reachable."""
    sizes: list[float] = [1.0]
    if extent > 0:
        targets = {math.floor(extent), round(extent), math.ceil(extent),
                   round(extent) - 1, round(extent) + 1}
        for e2 in sorted(targets):
            e2 = max(e2, min_feat, 1)
            f = e2 / extent
            if abs(f - 1.0) > 1e-9:
                sizes.append(f)
    return sizes


def _candidates(part_t: Part, degrees: list[int], extent_xy: tuple[float, float],
                min_feat: int, aniso: bool,
                narrow: bool = False,
                axis_reduce: bool = True,
                run2: bool = True
                ) -> list[tuple[Part, int, float, float, float, float]]:
    """Deterministic candidate list: degree asc × extent target × snap mode.
    Extent targets give implicit parity control: an integer covered extent plus
    a grid snap lands vertices on integers, the crisp-fill case. Adds also get
    ANISOTROPIC pairs (per-axis round±1 targets) — the reference's tie-inclusive
    coverage can stretch one axis by a pixel independently of the other."""
    ex, ey = extent_xy
    sizes: list[tuple[float, float]] = [(f, f) for f in
                                        _size_factors(min(ex, ey), min_feat)]
    # round 7 — a Circle is ROUND by identity: anisotropic per-axis pairs turn
    # it into an oval (figma's size-grouped circles rendered f≈1.07×0.92), and
    # the differing oval tuples are what defeat the size-group harmonization's
    # absolute-extent match (the members can't agree on one (round(fx·ex),
    # round(fy·ey))). Locking circles to isotropic sizes restores roundness AND
    # makes the equality pass succeed. Represented-geometry: clean+symmetric
    # over a fitted oval (the authorized trade).
    if aniso and ex > 0 and ey > 0 and not isinstance(part_t.primitive, Circle):
        seen = {(round(fx, 6), round(fy, 6)) for fx, fy in sizes}
        for tx in (round(ex) - 1, round(ex), round(ex) + 1):
            for ty in (round(ey) - 1, round(ey), round(ey) + 1):
                fx = max(tx, min_feat, 1) / ex
                fy = max(ty, min_feat, 1) / ey
                key = (round(fx, 6), round(fy, 6))
                if abs(fx - fy) > 1e-9 and key not in seen:
                    seen.add(key)
                    sizes.append((fx, fy))
    # Wedge doctrine: rotation is a SUPPRESSIBLE degree of freedom. Offer an
    # orientation-normalized base (sector rotated onto the 45° grid, sweep
    # preserved) at EVERY chord-detail level, listed FIRST — strict pixel-
    # detail preference; the floor arbitrates when the rotation is identity-
    # bearing. D0/D1 re-snap it idempotently; D2/D3 gain grid-aligned straight
    # edges WITH full chord detail (the combination the coupled ladder lacked).
    bases = [part_t]
    pr = part_t.primitive
    if isinstance(pr, Wedge):
        a0s = _snap_angle(pr.a0, 45.0)
        if abs(a0s - pr.a0) > 1e-6:
            bases.insert(0, Part(Wedge(cx=pr.cx, cy=pr.cy, r=pr.r, a0=a0s,
                                       a1=a0s + (pr.a1 - pr.a0)),
                                 part_t.op, part_t.group))
    if narrow:
        # round 3 — pronounced-feature geometry: elongated subordinates gain
        # NARROW + slope-snapped variants ("narrower and less tilted — less
        # accurate but more true to the style"): width targets down to
        # min_feat, length preserved, on a digital-slope-grid base
        lo, hi = _local_dims(pr)
        if lo > 0 and hi >= 2.0 * lo:
            if isinstance(pr, (RotatedRectangle, RoundedRectangle,
                               RotatedEllipse)) \
                    and abs(snap_slope(pr.angle) - pr.angle % 180.0) > 1e-6:
                bases.append(Part(replace(pr, angle=snap_slope(pr.angle)),
                                  part_t.op, part_t.group))
            # which local axis is the short one for scale_about_center_xy?
            # fx scales w/rx, fy scales h/ry (local frame for rotated kinds)
            if isinstance(pr, (RotatedRectangle, RoundedRectangle)):
                short_is_x = pr.w <= pr.h
            elif isinstance(pr, (Ellipse, RotatedEllipse)):
                short_is_x = pr.rx <= pr.ry
            else:
                short_is_x = ex <= ey
            seen2 = {(round(fx, 6), round(fy, 6)) for fx, fy in sizes}
            for ws in range(int(min_feat), int(round(lo))):
                f_short = ws / lo
                for tl in (round(hi), round(hi) + 1):
                    f_long = max(tl, min_feat, 1) / hi
                    fx, fy = ((f_short, f_long) if short_is_x
                              else (f_long, f_short))
                    key = (round(fx, 6), round(fy, 6))
                    if key not in seen2:
                        seen2.add(key)
                        sizes.append((fx, fy))
    # round 6 — the run-≥2 "geometric extension" rung: a diagonal whose
    # nearest Θ member is 45° renders 1-px teeth at length (W_STEP prices
    # them, but no alternative base exists, so the true angle wins by
    # fidelity and ships irregular bump trains). Offer the nearest RUN-≥2
    # slope as ONE extra base — uniform ≥2-px step runs, zero staircase.
    # The fid floor / retention arbitrate whether the larger rotation is
    # affordable ("avoided unless completely necessary"). Excluded for
    # radial-family blades via the run2 flag (rotational identity outranks
    # the slope grid — the comb machinery's territory).
    if run2 and part_t.op == "add":
        if isinstance(pr, (RotatedRectangle, RoundedRectangle,
                           RotatedEllipse)):
            lo, hi = _local_dims(pr)
            ang_ax = min(pr.angle % 90.0, 90.0 - (pr.angle % 90.0))
            s45 = snap_slope(pr.angle) % 90.0
            if (lo > 0 and hi >= 2.0 * lo and ang_ax > JAG_ANGLE_TOL
                    and abs(s45 - 45.0) < 1e-6):
                r2 = snap_slope_run2(pr.angle)
                if abs(r2 - pr.angle % 180.0) > 1e-6:
                    bases.append(Part(replace(pr, angle=r2),
                                      part_t.op, part_t.group))
        elif isinstance(pr, Triangle):
            # a triangle edge longer than the K45 chamfer that D1 would
            # leave at (or snap to) the 45° class — offer the all-run-2
            # edge-snapped form built by re-intersection
            need = False
            verts = ((pr.v1, pr.v2), (pr.v2, pr.v3), (pr.v3, pr.v1))
            for (ax, ay), (bx, by) in verts:
                if math.hypot(bx - ax, by - ay) <= K45 * math.sqrt(2.0):
                    continue
                th = math.degrees(math.atan2(by - ay, bx - ax)) % 90.0
                if abs(th - 45.0) <= TRI_SNAP_TOL[1]:
                    need = True
                    break
            if need:
                pts2 = snap_triangle_run2(pr)
                if pts2 is not None:
                    bases.append(Part(Triangle(*pts2),
                                      part_t.op, part_t.group))
    out = []
    for base in bases:
        for d in degrees:
            for fx, fy in sizes:
                for mode in (0, 1, 2):
                    cand, dx, dy = _build_candidate(base, d, fx, fy, mode)
                    out.append((cand, d, fx, fy, dx, dy))
    # round 5 — the L-CORNER doctrine (user-drawn ideal): a thin DIAGONAL
    # stroke also offers its axis-run reductions (L / mirrored-L / Z), the
    # coarsest digital line — zero staircase, single corners. Direct-emit at
    # the pre-sized target width and d=0 (the maximal-reduction rung; the
    # `0 in degrees` gate respects round-4's deg_min floors, so comb blades
    # keep their D1 slope doctrine). J2 strongly prefers these; the fid
    # floor + battery self-regulate (a long faithful diagonal rejects the L
    # on fidelity, a sub-representable bar accepts it).
    if (axis_reduce and part_t.op == "add" and 0 in degrees
            and isinstance(pr, (RotatedRectangle, RoundedRectangle,
                                RotatedEllipse))):
        lo, hi = _local_dims(pr)
        ang_ax = min((pr.angle % 90.0), 90.0 - (pr.angle % 90.0))
        if (lo > 0 and hi >= 2.0 * lo and lo <= min_feat + 1.0
                and ang_ax > 10.0):
            cx, cy = prim_center(pr)
            if isinstance(pr, (RotatedRectangle, RoundedRectangle)):
                long_along_angle = pr.w >= pr.h
            else:
                long_along_angle = pr.rx >= pr.ry
            th = math.radians(pr.angle + (0.0 if long_along_angle else 90.0))
            hl = hi / 2.0
            x0e, y0e = cx - hl * math.cos(th), cy - hl * math.sin(th)
            x1e, y1e = cx + hl * math.cos(th), cy + hl * math.sin(th)
            w_t = float(max(1.0, round(lo)))
            stub_t = min(abs(x1e - x0e), abs(y1e - y0e)) / 2.0
            for formk in (("z" if stub_t <= 2.5 else "z2"), "l", "lr"):
                ptsr = stroke_axis_runs(x0e, y0e, x1e, y1e, w_t, formk)
                if ptsr is None:
                    continue
                basep = Part(Polygon(pts=ptsr), part_t.op, part_t.group)
                for mode in (0, 1, 2):
                    cand, dxs, dys = _build_candidate(basep, 0, 1.0, 1.0,
                                                      mode)
                    out.append((cand, 0, 1.0, 1.0, dxs, dys))
    return out


# constraints


def _inflation_ok(cand_solo: np.ndarray, orig_solo: np.ndarray) -> bool:
    grown = int(np.count_nonzero(cand_solo & ~orig_solo))
    return grown <= INFLATE_FRAC * int(np.count_nonzero(orig_solo)) + INFLATE_ABS


def _bite_ok(cand_solo: np.ndarray, orig_solo: np.ndarray,
             parent: np.ndarray) -> bool:
    """The 'rounding must not exceed its rect' clause: the straightened shave may
    not eat meaningfully more of its parent than the original arc did (1-px
    dilation tolerates grid snap)."""
    bite_o = orig_solo & parent
    nb = int(np.count_nonzero(bite_o))
    if nb == 0:
        return True
    tol = cv2.dilate(bite_o, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    grown = int(np.count_nonzero((cand_solo & parent) & ~tol))
    return grown <= BITE_GROW_FRAC * nb + BITE_ABS


def _pocket_masks(ref: np.ndarray) -> list[np.ndarray]:
    """Significant negative-space pockets of the reference — the background a
    glob fills in (vscode's X wedges, claude's inter-blade gaps where they ARE
    representable). close − ref captures border-open concavities hole topology
    misses; iterations=2 bridges 2–3-px mouths; borderValue=0 is load-bearing
    (the dilation step must not bridge the canvas edge)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(ref, cv2.MORPH_CLOSE, k, iterations=2,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    pock = closed & ~ref
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(pock, connectivity=8)
    return [((lbl == i).astype(np.uint8)) * 255
            for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= POCKET_MIN]


def _outline_gaps(ref: np.ndarray, parts: list[Part],
                  true_solos: list[np.ndarray], add_live: list[int],
                  radial_skip: set[int], target: int
                  ) -> list[tuple[np.ndarray, frozenset[int]]]:
    """Round 8 — the OUTLINE-SEPARATION rule (user: "features on render are
    given separation if they're separate on the outline of the item — the
    half-circle domes on the sides"). A DOME is a Circle/Ellipse; two domes
    sitting side by side leave a concave NOTCH on the silhouette even when
    their bodies overlap through the trunk (figma's left+right domes touch
    at the spine but the outline pinches between them). The newly-violated
    `sep`/`pocket` clauses can't reopen such a notch once the baseline fuses
    it, so these notch pixels are enforced ABSOLUTELY.

    Returns (notch_mask, flank_set) per qualifying silhouette defect:
    - a convexity defect ≥ OUTLINE_MIN_DEPTH px deep (an absolute floor — a
      real pinch, not raster noise; the two-round-flank gate below, not
      depth, is the over-carve discriminator),
    - whose far point is flanked by ≥ 2 distinct ROUND add parts
      (Circle/Ellipse — the user's "domes"; excludes vscode's RRect+Triangle
      crossing wedge, gmail's single-body V, discord's antennae),
    - radial-family members skipped (claude's blades).
    The notch_mask is the pinched reference-background near the far point
    (bg with fg on both sides within 2 px); the flank_set attributes the
    gap to the domes that can reopen it, so _global_fail constrains only
    their search (never blames the trunk/pill, which can't move the notch).
    Verified to fire ONLY on figma across all five seeds at 16/24/32."""
    round_add = [i for i in add_live if i not in radial_skip
                 and isinstance(parts[i].primitive,
                                (Circle, Ellipse, RotatedEllipse))]
    if len(round_add) < 2:
        return []
    cnts, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 4:
        return []
    hull = cv2.convexHull(c, returnPoints=False)
    if hull is None or len(hull) < 3:
        return []
    try:
        defs = cv2.convexityDefects(c, hull)
    except cv2.error:
        return []
    if defs is None:
        return []
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = {i: cv2.dilate(true_solos[i], se, iterations=2) for i in round_add}
    fg = ref > 0
    bg = ~fg
    pinch = np.zeros(ref.shape, bool)
    for d_ in (1, 2):                     # bg with fg on both sides = a neck
        pinch |= bg & np.roll(fg, d_, 1) & np.roll(fg, -d_, 1)
        pinch |= bg & np.roll(fg, d_, 0) & np.roll(fg, -d_, 0)
    dmin = OUTLINE_MIN_DEPTH
    out: list[tuple[np.ndarray, frozenset[int]]] = []
    for _s, _e, f, d in defs[:, 0, :]:
        if d / 256.0 < dmin:
            continue
        fx, fy = int(c[f, 0, 0]), int(c[f, 0, 1])
        near = [i for i in round_add if dil[i][fy, fx] > 0]
        if len(near) < 2:
            continue
        disk = np.zeros(ref.shape, np.uint8)
        cv2.circle(disk, (fx, fy), 2, 1, -1)
        notch = (pinch & (disk > 0)).astype(np.uint8) * 255
        if int(np.count_nonzero(notch)):
            out.append((notch, frozenset(near)))
    return out


# symmetrization


def _flip_mask(mask: np.ndarray, a2: int, vertical: bool) -> np.ndarray:
    """Exact integer mirror about the half-px axis coord a2/2 (x if vertical,
    else y): index map i' = a2 − i, out-of-range sources → 0. This — not a
    reflected-polygon raster — is how follower symmetry stays pixel-exact:
    fillPoly's round-half-up tie bias is not mirror-equivariant."""
    out = np.zeros_like(mask)
    n = mask.shape[1] if vertical else mask.shape[0]
    src = a2 - np.arange(n)
    valid = (src >= 0) & (src < n)
    if vertical:
        out[:, np.arange(n)[valid]] = mask[:, src[valid]]
    else:
        out[np.arange(n)[valid], :] = mask[src[valid], :]
    return out


SYM_AXIS_SEARCH = 6          # ± half-px steps searched for the true flip axis


def _best_flip_axis(u_mask: np.ndarray, vertical: bool) -> tuple[int, float] | None:
    """Search a ±SYM_AXIS_SEARCH half-px window (±3 px) around the silhouette's own
    centre for the integer flip axis a2 maximising IoU(U, flip(U)) — aligning the mirror
    with what was drawn. Returns (a2, best_iou) or None on an empty mask. (Lifted from
    refine/common.py — H5: cubify's R6 dual-axis detection and the stage-4 symmetrize
    machinery share it; refine re-imports it from here.)"""
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


def _sym_prep(parts: list[Part], solos: list[np.ndarray],
              canvas_hw: tuple[int, int], target: int
              ) -> tuple[bool, float, int, float, dict[int, int], set[int]] | None:
    """Global mirror-axis prep: detect (canonical res, near-axis-aligned only —
    a diagonal mirror cannot be enforced on an integer raster), snap the axis
    to the target half-px grid, and pair parts by reflection-matching.
    Returns (vertical, a_snapped_target, a2, delta, leader_of, self_paired)
    or None. The caller rigidly translates parts_t by delta along the
    axis-normal — the reference is NEVER shifted."""
    h, w = canvas_hw
    comp = _compound_from_masks(
        [(p.op, p.group, s) for p, s in zip(parts, solos)], h, w)
    ax = _mirror_axis(comp)
    if ax is None:
        return None
    p, u = ax
    tol = math.sin(math.radians(SYM_AXIS_TOL_DEG))
    if abs(u[0]) <= tol:
        vertical = True               # line direction ~(0,±1) → mirrors x
    elif abs(u[1]) <= tol:
        vertical = False
    else:
        # _mirror_axis's strict argmax can land on a PCA DIAGONAL when the
        # shape is 4-fold symmetric (all axes tie within raster noise —
        # nested_counter's disc+moat). Diagonals can't be enforced on an
        # integer grid, but an axis-aligned mirror may still hold: re-test
        # vertical/horizontal directly before giving up.
        best_al: tuple[bool, float] | None = None
        for vert, uu in ((True, (0.0, 1.0)), (False, (1.0, 0.0))):
            mat = _reflection_m(float(p[0]), float(p[1]), *uu)
            refl_al = cv2.warpAffine(comp, mat, (w, h),
                                     flags=cv2.INTER_NEAREST, borderValue=0)
            iou_al = _iou_bool(comp, refl_al)
            if iou_al >= SYM_MIN_IOU and (best_al is None
                                          or iou_al > best_al[1]):
                best_al = (vert, iou_al)
        if best_al is None:
            return None
        vertical = best_al[0]
    f = target / max(h, w)
    a = (p[0] if vertical else p[1]) * f
    a_t = round(2.0 * a) / 2.0
    a2 = int(round(2.0 * a_t))
    delta = a_t - a
    # pairing at CANONICAL res — target rasters of 2-px eyes are too coarse
    uu = (0.0, 1.0) if vertical else (1.0, 0.0)
    mat = _reflection_m(float(p[0]), float(p[1]), *uu)
    refl = [cv2.warpAffine(s, mat, (w, h), flags=cv2.INTER_NEAREST,
                           borderValue=0) for s in solos]
    leader_of: dict[int, int] = {}
    self_paired: set[int] = set()
    used: set[int] = set()
    for i in range(len(parts)):
        if i in used:
            continue
        if _iou_bool(refl[i], solos[i]) >= 0.9:
            self_paired.add(i)
            used.add(i)
            continue
        for j in range(i + 1, len(parts)):
            if j in used or parts[j].op != parts[i].op \
                    or parts[j].group != parts[i].group:
                continue
            if _iou_bool(refl[i], solos[j]) >= SYM_PAIR_IOU:
                leader_of[j] = i
                used.add(i)
                used.add(j)
                break
    return vertical, a_t, a2, delta, leader_of, self_paired


def _mirror_trivial(p: Primitive) -> bool:
    """True when local-pair coordination buys nothing: the primitive is its own
    mirror image (circles, axis-aligned boxes/ellipses), so forcing a flip
    relation only constrains positions — measured to COST quality on figma's
    stacked circles. Rotated kinds off the axis grid (vscode's ±50° X bars)
    are the real local-pair targets."""
    if isinstance(p, (Circle, Ellipse, Rectangle)):
        return True
    if isinstance(p, (RotatedRectangle, RotatedEllipse, RoundedRectangle)):
        return abs(_snap_angle(p.angle, 90.0) - p.angle) <= SYM_AXIS_TOL_DEG
    return False


def _local_pairs(parts: list[Part], solos: list[np.ndarray],
                 canvas_hw: tuple[int, int], target: int
                 ) -> tuple[dict[int, int], dict[int, tuple[bool, int]],
                            dict[int, tuple[float, float]]]:
    """LOCAL mirror pairs when no global axis exists (vscode's two X bars):
    same-kind same-op/group siblings of comparable area whose reflection about
    the vertical (then horizontal) axis through their canonical midpoint
    matches (IoU ≥ SYM_PAIR_IOU). Mirror-trivial kinds are skipped. Each pair
    gets its OWN snapped axis; both members shift by the same delta so the
    axis lands on the half-px grid.
    Returns (leader_of, pair_axis {idx → (vertical, a2)}, deltas {idx → (dx, dy)})."""
    h, w = canvas_hw
    f = target / max(h, w)
    leader_of: dict[int, int] = {}
    pair_axis: dict[int, tuple[bool, int]] = {}
    deltas: dict[int, tuple[float, float]] = {}
    areas = [int(np.count_nonzero(s)) for s in solos]
    cents = [_centroid(s) for s in solos]
    used: set[int] = set()
    for i in range(len(parts)):
        if (i in used or cents[i] is None or areas[i] == 0
                or _mirror_trivial(parts[i].primitive)):
            continue
        for j in range(i + 1, len(parts)):
            if (j in used or cents[j] is None
                    or parts[j].op != parts[i].op
                    or parts[j].group != parts[i].group
                    or type(parts[j].primitive) is not type(parts[i].primitive)
                    or not 0.8 <= areas[j] / areas[i] <= 1.25):
                continue
            midx = (cents[i][0] + cents[j][0]) / 2.0
            midy = (cents[i][1] + cents[j][1]) / 2.0
            hit = None
            for vertical, uu in ((True, (0.0, 1.0)), (False, (1.0, 0.0))):
                mat = _reflection_m(midx, midy, *uu)
                ri = cv2.warpAffine(solos[i], mat, (w, h),
                                    flags=cv2.INTER_NEAREST, borderValue=0)
                if _iou_bool(ri, solos[j]) >= SYM_PAIR_IOU:
                    hit = vertical
                    break
            if hit is None:
                continue
            a = (midx if hit else midy) * f
            a_t = round(2.0 * a) / 2.0
            a2 = int(round(2.0 * a_t))
            d = a_t - a
            leader_of[j] = i
            pair_axis[j] = (hit, a2)
            dxy = (d, 0.0) if hit else (0.0, d)
            deltas[i] = dxy
            deltas[j] = dxy
            used.add(i)
            used.add(j)
            break
    return leader_of, pair_axis, deltas


# label collapse


def _fit_single_label(mask: np.ndarray, h: int, w: int
                      ) -> tuple[Primitive, float] | None:
    """Top-1 single primitive approximating a cluster silhouette, by
    min(area, boundary) IoU over a FIXED fitter pool (H5 — common.py fitters;
    _fit_rounded_rect admitted at min_aspect=1.0 for squircle faces, with
    _fit_ellipse as the honest backstop when its solidity gates fire).
    None when the best fit < COLLAPSE_MIN_IOU — the 'well-approximates' gate."""
    contour = _largest_contour(mask)
    if contour is None:
        return None
    cands: list[tuple[Primitive, np.ndarray]] = []
    rr = _fit_rounded_rect(contour, h, w, min_aspect=1.0)
    if rr is not None:
        cands.append((rr[0], rr[1]))
    e = _fit_ellipse(contour, h, w, 0, 0)
    cands.append((e[0], e[1]))
    re_ = _fit_rotated_ellipse(contour, h, w)
    if re_ is not None:
        cands.append((re_[0], re_[1]))
    r = _fit_rectangle(contour, h, w, 0, 0)
    cands.append((r[0], r[1]))
    rt = _fit_rotated_rect(contour, h, w)
    if rt is not None:
        cands.append((rt[0], rt[1]))
    t = _fit_triangle(contour, h, w, 0, 0)
    if t[2] > 0:                      # never None — degenerate flagged by area
        cands.append((t[0], t[1]))
    ic = _cand_inscribed_circle(mask, h, w)
    if ic is not None:
        cands.append((ic[0], ic[1]))
    band = _band_width(mask)
    best: tuple[Primitive, float] | None = None
    for prim, fit in cands:
        s = min(_iou_bool(fit, mask), _boundary_iou(fit, mask, band))
        if best is None or s > best[1]:
            best = (prim, s)
    return best if best is not None and best[1] >= COLLAPSE_MIN_IOU else None


def _radial_family(parts: list[Part], idxs: list[int]
                   ) -> tuple[int, list[int], dict] | None:
    """Detect a rotational family among a group's adds: a Circle hub + ≥
    RADIAL_MIN_BLADES same-kind elongated blades equidistant from the hub at
    near-uniform angular spacing (mirrors _fit_rot_spurs' gates, re-derived
    geometrically from the Part list — sunburst parts carry no family tag).
    Returns (hub index, blade indices, star params) or None."""
    hubs = [i for i in idxs if isinstance(parts[i].primitive, Circle)]
    if len(hubs) != 1:
        return None
    hub = hubs[0]
    hp = parts[hub].primitive
    blades = [i for i in idxs if i != hub]
    if len(blades) < RADIAL_MIN_BLADES:
        return None
    kinds = {type(parts[i].primitive).__name__ for i in blades}
    if len(kinds) != 1:
        return None
    cents = [prim_center(parts[i].primitive) for i in blades]
    dists = [math.hypot(cx - hp.cx, cy - hp.cy) for cx, cy in cents]
    md = sum(dists) / len(dists)
    if md <= 0 or math.sqrt(sum((d - md) ** 2 for d in dists) / len(dists)) / md \
            > RADIAL_DIST_CV:
        return None
    angs = sorted(math.degrees(math.atan2(cy - hp.cy, cx - hp.cx)) % 360.0
                  for cx, cy in cents)
    gaps = [(angs[(k + 1) % len(angs)] - angs[k]) % 360.0 for k in range(len(angs))]
    mg = sum(gaps) / len(gaps)
    if mg <= 0 or math.sqrt(sum((g - mg) ** 2 for g in gaps) / len(gaps)) / mg \
            > RADIAL_GAP_CV:
        return None
    n = len(blades)
    longs = [_local_dims(parts[i].primitive)[1] for i in blades]
    shorts = [_local_dims(parts[i].primitive)[0] for i in blades]
    r_outer = sorted(d + ln / 2.0 for d, ln in zip(dists, longs))[n // 2]
    # phase = circular mean of blade angles, computed modulo the pitch 360/n
    sx = sum(math.cos(math.radians(a * n)) for a in angs)
    sy = sum(math.sin(math.radians(a * n)) for a in angs)
    phase = math.degrees(math.atan2(sy, sx)) / n % (360.0 / n)
    # spoke half-width: the blade's angular half-extent at mid-radius — drives
    # the asterisk flavors (thin spokes, not fat half-pitch triangles)
    r_mid = max(1.0, (hp.r + r_outer) / 2.0)
    w_med = sorted(shorts)[n // 2]
    spoke_hw = math.degrees(math.atan2(w_med / 2.0, r_mid))
    # per-blade spikes for the NON-UNIFORM asterisk (organic sunbursts whose
    # blades sit on no uniform pitch — claude deviates ±8° from any 30° comb)
    per = sorted(
        (math.degrees(math.atan2(cy - hp.cy, cx - hp.cx)) % 360.0,
         math.hypot(cx - hp.cx, cy - hp.cy) + ln / 2.0)
        for (cx, cy), ln in zip(cents, longs))
    return hub, blades, {"cx": hp.cx, "cy": hp.cy, "n": n,
                         "r_outer": r_outer, "r_inner": hp.r, "phase": phase,
                         "spoke_hw": spoke_hw, "width": w_med,
                         "spikes": tuple(per)}


def _local_dims(p: Primitive) -> tuple[float, float]:
    """(min, max) local-frame dims — _budget_extent's measurement, both values."""
    if isinstance(p, (RotatedRectangle, RoundedRectangle)):
        return min(p.w, p.h), max(p.w, p.h)
    if isinstance(p, Circle):
        return 2.0 * p.r, 2.0 * p.r
    if isinstance(p, (Ellipse, RotatedEllipse)):
        return 2.0 * min(p.rx, p.ry), 2.0 * max(p.rx, p.ry)
    x0, y0, x1, y1 = prim_bbox(p)
    return min(x1 - x0, y1 - y0), max(x1 - x0, y1 - y0)


def _subrep_pairs(tsol: dict[int, np.ndarray], order_b: list[int],
                  cx_t: float, cy_t: float, target: int) -> list[float]:
    """Round 4 — per consecutive (wrapping) blade pair: the fraction of the
    pair's shared RADIAL span that is fused at target scale. Ring-quantized
    radius histogram (np.bincount over a precomputed integer radius map —
    library-scout verdict: warpPolar aliases 1-px blades at T ≤ 32); the
    contact test is the existing dilate∩ idiom. tsol masks are TARGET-scale
    with the hub zone excluded; cx_t/cy_t are the hub center in TARGET coords
    (canonical coords would shift every ring by ~6 px at T=16)."""
    yy, xx = np.indices((target, target))
    r_map = np.hypot(xx - cx_t, yy - cy_t).astype(np.int32)
    r_max = int(r_map.max()) + 1
    se3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bins = {i: np.bincount(r_map[tsol[i] > 0], minlength=r_max)
            for i in order_b}
    out: list[float] = []
    for k in range(len(order_b)):
        a, b = order_b[k], order_b[(k + 1) % len(order_b)]
        both = (bins[a] > 0) & (bins[b] > 0)
        denom = int(both.sum())
        if denom == 0:
            out.append(0.0)               # no shared radial span — open
            continue
        contact = cv2.dilate(tsol[a], se3) & tsol[b]
        if not contact.any():
            out.append(0.0)
            continue
        cb = np.bincount(r_map[contact > 0], minlength=r_max)
        out.append(float(((cb > 0) & both).sum()) / denom)
    return out


def _regularity_groups(parts: list[Part], canvas_hw: tuple[int, int],
                       skip: frozenset[int] = frozenset()
                       ) -> tuple[dict, list[Part]] | None:
    """Round 4 — detect LATENT translational regularity among adds (figma's
    equal stacked circles), pure primitive-field arithmetic, no rasters.
    Size groups: same-kind adds, ≥ 3 members (2-member sets are mirror-twin
    territory), relative extent spread ≤ REG_EXT_TOL → snap to the median.
    Alignment: group-member center coords within REG_ALIGN_TOL × S → snap to
    the cluster mean. Pitch: ≥ 3 row/column positions (cross-kind add
    centers, clustered at the same tolerance) with spacing CV ≤ REG_PITCH_CV
    → snap to exact uniform pitch about the mean. Returns (groups, snapped
    EXTRAPOLATED part list — same length/order/ops) or None."""
    s_tol = REG_ALIGN_TOL * max(canvas_hw)
    adds = [i for i, p in enumerate(parts)
            if p.op == "add" and i not in skip]
    if len(adds) < 3:
        return None
    by_kind: dict[str, list[int]] = {}
    for i in adds:
        by_kind.setdefault(type(parts[i].primitive).__name__, []).append(i)
    size_groups: list[list[int]] = []
    snapped: list[Primitive] = [p.primitive for p in parts]
    for kind in sorted(by_kind):
        mem = by_kind[kind]
        if len(mem) < 3:
            continue
        dims = [_local_dims(parts[i].primitive) for i in mem]
        med0 = sorted(d[0] for d in dims)[len(mem) // 2]
        med1 = sorted(d[1] for d in dims)[len(mem) // 2]
        if med0 <= 0 or med1 <= 0:
            continue
        if max(abs(d[0] - med0) for d in dims) / med0 > REG_EXT_TOL:
            continue
        if max(abs(d[1] - med1) for d in dims) / med1 > REG_EXT_TOL:
            continue
        size_groups.append(sorted(mem))
        for i in mem:                     # snap extents to the group median
            p = parts[i].primitive
            if isinstance(p, Circle):
                snapped[i] = replace(p, r=med0 / 2.0)
            else:
                d0, d1 = _local_dims(p)
                f0 = med0 / d0 if d0 > 0 else 1.0
                f1 = med1 / d1 if d1 > 0 else 1.0
                if isinstance(p, (RotatedRectangle, RoundedRectangle)):
                    fw, fh = (f0, f1) if p.w <= p.h else (f1, f0)
                    snapped[i] = replace(p, w=p.w * fw, h=p.h * fh)
                else:                     # uniform fallback for other kinds
                    snapped[i] = scale_about_center(p, (f0 + f1) / 2.0)

    def _cluster(vals: list[tuple[float, int]]) -> list[list[tuple[float, int]]]:
        out_c: list[list[tuple[float, int]]] = []
        for v, i in sorted(vals):
            if out_c and v - out_c[-1][0][0] <= s_tol:
                out_c[-1].append((v, i))
            else:
                out_c.append([(v, i)])
        return out_c

    # alignment within size groups: snap clustered center coords to the mean.
    # STRUCTURAL alignment only: members must be distinct positions on the
    # orthogonal axis — concentric parts (vscode's crossing X bars share
    # their center) "align" trivially and form no grid.
    align_clusters: list[tuple[str, list[int]]] = []
    for grp in size_groups:
        for ax in (0, 1):                 # 0 = cx, 1 = cy
            cl = _cluster([(prim_center(snapped[i])[ax], i) for i in grp])
            for c in cl:
                if len(c) < 2:
                    continue
                ortho = [prim_center(snapped[i])[1 - ax] for _v, i in c]
                if max(ortho) - min(ortho) <= s_tol:
                    continue              # concentric, not a column/row
                mean_v = sum(v for v, _i in c) / len(c)
                for v, i in c:
                    dx = (mean_v - v, 0.0) if ax == 0 else (0.0, mean_v - v)
                    snapped[i] = translate_prim(snapped[i], *dx)
                align_clusters.append(("x" if ax == 0 else "y",
                                       sorted(i for _v, i in c)))
    # uniform pitch over row/column positions (cross-kind add centers)
    pitch_groups: list[tuple[str, list[list[int]]]] = []
    for ax in (0, 1):
        rows = _cluster([(prim_center(snapped[i])[ax], i) for i in adds])
        if len(rows) < 3:
            continue
        pos = [sum(v for v, _i in r) / len(r) for r in rows]
        gapsv = [pos[k + 1] - pos[k] for k in range(len(pos) - 1)]
        mgap = sum(gapsv) / len(gapsv)
        if mgap <= 0 or math.sqrt(
                sum((g - mgap) ** 2 for g in gapsv) / len(gapsv)) / mgap \
                > REG_PITCH_CV:
            continue
        mid = sum(pos) / len(pos)
        for k, r in enumerate(rows):
            new_v = mid + (k - (len(rows) - 1) / 2.0) * mgap
            for _v0, i in r:
                # shift the member so its row coordinate lands on the comb
                # (alignment snaps already moved it — shift by the row delta)
                dv = new_v - prim_center(snapped[i])[ax]
                dxy = (dv, 0.0) if ax == 0 else (0.0, dv)
                snapped[i] = translate_prim(snapped[i], *dxy)
        pitch_groups.append(("x" if ax == 0 else "y",
                             [sorted(i for _v, i in r) for r in rows]))
    # regularity = REPEATED SHAPES plus structure. A size group alone is
    # not a latent grid (near-equal but unaligned members), and a pitch run
    # alone is not either — vscode's X symmetry makes its heterogeneous
    # part centers accidentally equi-spaced in y. The grid that earns
    # extrapolation is figma's: equal shapes, aligned, on a pitch.
    aligned_members = {i for _ax, cl in align_clusters for i in cl}
    size_groups = [grp for grp in size_groups
                   if any(i in aligned_members for i in grp)]
    if not size_groups:
        return None
    groups = {"size": size_groups, "align": align_clusters,
              "pitch": pitch_groups}
    vparts = [Part(snapped[i], parts[i].op, parts[i].group)
              for i in range(len(parts))]
    return groups, vparts


def _collapse_variants(parts: list[Part], roles: list[Role],
                       solos: list[np.ndarray], canvas_hw: tuple[int, int],
                       target: int
                       ) -> list[tuple[list[Part], str, dict[int, str],
                                       dict[int, int], str]]:
    """Collapse candidates per positive group (≥ 2 adds): ONE label fitted to
    the group silhouette M_g = (∪ add solos) AND-NOT (∪ shave solos) — shaves
    ABSORBED into the silhouette, counters RETAINED to re-carve the label.
    Radial families also offer star-polygon labels and (round 4) EXTRAPOLATED
    uniform forms. Each variant: (canonical-coords part list, label, absorbed
    {orig idx → reason}, index_map {variant idx → orig idx, label → −1},
    vkind "label" | "star_organic" | "extrapolate" | "decimate"). vkind
    routes round-4 reference substitution: "extrapolate"/"decimate" runs are
    searched and floor-judged against their OWN reference (idealized geometry
    / survivors); the arbitration key stays full-reference. Empty above
    COLLAPSE_MAX_T."""
    if target > COLLAPSE_MAX_T:
        return []
    h, w = canvas_hw
    out = []
    for g in sorted({p.group for p in parts if p.group >= 0}):
        idxs = [i for i, p in enumerate(parts) if p.group == g]
        adds = [i for i in idxs if parts[i].op == "add"]
        if len(adds) < 2:
            continue
        shaves = [i for i in idxs if parts[i].op == "subtract"
                  and roles[i] == "shave"]
        counters = [i for i in idxs if parts[i].op == "subtract"
                    and roles[i] == "counter"]
        others = [i for i in range(len(parts)) if parts[i].group != g]
        m = np.zeros((h, w), np.uint8)
        for i in adds:
            m |= solos[i]
        for i in shaves:
            m &= ~solos[i]

        def assemble(prim: Primitive, kind: str, score: float, vkind: str):
            absorbed = {i: f"absorbed into collapse[g{g}] {kind}"
                        for i in adds + shaves}
            vparts = ([Part(prim, "add", g)] + [parts[i] for i in counters]
                      + [parts[i] for i in others])
            imap = {0: -1}
            for k, i in enumerate(counters + others, start=1):
                imap[k] = i
            label = (f"g{g}:{kind} fit {score:.2f} "
                     f"({len(adds)}add+{len(shaves)}shave→1)")
            return vparts, label, absorbed, imap, vkind

        fit = _fit_single_label(m, h, w)
        if fit is not None:
            out.append(assemble(fit[0], type(fit[0]).__name__, fit[1],
                                "label"))
        fam = _radial_family(parts, adds)
        if fam is not None:
            hub_i, blades_f, sp = fam
            pitch = 360.0 / sp["n"]
            cands: list[Polygon] = [
                # non-uniform asterisk from the ACTUAL blade angles/tips —
                # organic sunbursts sit on no uniform pitch (claude ±8°)
                Polygon(pts=asterisk_polygon(sp["cx"], sp["cy"], sp["spikes"],
                                             sp["r_inner"], sp["width"])),
            ]
            phases = []
            for ph in (sp["phase"], 90.0 % pitch, 0.0):
                if not any(abs(ph - q) < 1e-6 for q in phases):
                    phases.append(ph)
            for ph in phases:                # uniform flavors, phase-snapped
                for hw_ in (sp["spoke_hw"], None):
                    cands.append(Polygon(pts=star_polygon(
                        sp["cx"], sp["cy"], sp["n"], sp["r_outer"],
                        sp["r_inner"], ph, spoke_halfwidth_deg=hw_)))
            best_star = None
            for star in cands:
                sm = _raster_part(star, h, w)
                # AREA IoU only: the star is an IDEALIZATION — its boundary
                # is deliberately simplified (sharp points for round blade
                # tips), so the canonical boundary band cannot coincide; the
                # honest boundary test is the target-scale arbitration.
                s = _iou_bool(sm, m)
                if best_star is None or s > best_star[1]:
                    best_star = (star, s)
            if best_star is not None and best_star[1] >= COLLAPSE_MIN_IOU:
                out.append(assemble(best_star[0], f"star{sp['n']}",
                                    best_star[1], "star_organic"))
            # blade decimation (round 3) — round-4 EVIDENCE gate: a pair is
            # sub-representable iff its fused span covers ≥ GAP_FUSE_FRAC of
            # the pair's shared radial span (the old any-8-adjacency test
            # over-fired on claude@32's organic spacing — open item (b)).
            ft = target / max(h, w)
            hub_t = _raster_part(_scale_part(parts[hub_i], ft).primitive,
                                 target, target)
            se3l = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            hub_zone = cv2.dilate(hub_t, se3l, iterations=2)
            tsol = {i: _raster_part(_scale_part(parts[i], ft).primitive,
                                    target, target) & ~hub_zone
                    for i in blades_f}
            cx_t = parts[hub_i].primitive.cx * ft
            cy_t = parts[hub_i].primitive.cy * ft
            order_b = [i for _a, i in sorted(
                (math.degrees(math.atan2(
                    prim_center(parts[i].primitive)[1]
                    - parts[hub_i].primitive.cy,
                    prim_center(parts[i].primitive)[0]
                    - parts[hub_i].primitive.cx)) % 360.0, i)
                for i in blades_f)]
            fr = _subrep_pairs(tsol, order_b, cx_t, cy_t, target)
            sub = sum(1 for f_ in fr if f_ >= GAP_FUSE_FRAC)
            subrep = (sub * 2 >= len(order_b)
                      and len(order_b) >= RADIAL_MIN_BLADES)

            def decim(keep: list[int], depth: int):
                dropped_b = [i for i in order_b if i not in keep]
                vparts = ([parts[hub_i]] + [parts[i] for i in keep]
                          + [parts[i] for i in counters]
                          + [parts[i] for i in others])
                absorbed = {i: (f"decimated: {sub}/{len(order_b)} gaps "
                                f"sub-representable at T={target}")
                            for i in dropped_b}
                imap = {0: hub_i}
                for k2, i in enumerate(keep + counters + others, start=1):
                    imap[k2] = i
                return (vparts,
                        f"g{g}:decimate {len(order_b)}→{len(keep)} blades",
                        absorbed, imap, "decimate")

            if subrep:
                keep2 = order_b[::2]
                out.append(decim(keep2, 2))
                # ladder: re-test the KEPT subfamily at its doubled pitch
                fr2 = _subrep_pairs(tsol, keep2, cx_t, cy_t, target)
                sub2 = sum(1 for f_ in fr2 if f_ >= GAP_FUSE_FRAC)
                if sub2 * 2 >= len(keep2) and len(order_b[::3]) >= 3:
                    out.append(decim(order_b[::3], 3))
            # round 4: family-harmonized multi-part EXTRAPOLATION, for
            # IRREGULAR families only (a family already on an exact comb —
            # asterisk6 — needs no extrapolation; its round-3 paths stand).
            # Blades rebuilt UNIFORM (median dims, median radial distance,
            # comb angles) at the REPRESENTABLE width — max(median,
            # min-feature at target): extrapolating to a thinner-than-
            # representable ideal makes the variant's own reference
            # unsatisfiable by construction (pockets/exclusivity judged
            # against gaps the render can never keep). Depth follows the
            # SAME per-gap evidence as decimation: full-n comb only when
            # gaps are representable, n//2 comb only when they are not
            # (the user-locked 12@32 / 6@16 doctrine).
            gaps_o = [(  # organic gap deviations from the comb pitch
                sorted(math.degrees(math.atan2(
                    prim_center(parts[i].primitive)[1]
                    - parts[hub_i].primitive.cy,
                    prim_center(parts[i].primitive)[0]
                    - parts[hub_i].primitive.cx)) % 360.0
                    for i in blades_f))][0]
            gaps_d = [(gaps_o[(k + 1) % len(gaps_o)] - gaps_o[k]) % 360.0
                      for k in range(len(gaps_o))]
            wid_all = sorted(_local_dims(parts[i].primitive)[0]
                             for i in order_b)
            wmed0 = wid_all[len(wid_all) // 2]
            irregular = (max(abs(gv - pitch) for gv in gaps_d)
                         > RADIAL_REG_TOL_DEG
                         or (wmed0 > 0 and max(abs(wv - wmed0)
                                               for wv in wid_all) / wmed0
                             > REG_EXT_TOL))
            bkind = type(parts[blades_f[0]].primitive)
            if irregular and bkind in (RotatedRectangle, RoundedRectangle):
                hp = parts[hub_i].primitive
                dists_f = [math.hypot(
                    prim_center(parts[i].primitive)[0] - hp.cx,
                    prim_center(parts[i].primitive)[1] - hp.cy)
                    for i in order_b]
                d_med = sorted(dists_f)[len(dists_f) // 2]
                lens_f = sorted(_local_dims(parts[i].primitive)[1]
                                for i in order_b)
                len_med = lens_f[len(lens_f) // 2]
                # width floors differ by path: representable-evidence combs
                # keep THIN blades (the naive's 12 distinguishable spokes —
                # just enough to raster); the decimated comb exists because
                # space ran out, so its fewer blades take min-feature width
                wid_eff = max(sp["width"],
                              (1.0 if not subrep else _min_feature_px(target))
                              / ft)

                # blades extend INWARD past the hub edge — CSG union absorbs
                # the overlap, and attachment stops depending on raster phase
                # (inner ends exactly at the hub edge detach at thin widths)
                ext_in = hp.r * 0.75
                d_eff = d_med - ext_in / 2.0
                len_eff = len_med + ext_in

                def _comb_blade(theta_deg: float) -> Primitive:
                    th = math.radians(theta_deg)
                    bcx = hp.cx + d_eff * math.cos(th)
                    bcy = hp.cy + d_eff * math.sin(th)
                    ang = theta_deg % 180.0
                    if bkind is RoundedRectangle:
                        return RoundedRectangle(
                            cx=bcx, cy=bcy, w=len_eff, h=wid_eff, angle=ang,
                            corner_radius=wid_eff / 2.0)
                    return RotatedRectangle(cx=bcx, cy=bcy, w=len_eff,
                                            h=wid_eff, angle=ang)

                nfam = len(order_b)
                if not subrep:
                    vparts2 = list(parts)
                    for k2, i in enumerate(order_b):
                        vparts2[i] = Part(_comb_blade(sp["phase"]
                                                      + k2 * (360.0 / nfam)),
                                          parts[i].op, parts[i].group)
                    out.append((vparts2, f"g{g}:harmonize {nfam} blades",
                                {}, {k2: k2 for k2 in range(len(parts))},
                                "extrapolate"))
                elif nfam // 2 >= 3:
                    n2f = nfam // 2
                    keep_h = order_b[:n2f]    # rebuilt in place on the comb
                    drop_h = order_b[n2f:]
                    vparts3 = ([parts[hub_i]]
                               + [Part(_comb_blade(sp["phase"]
                                                   + k2 * (360.0 / n2f)),
                                       "add", g)
                                  for k2 in range(n2f)]
                               + [parts[i] for i in counters]
                               + [parts[i] for i in others])
                    absorbed3 = {i: (f"decimated: {sub}/{nfam} gaps "
                                     f"sub-representable at T={target}")
                                 for i in drop_h}
                    imap3 = {0: hub_i}
                    for k2, i in enumerate(keep_h + counters + others,
                                           start=1):
                        imap3[k2] = i
                    out.append((vparts3,
                                f"g{g}:harmonize {nfam}→{n2f} blades",
                                absorbed3, imap3, "decimate"))
    return out


# the grid-fit search


@dataclass
class _Run:
    """One construction's search outcome (incumbent or collapse variant)."""
    mask: np.ndarray
    records: list[TweakRecord]
    drops: list[str]
    violations: list[str]
    sym: float | None
    axis: tuple[str, float] | None
    n_live: int
    pairs: int = 0
    battery: bool = True      # ABSOLUTE sep/pocket/pron state of the final mask
    glob_offenders: list[int] = field(default_factory=list)   # round 4: adds
    #                           implicated in the glob (run-local indices)
    viol_class: dict[int, str] = field(default_factory=dict)  # round 4:
    #                           part → "glob" | "floor" | "other"


def render_debug(parts: list[Part], canvas_hw: tuple[int, int],
                 target: int) -> RenderResult:
    """Full render with per-part tweak records (the harness's entry point).
    parts: Decomposition.parts (canonical padded canvas coords, SQUARE canvas).
    target: hicolor size. The incumbent multi-part render and (at T ≤
    COLLAPSE_MAX_T) any collapse variants each run the same pair-aware search
    and compete on (feasible, jag, fidelity, fewer-parts)."""
    h, w = canvas_hw
    if h != w:
        raise ValueError(f"render expects a square canvas, got {canvas_hw}")
    ref = reference_mask(parts, canvas_hw, target)
    naive = naive_mask(parts, canvas_hw, target)
    floor = floor_for(target)
    min_feat = _min_feature_px(target)
    band = _band_width(ref)

    if not parts:
        blank = np.zeros((target, target), np.uint8)
        return RenderResult(blank, ref, naive, [], 0.0, jag_score(naive), 0.0,
                            jag_norm(naive), 0.0, 0.0, floor)

    f = target / max(h, w)
    canon_solos = [_raster_part(p.primitive, h, w) for p in parts]
    roles_orig = infer_roles(parts, canvas_hw)
    topo_ref_v = _topology(ref)
    topo_canon_v = _topology(_compound_mask(parts, h, w))
    pockets = _pocket_masks(ref)          # anti-globbing: the protected background
    naive_jag = jag_score(naive)          # the refiner fire-gate's ratio anchor

    # Feed-resolution dispatch: T <= 24 renders via the pixel-cubify operator
    # (naive raster -> 1px-brush local search), NOT geometry straightening. The
    # decomposition/roles/pockets above become cubify's constraints; the geometry
    # _run/arbitration machinery below is retained only for T > 24 (synthetic
    # regression). Lazy import breaks the render<->cubify cycle.
    if target <= CUBIFY_MAX_T:
        from .cubify import _glyph_stroke_relayout, cubify
        # W4 — stroke-glyph re-layout (medium's M): a deliberate idealization is
        # scored against its OWN reference (the round-4/5 ref-substitution
        # doctrine); fidelity vs the ORIGINAL reference is reported alongside
        # (fid_ref / "extrapolate") so the idealization cost stays auditable.
        pp = _glyph_stroke_relayout(parts, canvas_hw, target)
        fid_orig: float | None = None
        kind: str | None = None
        oref = ref
        if pp is not parts:
            ref = reference_mask(pp, canvas_hw, target)
            naive = naive_mask(pp, canvas_hw, target)
            band = _band_width(ref)
            topo_ref_v = _topology(ref)
            topo_canon_v = _topology(_compound_mask(pp, h, w))
            naive_jag = jag_score(naive)
        # Cross-resolution structure constraint: a smaller feed size uses the largest feed
        # size (@24) as its structural reference (the spine connected throughout). @24 itself
        # has no larger sibling → renders normally (structure_ref=None, byte-identical).
        structure_ref = (cubify(_glyph_stroke_relayout(parts, canvas_hw, CUBIFY_MAX_T)
                                if pp is not parts else parts,
                                canvas_hw, CUBIFY_MAX_T)
                         if target < CUBIFY_MAX_T else None)
        cmask = cubify(pp, canvas_hw, target, structure_ref)
        if pp is not parts:
            fid_orig = min(_iou_bool(cmask, oref),
                           _boundary_iou(cmask, oref, _band_width(oref)))
            kind = "extrapolate"
        return RenderResult(
            mask=cmask, ref=ref, naive=naive, records=[],
            jag=jag_score(cmask), jag_naive=naive_jag,
            jag_norm=jag_norm(cmask), jag_norm_naive=jag_norm(naive),
            area_iou=_iou_bool(cmask, ref),
            boundary_iou=_boundary_iou(cmask, ref, band),
            floor=floor,
            topo_render=_topology(cmask),
            topo_ref=topo_ref_v, topo_canon=topo_canon_v,
            staircase=_staircase_total(cmask),
            fid_ref=fid_orig, ref_kind=kind,
        )

    coverage = reference_coverage(parts, canvas_hw, target)   # refiner target
    # radial family per positive group (spacing harmonization + decimation)
    radial = None
    for g in sorted({p.group for p in parts if p.group >= 0}):
        adds_g = [i for i, p in enumerate(parts)
                  if p.group == g and p.op == "add"]
        fam = _radial_family(parts, adds_g)
        if fam is not None:
            radial = fam
            break

    def fidelity(mask: np.ndarray) -> float:
        return min(_iou_bool(mask, ref), _boundary_iou(mask, ref, band))

    # round 4 — latent translational regularity (figma's equal stacked
    # circles): detected once on the ORIGINAL parts; the snapped part list
    # becomes the extrapolated variant, the groups drive the render audit.
    # A radial family OWNS its blades — they are the comb machinery's
    # territory, not the translational detector's (asterisk blades would
    # otherwise double-register as a size group).
    fam_skip = (frozenset({radial[0], *radial[1]}) if radial is not None
                else frozenset())
    reg_detect = _regularity_groups(parts, canvas_hw, skip=fam_skip)

    def _run(pp: list[Part], pp_roles: list[Role], pp_solos: list[np.ndarray],
             tag, enable_sym: bool = True,
             ref_run: np.ndarray | None = None,
             pockets_run: list[np.ndarray] | None = None,
             topo_ref_run: tuple[int, int] | None = None,
             reg_check: bool = False) -> _Run:
        # round 4 — reference substitution: extrapolated/decimated/removal
        # variants are SEARCHED and floor-judged against their OWN reference
        # (a uniform star fills different gap-pockets than the organic
        # source); the arbitration key + report stay full-reference. Defaults
        # keep every pre-round-4 caller byte-identical.
        ref_l = ref if ref_run is None else ref_run
        band_l = band if ref_run is None else _band_width(ref_l)
        pockets_l = pockets if pockets_run is None else pockets_run
        topo_l = topo_ref_v if topo_ref_run is None else topo_ref_run
        # fam_run: this reference-substituted run renders a radial family
        # (comb). The comb-scale battery relaxations below are scoped to it —
        # they must NOT leak to non-family extrapolated runs (figma's
        # regularize needs the per-pocket veto + boundary policing).
        fam_run = False
        fam_blades: list[int] = []
        if ref_run is not None and target <= COLLAPSE_MAX_T:
            for g2f in sorted({p.group for p in pp if p.group >= 0}):
                adds_g2f = [q for q in range(len(pp))
                            if pp[q].group == g2f and pp[q].op == "add"]
                famr0 = _radial_family(pp, adds_g2f)
                if famr0 is not None:
                    fam_run = True
                    fam_blades.extend(famr0[1])
        if fam_run:
            # significance floor: a comb pocket below 2×POCKET_MIN cannot
            # stay half-open once blades render at min-feature widths —
            # keeping it in the battery fails every honest comb render
            pockets_l = [pk for pk in pockets_l
                         if int(np.count_nonzero(pk)) >= 2 * POCKET_MIN]
        pk_ns = [int(np.count_nonzero(pk)) for pk in pockets_l]
        pk_total = max(1, sum(pk_ns))

        def fidelity_l(mask: np.ndarray) -> float:
            return min(_iou_bool(mask, ref_l), _boundary_iou(mask, ref_l, band_l))

        n = len(pp)
        violations: list[str] = []
        parts_t = [_scale_part(_floatify(p), f) for p in pp]

        # symmetry prep: global axis, else local mirror pairs
        leader_of: dict[int, int] = {}
        self_paired: set[int] = set()
        pair_axis: dict[int, tuple[bool, int]] = {}
        glob: tuple[bool, float, int] | None = None   # (vertical, a_t, a2)
        if enable_sym:
            sp = _sym_prep(pp, pp_solos, canvas_hw, target)
            if sp is not None:
                vertical, a_t, a2, delta, leader_of, self_paired = sp
                glob = (vertical, a_t, a2)
                if abs(delta) > 1e-9:
                    dxy = (delta, 0.0) if vertical else (0.0, delta)
                    parts_t = [Part(translate_prim(p.primitive, *dxy),
                                    p.op, p.group) for p in parts_t]
                pair_axis = {j: (vertical, a2) for j in leader_of}
            else:
                leader_of, pair_axis, pdeltas = _local_pairs(
                    pp, pp_solos, canvas_hw, target)
                for i, dxy in pdeltas.items():
                    if abs(dxy[0]) > 1e-9 or abs(dxy[1]) > 1e-9:
                        parts_t[i] = Part(translate_prim(parts_t[i].primitive,
                                                         *dxy),
                                          parts_t[i].op, parts_t[i].group)
        followers = set(leader_of)
        followers_of: dict[int, list[int]] = {}
        for j in sorted(leader_of):
            followers_of.setdefault(leader_of[j], []).append(j)

        true_solos = [_raster_part(p.primitive, target, target) for p in parts_t]
        extents = []
        extents_xy = []
        for p in parts_t:
            x0, y0, x1, y1 = prim_bbox(p.primitive)
            extents_xy.append((max(0.0, x1 - x0), max(0.0, y1 - y0)))
            extents.append(max(0.0, min(x1 - x0, y1 - y0)))

        # group add-unions of TRUE target-scale forms (bite/parent reference)
        group_union: dict[int, np.ndarray] = {}
        for g in sorted({p.group for p in pp}):
            u = np.zeros((target, target), np.uint8)
            for i, p in enumerate(pp):
                if p.group == g and p.op == "add":
                    u |= true_solos[i]
            group_union[g] = u

        # min-feature pass: drop / clamp / force-merge BEFORE the search;
        # PAIR-COHERENT — the leader decides for its mirror twin
        drop_reason: dict[int, str] = {}

        def _drop(i: int, reason: str) -> None:
            drop_reason[i] = reason
            for jj in followers_of.get(i, []):
                drop_reason[jj] = (f"{tag(jj)} dropped with mirrored leader "
                                   f"{tag(i)}")

        for i, p in enumerate(pp):
            if i in followers:
                continue                  # leader decides for the pair
            if extents[i] >= min_feat and np.count_nonzero(true_solos[i]):
                continue
            role = pp_roles[i]
            if role in ("dominant", "lobe"):
                continue                  # size candidates clamp UP, never drop
            if role == "accent" and p.op == "add":
                # round 3 — pronouncedness exemption: a thin feature whose
                # PROTRUSION beyond the other ink is significant is identity
                # (tab_body's tab, vscode's tails) — clamp up, never drop
                body = np.zeros((target, target), np.uint8)
                for q, p2 in enumerate(pp):
                    if q != i and p2.op == "add" and q not in drop_reason:
                        body |= true_solos[q]
                if np.count_nonzero(body) and np.count_nonzero(true_solos[i]):
                    dt = cv2.distanceTransform(cv2.bitwise_not(body),
                                               cv2.DIST_L2,
                                               cv2.DIST_MASK_PRECISE)
                    if float(dt[true_solos[i] > 0].max()) >= PRON_MIN_DEPTH:
                        continue
            if role == "counter":
                # force-merge: hole rounded up to min_feat must leave ≥1 px of
                # surrounding material (erode parent 3×3); else drop the hole.
                up = scale_about_center(parts_t[i].primitive,
                                        min_feat / extents[i]
                                        if extents[i] > 0 else 1.0)
                hole = _raster_part(up, target, target)
                se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                shell = cv2.erode(group_union[p.group], se,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
                if np.count_nonzero(hole & ~shell) == 0 and np.count_nonzero(hole):
                    continue              # merge survives; search clamps the size
                _drop(i, f"{tag(i)} counter below min-feature "
                         f"(topology would break)")
            else:
                _drop(i, f"{tag(i)} {role} below min-feature")
        dropped = [i in drop_reason for i in range(n)]
        live = [i for i in range(n) if not dropped[i]]

        # candidate sets + initial state (most faithful within budget).
        # Degrees ABOVE the budget stay as ESCALATION options (the floor
        # outranks the straightening doctrine). Budgets key on LOCAL-frame
        # extents (length for elongated strokes); followers carry no candidate
        # set — their solo is always the flip of their leader's.
        bud_ext = [_budget_extent(p.primitive) for p in parts_t]
        budgets = [allowed_degree(pp_roles[i], bud_ext[i]) for i in range(n)]
        # round 4 — round-identity curvature floor: a non-dominant Circle in
        # a rectilinear-dominant construction is the distinguishing round
        # form (figma's circle squared at 32) — it renders at least as the
        # chamfered octagon where the extent permits; below the extent floor
        # the grid cannot show roundness and the rule stands down.
        deg_min = [0] * n
        dom_kind: dict[int, type] = {}
        for g2 in sorted({p.group for p in pp}):
            di = next((q for q in range(n) if pp[q].group == g2
                       and pp_roles[q] == "dominant"), None)
            if di is not None:
                dom_kind[g2] = type(pp[di].primitive)
        for i in range(n):
            if (pp[i].op == "add" and pp_roles[i] in ("lobe", "accent")
                    and isinstance(pp[i].primitive, Circle)
                    and dom_kind.get(pp[i].group) in (
                        Rectangle, RotatedRectangle, RoundedRectangle,
                        Polygon)):
                if extents[i] >= ROUND_ID_MIN_EXTENT:
                    deg_min[i] = 2
                    budgets[i] = max(budgets[i], 2)
                elif extents[i] >= 6.0:
                    # round 7 — represented geometry: between 6 px and the
                    # octagon floor a CHAMFER is still representable (a
                    # 1-px corner cut leaves ≥2-px straight runs); a bare
                    # D0 square erases the circle's identity (figma's
                    # 6.5-px circles at 24 rendered as squares). Below 6
                    # the cut IS the edge — chamfered 4-px circles measured
                    # as mush (figma@16 0.706 → 1.132 LOSS at a 4.0 bound).
                    deg_min[i] = 1
                    budgets[i] = max(budgets[i], 1)
        # round 4 — comb identity is ROTATIONAL: family blades in
        # reference-substituted runs keep at least D1 (slope-grid Θ angle);
        # D0's 90° snap axis-aligns every blade and the comb collapses into
        # a "#" grid. fam_run scopes the comb-scale battery relaxations —
        # they must NOT leak to non-family extrapolated runs (figma's
        # regularize needs the per-pocket veto + boundary term that police
        # silhouette spill).
        for q in fam_blades:
            deg_min[q] = max(deg_min[q], 1)
            budgets[q] = max(budgets[q], 1)
        # round 4 — rounded identity (the RRect sibling of the round-
        # identity floor): an RRect add whose corner radius is representable
        # at target keeps at least the D1 chamfer — D0's bare box erases the
        # corners, and figma's pill corner IS the upper half of its armpit
        # dent. Round 7 — the REPRESENTED-GEOMETRY heuristic (user ruling):
        # the floor describes the PART, not the run kind — scoping it to
        # ref-runs made figma's pill chamfer at 16/24 (regularize won) but
        # box at 32 (incumbent won); identity-bearing curvature renders at
        # every scale at its representable degree because the geometry is
        # what separates components visually. cr < 1 px stays the only
        # stand-down.
        for q in range(n):
            if (pp[q].op == "add"
                    and isinstance(pp[q].primitive, RoundedRectangle)
                    and parts_t[q].primitive.corner_radius >= 1.0):
                deg_min[q] = max(deg_min[q], 1)
                budgets[q] = max(budgets[q], 1)
        degree_sets = [
            [DEG_TRUE] if bud_ext[i] >= FORCE_TRUE_EXTENT
            else [d for d in range(DEG_TRUE + 1) if d >= deg_min[i]]
            for i in range(n)
        ]
        # round 5 — radial families are the comb machinery's territory: their
        # blades do NOT get L-corner reductions (rotational identity; the
        # asterisk6/claude byte-compare guard). Ref-runs are already excluded
        # via the deg_min ≥ 1 blade floor (`0 in degrees` gate). Round 6 —
        # the same exclusion scopes the run-≥2 slope bases: snapping blades
        # off the comb angles would break the family's angular pitch.
        no_lcorner: set[int] = set()
        if pp is parts and radial is not None:
            no_lcorner = {q for q in radial[1] if q < n}
        cand_sets: list[list] = [
            _candidates(parts_t[i], degree_sets[i], extents_xy[i], min_feat,
                        pp[i].op == "add",
                        narrow=(pp[i].op == "add"
                                and pp_roles[i] in ("lobe", "accent")),
                        axis_reduce=i not in no_lcorner,
                        run2=i not in no_lcorner)
            if not dropped[i] and i not in followers else []
            for i in range(n)
        ]
        # round 4 — extrapolated runs render the IDEALIZED geometry: the
        # snapped/comb form IS the answer, so the search's only legitimate
        # freedoms are degree and raster phase. Free sizes and drifts are
        # how jag-first re-globs the ideal (figma@16: the pill shrank to
        # fy 0.68, parent-follow dragged the circles up, the silhouette
        # step filled). Keep: uniform sizes at 1.0 (plus the up-clamp for
        # sub-representable parts), half-px snaps.
        if ref_run is not None:
            for i in range(n):
                if not cand_sets[i]:
                    continue
                poly5 = isinstance(pp[i].primitive, Polygon)
                kept_c = [c for c in cand_sets[i]
                          if abs(c[2] - c[3]) < 1e-9
                          # ≤1px extent adjustment: the round±1 PARITY
                          # candidates stay (a 4.4px circle must be able to
                          # raster 4px, not 5) — aggressive shrinks go.
                          # Polygon parts are PRE-IDEALIZED (round-5 grid-
                          # snapped reductions): exactly as built, f = 1
                          and ((abs(c[2] - 1.0) < 1e-9) if poly5 else
                               (abs(c[2] * extents[i] - extents[i])
                                <= 1.0 + 1e-9
                                or (extents[i] < min_feat and c[2] > 1.0)))
                          and ((abs(c[4]) < 1e-9 and abs(c[5]) < 1e-9)
                               if poly5 else
                               (abs(c[4]) <= 0.5 and abs(c[5]) <= 0.5))]
                if kept_c:
                    cand_sets[i] = kept_c
        cand_rasters: list[list[np.ndarray]] = [
            [_raster_part(c.primitive, target, target) for c, *_r in cand_sets[i]]
            for i in range(n)
        ]

        # self-paired (on-axis) parts: only flip-invariant candidates may
        # render — an even-width box on an integer axis CANNOT raster
        # symmetric; the round±1 parity targets guarantee an alternative.
        allowed_k: dict[int, set[int]] = {}
        if glob is not None:
            for i in sorted(self_paired):
                if dropped[i] or not cand_sets[i]:
                    continue
                ks = {k for k, r in enumerate(cand_rasters[i])
                      if np.array_equal(r, _flip_mask(r, glob[2], glob[0]))}
                if ks:
                    allowed_k[i] = ks
                else:
                    violations.append(f"{tag(i)}: no flip-invariant candidate "
                                      f"at T={target} — symmetry filter disabled")

        choice: list[int] = []
        for i in range(n):
            if dropped[i] or not cand_sets[i]:
                choice.append(-1)
                continue
            want = (budgets[i], 1.0, 1.0)
            init = next((k for k, (_c, d, fx, fy, dx, dy)
                         in enumerate(cand_sets[i])
                         if (d, fx, fy) == want and dx == 0.0 and dy == 0.0
                         and (i not in allowed_k or k in allowed_k[i])), None)
            if init is None:
                init = min(allowed_k[i]) if i in allowed_k else 0
            choice.append(init)

        def solo_now(i: int, over: int | None = None,
                     cand_mask: np.ndarray | None = None) -> np.ndarray:
            if i in followers:
                ld = leader_of[i]
                v, a2p = pair_axis[i]
                base = cand_mask if ld == over else cand_rasters[ld][choice[ld]]
                return _flip_mask(base, a2p, v)
            return cand_mask if i == over else cand_rasters[i][choice[i]]

        def compound(over: int | None = None,
                     cand_mask: np.ndarray | None = None) -> np.ndarray:
            return _compound_from_masks(
                [(pp[i].op, pp[i].group, solo_now(i, over, cand_mask))
                 for i in live], target, target)

        def chosen_parts() -> list[Part]:          # records only, never scored
            return [cand_sets[i][choice[i]][0] for i in live
                    if i not in followers and cand_sets[i]]

        def retention_ok(global_mask: np.ndarray,
                         narrowed: set[int] = frozenset()) -> bool:
            for q in live:
                if pp[q].op != "add":
                    continue
                # round 4 — significance floor (feasibility-saturation
                # guard): in reference-substituted runs, sub-min-feature
                # parts' pixel-count retention measures rasterization PHASE,
                # not identity — min-feature/pronouncedness govern them.
                # LOCAL short dim, not bbox: a diagonal thin blade has a fat
                # bbox but is still a sub-min-feature stroke. ref-run wide,
                # NOT fam-scoped: 1-px axis-run strokes (round 5) are sub-
                # min-feature BY DESIGN; the < min_feat gate already keeps
                # figma-class parts fully protected
                if (ref_run is not None
                        and _local_dims(parts_t[q].primitive)[0] < min_feat):
                    continue
                solo = true_solos[q]
                tot = int(np.count_nonzero(solo))
                if tot == 0:
                    continue
                vis_ref = int(np.count_nonzero(ref_l & solo)) / tot
                vis = int(np.count_nonzero(global_mask & solo)) / tot
                # narrowed-with-pronouncedness-held parts trade area for
                # protrusion (round 3): the relief that lets a wing render
                # thin without tripping the coverage floor
                frac = RETAIN_NARROW if q in narrowed else RETAIN_FRAC
                if vis < frac * vis_ref:
                    return False
            return True

        # distinctness: retention measures COVERAGE, not identity — a part
        # absorbed into a sibling is still "covered". Every add must keep
        # ≥ RETAIN_FRAC of its EXCLUSIVE territory (pixels only it covers in
        # the true forms), or the jag-first objective fuses identity-bearing
        # twins (figma's stacked circles, vscode's X bars) the moment pairing
        # lets one choice move both at once.
        add_live = [i for i in live if pp[i].op == "add"]
        excl_ref: dict[int, int] = {}
        if len(add_live) >= 2:
            cnt0 = np.zeros((target, target), np.uint16)
            for q in add_live:
                cnt0 += (true_solos[q] > 0).astype(np.uint16)
            for q in add_live:
                excl_ref[q] = int(np.count_nonzero(
                    (true_solos[q] > 0) & (cnt0 == 1)))

        def _excl_now(over: int | None = None,
                      cand_mask: np.ndarray | None = None) -> dict[int, int]:
            solos = {q: solo_now(q, over, cand_mask) for q in add_live}
            cnt = np.zeros((target, target), np.uint16)
            for q in add_live:
                cnt += (solos[q] > 0).astype(np.uint16)
            return {q: int(np.count_nonzero((solos[q] > 0) & (cnt == 1)))
                    for q in add_live}

        def distinct_ok(own: set[int], baseline: dict[int, int],
                        over: int | None = None,
                        cand_mask: np.ndarray | None = None) -> bool:
            """NEWLY-VIOLATED semantics: a candidate is infeasible iff it breaks
            ITS OWN parts' exclusivity (`own` = the evaluated part + its
            followers) or pushes another add below threshold that was above it
            in the `baseline` state. Gating on the whole state deadlocks the
            descent when the INITIAL state is already fused (claude's 12
            D0-snapped blades) — every candidate of every part 'fails' and the
            search collapses into fallback."""
            if len(add_live) < 2:
                return True
            now = _excl_now(over, cand_mask)
            for q in add_live:
                lim = RETAIN_FRAC * excl_ref.get(q, 0)
                if excl_ref.get(q, 0) < 4:
                    continue              # nothing meaningful to protect
                if (ref_run is not None
                        and _local_dims(parts_t[q].primitive)[0] < min_feat):
                    continue              # round 4 significance floor (see
                    #                       retention_ok)
                if now[q] >= lim:
                    continue
                if q in own:
                    # own-hard, unconditionally. Round 6 probed a newly-
                    # violated-or-worsened soften here (for figma@32's
                    # spine deadlock) and REVERTED it: it never freed the
                    # spine (the regularize variant's fid_ref floor blocks
                    # that cell regardless — the harmonization fix is what
                    # greened figma@24), and it perturbed accepted cells
                    # (claude@16's comb blades drifted to viol; thin_x@16's
                    # accent drifted to battery [retention]/[conn]). A
                    # part's own exclusivity is its identity.
                    if os.environ.get("R1_DEBUG"):
                        print(f"          distinct-own fail {tag(q)} "
                              f"excl_ref={excl_ref.get(q, 0)} "
                              f"now={now[q]} lim={lim:.1f} "
                              f"base={baseline.get(q, 0)}")
                    return False
                if baseline.get(q, 0) >= lim:
                    if os.environ.get("R1_DEBUG"):
                        print(f"          distinct-sib fail {tag(q)} "
                              f"now={now[q]} lim={lim:.1f} "
                              f"base={baseline.get(q, 0)}")
                    return False          # newly broke a sibling
            return True

        # round-3 constraint battery: separation, pockets, pronouncedness —
        # all newly-violated vs a per-part-optimization snapshot
        se3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sep_pairs: list[tuple[int, int]] = []
        if len(add_live) >= 2:
            dil0 = {q: cv2.dilate(true_solos[q], se3) for q in add_live}
            for ii in range(len(add_live)):
                for jj in range(ii + 1, len(add_live)):
                    a, b = add_live[ii], add_live[jj]
                    if not np.count_nonzero(dil0[a] & true_solos[b]):
                        sep_pairs.append((a, b))     # well separated: G ≥ 1 px

        # round 8 — the subset of sep_pairs that are also SEPARATE ON THE
        # OUTLINE (a deep silhouette defect between them): enforced ABSOLUTELY
        # below, because the ordinary newly-violated sep clause cannot reopen
        # a gap the baseline already fused (figma's domes at 16/24).
        # round 8 — outline-separation is scoped to the LOWEST resolution
        # (user: "on the lowest resolution"). At T ≥ 24 the domes are large
        # enough that the ordinary representable-separation machinery applies;
        # forcing a 1-px notch open there only breaks feasibility (figma@24
        # domes fall [glob], the cell was already separable/green). 16 is
        # where the grid genuinely cannot show the split without help.
        outline_gaps = _outline_gaps(
            ref_l, pp, true_solos, add_live,
            set(radial[1]) if radial is not None else set(), target
            ) if target <= 16 else []
        if os.environ.get("R1_DEBUG") and outline_gaps:
            print(f"          outline_gaps={[(int((g>0).sum()), sorted(fl)) for g,fl in outline_gaps]} "
                  f"ref_run={ref_run is not None}")

        def _adjacent(ma: np.ndarray, mb: np.ndarray) -> bool:
            return bool(np.count_nonzero(cv2.dilate(ma, se3) & mb))

        def _pocket_open(trial: np.ndarray, pk: np.ndarray) -> float:
            return int(np.count_nonzero(pk & ~trial)) / int(np.count_nonzero(pk))

        # pronouncedness references: subordinate adds whose protrusion beyond
        # the OTHER parts' true ink is significant — the feature's identity
        pron_ref: dict[int, float] = {}
        if len(add_live) >= 2:
            for q in add_live:
                if pp_roles[q] not in ("lobe", "accent"):
                    continue
                if excl_ref.get(q, 0) < 4 or not np.count_nonzero(true_solos[q]):
                    continue
                body = np.zeros((target, target), np.uint8)
                for r_ in add_live:
                    if r_ != q:
                        body |= true_solos[r_]
                if not np.count_nonzero(body):
                    continue
                dt = cv2.distanceTransform(cv2.bitwise_not(body), cv2.DIST_L2,
                                           cv2.DIST_MASK_PRECISE)
                d = float(dt[true_solos[q] > 0].max())
                if d >= PRON_MIN_DEPTH:
                    pron_ref[q] = d

        def _pron_depth(q: int, over: int | None = None,
                        cand_mask: np.ndarray | None = None) -> float:
            sol = solo_now(q, over, cand_mask)
            if not np.count_nonzero(sol):
                return 0.0
            body = np.zeros((target, target), np.uint8)
            for r_ in add_live:
                if r_ != q:
                    body |= solo_now(r_, over, cand_mask)
            if not np.count_nonzero(body):
                return float(target)      # no body → trivially pronounced
            dt = cv2.distanceTransform(cv2.bitwise_not(body), cv2.DIST_L2,
                                       cv2.DIST_MASK_PRECISE)
            return float(dt[sol > 0].max())

        def _ncc(mask: np.ndarray) -> int:
            return int(cv2.connectedComponents((mask > 0).astype(np.uint8),
                                               connectivity=8)[0]) - 1

        def _snapshot() -> dict:
            cur = compound()
            return {
                "excl": _excl_now(),
                "adj": {pr: _adjacent(solo_now(pr[0]), solo_now(pr[1]))
                        for pr in sep_pairs},
                "pock": [_pocket_open(cur, pk) for pk in pockets_l],
                "pron": {q: _pron_depth(q) for q in pron_ref},
                "ncc": _ncc(cur),
            }

        def _global_fail(own: set[int], base: dict, over: int | None = None,
                         cand_mask: np.ndarray | None = None,
                         trial: np.ndarray | None = None,
                         narrowed: set[int] = frozenset()) -> str | None:
            """The full global feasibility battery (retention, distinctness,
            separation, pocket retention, pronouncedness) — newly-violated
            semantics throughout, own-part hard where the part controls it.
            Round 4: returns the FAILING CLASS ("retention" | "distinct" |
            "conn" | "sep" | "pocket" | "pron") or None on pass — sep/pocket
            classify a part violation as glob-evidenced."""
            if trial is None:
                trial = compound(over, cand_mask)
            if not retention_ok(trial, narrowed):
                return "retention"
            if not distinct_ok(own, base["excl"], over, cand_mask):
                return "distinct"
            # connectivity (anti-FRAGMENTATION, the dual of anti-globbing):
            # separation/pocket carving must not split a connected mark into
            # more pieces than the reference has (vscode@16's floating chunk)
            ncc = _ncc(trial)
            if ncc > topo_l[0] and ncc > base.get("ncc", ncc):
                return "conn"
            for pr in sep_pairs:
                a, b = pr
                if a not in own and b not in own and over not in (a, b):
                    continue              # candidate cannot change this pair
                if _adjacent(solo_now(a, over, cand_mask),
                             solo_now(b, over, cand_mask)) \
                        and not base["adj"][pr]:
                    return "sep"          # newly fused a separated pair
            # round 8 — outline-separation: ABSOLUTE pocket on the dome notch
            # (no newly-violated escape — the baseline already fused it). The
            # domes connect through the trunk; only the NOTCH must stay open,
            # so this is a pocket clause, not part non-adjacency. Attributed
            # to the flanking domes (own/over ∩ flank) so only their search
            # is constrained — the trunk/pill can't move the notch and are
            # never blamed. A dome shrinks until the 1-px neck reappears;
            # else it falls to fallback and the glob ships flagged.
            for gp, flank in outline_gaps:
                if not (own & flank or over in flank):
                    continue
                if _pocket_open(trial, gp) < OUTLINE_RETAIN:
                    return "sep"
            if fam_run and pockets_l:
                # round 4 — AGGREGATE pocket grain for reference-substituted
                # runs: a 12-wedge comb must not be vetoed by one marginal
                # wedge; systemic globbing still fails the total. (Organic
                # runs keep the per-pocket veto — protecting THE pocket is
                # the round-3 doctrine, e.g. vscode's X wedges.)
                o_now = (sum(int(np.count_nonzero(pk & ~trial))
                             for pk in pockets_l) / pk_total)
                b_now = (sum(bo * n_ for bo, n_ in zip(base["pock"], pk_ns))
                         / pk_total)
                if o_now < POCKET_RETAIN_REF and b_now >= POCKET_RETAIN_REF:
                    return "pocket"
            else:
                for pk, bo in zip(pockets_l, base["pock"]):
                    o = _pocket_open(trial, pk)
                    if o < POCKET_RETAIN and bo >= POCKET_RETAIN:
                        return "pocket"   # newly filled a protected pocket
            for q, dref in pron_ref.items():
                lim = PRON_FRAC * dref
                d = _pron_depth(q, over, cand_mask)
                if d >= lim:
                    continue
                if q in own:
                    return "pron"         # must keep its own protrusion
                if base["pron"].get(q, 0.0) >= lim:
                    return "pron"         # newly flattened a sibling
            return None

        def global_ok(own: set[int], base: dict, over: int | None = None,
                      cand_mask: np.ndarray | None = None,
                      trial: np.ndarray | None = None,
                      narrowed: set[int] = frozenset()) -> bool:
            return _global_fail(own, base, over, cand_mask, trial,
                                narrowed) is None

        part_violation: dict[int, str] = {}
        part_viol_class: dict[int, str] = {}   # round 4: "glob"|"floor"|"other"
        stair_cache: dict[tuple[int, int], float] = {}

        def _stair_of(i: int, k: int) -> float:
            key = (i, k)
            if key not in stair_cache:
                stair_cache[key] = _staircase_total(cand_rasters[i][k])
            return stair_cache[key]

        _INFL_ROLES = ("shave", "counter", "accent", "lobe")   # round 3: lobes
        #                capped too — the discord forced-extension bug-class

        def _optimize_part(i: int) -> None:
            best_in: tuple | None = None        # feasible, degree ≤ budget
            best_in_k = choice[i]
            best_esc: tuple | None = None       # feasible, escalated
            best_esc_k = choice[i]
            fallback_key: tuple | None = None
            fallback_k = choice[i]
            flw = [j for j in followers_of.get(i, []) if not dropped[j]]
            own = {i, *flw}
            base = _snapshot()                  # constraint baselines, fixed
            #                                     across i's candidates
            # round 4 — reference-substituted runs use NEWLY-VIOLATED floor
            # semantics: a transient sub-floor GLOBAL state must not flag
            # every part as "no feasible candidate" when no single part can
            # rescue the whole (the harmonized-comb mass-fallback deadlock)
            base_fid = fidelity_l(compound()) if ref_run is not None else None
            # narrowed siblings (chosen f < 1, pronouncedness held at baseline)
            base_narrow = {q for q in pron_ref
                           if q not in own and choice[q] >= 0 and cand_sets[q]
                           and cand_sets[q][choice[q]][2]
                           * cand_sets[q][choice[q]][3] < 1.0 - 1e-9
                           and base["pron"].get(q, 0.0)
                           >= PRON_FRAC * pron_ref[q]}
            # staircase-escalation clause: if every in-budget candidate's solo
            # carries teeth, one extra degree counts as in-budget (else the
            # J2 penalty deadlocks against the budget — circle regresses to a
            # bare square)
            budget_eff = budgets[i]
            in_b = [k for k, (_c, d, *_r) in enumerate(cand_sets[i])
                    if d <= budgets[i]
                    and (i not in allowed_k or k in allowed_k[i])]
            if in_b and all(_stair_of(i, k) > 0 for k in in_b):
                budget_eff = budgets[i] + 1
            glob_seen = False              # round 4: a candidate cleared the
            best_fid_seen = 0.0            # per-part checks + floor and failed
            #                                ONLY on sep/pocket → class "glob"
            gf_seen: dict[str, int] = {}   # diagnostic: failure-class counts
            # round 4 — a SUB-REPRESENTABLE part clamping up to min-feature
            # is LEGAL inflation (the min-feature doctrine demands it); the
            # forced-extension cap applies only beyond that clamp
            def _clamp_exempt(q: int, solo_c: np.ndarray) -> bool:
                if extents[q] >= min_feat:
                    return False
                ys, xs = np.nonzero(solo_c)
                if not len(xs):
                    return False
                return min(int(xs.max()) - int(xs.min()),
                           int(ys.max()) - int(ys.min())) + 1 <= min_feat + 1

            # round 6 — baseline-relative proportion guard: per degree, the
            # min spill outside ref_l over this part's f=1 candidates. Extra
            # spill beyond the same-degree f=1 twin is PROPORTION DRIFT
            # (vscode@16's +1-px column; gmail's 4-wide bar), not
            # straightening quantization (a D0 square over a circle ref
            # spills its corners at f=1 too → extra = 0 by construction).
            # Polygon candidates over non-Polygon parts (round-5 axis-run
            # forms) are different geometry — excluded from baseline AND
            # check.
            # NOTE: no Polygon-type exemption here — straighten() emits
            # Polygon forms for EVERY primitive at every degree, so a type
            # test would match every normal candidate and neuter the guard
            # (the first-cut bug). Round-5 axis-run candidates need no
            # carve-out: against the LIVE ref their spill prices them out
            # (they never won incumbent runs anyway — sub-floor by design),
            # and in their own axisreduce ref-run extra ≈ 0.
            # Baselines are PER (part, degree): a follower's twin floor is
            # the FLIPPED f=1 raster vs ITS OWN true form — the pair-flip
            # phase error is constant across candidates, so it must land in
            # the baseline too or every candidate fails on the follower
            # (gmail@16's mirrored V counters).
            base_spl6: dict[tuple[int, int], int] = {}
            is_add6 = pp[i].op == "add"
            for k6, (c6, d6, fx6, fy6, _dx6, _dy6) in \
                    enumerate(cand_sets[i]):
                if abs(fx6 - 1.0) > 1e-9 or abs(fy6 - 1.0) > 1e-9:
                    continue
                r6 = cand_rasters[i][k6]
                for q6 in (i, *flw):
                    if q6 == i:
                        rq6 = r6
                    else:
                        v6f, a2f = pair_axis[q6]
                        rq6 = _flip_mask(r6, a2f, v6f)
                    s6 = (int(np.count_nonzero(rq6 & ~ref_l)) if is_add6
                          else int(np.count_nonzero(true_solos[q6] & ~rq6
                                                    & ~ref_l)))
                    if (q6, d6) not in base_spl6 \
                            or s6 < base_spl6[(q6, d6)]:
                        base_spl6[(q6, d6)] = s6

            def _prop_ok(q: int, d6: int, solo_c: np.ndarray,
                         cand_prim) -> bool:
                # INCUMBENT runs only: reference-substituted runs already
                # carry the round-4 idealized-candidate lock + the
                # SPILL_FRAC_REF cap — stacking this guard there double-
                # polices and broke figma@16's regularize equilibrium
                # (snap-harmonization tuples rejected → group unequal).
                if ref_run is not None:
                    return True
                if (q, d6) not in base_spl6:
                    return True
                if _clamp_exempt(q, solo_c):
                    return True
                # rotated thin strokes: extents[] and the solo-bbox test are
                # both blind to stroke WIDTH on rotated geometry (a 63° bar's
                # bbox is half its length) — a sub-min-feature-width part's
                # width clamp-up is the legal min-feature inflation, same
                # doctrine as _clamp_exempt (long_diag's 1.5-px bars at 16)
                if _local_dims(parts_t[q].primitive)[0] < min_feat:
                    return True
                # radial-family blades: family machinery (comb harmonization,
                # decimation, family ladder) polices them — the guard
                # rejecting a blade's hub-reaching inflation fragmented
                # asterisk6@64 into 3 components (blades only clear the
                # min-feature width at 64, where no harmonization re-fuses)
                if (pp is parts and radial is not None
                        and q in radial[1]):
                    return True
                if pp[q].op == "add":
                    v6 = int(np.count_nonzero(solo_c & ~ref_l))
                    ink6 = int(np.count_nonzero(solo_c))
                else:
                    # the subtract mirror (round 6): px the TRUE carve
                    # claims in reference-EMPTY zones that the candidate
                    # fails to carve — a shrunken counter leaves compound
                    # ink where the reference is empty (gmail@16's V
                    # closed at f=0.68 because jag-first liked the shorter
                    # boundary; _bite_ok polices over-carving, nothing
                    # policed under-carving)
                    v6 = int(np.count_nonzero(true_solos[q] & ~solo_c
                                              & ~ref_l))
                    ink6 = int(np.count_nonzero(true_solos[q]))
                ok6 = v6 - base_spl6[(q, d6)] <= max(
                    2.0, PROP_SPILL_FRAC * ink6)
                if not ok6 and os.environ.get("R1_DEBUG_PROP"):
                    print(f"            prop fail {tag(q)} d={d6} v6={v6} "
                          f"base={base_spl6[(q, d6)]} ink={ink6}")
                return ok6

            for k, (cand, d, fx_c, fy_c, dx, dy) in enumerate(cand_sets[i]):
                if i in allowed_k and k not in allowed_k[i]:
                    continue
                cand_solo = cand_rasters[i][k]
                per_part_ok = True
                prop_fail = False
                if not _prop_ok(i, d, cand_solo, cand.primitive):
                    per_part_ok = False        # round 6 — proportion drift
                    prop_fail = True
                if (per_part_ok
                        and ref_run is not None and pp[i].op == "add"
                        and _local_dims(parts_t[i].primitive)[0]
                        >= min_feat):
                    # background-spill cap (round 4): the ideal is known —
                    # don't paint meaningfully outside it. Open-corner
                    # negative space (figma's step) has no pocket guard;
                    # this is the constraint that polices it. Sub-min-
                    # feature parts are exempt: their spill is parity skin
                    # over a part the clamp machinery governs.
                    spl = int(np.count_nonzero(cand_solo & ~ref_l))
                    per_part_ok = spl <= max(
                        2.0, SPILL_FRAC_REF
                        * int(np.count_nonzero(cand_solo)))
                if per_part_ok and pp_roles[i] in _INFL_ROLES \
                        and not _clamp_exempt(i, cand_solo):
                    per_part_ok = _inflation_ok(cand_solo, true_solos[i])
                if per_part_ok and pp_roles[i] == "shave":
                    per_part_ok = _bite_ok(cand_solo, true_solos[i],
                                           group_union[pp[i].group])
                # the pair is judged jointly: the follower's flipped solo must
                # satisfy ITS constraints against its own true form
                for j in flw:
                    if not per_part_ok:
                        break
                    v, a2p = pair_axis[j]
                    fsolo = _flip_mask(cand_solo, a2p, v)
                    # round 6 — follower proportion drift, judged on the
                    # LEADER's f=1 baseline (same geometry, mirrored)
                    if not _prop_ok(j, d, fsolo, cand.primitive):
                        per_part_ok = False
                        prop_fail = True
                        break
                    if pp_roles[j] in _INFL_ROLES \
                            and not _clamp_exempt(j, fsolo):
                        per_part_ok = _inflation_ok(fsolo, true_solos[j])
                    if per_part_ok and pp_roles[j] == "shave":
                        per_part_ok = _bite_ok(fsolo, true_solos[j],
                                               group_union[pp[j].group])
                trial = compound(over=i, cand_mask=cand_solo)
                fid = fidelity_l(trial)
                best_fid_seen = max(best_fid_seen, fid)
                j_ = jag_score(trial)
                key = (j_, -fid, d, abs(dx) + abs(dy), k)
                narrowed = set(base_narrow)
                if (fx_c * fy_c < 1.0 - 1e-9 and i in pron_ref
                        and _pron_depth(i, over=i, cand_mask=cand_solo)
                        >= PRON_FRAC * pron_ref[i]):
                    narrowed |= own
                fid_ok = fid >= floor or (base_fid is not None
                                          and fid >= base_fid - 1e-9)
                feasible = False
                if per_part_ok and fid_ok:
                    gf = _global_fail(own, base, over=i, cand_mask=cand_solo,
                                      trial=trial, narrowed=narrowed)
                    feasible = gf is None
                    if gf is not None:
                        gf_seen[gf] = gf_seen.get(gf, 0) + 1
                    if gf in ("sep", "pocket"):
                        glob_seen = True
                elif not per_part_ok:
                    cls6 = "prop" if prop_fail else "per_part"
                    gf_seen[cls6] = gf_seen.get(cls6, 0) + 1
                else:
                    gf_seen["fid"] = gf_seen.get("fid", 0) + 1
                if feasible:
                    if d <= budget_eff and (best_in is None or key < best_in):
                        best_in, best_in_k = key, k
                    elif d > budget_eff and (best_esc is None or key < best_esc):
                        best_esc, best_esc_k = key, k
                fb_key = (-fid, j_, k)     # fallback: most faithful first
                if fallback_key is None or fb_key < fallback_key:
                    fallback_key, fallback_k = fb_key, k
            if best_in is not None:
                choice[i] = best_in_k
                part_violation.pop(i, None)
                part_viol_class.pop(i, None)
            elif best_esc is not None:
                choice[i] = best_esc_k        # floor outranks the budget
                part_violation.pop(i, None)
                part_viol_class.pop(i, None)
            else:
                choice[i] = fallback_k
                cls = ("glob" if glob_seen
                       else ("floor" if best_fid_seen < floor else "other"))
                part_viol_class[i] = cls
                part_violation[i] = (f"{tag(i)} {pp_roles[i]}: no feasible "
                                     f"candidate at T={target} — most-faithful "
                                     f"fallback [{cls}]")
                if os.environ.get("R1_DEBUG"):
                    print(f"          fallback {tag(i)} classes={gf_seen}")

        order = [i for i in sorted(
            live, key=lambda i: (-int(np.count_nonzero(true_solos[i])), i))
            if i not in followers and cand_sets[i]]

        for i in order:                       # pass 1: static grid-snap candidates
            _optimize_part(i)

        # parent-follow candidates (the composition clause): a subordinate
        # must cut the SNAPPED parent, not the absolute grid. Followers are
        # excluded — their geometry mirrors the leader's follow automatically.
        group_dom: dict[int, int] = {}
        for g in sorted({p.group for p in pp}):
            adds = [i for i in live if pp[i].group == g and pp[i].op == "add"
                    and i not in followers]
            if adds:
                group_dom[g] = sorted(
                    adds, key=lambda i: (-int(np.count_nonzero(true_solos[i])),
                                         i))[0]
        for i in (live if ref_run is None else ()):   # extrapolated runs:
            #                       geometry is ideal — no follow drift
            di = group_dom.get(pp[i].group)
            if di is None or di == i or i in followers or not cand_sets[i]:
                continue
            _dc, _dd, dfx, dfy, ddx, ddy = cand_sets[di][choice[di]]
            pcx, pcy = prim_center(parts_t[di].primitive)
            bcx, bcy = prim_center(parts_t[i].primitive)
            tx = (pcx + dfx * (bcx - pcx) + ddx) - bcx
            ty = (pcy + dfy * (bcy - pcy) + ddy) - bcy
            for d in degree_sets[i]:
                for f2 in _size_factors(extents[i], min_feat):
                    form = straighten(scale_about_center(parts_t[i].primitive,
                                                         f2), d)
                    cand = Part(primitive=translate_prim(form, tx, ty),
                                op=pp[i].op, group=pp[i].group)
                    cand_sets[i].append((cand, d, f2, f2, tx, ty))
                    cand_rasters[i].append(
                        _raster_part(cand.primitive, target, target))

        for i in order:                       # pass 2 sees the follow candidates
            _optimize_part(i)
        violations.extend(part_violation[i] for i in sorted(part_violation))

        # twin harmonization: identical-shape siblings must not end with
        # MIXED degrees from fractional-phase ties. Leaders only — followers
        # mirror automatically.
        twin_sets: dict[tuple, list[int]] = {}
        for i in order:
            p = pp[i].primitive
            if ref_run is not None:
                # round 4 — rotation-invariant twin key for reference-
                # substituted runs: comb blades at different angles have
                # different BBOXES but identical local geometry; one twin
                # set means the uniform tuple is judged SET-WIDE (a drift
                # that is noise per-blade fails the floor jointly)
                ld0, ld1 = _local_dims(parts_t[i].primitive)
                key_t = (pp[i].op, pp[i].group, type(p).__name__,
                         round(ld0 * 2.0) / 2.0, round(ld1 * 2.0) / 2.0)
            else:
                key_t = (pp[i].op, pp[i].group, type(p).__name__,
                         round(extents[i] * 2.0) / 2.0)
            twin_sets.setdefault(key_t, []).append(i)
        for members in (m for _k, m in sorted(twin_sets.items(),
                                              key=lambda kv: kv[1][0])
                        if len(m) > 1):
            cur = compound()
            cur_fid0 = fidelity_l(cur)
            best_key = (jag_score(cur), cur_fid0 * -1.0)
            best_choice = list(choice)
            harmonize_base = _snapshot()      # constraint baselines for the set
            # round 3: twins must render IDENTICAL up to placement — uniform
            # (degree, fx, fy), not just uniform degree; snaps stay free
            tuples: set[tuple] | None = None
            for i in members:
                mine = {(d, round(fx, 6), round(fy, 6))
                        for k, (_c, d, fx, fy, _dx, _dy)
                        in enumerate(cand_sets[i])
                        if i not in allowed_k or k in allowed_k[i]}
                tuples = mine if tuples is None else (tuples & mine)
            for tup in sorted(tuples or ()):
                saved = list(choice)
                ok = True
                for i in members:
                    m_best: tuple | None = None
                    m_k = None
                    flw = [j for j in followers_of.get(i, [])
                           if not dropped[j]]
                    for k, (cand, cd, fx, fy, dx, dy) in enumerate(cand_sets[i]):
                        if (cd, round(fx, 6), round(fy, 6)) != tup:
                            continue
                        if i in allowed_k and k not in allowed_k[i]:
                            continue
                        cand_solo = cand_rasters[i][k]
                        per_part_ok = True
                        if pp_roles[i] in _INFL_ROLES:
                            per_part_ok = _inflation_ok(cand_solo, true_solos[i])
                        if per_part_ok and pp_roles[i] == "shave":
                            per_part_ok = _bite_ok(cand_solo, true_solos[i],
                                                   group_union[pp[i].group])
                        for j in flw:
                            if not per_part_ok:
                                break
                            v, a2p = pair_axis[j]
                            fsolo = _flip_mask(cand_solo, a2p, v)
                            if pp_roles[j] in _INFL_ROLES:
                                per_part_ok = _inflation_ok(fsolo, true_solos[j])
                            if per_part_ok and pp_roles[j] == "shave":
                                per_part_ok = _bite_ok(fsolo, true_solos[j],
                                                       group_union[pp[j].group])
                        if not per_part_ok:
                            continue
                        # floor/battery judged on the FINAL uniform state —
                        # gating half-switched states locks the asymmetry in
                        trial = compound(over=i, cand_mask=cand_solo)
                        kk = (jag_score(trial), -fidelity(trial),
                              abs(dx) + abs(dy), k)
                        if m_best is None or kk < m_best:
                            m_best, m_k = kk, k
                    if m_k is None:
                        ok = False
                        break
                    choice[i] = m_k
                if ok:
                    trial = compound()
                    fid = fidelity_l(trial)
                    fid_ok = fid >= floor or (ref_run is not None
                                              and fid >= cur_fid0 - 1e-9)
                    if fid_ok and global_ok(set(members), harmonize_base,
                                            trial=trial):
                        kk = (jag_score(trial), -fid)
                        if kk < best_key:
                            best_key, best_choice = kk, list(choice)
                choice[:] = saved
            choice[:] = best_choice

        # round 4 — regularity snap harmonization: members of a detected
        # size group share their FULL tweak tuple (degree, fx, fy, dx, dy)
        # when a matching candidate exists — identical primitives (the
        # extrapolated variant) then raster identically up to translation.
        # reg_check is set ONLY for runs built from the original or snapped
        # TRANSLATIONAL geometry (incumbent, regularize variant) — a radial-
        # harmonized comb legitimately moves the members the original
        # clusters were measured on
        reg_active = (reg_check and reg_detect is not None
                      and n == len(parts) and target <= COLLAPSE_MAX_T)
        if reg_active:
            for grp in reg_detect[0]["size"]:
                mem = [i for i in grp if i in live and i not in followers
                       and cand_sets[i] and choice[i] >= 0]
                if len(mem) < 2:
                    continue
                # round 7 — harmonize to the MEDIAN member's target extent,
                # not mem[0]'s: a lone outlier (figma@24's one circle at
                # f=1.07 vs the other two at 0.92) must snap DOWN to the
                # majority, not drag the majority UP to it. Every member
                # then matches the consensus tuple — equality by
                # construction, minimal disruption.
                def _te(i: int) -> tuple[int, int]:
                    _c, _d, fxi, fyi, _dx, _dy = cand_sets[i][choice[i]]
                    exi0, eyi0 = extents_xy[i]
                    return (round(fxi * exi0), round(fyi * eyi0))
                lead = sorted(mem, key=lambda i: (_te(i), i))[len(mem) // 2]
                snap_base = _snapshot()
                pre_fid = fidelity_l(compound())
                _c0, d0_, fx0, fy0, dx0, dy0 = cand_sets[lead][choice[lead]]
                # round 6 — match on ABSOLUTE target extents, not raw ratio
                # tuples: fx/fy are ratios relative to each member's OWN
                # extent, so a genuinely-perturbed group (figma's r=26.3/
                # 26.0/25.7 circles) could never match — equal ratios give
                # unequal pixels and equal pixels give unequal ratios; the
                # ratio matcher was structurally a no-op for exactly the
                # groups it exists for. Same degree, same integer pixel
                # extents; snaps stay free (each member lands its own
                # phase — REG_RENDER_TOL audits the result), tie-broken by
                # smallest snap then index (H8).
                exl, eyl = extents_xy[lead]
                tgt0 = (d0_, round(fx0 * exl), round(fy0 * eyl))
                saved = list(choice)
                ok = True
                for i in mem:
                    if i == lead:
                        continue
                    exi, eyi = extents_xy[i]
                    kf = min(
                        (k for k, (_c, d, fx, fy, dx, dy)
                         in enumerate(cand_sets[i])
                         if (d, round(fx * exi), round(fy * eyi)) == tgt0
                         and abs(dx) <= 0.5 + 1e-9 and abs(dy) <= 0.5 + 1e-9
                         and (i not in allowed_k or k in allowed_k[i])),
                        key=lambda k: (abs(cand_sets[i][k][4])
                                       + abs(cand_sets[i][k][5]), k),
                        default=None)
                    if kf is None:
                        ok = False
                        break
                    choice[i] = kf
                if ok and any(choice[i] != saved[i] for i in mem):
                    trial = compound()
                    fid_t = fidelity_l(trial)
                    fid_ok = fid_t >= floor or (ref_run is not None
                                                and fid_t >= pre_fid - 1e-9)
                    if not (fid_ok
                            and global_ok(set(mem), snap_base, trial=trial)):
                        ok = False
                        if os.environ.get("R1_DEBUG"):
                            print(f"          reg-harmonize REVERT grp={mem} "
                                  f"tgt={tgt0} fid_t={fid_t:.3f} "
                                  f"fid_ok={fid_ok}")
                elif not ok and os.environ.get("R1_DEBUG"):
                    print(f"          reg-harmonize NOMATCH grp={mem} "
                          f"tgt={tgt0}")
                if not ok:
                    choice[:] = saved

        # radial spacing harmonization (round 3): for a kept blade family,
        # offer the comb (each blade rotated about the hub onto a uniform
        # pitch) as a SET-WIDE alternative — offered and arbitrated, never
        # forced (claude's organic ±8° may honestly win on fidelity).
        # `radial` indices refer to the ORIGINAL parts — incumbent run only.
        if radial is not None and pp is parts:
            hub_i, blades_r, _spr = radial
            kept = [i for i in blades_r
                    if i < n and i in live and i not in followers
                    and cand_sets[i] and choice[i] >= 0]
            if len(kept) >= 3 and hub_i < n:
                pitch = 360.0 / len(kept)
                hubc = prim_center(parts_t[hub_i].primitive)
                angs_k = sorted(
                    (math.degrees(math.atan2(
                        prim_center(parts_t[i].primitive)[1] - hubc[1],
                        prim_center(parts_t[i].primitive)[0] - hubc[0]))
                     % 360.0, i) for i in kept)
                base_phase = angs_k[0][0]
                comb_k: dict[int, int] = {}
                for idx, (ang, i) in enumerate(angs_k):
                    dtheta = ((base_phase + idx * pitch - ang + 180.0)
                              % 360.0 - 180.0)
                    form = cand_sets[i][choice[i]][0].primitive
                    if not isinstance(form, Polygon):
                        comb_k = {}
                        break
                    if abs(dtheta) < 1e-6:
                        comb_k[i] = choice[i]
                        continue
                    th = math.radians(dtheta)
                    cth, sth = math.cos(th), math.sin(th)
                    rot = tuple(
                        (hubc[0] + (x - hubc[0]) * cth - (y - hubc[1]) * sth,
                         hubc[1] + (x - hubc[0]) * sth + (y - hubc[1]) * cth)
                        for x, y in form.pts)
                    _c0, d0_, fx0, fy0, _dx0, _dy0 = cand_sets[i][choice[i]]
                    cand_sets[i].append((Part(Polygon(pts=rot), pp[i].op,
                                              pp[i].group), d0_, fx0, fy0,
                                         0.0, 0.0))
                    cand_rasters[i].append(_raster_part(Polygon(pts=rot),
                                                        target, target))
                    comb_k[i] = len(cand_sets[i]) - 1
                if comb_k and any(comb_k[i] != choice[i] for i in comb_k):
                    comb_base = _snapshot()
                    cur = compound()
                    cur_key = (jag_score(cur), -fidelity_l(cur))
                    saved = list(choice)
                    for i, k in comb_k.items():
                        choice[i] = k
                    trial = compound()
                    fid = fidelity_l(trial)
                    kk = (jag_score(trial), -fid)
                    if not (fid >= floor and kk < cur_key
                            and global_ok(set(comb_k), comb_base, trial=trial)):
                        choice[:] = saved

        # render_fit refiner (optional torch): polish leader deltas when
        # the deterministic result is flagged; decoded state must beat the
        # incumbent under the SAME hard key + full battery or it is discarded
        if enable_sym and live and ref_run is None:
            cur = compound()
            fid_cur = fidelity_l(cur)
            if (part_violation or fid_cur < floor + 0.05
                    or jag_score(cur) > 0.95 * max(naive_jag, 1e-9)):
                try:
                    from iconograph.application.compress.render_fit import (
                        refine_tweaks,
                    )
                except Exception:           # torch absent → silent skip
                    refine_tweaks = None
                deltas: dict = {}
                if refine_tweaks is not None:
                    leaders = [(i, cand_sets[i][choice[i]][0]) for i in live
                               if i not in followers and cand_sets[i]
                               and choice[i] >= 0]
                    try:
                        deltas = refine_tweaks(leaders, leader_of, pair_axis,
                                               coverage, pockets, target)
                    except NotImplementedError:
                        deltas = {}
                if deltas:
                    base = _snapshot()
                    cur_key = (jag_score(cur), -fid_cur)
                    saved = list(choice)
                    applied: set[int] = set()
                    leader_ids = {i for i, _f in leaders}
                    for i in sorted(deltas):
                        if i not in leader_ids:
                            continue
                        ddx, ddy, dfx, dfy = deltas[i]
                        # round 8 — make the decoded NN output CONFORM to
                        # policy (user ruling: conform, don't relax). The
                        # soft-optimized deltas move SELF-PAIRED parts off the
                        # mirror axis by a fraction of a pixel, which breaks the
                        # exact-symmetry IoU and dropped the stronger candidate.
                        # A self-paired part must stay centred on the axis, so
                        # its ACROSS-axis shift is illegal: zero it (glob[0]
                        # vertical → mirrors x → across-axis is x).
                        if glob is not None and i in self_paired:
                            if glob[0]:
                                ddx = 0.0
                            else:
                                ddy = 0.0
                        c0, d0_, fx0, fy0, dx0, dy0 = cand_sets[i][choice[i]]
                        form = translate_prim(
                            scale_about_center_xy(c0.primitive, dfx, dfy),
                            ddx, ddy)
                        r_new = _raster_part(form, target, target)
                        # min-feature clamp: the refiner must not shrink a part
                        # under the floor (a decoded sub-min-feature form is
                        # not renderable per policy)
                        ys, xs = np.nonzero(r_new)
                        if len(xs) and min(int(xs.max()) - int(xs.min()),
                                           int(ys.max()) - int(ys.min())) \
                                + 1 < min_feat:
                            continue
                        # self-paired parts must raster EXACTLY flip-invariant
                        # (the policy the sym_kept check enforces) — accept the
                        # decoded candidate only if it does, else keep the
                        # part's pre-refiner choice (revert just this delta)
                        if (glob is not None and i in self_paired
                                and not np.array_equal(
                                    r_new, _flip_mask(r_new, glob[2], glob[0]))):
                            continue
                        cand_sets[i].append((Part(form, pp[i].op, pp[i].group),
                                             d0_, fx0 * dfx, fy0 * dfy,
                                             dx0 + ddx, dy0 + ddy))
                        cand_rasters[i].append(r_new)
                        choice[i] = len(cand_sets[i]) - 1
                        applied.add(i)
                    trial = compound()
                    fid2 = fidelity_l(trial)
                    kk = (jag_score(trial), -fid2)
                    sym_kept = (glob is None
                                or _iou_bool(trial, _flip_mask(trial, glob[2],
                                                               glob[0])) >= 1.0)
                    if (applied and fid2 >= floor and kk < cur_key and sym_kept
                            and global_ok(applied, base, trial=trial)):
                        # the refiner repaired the flagged state — the search's
                        # fallback flags no longer describe the shipped render
                        for msg in part_violation.values():
                            if msg in violations:
                                violations.remove(msg)
                        part_violation.clear()
                    else:
                        choice[:] = saved

        final = compound() if live else np.zeros((target, target), np.uint8)

        # counter-integrity audit (pair-coherent): a fused/vanished island
        # is dropped — with its mirror twin — and the compound re-rendered.
        island_changed = False
        for i in list(live):
            if pp[i].op != "subtract" or pp[i].group >= 0 or i in followers:
                continue
            solo = solo_now(i)
            kept = solo & final
            n_kept = int(np.count_nonzero(kept))
            if n_kept:
                _n, labels = cv2.connectedComponents(
                    (final > 0).astype(np.uint8), connectivity=8)
                labs = [int(v) for v in np.unique(labels[kept > 0]) if v != 0]
                blob = int(np.count_nonzero(np.isin(labels, labs)))
                if blob <= n_kept * 2 + 4:
                    continue              # intact: its own small component
            _drop(i, f"{tag(i)} counter island fused/vanished at T={target} — "
                     f"dropped (annulus degradation)")
            for jj in (i, *followers_of.get(i, [])):
                if jj in live:
                    live.remove(jj)
                    dropped[jj] = True
            island_changed = True
        if island_changed:
            final = compound() if live else np.zeros((target, target), np.uint8)

        # final-mask audit (round 3): unclaimed sub-min-feature components
        # are stray pixels (the gmail@32 class) — removed AND recorded, never
        # silently erased; sym is measured AFTER the audit
        extra_drops: list[str] = []
        if live:
            n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(
                final, connectivity=8)
            for ci in range(1, n_cc):
                sub_mf = min(int(stats[ci, cv2.CC_STAT_WIDTH]),
                             int(stats[ci, cv2.CC_STAT_HEIGHT])) < min_feat
                area = int(stats[ci, cv2.CC_STAT_AREA])
                # round 6 — the claimed-stray clause also covers AT-min-
                # feature debris (gmail@24/32's 3×2 sliver: min dim ==
                # min_feat, area == min_feat²); the unclaimed branch keeps
                # the round-3 sub-min-feature scope exactly
                if not (sub_mf or area <= min_feat * min_feat):
                    continue
                comp = lbl == ci
                claimers = [q for q in live
                            if int(np.count_nonzero(comp & (solo_now(q) > 0)))
                            * 2 >= area]
                claimed = bool(claimers)
                if not claimed and not sub_mf:
                    continue
                # at-min-feature extension scope: a claimer that is itself a
                # sub-min-feature-width STROKE legitimately rasters in
                # min_feat²-sized fragments (claude@16's 1-px comb blades) —
                # only debris claimed exclusively by WIDE parts is stray
                if (claimed and not sub_mf
                        and any(_local_dims(parts_t[q].primitive)[0]
                                < min_feat for q in claimers)):
                    continue
                # round 6 — a CLAIMED component lying mostly OUTSIDE the
                # reference is still a stray (gmail@24/32: the dominant's
                # sliver between the mirrored V counter tips). Removed AND
                # recorded as a [stray] violation — a new feasibility class,
                # so a stray-free run outranks a cleaned one in arbitration.
                # Unclaimed components keep the round-3 semantics exactly
                # (drops-only: they are raster noise, not misfit evidence).
                out6 = int(np.count_nonzero(comp & (ref_l == 0)))
                if claimed and out6 < STRAY_OUT_FRAC * area:
                    continue              # a live part claims it — legitimate
                final[comp] = 0
                if claimed:
                    extra_drops.append(
                        f"stray component removed ({area} px, {out6}/{area} "
                        f"outside ref) — final-mask audit")
                    violations.append("final-mask: claimed sub-min-feature "
                                      "stray outside reference [stray]")
                else:
                    extra_drops.append(
                        f"stray component removed ({area} px) — "
                        f"final-mask audit")

        sym = (_iou_bool(final, _flip_mask(final, glob[2], glob[0]))
               if glob is not None else None)

        # safety valve: a symmetrized render below the floor yields to a
        # feasible plain render (recorded either way) — but symmetry is never
        # arbitrated against jag/fidelity head-to-head (that recreates the
        # blindness this pass exists to fix).
        # fam_run valve trigger is AREA-based, matching the comb's fid_ref
        # doctrine (boundary IoU at min-feature blade scale is raster-phase
        # noise — it must not fire the symmetrize→plain fallback either)
        valve_fid = (_iou_bool(final, ref_l) if fam_run
                     else fidelity_l(final))
        if glob is not None and enable_sym and valve_fid < floor:
            plain = _run(pp, pp_roles, pp_solos, tag, enable_sym=False,
                         ref_run=ref_run, pockets_run=pockets_run,
                         topo_ref_run=topo_ref_run)
            if fidelity_l(plain.mask) >= floor and not plain.violations:
                plain.violations.append("symmetrized render below floor — "
                                        "plain fallback shipped")
                return plain
            violations.append("symmetrized render below floor "
                              "(plain also infeasible — symmetric kept)")

        records: list[TweakRecord] = []
        for i in range(n):
            if dropped[i]:
                records.append(TweakRecord(i, pp_roles[i], extents[i],
                                           budgets[i], DEG_TRUE, 1.0, 1.0,
                                           0.0, 0.0, drop_reason[i], None))
                continue
            if i in followers:
                ld = leader_of[i]
                _c, d, fx, fy, _dx, _dy = cand_sets[ld][choice[ld]]
                records.append(TweakRecord(i, pp_roles[i], extents[i],
                                           budgets[i], d, fx, fy, 0.0, 0.0,
                                           None, None, mirrored_of=ld))
                continue
            _cand, d, fx, fy, dx, dy = cand_sets[i][choice[i]]
            records.append(TweakRecord(i, pp_roles[i], extents[i], budgets[i],
                                       d, fx, fy, dx, dy, None,
                                       part_violation.get(i)))

        # round 4 — the EXTRAPOLATED-SYMMETRY feasibility rule: a render
        # that breaks a DETECTED family's regularity is infeasible, full
        # stop. Offering symmetric variants is not enough — the lopsided
        # cross has LOWER raw jag than a clean star, so the regular form can
        # only win through the feasibility term.
        if target <= COLLAPSE_MAX_T and live:
            # rotational: a radial family must render comb-regular — uniform
            # gaps AND uniform blade widths (asterisk6 is exact and passes;
            # claude's organic ±8°/21–46 px widths fail)
            for g2 in sorted({p.group for p in pp if p.group >= 0}):
                adds_g2 = [q for q in range(n)
                           if pp[q].group == g2 and pp[q].op == "add"
                           and q not in drop_reason]
                famr = _radial_family(pp, adds_g2)
                if famr is None:
                    continue
                _h2, blades2, sp2 = famr
                pitch2 = 360.0 / sp2["n"]
                hp2 = pp[_h2].primitive
                angs2 = sorted(
                    math.degrees(math.atan2(
                        prim_center(pp[q].primitive)[1] - hp2.cy,
                        prim_center(pp[q].primitive)[0] - hp2.cx)) % 360.0
                    for q in blades2)
                gaps2 = [(angs2[(k2 + 1) % len(angs2)] - angs2[k2]) % 360.0
                         for k2 in range(len(angs2))]
                gdev = max(abs(gv - pitch2) for gv in gaps2)
                dims2 = [_local_dims(pp[q].primitive)[0] for q in blades2]
                m0 = sorted(dims2)[len(dims2) // 2]
                wdev = (max(abs(dv - m0) for dv in dims2) / m0
                        if m0 > 0 else 0.0)
                if gdev > RADIAL_REG_TOL_DEG or wdev > REG_EXT_TOL:
                    violations.append(
                        f"radial regularity: {sp2['n']}-blade family rendered "
                        f"irregular (gap dev {gdev:.1f} deg, width dev "
                        f"{wdev:.2f}) at T={target} [reg]")
        if reg_active and live:
            # translational: detected size groups render identical (bbox
            # dims), aligned clusters stay aligned, pitch rows stay uniform
            def _bb(mask: np.ndarray):
                ys, xs = np.nonzero(mask)
                if not len(xs):
                    return None
                return (float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max()))

            groups_r = reg_detect[0]
            reg_msgs: list[str] = []
            bbs = {i: _bb(solo_now(i)) for i in live if pp[i].op == "add"}
            for grp in groups_r["size"]:
                mem = [i for i in grp if bbs.get(i) is not None]
                if len(mem) < 3:
                    continue
                # AREA equality, not bbox — rotated same-kind members
                # (asterisk6's blades) legitimately differ in bbox
                areas_r = [int(np.count_nonzero(solo_now(i))) for i in mem]
                med_a = sorted(areas_r)[len(areas_r) // 2]
                if max(abs(a - med_a) for a in areas_r) \
                        > max(2.0, 0.1 * med_a):
                    reg_msgs.append(f"size group {mem} unequal at T={target}")
            for ax, cl in groups_r["align"]:
                mem = [i for i in cl if bbs.get(i) is not None]
                if len(mem) < 2:
                    continue
                k_ax = (0, 2) if ax == "x" else (1, 3)
                cs = [(bbs[i][k_ax[0]] + bbs[i][k_ax[1]]) / 2.0 for i in mem]
                if max(cs) - min(cs) > REG_RENDER_TOL:
                    reg_msgs.append(f"{ax}-alignment {mem} drifted "
                                    f"{max(cs) - min(cs):.1f} px")
            for ax, rows in groups_r["pitch"]:
                k_ax = (0, 2) if ax == "x" else (1, 3)
                pos_r = []
                for row in rows:
                    mem = [i for i in row if bbs.get(i) is not None]
                    if mem:
                        pos_r.append(sum(
                            (bbs[i][k_ax[0]] + bbs[i][k_ax[1]]) / 2.0
                            for i in mem) / len(mem))
                if len(pos_r) < 3:
                    continue
                gapsr = [pos_r[k2 + 1] - pos_r[k2]
                         for k2 in range(len(pos_r) - 1)]
                mg = sum(gapsr) / len(gapsr)
                if max(abs(gv - mg) for gv in gapsr) > REG_RENDER_TOL:
                    reg_msgs.append(f"{ax}-pitch drifted at T={target}")
            if reg_msgs:
                violations.append("regularity: detected group rendered "
                                  "irregular (" + "; ".join(reg_msgs)
                                  + ") [reg]")

        drops = [drop_reason[i] for i in sorted(drop_reason)] + extra_drops
        axis_out = ((("x" if glob[0] else "y"), glob[1])
                    if glob is not None else None)
        # ABSOLUTE battery for the arbitration: a base where everything held,
        # so newly-violated == absolute — this is how a clean variant honestly
        # beats a globbed incumbent (the in-run constraints can only prevent
        # WORSENING from a globbed init, never force recovery)
        base_abs = {
            "excl": dict(excl_ref),
            "adj": {pr: False for pr in sep_pairs},
            "pock": [1.0] * len(pockets_l),
            "pron": dict(pron_ref),
            "ncc": topo_l[0],
        }
        narrowed_final = {q for q in pron_ref
                          if q in live and q not in followers
                          and cand_sets[q] and choice[q] >= 0
                          and cand_sets[q][choice[q]][2]
                          * cand_sets[q][choice[q]][3] < 1.0 - 1e-9
                          and _pron_depth(q) >= PRON_FRAC * pron_ref[q]}
        batt_class = (None if not live else
                      _global_fail(set(), base_abs, trial=final,
                                   narrowed=narrowed_final))
        battery = bool(live) and batt_class is None
        if os.environ.get("R1_DEBUG") and batt_class is not None:
            print(f"          battery fail [{batt_class}]")
        # round 4 — glob attribution for the removal phase: which live
        # adds assembled the glob (absolutely-fused separated pairs; accents
        # covering a filled protected pocket). Computed only on flagged runs
        # — zero cost on clean ones. Indices are in THIS run's part list.
        glob_offenders: list[int] = []
        if live and (part_violation or not battery):
            off: set[int] = set()
            for pr in sep_pairs:
                if (pr[0] in live and pr[1] in live
                        and _adjacent(solo_now(pr[0]), solo_now(pr[1]))):
                    off |= {pr[0], pr[1]}
            for pk in pockets_l:
                filled = pk & final
                nf = int(np.count_nonzero(filled))
                if nf and _pocket_open(final, pk) < POCKET_RETAIN:
                    for q in add_live:
                        if q in live and int(np.count_nonzero(
                                solo_now(q) & filled)) >= GLOB_ATTR_FRAC * nf:
                            off.add(q)
            glob_offenders = sorted(off)
        return _Run(mask=final, records=records, drops=drops,
                    violations=violations, sym=sym, axis=axis_out,
                    n_live=len(live), pairs=len(leader_of), battery=battery,
                    glob_offenders=glob_offenders,
                    viol_class=dict(part_viol_class))

    # incumbent + collapse variants, arbitrated honestly
    incumbent = _run(parts, roles_orig, canon_solos, lambda i: f"#{i}",
                     reg_check=True)
    runs: list[tuple[_Run, str | None, dict | None, dict | None,
                     dict | None]] = [(incumbent, None, None, None, None)]

    # the family's organic irregularity — flags the organic-asterisk label,
    # whose single-add run cannot re-detect the family it idealizes
    fam_irregular = False
    if radial is not None:
        _hf, blades_h, sp_h = radial
        pitch_h = 360.0 / sp_h["n"]
        hp_h = parts[_hf].primitive
        angs_h = sorted(
            math.degrees(math.atan2(
                prim_center(parts[q].primitive)[1] - hp_h.cy,
                prim_center(parts[q].primitive)[0] - hp_h.cx)) % 360.0
            for q in blades_h)
        gaps_h = [(angs_h[(k + 1) % len(angs_h)] - angs_h[k]) % 360.0
                  for k in range(len(angs_h))]
        fam_irregular = (max(abs(gv - pitch_h) for gv in gaps_h)
                         > RADIAL_REG_TOL_DEG)

    def _start_variant(vparts: list[Part], vlabel: str, absorbed: dict,
                       imap: dict, vkind: str, reg_check: bool = False,
                       roles_override: list | None = None,
                       enable_sym: bool = True,
                       fid_ref_area: bool = False):
        """Run one variant. round 4: "extrapolate"/"decimate"/"removal" runs
        are searched against their OWN reference and carry rinfo (fid_ref =
        floor-substitute fidelity; removed_px for removal kinds). The
        arbitration key stays full-reference."""
        v_solos = [_raster_part(p.primitive, h, w) for p in vparts]
        # round 5: substitution variants whose parts map 1:1 onto the
        # originals INHERIT roles — re-inference on reduced 1-px forms
        # inverts dominance (the accent square outweighing the X strokes)
        v_roles = (roles_override if roles_override is not None
                   else infer_roles(vparts, canvas_hw))

        def vtag(k: int, _imap=imap) -> str:
            oi = _imap.get(k, k)
            return "L" if oi == -1 else f"#{oi}"

        if vkind in ("extrapolate", "decimate", "removal"):
            vref = reference_mask(vparts, canvas_hw, target)
            r = _run(vparts, v_roles, v_solos, vtag, ref_run=vref,
                     pockets_run=_pocket_masks(vref),
                     topo_ref_run=_topology(vref), reg_check=reg_check,
                     enable_sym=enable_sym)
            # fid_ref: AREA IoU only for FAMILY (comb) variants — at
            # min-feature blade scale the boundary term measures raster
            # phase, not structure. Non-family variants keep the boundary
            # term: it is what polices silhouette spill (figma's step).
            # Round 6: fid_ref_area extends the same doctrine to slopereg
            # runs whose SUBSTITUTED parts are all at thin-stroke scale —
            # the caller computes that scope; figma-class geometry (9-px
            # circles) never qualifies, so the round-4 leak stays closed.
            fam_v = any(
                _radial_family(vparts, [q for q, p_ in enumerate(vparts)
                                        if p_.group == g_ and p_.op == "add"])
                is not None
                for g_ in sorted({p_.group for p_ in vparts if p_.group >= 0}))
            fr_ = (_iou_bool(r.mask, vref) if (fam_v or fid_ref_area)
                   else min(_iou_bool(r.mask, vref),
                            _boundary_iou(r.mask, vref, _band_width(vref))))
            rinfo: dict = {"kind": ("extrapolate" if vkind == "extrapolate"
                                    else "removal"),
                           "fid_ref": fr_}
            rinfo["topo_ref"] = _topology(vref)
            # the canonical escape, like the main path's topo_canon: a thin-
            # bladed comb's REFERENCE can degenerate at tiny T while the
            # render is cleanly connected
            rinfo["topo_canon"] = _topology(_compound_mask(vparts, h, w))
            if vkind in ("decimate", "removal") and absorbed:
                ftv = target / max(h, w)
                rem = np.zeros((target, target), np.uint8)
                for i in absorbed:
                    rem |= _raster_part(_scale_part(parts[i], ftv).primitive,
                                        target, target)
                rinfo["removed_px"] = int(np.count_nonzero(ref & rem))
            return r, vlabel, absorbed, imap, rinfo
        r = _run(vparts, v_roles, v_solos, vtag)
        if (vkind == "star_organic" and fam_irregular
                and target <= COLLAPSE_MAX_T):
            r.violations.append("radial regularity: organic idealization of "
                                "an irregular family [reg]")
        return r, vlabel, absorbed, imap, None

    for vparts, vlabel, absorbed, imap, vkind in _collapse_variants(
            parts, roles_orig, canon_solos, canvas_hw, target):
        runs.append(_start_variant(vparts, vlabel, absorbed, imap, vkind))

    # round 4 — translational EXTRAPOLATED variant (figma's latent grid):
    # the snapped part list competes like any variant; imap is identity
    if reg_detect is not None and target <= COLLAPSE_MAX_T:
        groups_r0, snapped_parts = reg_detect
        runs.append(_start_variant(
            snapped_parts,
            f"regularize {len(groups_r0['size'])}grp"
            f"/{len(groups_r0['pitch'])}pitch",
            {}, {k: k for k in range(len(parts))}, "extrapolate",
            reg_check=True))

    # round 5 — the L-CORNER doctrine as an EXTRAPOLATED variant (user-drawn
    # ideal: "diagonals reduced into L corners"): thin diagonal strokes
    # replaced by their Z axis-run reductions, searched against their own
    # reference like every extrapolation — the candidate-level forms alone
    # cannot beat the most-faithful fallback in sub-floor territory, and
    # "less accurate but more true to the style" is the point.
    if target <= COLLAPSE_MAX_T:
        ft5 = target / max(h, w)
        quals: dict[int, tuple[float, float, float, float, float]] = {}
        for i5, p5 in enumerate(parts):
            pr5 = p5.primitive
            if p5.op != "add" or not isinstance(
                    pr5, (RotatedRectangle, RoundedRectangle, RotatedEllipse)):
                continue
            if radial is not None and i5 in radial[1]:
                continue              # comb territory, never L-reduced
            lo5, hi5 = _local_dims(pr5)
            ang5 = min(pr5.angle % 90.0, 90.0 - (pr5.angle % 90.0))
            if not (lo5 > 0 and hi5 >= 2.0 * lo5
                    and lo5 * ft5 <= _min_feature_px(target) + 1.0
                    and ang5 > 10.0):
                continue
            if isinstance(pr5, (RotatedRectangle, RoundedRectangle)):
                long_along = pr5.w >= pr5.h
            else:
                long_along = pr5.rx >= pr5.ry
            th5 = math.radians(pr5.angle + (0.0 if long_along else 90.0))
            cx5, cy5 = prim_center(pr5)
            hl5 = hi5 / 2.0
            # 1-px floor, NOT min-feature: the reduction's whole point is
            # that 1-px AXIS runs are stable, refinement-safe forms (min-
            # feature polices 1-px DIAGONAL steps) — the drawn ideal is a
            # 1-px stroke
            w5 = max(lo5, 1.0 / ft5)
            quals[i5] = (cx5 - hl5 * math.cos(th5),
                         cy5 - hl5 * math.sin(th5),
                         cx5 + hl5 * math.cos(th5),
                         cy5 + hl5 * math.sin(th5), w5)
        # crossing strokes share ONE band at the crossing's transverse
        # coordinate, in the frame of the TIP SPREAD — the drawn X's band
        # lies on the pair axis; two per-stroke midline bands render as
        # parallel rails, not an X
        band_for: dict[int, tuple[str, float]] = {}
        by_grp: dict[int, list[int]] = {}
        for i5 in sorted(quals):
            by_grp.setdefault(parts[i5].group, []).append(i5)
        for g5 in sorted(by_grp):
            mem5 = by_grp[g5]
            if len(mem5) < 2:
                continue
            cents = {i5: ((quals[i5][0] + quals[i5][2]) / 2.0,
                          (quals[i5][1] + quals[i5][3]) / 2.0)
                     for i5 in mem5}
            span5 = min(math.hypot(quals[i5][2] - quals[i5][0],
                                   quals[i5][3] - quals[i5][1])
                        for i5 in mem5)
            cxm = sum(c[0] for c in cents.values()) / len(mem5)
            cym = sum(c[1] for c in cents.values()) / len(mem5)
            if not all(math.hypot(c[0] - cxm, c[1] - cym) <= 0.3 * span5
                       for c in cents.values()):
                continue              # not a crossing cluster
            # band axis = along the composite's TIP SPREAD (a tall X takes
            # a vertical center band — the transposed drawn form). The band
            # sits on the crossing center, which the pair flip maps to
            # itself, so the strokes SHARE it.
            tx5 = [v for i5 in mem5 for v in (quals[i5][0], quals[i5][2])]
            ty5 = [v for i5 in mem5 for v in (quals[i5][1], quals[i5][3])]
            ba = (("x", cym, cxm)
                  if max(tx5) - min(tx5) >= max(ty5) - min(ty5)
                  else ("y", cxm, cym))
            for i5 in mem5:
                band_for[i5] = ba
        red_parts = list(parts)
        red_idx: list[int] = []
        for i5 in sorted(quals):
            x0e, y0e, x1e, y1e, w5 = quals[i5]
            baxis, bat, brc = band_for.get(i5, (None, None, None))
            eff_axis = baxis or ("x" if abs(x1e - x0e) >= abs(y1e - y0e)
                                 else "y")
            tspan = abs(y1e - y0e) if eff_axis == "x" else abs(x1e - x0e)
            # form selection (user-locked): the drawn single-corner Z while
            # its stubs stay short at target; near-square strokes take the
            # double-corner 3-run form so the crossing identity survives
            form5 = "z" if tspan * ft5 / 2.0 <= 2.5 else "z2"
            # the idealization is built ON THE TARGET GRID: endpoints, band
            # and width snap to integer pixel centers, so every run rasters
            # crisp (a float-positioned 1-px run straddles a boundary and
            # fattens to 2 px, which its own coverage-reference then counts
            # as spill)
            # 1-px runs build at width 1.6 (half-thickness 0.3): the
            # quarter-px Polygon raster quantizes 0.3 -> 0.25 so the run
            # stays 1 px, while the >=0.5 coverage reference sees 0.6 and
            # keeps it — at width exactly 1 the form is a zero-thickness
            # polyline and the reference comes out EMPTY (coverage 1/k),
            # flooring every candidate; at 0.49 the quarter-px grid rounds
            # the corner to the 0.5 TIE, which fills outward to 2 px
            w_run5 = max(1, round(w5 * ft5))
            pts5t = stroke_axis_runs(
                float(round(x0e * ft5)), float(round(y0e * ft5)),
                float(round(x1e * ft5)), float(round(y1e * ft5)),
                1.6 if w_run5 == 1 else float(w_run5), form5,
                band_axis=baxis,
                band_at=None if bat is None else float(round(bat * ft5)),
                run_center=None if brc is None
                else float(round(brc * ft5)))
            if pts5t is None:
                continue
            red_parts[i5] = Part(
                Polygon(pts=tuple((x / ft5, y / ft5) for x, y in pts5t)),
                parts[i5].op, parts[i5].group)
            red_idx.append(i5)
        if red_idx:
            # round 8 — axisreduce symmetry predicate. Round 5 disabled sym
            # unconditionally because for thin_x the reduced crossing pair IS
            # the whole shape (exact mirrors by construction; the off-axis
            # accent kills the global axis) and the pair-flip half-px
            # detection was the only asymmetry left. But vscode reduces only
            # 2 of its parts; the other 5 (spine/triangles/ellipse) then
            # render with NO global symmetry → fall back, asymmetric. Enable
            # sym iff there are non-reduced live adds AND a real global axis
            # exists on the reduced part list — gate on _sym_prep presence
            # (never let _run fall through to _local_pairs, which would
            # re-pair thin_x's two bars and break its byte-locked win).
            ar_reduced = set(red_idx)
            ar_nonreduced = [q for q, p_ in enumerate(parts)
                             if p_.op == "add" and q not in ar_reduced]
            ar_solos = [_raster_part(p_.primitive, h, w) for p_ in red_parts]
            ar_sym = (bool(ar_nonreduced)
                      and _sym_prep(red_parts, ar_solos, canvas_hw, target)
                      is not None)
            runs.append(_start_variant(
                red_parts,
                "axisreduce " + ",".join(f"#{i5}" for i5 in red_idx),
                {}, {k: k for k in range(len(parts))}, "extrapolate",
                roles_override=list(roles_orig),
                enable_sym=ar_sym))

    # round 6 — run-≥2 GEOMETRIC EXTENSION as an EXTRAPOLATED variant: a
    # diagonal whose nearest Θ member is 45° renders 1-px teeth at length
    # (or, unsnapped, irregular bump trains — "many bumps"); the run-2-
    # snapped form is the style ideal ("avoided unless completely
    # necessary") but rotating long thin geometry breaks retention/fid at
    # candidate level (thin_x@32/64: the run-2 base is OFFERED and floored).
    # Substitute the snapped geometry and judge it against ITS OWN
    # reference, like every extrapolation; the arbitration key stays
    # full-reference. Radial blades exempt (comb angular pitch outranks
    # the slope grid); 1:1 substitution → roles inherited.
    # T ≥ 32 only: the L-corner reduction owns thin diagonals at tiny T
    # (round-5 locked, user-drawn ideal — "heavy 1px stepping" is the
    # tiny-T complaint); the slope grid owns the sizes where uniform
    # run-words actually read as lines. At 16/24 the sym-off slopereg
    # form measured fid_ref 0.289/0.544 — garbage that would displace
    # the locked doctrine on raw J.
    ft6 = target / max(h, w)
    sub6 = list(parts)
    sub_idx6: list[int] = []
    for i6, p6 in (enumerate(parts) if target >= 32 else ()):
        pr6 = p6.primitive
        if p6.op != "add":
            continue
        if radial is not None and i6 in radial[1]:
            continue
        if isinstance(pr6, (RotatedRectangle, RoundedRectangle,
                            RotatedEllipse)):
            lo6, hi6 = _local_dims(pr6)
            ang6 = min(pr6.angle % 90.0, 90.0 - (pr6.angle % 90.0))
            if not (lo6 > 0 and hi6 >= 2.0 * lo6 and ang6 > JAG_ANGLE_TOL
                    and abs(snap_slope(pr6.angle) % 90.0 - 45.0) < 1e-6):
                continue
            r2a = snap_slope_run2(pr6.angle)
            if abs(r2a - pr6.angle % 180.0) <= 1e-6:
                continue
            sub6[i6] = Part(replace(pr6, angle=r2a), p6.op, p6.group)
            sub_idx6.append(i6)
        elif isinstance(pr6, Triangle):
            need6 = False
            for (ax6, ay6), (bx6, by6) in ((pr6.v1, pr6.v2),
                                           (pr6.v2, pr6.v3),
                                           (pr6.v3, pr6.v1)):
                if (math.hypot(bx6 - ax6, by6 - ay6) * ft6
                        <= K45 * math.sqrt(2.0)):
                    continue
                th6 = math.degrees(math.atan2(by6 - ay6, bx6 - ax6)) % 90.0
                if abs(th6 - 45.0) <= TRI_SNAP_TOL[1]:
                    need6 = True
                    break
            if not need6:
                continue
            pts6 = snap_triangle_run2(pr6)
            if pts6 is None:
                continue
            sub6[i6] = Part(Triangle(*pts6), p6.op, p6.group)
            sub_idx6.append(i6)
    if sub_idx6:
        # thin-stroke scope for the area-only fid_ref (round-4 doctrine:
        # boundary IoU at ≤min-feature stroke scale is raster phase, not
        # structure — thin_x@32's bars are 1.5 px wide)
        thin6 = all(_local_dims(sub6[q].primitive)[0] * ft6
                    <= _min_feature_px(target) + 1.0 for q in sub_idx6)
        runs.append(_start_variant(
            sub6, "slopereg " + ",".join(f"#{i6}" for i6 in sub_idx6),
            {}, {k: k for k in range(len(parts))}, "extrapolate",
            roles_override=list(roles_orig), fid_ref_area=thin6,
            enable_sym=False))

    # round 9 — SUBTRACT-side diagonal reduction: gmail's inner-V and vscode's
    # interior chevron carve are SUBTRACT Triangles — invisible to the
    # add-only axisreduce/slopereg, so their diagonals never reduce (gmail's V
    # renders raw; vscode's raw carve cuts through the reduced add-bars and
    # fragments them into holes). Reduce qualifying subtract Triangles via
    # snap_triangle_run2 (op-agnostic edge→Θ snap — a subtract carve has NO
    # add-stroke width-duality, so it reaches T=16 unlike slopereg's strokes)
    # so the negative space reads as clean run-≥2/axis edges. Triangle-only
    # (serif RRect/Ellipse subtracts excluded — out of the V's scope); the
    # SAME gate the add path uses (edge > K45·√2, |edge%90−45| ≤ TRI_SNAP_TOL
    # [1]) — which keeps discord's >13.5°-off subtract triangles out; radial-
    # skipped; CSG-correct (op="subtract"/group preserved, judged vs own ref).
    subr = list(parts)
    subr_idx: list[int] = []
    if target <= 16:
        # round 9 — scoped to the LOWEST resolution (all three round-9
        # complaints are 16×16). At T ≥ 24 the diagonal carve is more
        # representable and the reduction's fidelity cost isn't worth it —
        # gmail@32 subreduce dropped fid to 0.57 (< 0.65 floor, shipping only
        # via the extrapolation waiver, the M's central peak melting). Same
        # tiny-T scope as the figma outline-separation and the round-5/6
        # L-corner doctrine.
        for i9, p9 in enumerate(parts):
            pr9 = p9.primitive
            if p9.op != "subtract" or not isinstance(pr9, Triangle):
                continue
            if radial is not None and i9 in radial[1]:
                continue
            need9 = False
            for (ax9, ay9), (bx9, by9) in ((pr9.v1, pr9.v2), (pr9.v2, pr9.v3),
                                           (pr9.v3, pr9.v1)):
                if (math.hypot(bx9 - ax9, by9 - ay9) * ft6
                        <= K45 * math.sqrt(2.0)):
                    continue
                th9 = math.degrees(math.atan2(by9 - ay9, bx9 - ax9)) % 90.0
                if abs(th9 - 45.0) <= TRI_SNAP_TOL[1]:
                    need9 = True
                    break
            if not need9:
                continue
            pts9 = snap_triangle_run2(pr9)
            if pts9 is None:
                continue
            subr[i9] = Part(Triangle(*pts9), p9.op, p9.group)
            subr_idx.append(i9)
    if subr_idx:
        runs.append(_start_variant(
            subr, "subreduce " + ",".join(f"#{i9}" for i9 in subr_idx),
            {}, {k: k for k in range(len(parts))}, "extrapolate",
            roles_override=list(roles_orig)))

    def _feasible(r: _Run, absorbed: dict | None = None,
                  rinfo: dict | None = None) -> bool:
        fid_ok = fidelity(r.mask) >= floor
        if (not fid_ok and rinfo is not None
                and rinfo.get("fid_ref") is not None):
            # round 4 floor waiver: removal variants judge the floor vs the
            # survivors' reference, extrapolated variants vs the idealized
            # geometry; fid_ref NEVER enters the arbitration key
            fid_ok = rinfo["fid_ref"] >= floor
        topo_ok = (_topology(r.mask) in (topo_ref_v, topo_canon_v)
                   or bool(r.drops) or bool(absorbed)
                   # reference-substituted runs are judged against their OWN
                   # reference's topology (a comb's holes differ from the
                   # organic source's), with the same canonical escape the
                   # main path has
                   or (rinfo is not None
                       and _topology(r.mask) in (rinfo.get("topo_ref"),
                                                 rinfo.get("topo_canon"))))
        return fid_ok and not r.violations and r.battery and topo_ok

    # round 4 — phase B, removal on globbing evidence: ONLY when nothing
    # is feasible AND the incumbent carries glob evidence; accent-role
    # offenders only (lobes/dominant are identity — claude's blades are
    # LOBES: family-ladder territory, never this path); twin-closed; one
    # removal phase, no recursion; appended LAST so the key's index term
    # tie-breaks toward the incumbent.
    if not any(_feasible(r, ab, ri) for r, _v, ab, _im, ri in runs):
        cand_off = sorted(set(incumbent.glob_offenders)
                          | {i for i, c in incumbent.viol_class.items()
                             if c == "glob"})
        offenders = [i for i in cand_off
                     if i < len(parts) and parts[i].op == "add"
                     and roles_orig[i] == "accent"]
        if offenders:
            spz = _sym_prep(parts, canon_solos, canvas_hw, target)
            if spz is not None:
                lo = spz[4]
            else:
                lo, _pa, _pd = _local_pairs(parts, canon_solos, canvas_hw,
                                            target)

            def _closure(i: int) -> frozenset:
                s = {i}
                if i in lo:
                    s.add(lo[i])
                s |= {j for j, ld in lo.items() if ld in s}
                return frozenset(s)

            rm_sets: list[list[int]] = []
            seen: set[frozenset] = set()
            for i in offenders:
                cs = _closure(i)
                if cs not in seen:
                    seen.add(cs)
                    rm_sets.append(sorted(cs))
            rm_sets = rm_sets[:3]
            if len(rm_sets) >= 2:
                rm_sets.append(sorted({q for s_ in rm_sets for q in s_}))
            ref_ink = int(np.count_nonzero(ref))
            ftv = target / max(h, w)
            for rm in rm_sets:
                rem = np.zeros((target, target), np.uint8)
                for i in rm:
                    rem |= _raster_part(_scale_part(parts[i], ftv).primitive,
                                        target, target)
                if int(np.count_nonzero(ref & rem)) \
                        > REMOVE_MAX_FRAC * ref_ink:
                    continue              # too much identity to spend
                keep_idx = [i for i in range(len(parts)) if i not in rm]
                runs.append(_start_variant(
                    [parts[i] for i in keep_idx],
                    "remove " + ",".join(f"#{i}" for i in rm),
                    {i: f"removed: presence globs at T={target} "
                        f"(sep/pocket evidence)" for i in rm},
                    {k: i for k, i in enumerate(keep_idx)}, "removal"))

    if os.environ.get("R1_DEBUG"):        # arbitration trace (debug harness)
        for k, (r_, vl_, ab_, _im_, ri_) in enumerate(runs):
            fr_d = f"{ri_['fid_ref']:.3f}" if ri_ else "-"
            print(f"        run[{k}] {vl_ or 'incumbent':<34} "
                  f"feas={int(_feasible(r_, ab_, ri_))} "
                  f"J={jag_score(r_.mask):6.1f} fid={fidelity(r_.mask):.3f} "
                  f"fid_ref={fr_d} "
                  f"viol={len(r_.violations)} batt={int(r_.battery)} "
                  f"n={r_.n_live}")
            for v_ in r_.violations:
                print(f"            VIOL: {v_[:100]}")
    # round 9 — hole guard in the arbitration key: a render that opens
    # INTERIOR VOIDS absent from the reference (vscode's fragmented chevron —
    # the raw subtract carve cutting the reduced bars into 2 holes) must never
    # win on jag alone (the jag metric is blind to interior topology). Penalize
    # holes beyond what the reference AND the canonical decomposition have (so
    # legitimate counters — nested_counter, annulus — and a variant's own ref
    # holes are never charged); sorts right after feasibility, before jag.
    def _excess_holes(k: int) -> int:
        # bar = the MAIN reference's hole count (the true silhouette resolved
        # at THIS target). NOT the canonical full-res compound (vscode's
        # canonical has 2 holes but the 16-px ref resolves 1 — a render
        # opening 2 there is the artifact) and NOT the variant's own ref
        # (the full-reference arbitration judges against the true silhouette).
        # Under-holed renders (nested_counter: 1 < ref 2) score 0 — only
        # EXCESS interior voids are charged.
        return max(0, _topology(runs[k][0].mask)[1] - topo_ref_v[1])

    best_i = min(range(len(runs)), key=lambda k: (
        not _feasible(runs[k][0], runs[k][2], runs[k][4]),
        _excess_holes(k),
        jag_score(runs[k][0].mask),
        -fidelity(runs[k][0].mask), runs[k][0].n_live, k))
    best, blabel, babsorbed, bimap, brinfo = runs[best_i]

    if blabel is None:
        records, drops, violations = best.records, best.drops, best.violations
    else:
        # remap the winning variant's records onto ORIGINAL part indices;
        # absorbed parts get 1:1-accounted records, the label a synthetic one
        rec_by_orig: dict[int, TweakRecord] = {}
        label_rec: TweakRecord | None = None
        for k, rec in enumerate(best.records):
            oi = bimap[k]
            if oi == -1:
                label_rec = replace(rec, index=len(parts))
            else:
                rec_by_orig[oi] = replace(rec, index=oi)
        records = []
        for i in range(len(parts)):
            if i in babsorbed:
                records.append(TweakRecord(i, roles_orig[i], 0.0, 0, DEG_TRUE,
                                           1.0, 1.0, 0.0, 0.0, babsorbed[i],
                                           None))
            elif i in rec_by_orig:
                records.append(rec_by_orig[i])
        if label_rec is not None:
            records.append(label_rec)
        if babsorbed:
            prefix = ("removal win" if brinfo is not None
                      and brinfo.get("kind") == "removal"
                      and blabel.startswith("remove") else "collapse win")
            drops = best.drops + [
                f"{prefix}: {blabel} — absorbed "
                + ",".join(f"#{i}" for i in sorted(babsorbed))]
        else:
            drops = best.drops + [f"variant win: {blabel}"]
        violations = best.violations

    final = best.mask
    return RenderResult(
        mask=final, ref=ref, naive=naive, records=records,
        jag=jag_score(final), jag_naive=jag_score(naive),
        jag_norm=jag_norm(final), jag_norm_naive=jag_norm(naive),
        area_iou=_iou_bool(final, ref),
        boundary_iou=_boundary_iou(final, ref, band),
        floor=floor, violations=violations, drops=drops,
        topo_render=_topology(final), topo_ref=topo_ref_v,
        topo_canon=topo_canon_v,
        sym=best.sym, axis=best.axis, collapsed=blabel,
        pairs=best.pairs, staircase=_staircase_total(final),
        fid_ref=(brinfo.get("fid_ref") if brinfo is not None else None),
        ref_kind=(brinfo.get("kind") if brinfo is not None else None),
        removed_ink=(brinfo.get("removed_px", 0) if brinfo is not None else 0),
        reg=("radial" if radial is not None
             else ("group" if reg_detect is not None else None)),
    )


def render(parts: list[Part], canvas_hw: tuple[int, int], target: int) -> np.ndarray:
    """The R1 render: (target, target) uint8 {0,255}. Deterministic (H8)."""
    return render_debug(parts, canvas_hw, target).mask


# the cross-construction ASSEMBLY layer + construction builder (production;
#    lifted from the render_debug harness so application/pipeline.py never
# imports a debug module — the harness imports these back)

_FX_BG = (238, 238, 238)   # composite background: white-on-white features stay visible


def build_constructions(fs, sub_case: str, tag: str = "?"):
    """Per-segment CSG constructions in FULL-canvas coords from an L2
    FeatureSegmentation (PURE — no disk I/O; the production build and the
    render_debug harness both call this; the harness's _fixture_constructions
    is the thin pkl-reading wrapper). Classified -> 1-part add; Feature ->
    decompose_shape (computed once, target-independent) offset by (bbox - pad),
    label-cleaned. Returns (S, cons, src): cons = [(parts, color_rgb, label)],
    src = the SOURCE silhouette at S (the union of the segments' own masks —
    the _bridge_source_gaps connectivity ground truth)."""
    S = int(fs.image_shape[0])
    min_feat = _min_feature_px(S)
    cons: list[tuple[list[Part], tuple[int, int, int], str]] = []
    src = np.zeros((S, S), np.uint8)
    n_residual = 0
    for si, seg in enumerate(fs.segments):
        if isinstance(seg, Classified):
            src |= _raster_part(seg.primitive, S, S)
            cons.append(([Part(seg.primitive, "add")], seg.color_rgb,
                         f"#{si} L1 {type(seg.primitive).__name__}"))
            continue
        if not isinstance(seg, Feature):
            continue
        _bx, _by, _bw, _bh = seg.bbox
        src[_by:_by + _bh, _bx:_bx + _bw] |= (seg.mask > 0).astype(np.uint8) * 255
        t0 = time.time()
        d = decompose_shape(seg.mask, min_feat, seg.shape_hint,
                            ring=seg.ring_geom, bbox_xy=(seg.bbox[0], seg.bbox[1]))
        # WS2 label-cleaner: prune/merge/re-fit/normalize/re-select before render.
        # Rebuild the SAME padded target the decomposer scored against
        # (decompose.py builds target from seg.mask at d.pad / d.shape).
        _th, _tw = seg.mask.shape[:2]
        _src = np.zeros(d.shape, np.uint8)
        _src[d.pad:d.pad + _th, d.pad:d.pad + _tw] = seg.mask
        d = clean_labels(d, _src, sub_case, min_feat)
        dt = time.time() - t0
        if d.residual is not None:
            n_residual += 1
        if not d.parts:
            print(f"  {tag}#{si}: decompose returned no parts — deferred "
                  f"(R3 residual territory) [{dt:.1f}s]")
            continue
        ox, oy = seg.bbox[0] - d.pad, seg.bbox[1] - d.pad
        parts = [_offset_part(p, ox, oy) for p in d.parts]
        cons.append((parts, seg.color_rgb, f"#{si} D2 n={len(parts)}"))
        print(f"  {tag}#{si} decomposed: {len(parts)} parts "
              f"cov a{d.coverage:.2f}/b{d.boundary_cov:.2f} [{dt:.1f}s]")
    if n_residual:
        print(f"  {tag}: {n_residual} feature(s) carry a residual "
              f"(R3 tier, not rendered)")
    return S, cons, src




SEP_MIN_CONS = 4   # divots fire only on OVERREPRESENTED multi-element marks (>= this many
#                    constructions — slack's 8-pill pinwheel). A 2-3 piece glyph (adobe's
#                    A + i-stem + i-dot) is NOT force-separated: the i-dot stays attached to
#                    its stem (user: 'dot attached to stem'). A size guard can't do this —
#                    at 16px slack's short pills are the SAME size as the i-dot — so the
#                    discriminator is piece COUNT (the user's 'overrepresented' principle).


HARM_COPY_IOU = 0.95      # centered solo-raster IoU gate: group members must be translated COPIES


def _harmonize_singles(masks: list[np.ndarray], cons, S: int, target: int
                       ) -> tuple[list[np.ndarray], frozenset]:
    """R1 cross-construction stamp harmonization (docker's nine container squares):
    constructions that are single-add-part translated COPIES with near-equal canonical
    extents — detected by `render._regularity_groups` on the flat single-part list (H5
    reuse) — are re-rendered as ONE shared stamp placed on per-axis integer pitch
    combs, so every member rasterizes the SAME HxW at the same grid parity
    (independent per-construction rounding made docker's squares 2x1…3x2, some not
    even square). Returns (masks, protected construction indices); protected members
    must not be eaten by _separate_divots' contact-clear (the whale render spills into
    the container rows — probe-proven). Fires only when >= 3 single-part constructions
    form an aligned size group (probe: docker alone across the fixture set).
    Deterministic (H8: sorted orders, median/first-cluster anchors, integer combs)."""
    singles = [i for i, (parts, _c, _l) in enumerate(cons)
               if len(parts) == 1 and parts[0].op == "add"]
    if len(singles) < 3:
        return masks, frozenset()
    flat = [cons[i][0][0] for i in singles]
    det = _regularity_groups(flat, (S, S))
    if det is None:
        return masks, frozenset()
    groups, _snapped = det
    out = list(masks)
    protected: set[int] = set()
    tol = REG_ALIGN_TOL * S

    def _cluster_means(vals: list[float]) -> list[float]:
        means: list[list[float]] = []
        for v in sorted(vals):
            if means and v - means[-1][-1] <= tol:
                means[-1].append(v)
            else:
                means.append([v])
        return [sum(c) / len(c) for c in means]

    for grp in groups["size"]:
        mem = [singles[j] for j in grp]                    # construction indices
        if len(mem) < 3:
            continue
        solos = [_compound_mask(cons[ci][0], S, S) for ci in mem]
        cents, boxes, ok = [], [], True
        for m in solos:
            ys, xs = np.nonzero(m)
            if xs.size == 0:
                ok = False
                break
            cents.append((float(ys.mean()), float(xs.mean())))
            boxes.append((int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)))
        if not ok:
            continue

        def _centered(m: np.ndarray, cy: float, cx: float) -> np.ndarray:
            dy, dx = int(round(S / 2 - cy)), int(round(S / 2 - cx))
            c = np.zeros_like(m)
            ys, xs = np.nonzero(m)
            c[np.clip(ys + dy, 0, S - 1), np.clip(xs + dx, 0, S - 1)] = 255
            return c

        ref0 = _centered(solos[0], *cents[0])
        if any(_iou_bool(_centered(solos[q], *cents[q]) > 0, ref0 > 0) < HARM_COPY_IOU
               for q in range(1, len(mem))):
            continue                                       # not translated copies
        row_means = _cluster_means([c[0] for c in cents])
        col_means = _cluster_means([c[1] for c in cents])
        f = target / S

        def _comb(means: list[float]) -> dict[int, int]:
            """cluster index -> target-space position on an integer pitch comb."""
            base = int(round(means[0] * f))
            if len(means) < 2:
                return {0: base}
            pitch = max(1, int(round((means[-1] - means[0]) / (len(means) - 1) * f)))
            return {q: base + q * pitch for q in range(len(means))}

        rows_t, cols_t = _comb(row_means), _comb(col_means)
        med_h = sorted(b[0] for b in boxes)[len(boxes) // 2]
        med_w = sorted(b[1] for b in boxes)[len(boxes) // 2]
        row_pitch = (max(rows_t.values()) - min(rows_t.values())) // max(1, len(rows_t) - 1) \
            if len(rows_t) > 1 else target
        col_pitch = (max(cols_t.values()) - min(cols_t.values())) // max(1, len(cols_t) - 1) \
            if len(cols_t) > 1 else target
        sh = min(max(1, int(round(med_h * f))), max(1, row_pitch - 1))
        sw = min(max(1, int(round(med_w * f))), max(1, col_pitch - 1))

        def _nearest(q_means: list[float], v: float) -> int:
            return min(range(len(q_means)), key=lambda q: (abs(q_means[q] - v), q))

        for ci, (cy, cx) in zip(mem, cents):               # sorted flat order (mem ascending)
            py = rows_t[_nearest(row_means, cy)]
            px = cols_t[_nearest(col_means, cx)]
            y0 = min(max(0, py - sh // 2), target - sh)
            x0 = min(max(0, px - sw // 2), target - sw)
            stamp = np.zeros((target, target), np.uint8)
            stamp[y0:y0 + sh, x0:x0 + sw] = 255
            out[ci] = stamp
            protected.add(ci)
    return out, frozenset(protected)


def _canon_masks(cons, S: int) -> list[np.ndarray]:
    """Per-construction canonical (S,S) compound masks — the source-truth footprints
    the cross-construction assembly steps key on (divot separation's abut test, the
    source-gap bridge's component vote). One helper, not per-step comprehensions (H5)."""
    return [_compound_mask(parts, S, S) for parts, _c, _l in cons]


def _separate_divots(masks: list[np.ndarray], cons, S: int,
                     protected: frozenset = frozenset()) -> list[np.ndarray]:
    """Re-open 1px divots between constructions that are SEPARATE in the source but whose
    rendered masks fused at the feed resolution (slack's 8 pinwheel pills 'glob together
    for no good reason'). Gated to OVERREPRESENTED marks (>= SEP_MIN_CONS constructions);
    a 2-3 piece glyph is left as rendered. Source-separate iff CANONICAL footprints are
    8-disconnected at canvas res (unambiguous, before the feed-res downsample globs the
    gaps — slack's gaps are sub-pixel in the reference at 16, so a reference-based test
    under-separates). Where such a pair's rendered masks touch, the contact pixels are
    cleared from the later (area-sorted smaller, drawn on top) construction so each label
    keeps a >=1px gap; pairs that genuinely abut in the source are left fused. `protected`
    (R1 harmonized-stamp members) are never eaten: a protected-vs-unprotected contact
    clears from the UNPROTECTED side (the whale render spills into docker's container
    rows and would otherwise erase the 1px stamps); protected-vs-protected contacts are
    left (stamps of one comb never touch by construction). (H8)."""
    if len(masks) < SEP_MIN_CONS:
        return masks
    k = np.ones((3, 3), np.uint8)
    canon = _canon_masks(cons, S)
    out = [m.copy() for m in masks]
    for i in range(len(masks)):
        di = cv2.dilate(masks[i], k)
        ci = cv2.dilate(canon[i], k)
        for j in range(i + 1, len(masks)):
            if np.any((ci > 0) & (canon[j] > 0)):
                continue                              # abut in source — preserve the join
            contact = (di > 0) & (masks[j] > 0)
            if not np.any(contact):
                continue
            if j in protected and i not in protected:
                back = (cv2.dilate(masks[j], k) > 0) & (out[i] > 0)
                out[i] = out[i].copy()
                out[i][back] = 0                      # clear from the unprotected side
            elif not (i in protected and j in protected):
                out[j] = out[j].copy()
                out[j][contact] = 0
    return out


NEST_MIN_GAP = 3      # min inter-ring band (px) between nested label rings — the user's
#                       "enough space for a 3x3 square between them": a 1px accent then
#                       clears both rings by 1px inside the band's corner pocket
NEST_ACC3_GAP = 5     # band width that hosts the RATIFIED 3x3-OUTLINE accent (@24): 3px
#                       dot + 1px clearance to each ring; falls back to a 1px accent in a
#                       NEST_MIN_GAP band when the interior can't afford 2x5 + a viable ring
NEST_ACC_DIM_FRAC = 0.2   # accent's canonical max-dim <= this x min(outer box dims) — a
#                           sub-speck mark (instagram's camera dot), never a sibling glyph


def _coordinate_nested_rings(masks: list[np.ndarray], cons, S: int, target: int
                             ) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Coordinate NESTED label-ring constructions + their inter-ring accent (instagram:
    body ring ⊃ lens ring + camera dot). Per-construction cubify renders each ring alone,
    so nothing guarantees the composed geometry: the lens lands off-center (unequal gaps),
    and the dot's floor-mapped center collides with a ring corner (it OVERLAPPED both
    rings @24). This assembly pass re-derives the composition from the same `label_ring_spec`
    the label path renders (one source of truth): the inner ring is RE-CENTERED in the
    outer's interior (equal per-axis gaps, parity-matched), SHRUNK when the natural band
    is under NEST_MIN_GAP (the user's 3x3-pocket floor) or when the RATIFIED 3x3-OUTLINE
    accent needs a NEST_ACC3_GAP band, and the accent is PLACED in the interior-corner
    pocket nearest its canonical centroid with >= 1px clearance to both rings — the
    user-ratified forms: 3x3 hollow square (1px stroke) when the band affords 5px, else a
    single px. Fires only when exactly TWO constructions carry ring specs, one strictly
    containing the other, BOTH rendered as their spec rings (the label path actually won),
    and a sub-speck single-add accent lives in the canonical inter-ring band — otherwise
    byte-identical passthrough (youtube-music: one spec; android: zero).

    Returns (masks, accent_protected): accent_protected = the FILLED dxd accent bbox
    (uint8 255; the 3x3 outline's 1x1 interior hole is protected territory too) — the
    zone refine_perimeter carves out and restores, since refine's reference is empty at
    the placed position (composite-ref inseparability, the documented @16 residual).
    Deterministic (H8): spec-derived integer boxes, fixed corner order TL<TR<BL<BR."""
    from .cubify import _map_coords, label_ring_spec, ring_mask  # lazy: breaks the render<->cubify cycle
    specs = [label_ring_spec(parts, (S, S), target) for parts, _c, _l in cons]
    ringed = [i for i, sp in enumerate(specs) if sp is not None]
    if len(ringed) != 2:
        return masks, None

    def _contains(ob, ib) -> bool:
        oy, ox, oh, ow = ob
        iy, ix, ih, iw = ib
        return (oy < iy and ox < ix
                and iy + ih < oy + oh and ix + iw < ox + ow)

    a, b = ringed
    if _contains(specs[a][0], specs[b][0]):
        o, inr = a, b
    elif _contains(specs[b][0], specs[a][0]):
        o, inr = b, a
    else:
        return masks, None
    if (not np.array_equal(masks[o], ring_mask(*specs[o], target))
            or not np.array_equal(masks[inr], ring_mask(*specs[inr], target))):
        return masks, None                 # a ring lost the label path — don't compose blind
    # accent: sub-speck single-add construction whose canonical centroid sits in the
    # inter-ring band (inside the OUTER ring's subtract box, outside the INNER's add box)
    h_c, w_c = S, S
    ob_can = cv2.boundingRect(_raster_part(cons[o][0][1].primitive, h_c, w_c))
    ib_can = cv2.boundingRect(_raster_part(cons[inr][0][0].primitive, h_c, w_c))
    oo_can = cv2.boundingRect(_raster_part(cons[o][0][0].primitive, h_c, w_c))
    canon = _canon_masks(cons, S)
    acc = None
    for ci, (parts, _c, _l) in enumerate(cons):
        if ci in (o, inr) or len(parts) != 1 or parts[0].op != "add":
            continue
        ys, xs = np.nonzero(canon[ci])
        if xs.size == 0:
            continue
        bh = int(ys.max() - ys.min() + 1)
        bw = int(xs.max() - xs.min() + 1)
        if max(bh, bw) > NEST_ACC_DIM_FRAC * min(oo_can[2], oo_can[3]):
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        in_outer = (ob_can[1] <= cy < ob_can[1] + ob_can[3]
                    and ob_can[0] <= cx < ob_can[0] + ob_can[2])
        in_inner = (ib_can[1] <= cy < ib_can[1] + ib_can[3]
                    and ib_can[0] <= cx < ib_can[0] + ib_can[2])
        if in_outer and not in_inner:
            acc = (ci, cy, cx)
            break                          # lowest index wins (deterministic)
    if acc is None:
        return masks, None
    ai, acy, acx = acc
    (oy, ox, oh, ow), wo = specs[o]
    iy0, ix0 = oy + wo, ox + wo            # outer ring interior
    ih_i, iw_i = oh - 2 * wo, ow - 2 * wo
    (_ny, _nx, nih, niw), wi = specs[inr]  # natural inner ring size
    d = 3 if min(ih_i, iw_i) - 2 * NEST_ACC3_GAP >= 2 * wi + 1 else 1
    g_req = NEST_ACC3_GAP if d == 3 else NEST_MIN_GAP
    lh = min(nih, ih_i - 2 * g_req)
    lw = min(niw, iw_i - 2 * g_req)
    if (ih_i - lh) % 2:
        lh -= 1                            # parity-match so the gaps split evenly
    if (iw_i - lw) % 2:
        lw -= 1
    if min(lh, lw) < 2 * wi + 1:
        return masks, None                 # honest partial: no viable inner ring
    gy, gx = (ih_i - lh) // 2, (iw_i - lw) // 2
    out = list(masks)
    out[inr] = ring_mask((iy0 + gy, ix0 + gx, lh, lw), wi, target)
    # corner pocket (gy x gx at the interior corners) nearest the accent's floor-mapped
    # canonical centroid; ties in fixed TL<TR<BL<BR order
    ty, tx = _map_coords(np.array([acy]), np.array([acx]), S, target, True)
    ty, tx = float(ty[0]), float(tx[0])
    corners = (
        (iy0, ix0, "t", "l"), (iy0, ix0 + iw_i - gx, "t", "r"),
        (iy0 + ih_i - gy, ix0, "b", "l"), (iy0 + ih_i - gy, ix0 + iw_i - gx, "b", "r"),
    )
    py, px, vk, hk = min(
        corners,
        key=lambda c: ((c[0] + (gy - 1) / 2.0 - ty) ** 2
                       + (c[1] + (gx - 1) / 2.0 - tx) ** 2))
    sy, sx = gy - d, gx - d                # slack biased toward the OUTER corner
    dy0 = py + (sy // 2 if vk == "t" else sy - sy // 2)
    dx0 = px + (sx // 2 if hk == "l" else sx - sx // 2)
    dot = np.zeros((target, target), np.uint8)
    dot[dy0:dy0 + d, dx0:dx0 + d] = 255
    if d == 3:
        dot[dy0 + 1, dx0 + 1] = 0          # the RATIFIED hollow form (1px stroke)
    out[ai] = dot
    prot = np.zeros((target, target), np.uint8)
    prot[dy0:dy0 + d, dx0:dx0 + d] = 255   # FILLED bbox: the hole px are protected too
    return out, prot


def _bridge_source_gaps(masks: list[np.ndarray], cons, S: int,
                        src: np.ndarray | None) -> list[np.ndarray]:
    """Re-CONNECT constructions the SOURCE connects but the rendered composite left
    apart (android@24's left antenna: the decomposition opened a ~2px hairline at S —
    sub-pixel at feed res — and independent per-construction rounding blew it up to a
    2px gap). The dual of `_separate_divots`: divots separates only source-8-DISCONNECTED
    pairs, this bridges only within a source-8-CONNECTED group — disjoint scopes, no
    oscillation. Constructions vote their source component by majority canonical-px
    label (bincount argmax = lowest label on ties); a group whose rendered union splits
    into >1 c8 components gets `mst_bridges` 1px `LINE_4` bridges, each drawn into the
    lowest-index endpoint-owning construction's mask. Keyed on source-at-S connectivity,
    NEVER a downsampled source (INTER_AREA fuses adobe's dots / shatters youtube-music)
    and NEVER canonical proximity (instagram's dot sits sub-pixel-close at feed res —
    bridging it would fight the separation doctrine). Deterministic (H8). Blast radius
    at introduction: android-head_flat@24 alone across the fixture set."""
    if src is None or len(masks) < 2:
        return masks
    lbl = cv2.connectedComponents((src > 0).astype(np.uint8), connectivity=8)[1]
    canon = _canon_masks(cons, S)
    groups: dict[int, list[int]] = {}
    for ci in range(len(cons)):
        if not np.any(masks[ci]):
            continue
        labs = lbl[canon[ci] > 0]
        labs = labs[labs > 0]                     # drop bg label 0 — it must never win the vote
        if labs.size == 0:
            continue
        groups.setdefault(int(np.bincount(labs).argmax()), []).append(ci)
    out = list(masks)
    for g in sorted(groups):
        mem = groups[g]
        if len(mem) < 2:
            continue
        union = np.zeros_like(out[mem[0]])
        for ci in mem:
            union |= out[ci]
        nc, clbl = cv2.connectedComponents((union > 0).astype(np.uint8), connectivity=8)
        if nc - 1 <= 1:
            continue
        for pa, pb in mst_bridges([clbl == q for q in range(1, nc)]):
            owner = min(ci for ci in mem
                        if out[ci][pa[0], pa[1]] or out[ci][pb[0], pb[1]])
            om = out[owner].copy()
            cv2.line(om, (pa[1], pa[0]), (pb[1], pb[0]), 255, 1, cv2.LINE_4)   # (x,y)
            out[owner] = om
    return out


RAD_MIN_MEMBERS = 6   # >= this many constructions on a shared radius = a radial family
#                       (signal 15; slack's 4 corner droplets excluded — the only other
#                       radial arrangement in the 30-fixture scan; adobe's 3 dots below any bar).
#                       Membership keys on MASK geometry (radius/size/slot), NOT part
#                       structure — signal's dashes decompose into 1-7 part mixed-op
#                       constructions (only 4 are single-add)
RAD_R_TOL = 0.15      # |r - median_r| / median_r membership band. Signal's dashes sit at
#                       +-3.1%; the annular-sector corner piece at +21.6% is EXCLUDED — which
#                       is what keeps the tail slot empty (it keeps its own construction render)
RAD_SIZE_TOL = 1.5    # member area within [1/tol, tol] x median (signal's long flanks: 1.33)
RAD_SLOT_TOL = 0.25   # member-to-slot angular deviation / pitch (signal measured max 0.093)


def _stamp_slot_bar(canvas: np.ndarray, cy: int, cx: int, ang: int, length: int) -> None:
    """Draw a 1px-thick LENGTH-px bar through (cy, cx) at slope-grid angle
    `ang` in {0, 45, 90, 135} — numpy index arrays, NOT cv2.line (LINE_4
    staircases a true diagonal into 3px Ls). Bounds via a boolean mask on BOTH
    index arrays — never np.clip (it silently pins duplicate edge px)."""
    t = canvas.shape[0]
    k = np.arange(length) - length // 2
    if ang == 0:
        ys, xs = np.full_like(k, cy), cx + k
    elif ang == 90:
        ys, xs = cy + k, np.full_like(k, cx)
    elif ang == 45:
        ys, xs = cy - k, cx + k
    else:
        ys, xs = cy + k, cx + k
    valid = (ys >= 0) & (ys < t) & (xs >= 0) & (xs < t)
    canvas[ys[valid], xs[valid]] = 255


def _harmonize_radial(masks: list[np.ndarray], cons, S: int, target: int
                      ) -> tuple[list[np.ndarray], frozenset, int, np.ndarray | None]:
    """Radial sibling of _harmonize_singles (signal's 15 burst dashes): constructions
    whose canonical masks sit on a shared radius at a near-uniform angular pitch
    (part structure irrelevant — the dashes decompose 1-7 parts each) re-render as
    ONE shared 1px bar stamped on a uniform angular comb — an orderly dashed ring
    instead of the ~12-component scatter independent rounding produced. Slots with
    no member (signal's tail gap at ~135deg — the corner piece fails RAD_R_TOL by
    design) are never stamped. Sub-representable combs decimate by halving the
    slot count while the per-slot dash length is < 2px (the blade-decimation
    doctrine); returns (masks, protected member indices, n_decimated) where
    n_decimated = stamps lost to decimation, fed to the topo audit's drops escape.
    Detection is pure numpy/math trigonometry — NO cv2.warpPolar (H8; the
    documented thread-nondeterminism hazard). Deterministic: sorted orders,
    median anchors, argmin-with-fixed-tie parity. Fires on signal alone
    (30-fixture scan)."""
    if len(cons) < RAD_MIN_MEMBERS + 1:      # family + at least the mark it orbits
        return masks, frozenset(), 0, None
    canon = _canon_masks(cons, S)
    comp = np.zeros((S, S), np.uint8)
    for cm in canon:
        comp |= cm
    ys, xs = np.nonzero(comp)
    if xs.size == 0:
        return masks, frozenset(), 0, None
    cy0, cx0 = float(ys.mean()), float(xs.mean())
    mem, radii, angs, areas = [], [], [], []
    for ci in range(len(cons)):
        mys, mxs = np.nonzero(canon[ci])
        if mxs.size == 0:
            continue
        mem.append(ci)
        my, mx = float(mys.mean()), float(mxs.mean())
        radii.append(math.hypot(my - cy0, mx - cx0))
        angs.append(math.degrees(math.atan2(my - cy0, mx - cx0)) % 360.0)
        areas.append(float(mxs.size))
    if len(mem) < RAD_MIN_MEMBERS:
        return masks, frozenset(), 0, None
    med_r = float(np.median(radii))
    med_a = float(np.median(areas))
    if med_r <= 0.0 or med_a <= 0.0:
        return masks, frozenset(), 0, None
    keep = [k for k in range(len(mem))
            if abs(radii[k] - med_r) <= RAD_R_TOL * med_r
            and med_a / RAD_SIZE_TOL <= areas[k] <= med_a * RAD_SIZE_TOL]
    if len(keep) < RAD_MIN_MEMBERS:
        return masks, frozenset(), 0, None
    kang = sorted(angs[k] for k in keep)
    gaps = [(kang[(i + 1) % len(kang)] - kang[i]) % 360.0 for i in range(len(kang))]
    med_gap = float(np.median(gaps))
    if med_gap <= 0.0:
        return masks, frozenset(), 0, None
    n_slots = max(2, int(round(360.0 / med_gap)))

    def _phase(n: int) -> float:
        # circular mean at n-fold — render.py:1911-1913's formula verbatim (a
        # private symbol there, and _radial_family needs a hub Part; the x n
        # fold is what maps n-fold-equivalent angles onto one direction)
        sx = sum(math.cos(math.radians(a * n)) for a in kang)
        sy = sum(math.sin(math.radians(a * n)) for a in kang)
        return math.degrees(math.atan2(sy, sx)) / n % (360.0 / n)

    pitch = 360.0 / n_slots
    phase = _phase(n_slots)

    def _dev(a: float, ph: float, pt: float) -> float:
        d = (a - ph) % pt
        return min(d, pt - d)

    if any(_dev(a, phase, pitch) > RAD_SLOT_TOL * pitch for a in kang):
        return masks, frozenset(), 0, None
    # shared dash form + the decimation ladder (per-slot length must be >= 2px)
    longs = [max(cv2.minAreaRect(cv2.findNonZero(canon[mem[k]]))[1]) for k in keep]
    ink_frac = min(0.95, float(np.median(longs)) * n_slots / (2.0 * math.pi * med_r))
    f = target / S
    r_t = med_r * f
    n_full = n_slots
    while True:
        period = 2.0 * math.pi * r_t / n_slots
        len_t = min(int(round(period * ink_frac)), max(1, int(math.floor(period)) - 1))
        if len_t >= 2 or n_slots <= 2:
            break
        n_slots //= 2
    if len_t < 2:
        return masks, frozenset(), 0, None
    if n_slots != n_full:
        pitch_full = 360.0 / n_full
        pitch = 360.0 / n_slots
        # decimation parity: the halved comb sits on the even or odd full-comb
        # slots; argmin total member->slot cost, tie -> parity 0
        cands = [phase % pitch, (phase + pitch_full) % pitch]
        costs = [sum(_dev(a, ph, pitch) for a in kang) for ph in cands]
        phase = cands[int(np.argmin(costs))]
    # 1px-thick bars: median canonical thickness x f rounds to 1 at both feed
    # sizes (signal ~46px x 0.014/0.021); the all-1px-line doctrine's width
    # stamp per OCCUPIED slot; each stamp lands in the nearest member's mask
    # (lowest index on ties); unassigned members are cleared (decimation)
    slot_members: dict[int, list[int]] = {}
    for k in keep:
        q = int(round(((angs[k] - phase) % 360.0) / pitch)) % n_slots
        slot_members.setdefault(q, []).append(k)
    out = list(masks)
    for k in keep:                       # clear all members first (deterministic)
        out[mem[k]] = np.zeros((target, target), np.uint8)
    n_stamped = 0
    for q in sorted(slot_members):
        theta = math.radians(phase + q * pitch)
        py = int(round((cy0 + med_r * math.sin(theta)) * f))
        px = int(round((cx0 + med_r * math.cos(theta)) * f))
        tang = (math.degrees(theta) + 90.0) % 180.0
        ang = min((0, 45, 90, 135), key=lambda a: min(abs(tang - a), 180 - abs(tang - a)))
        owner = min(slot_members[q],
                    key=lambda k: (_dev(angs[k], phase + q * pitch, 360.0), mem[k]))
        stamp = out[mem[owner]]
        _stamp_slot_bar(stamp, py, px, ang, len_t)
        n_stamped += 1
    print(f"      harmonize_radial n={len(keep)} slots={n_slots} stamped={n_stamped} "
          f"len={len_t}px decimated={max(0, len(keep) - n_stamped)}")
    zone = np.zeros((target, target), np.uint8)
    for k in keep:
        zone |= out[mem[k]]
    # the comb is a DELIBERATE assembly decision refine's ref-oracles cannot
    # see (the dash references are EMPTY at 16, sub-0.5-coverage — the
    # instagram-dot class): its stamps join the protected accent union and are
    # carved out of the chain + restored verbatim (despeck's ref-less peel ate
    # them, and symmetrize churned on the deliberately-asymmetric ring).
    return out, frozenset(mem[k] for k in keep), max(0, len(keep) - n_stamped), zone


SUBREP_ACC_MIN_DIM = 4   # canonical minAreaRect short side of a part's EXCLUSIVE zone >=
#                          this px = a real identity lobe (signal's tail ~95px; a fit
#                          hairline like adobe's RotatedEllipse sliver ~1.6px is noise)
SUBREP_ACC_MAX_PX = 8    # cov>=0.5 target footprint <= this px = sub-representable
#                          (the tail: 3px at both feed sizes; a representable lobe is 10s+)


def _stamp_subrep_accents(masks: list[np.ndarray], cons, S: int, target: int,
                          skip: frozenset = frozenset()
                          ) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Represented-geometry minimum for SUB-REPRESENTABLE IDENTITY PARTS fused
    inside a construction (signal's speech-bubble tail: a first-class Triangle
    add whose feed-res footprint is a 1-2px nub that greedy tooth-clear erodes
    and despeck then eats — the instagram accent class, but structurally outside
    the ring-nest machinery). Per add part of a multi-add construction, the
    part's EXCLUSIVE canonical zone (its raster minus the other adds') is
    stamped at its >= _min_feature_px(target)^2 coverage footprint, USING ONLY
    px outside the construction's already-rendered mask — interior lobes
    (adobe's crossbar) have an empty outside-parent set, making the pass a
    structural byte-identical no-op there: that property IS the blast-radius
    guard. The stamp ORs into the mask (bridged by one 1px LINE_4 segment if
    not already 8-adjacent) and its outside-parent zone joins the PROTECTED
    accent union that refine carves out and restores. Deterministic (H8):
    coverage-desc/row-major inflation, fixed part order."""
    from .cubify import _glyph_stroke_relayout  # lazy: render<->cubify cycle
    min_feat = _min_feature_px(target)
    need = min_feat * min_feat
    out = None
    accent = None
    for ci, (parts, _c, _l) in enumerate(cons):
        if ci in skip:
            continue                 # radial-harmonized members: the stamp IS their form
        adds = [p for p in parts if p.op == "add"]
        if len(adds) != 2:
            continue                 # the DIAGNOSED class only: one accent riding one
            #                          single-primitive body (signal's Ellipse+Triangle).
            #                          Multi-add unions (reddit's ears, docker's fin,
            #                          whatsapp/homebrew/obsidian/spotify protrusions)
            #                          pass the hull gate too but regress cells (reddit@24
            #                          PASS->FAIL) — the 3rd no-geometric-signal instance;
            #                          documented follow-on, not this round's scope
        if _radial_family(parts, [i for i, q in enumerate(parts)
                                  if q.op == "add"]) is not None:
            continue                 # radial families are comb-policed (claude's
            #                          blades decimate BY DESIGN — a stamp fights it)
        if _glyph_stroke_relayout(parts, (S, S), target) is not parts:
            continue                 # relayout-fired construction: the re-layout IS the
            #                          representation decision — raw-part zones no longer
            #                          align with the moved render (medium's serif)
        rasters = [_raster_part(p.primitive, S, S) for p in adds]
        parent = masks[ci] > 0
        for pi in range(len(adds)):
            excl = rasters[pi] > 0
            for qi in range(len(adds)):
                if qi != pi:
                    excl &= ~(rasters[qi] > 0)
            if not np.any(excl):
                continue
            (rw, rh) = cv2.minAreaRect(cv2.findNonZero(excl.astype(np.uint8)))[1]
            if min(rw, rh) < SUBREP_ACC_MIN_DIM:
                continue
            # PARENT-REPRESENTABILITY gate: an accent hangs off a REPRESENTABLE
            # body (signal's bubble ~200px @16). When the rest of the
            # construction is itself near-sub-representable (a burst dash's
            # other half: 2-4px), the "accent" is just a part split — skip
            # (this is what stopped the pass from stamping every dash part).
            par = np.zeros((S, S), bool)
            for qi in range(len(adds)):
                if qi != pi:
                    par |= rasters[qi] > 0
            pcov = cv2.resize(par.astype(np.uint8) * 255, (target, target),
                              interpolation=cv2.INTER_AREA)
            if int(np.count_nonzero(pcov >= 128)) <= SUBREP_ACC_MAX_PX:
                continue
            # HULL-PROTRUSION gate: the accent must protrude OUTSIDE the
            # parent's convex hull (signal's tail off the disc). A dominantly-
            # exclusive part INSIDE the hull is a union constituent, not an
            # appendage (adobe's crossbar 0.985 exclusive yet interior;
            # discord/firefox/apple/mathematica sliver zones likewise).
            eys, exs = np.nonzero(excl)
            hull = cv2.convexHull(cv2.findNonZero(par.astype(np.uint8)))
            if cv2.pointPolygonTest(
                    hull, (float(exs.mean()), float(eys.mean())), False) >= 0:
                continue
            cov = cv2.resize(excl.astype(np.uint8) * 255, (target, target),
                             interpolation=cv2.INTER_AREA)
            n50 = int(np.count_nonzero(cov >= 128))
            if n50 == 0 or n50 > SUBREP_ACC_MAX_PX:
                continue
            core = (cov >= 128) & ~parent
            if not np.any(core):
                continue                 # fully rendered already — the structural no-op
            # (a missing-fraction gate was probed and DROPPED: the zone hugs the
            #  parent rim, so rendered rim px dilute the measure — the HULL-
            #  PROTRUSION gate below is the real appendage/constituent separator)
            stamp = core.copy()
            if int(stamp.sum()) < need:
                # inflate from cov>0 outside-parent px: coverage desc, then
                # row-major (argsort stable on the negated flat array)
                pool = (cov > 0) & ~parent & ~stamp
                order = np.argsort(-cov.ravel().astype(np.int32), kind="stable")
                flat_pool = pool.ravel()
                st = stamp.ravel()
                tt = core.shape[0]

                def _makes_block(flat, idx):
                    # would setting idx complete a 2x2 all-ink block? (the
                    # tail must stay an L, not a blot square)
                    y, x = divmod(int(idx), tt)
                    m = flat.reshape(tt, tt)
                    for oy in (-1, 0):
                        for ox in (-1, 0):
                            y0, x0 = y + oy, x + ox
                            if 0 <= y0 < tt - 1 and 0 <= x0 < tt - 1:
                                blk = m[y0:y0 + 2, x0:x0 + 2].copy()
                                blk[y - y0, x - x0] = True
                                if blk.all():
                                    return True
                    return False

                for pass_blocks in (False, True):   # non-blocking px first
                    for idx in order:
                        if int(st.sum()) >= need:
                            break
                        if flat_pool[idx] and not st[idx] and (
                                pass_blocks or not _makes_block(st, idx)):
                            st[idx] = True
                    if int(st.sum()) >= need:
                        break
                stamp = st.reshape(core.shape)
            stamp_u8 = stamp.astype(np.uint8) * 255
            if out is None:
                out = list(masks)
            pre = out[ci]
            newm = pre | stamp_u8
            if pre.any() and not masks_8adjacent(stamp_u8, pre):
                pa, pb, _d = nearest_gap(np.argwhere(stamp), np.argwhere(pre > 0))
                cv2.line(newm, (pa[1], pa[0]), (pb[1], pb[0]), 255, 1, cv2.LINE_4)
            out[ci] = newm
            az = stamp_u8.copy()
            accent = az if accent is None else (accent | az)
            print(f"      subrep_accent cons#{ci} part{pi} "
                  f"{int(stamp.sum())}px @{target}")
    return (masks if out is None else out), accent


def _assemble_masks(base_masks: list[np.ndarray], cons, S: int, target: int,
                    src: np.ndarray | None):
    """The cross-construction ASSEMBLY chain — ONE orchestration, two callers
    (`assemble_render` and `_process_fixture`; H5): stamp harmonization → radial
    harmonization → divot separation → nested-ring coordination → sub-representable
    accent stamps → source-gap bridge. Returns (masks, accent, protected, n_dec):
    `accent` = the union of deliberately-placed protected-accent zones (None when
    neither the nest nor the subrep pass fired); `protected` = harmonized-stamp
    construction indices (the szspread gate column's input — the RADIAL member set
    is divot-protected internally but NOT returned here, its diagonal stamp bboxes
    would pollute szspread); `n_dec` = radial slots lost to decimation (the topo
    audit's drops escape)."""
    hmasks, protected = _harmonize_singles(base_masks, cons, S, target)
    rmasks, prot_rad, n_dec, rad_zone = _harmonize_radial(hmasks, cons, S, target)
    masks = _separate_divots(rmasks, cons, S, protected | prot_rad)
    masks, accent = _coordinate_nested_rings(masks, cons, S, target)
    masks, acc2 = _stamp_subrep_accents(masks, cons, S, target, skip=prot_rad)
    for az in (acc2, rad_zone):
        if az is not None:
            accent = az if accent is None else (accent | az)
    masks = _bridge_source_gaps(masks, cons, S, src)
    return masks, accent, protected, n_dec


def assemble_render(cons, S: int, target: int, src: np.ndarray | None = None):
    """Add all the render pieces together for ONE target size.

    cons: [(parts, color_rgb, label), ...] from _fixture_constructions (decomposed
    + label-cleaned once, target-independent). Renders every construction
    INDEPENDENTLY on the shared (S, S) grid (per-feature, no cross-feature CSG
    group collisions — the strip path's contract), composites the
    per-construction R1 masks in segment order (area-sorted upstream → smaller
    features draw on top), then runs the cross-construction assembly chain
    (_assemble_masks — one orchestration, two callers): stamp harmonization →
    RADIAL harmonization → divot separation → nested-ring coordination →
    sub-representable accent stamps → the source-gap bridge (`src` = the source
    silhouette at S from _fixture_constructions; None skips the bridge).

    Returns (masks, icon, accent): masks = per-construction (target, target)
    uint8 {0,255} R1 masks; icon = (target, target, 3) uint8 composited render;
    accent = the UNION of deliberately-placed protected zones — the nested-ring
    accent, the radial-comb stamps, and the sub-representable accent stamps (None
    when none fired) — pass it to refine_perimeter's `protected=` so refine
    honors the placements. The tuple
    (cons, masks, (S, S), target) is exactly what stage-4
    refine.common.refine_outline consumes — this function IS the seam refine
    touches up next."""
    rrs = [render_debug(parts, (S, S), target) for parts, _c, _l in cons]
    masks, accent, _protected, _n_dec = _assemble_masks(
        [r.mask for r in rrs], cons, S, target, src)
    icon = np.full((target, target, 3), _FX_BG, np.uint8)
    for m, (_p, c, _l) in zip(masks, cons):
        icon[m > 0] = c
    return masks, icon, accent
