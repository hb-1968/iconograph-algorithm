"""Inspect synthetic constructions or fixture renderings.

Use --fixtures for real logos, --final for assembled products, --all for the
fixture set and --twice to compare repeated output bytes. Synthetic and
fixture grids report fidelity, topology, drops and feasibility separately.
Outputs go under testbed/render/. The startup check pins fillPoly's
outline-inclusive, round-half-up raster convention.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import contextlib  # noqa: E402
import io  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from iconograph.application.compress import Classified, Feature  # noqa: E402
from iconograph.application.compress.common import (  # noqa: E402
    Circle,
    Part,
    Polygon,
    Rectangle,
    RotatedRectangle,
    RoundedRectangle,
    Triangle,
    Wedge,
    _band_width,
    _boundary_iou,
    _compound_mask,
    _iou_bool,
    _min_feature_px,
    _offset_part,
    _raster_part,
    _scale_part,
)
from iconograph.application.compress.decompose import decompose_shape  # noqa: E402
from iconograph.application.compress.label_clean import clean_labels  # noqa: E402
from iconograph.application.compress.render import (  # noqa: E402
    CUBIFY_MAX_T,
    _FX_BG,
    _assemble_masks,
    _topology,
    assemble_render,
    build_constructions,
    floor_for,
    jag_norm,
    jag_score,
)
from iconograph.application.compress.render import render_debug as r1_render  # noqa: E402
from iconograph.application.compress.straighten import DEG_TRUE, straighten_part  # noqa: E402
from iconograph.cache import CACHE_ROOT, TESTBED_ROOT  # noqa: E402
from iconograph.debug_common import hstack, labeled  # noqa: E402
from iconograph.application.refine.common import (  # noqa: E402
    cleanup_scene, outline_scene, refine_outline)

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMP_DIR = TESTBED_ROOT / "render"
# Render strips are filed by CONTEXT (H7 — testbed never accumulates a flat pile):
SYNTH_DIR = DUMP_DIR / "synthetic"  # synthetic shape regression strips (geometry path,
#                                     T ∈ {16,24,32,64}) — the default/synthetic mode
FX_DIR = DUMP_DIR / "fixtures"      # --fixtures: per-fixture feed-res gate strips
FINAL_DIR = DUMP_DIR / "final"      # --final: assembled render products (the
#                                     deliverable + stage-4 refine's input)

DEFAULT_SIZES = (16, 24, 32, 64)
# Feed-resolution lock (user ruling): the fixture gate is the seed set at
# T ∈ {16, 24} — the only sizes fed onward to stage-4 refinement, and the
# sizes the pixel-cubify render owns. Synthetics keep 32/64 in DEFAULT_SIZES
# as the geometry-path (T > 24) regression guard.
DEFAULT_FIXTURES = ("discord", "gmail", "vscode", "claude", "figma")
GATE_SIZES = (16, 24)

# Round-3 axis-fire audit: what symmetry machinery SHOULD detect per case.
# "axis" = global mirror axis must fire; "pairs" = local mirror pairs only;
# "none" = guard (any fire is spurious). Gate: zero expected-but-missed AND
# zero spurious fires; SYM constants loosen only against a named miss.
EXPECTED_SYM = {
    "circle": "axis", "annulus": "axis", "bevel_rect": "axis",
    "lobe_union": "axis", "wedge": "none", "notch_rect": "axis",
    "capsule": "axis", "nested_counter": "axis", "competition": "none",
    "long_diag": "axis", "asterisk6": "axis",
    # round 5: thin_x's axisreduce winner is symmetric BY CONSTRUCTION
    # (grid-snapped mirror polygons) with sym detection off — pairs fire
    # only when the staircase incumbent wins (@32)
    "thin_x": "any",
    "tab_body": "axis",
    "discord": "axis", "gmail": "axis", "vscode": "axis",
    # round 4: a phase-snapped EXTRAPOLATED star may legitimately fire a
    # mirror axis — neither a fire nor a non-fire is an audit failure
    "claude": "any", "asterisk12": "any",
    "figma": "none", "row_stack": "none",
}

# Round-4 regularity audit: what the extrapolated-symmetry detectors SHOULD
# find per case. "radial" = _radial_family fires; "group" = _regularity_groups
# fires (size/align/pitch); "none" = guard. Same gate shape as EXPECTED_SYM:
# zero expected-but-missed AND zero spurious; REG constants loosen only
# against a named miss. (asterisk6's six identical blades legitimately form
# BOTH a radial family and a size group — "radial" is the primary signal.)
EXPECTED_REG = {
    "circle": "none", "annulus": "none", "bevel_rect": "none",
    "lobe_union": "none", "wedge": "none", "notch_rect": "none",
    "capsule": "none", "nested_counter": "none", "competition": "none",
    "long_diag": "none", "thin_x": "none", "tab_body": "none",
    "asterisk6": "radial", "asterisk12": "radial", "claude": "radial",
    "figma": "group", "row_stack": "group",
    "discord": "none", "gmail": "none", "vscode": "none",
}
# Identity-notch witness (closes the render==own-primitive-reference blindness
# class: vue PASSED Ĵ 0.201/0.122 while its whole nested-V was missing — the
# gate graded a triangle against a triangle). notch = hull(ren)∖ren px in the
# TOP HALF of ren's bbox; listed fixtures FAIL below their floor (measured:
# vue 4 @16 / 6 @24), everyone else report-only.
EXPECTED_NOTCH: dict[str, dict[int, int]] = {"vue": {16: 3, 24: 5}}
PANEL_T = 256              # upscale targets to ~this for visibility
_INK = (40, 40, 40)
_UNDER = (230, 25, 75)     # reference pixels the render missed — red
_OVER = (67, 99, 216)      # render spill past the reference — blue

S = 192                    # synthetic canonical canvas (square, like padded crops)


def _shapes() -> dict[str, Callable[[], tuple[list[Part], tuple[int, int]]]]:
    """The 9 synthetic cases: 'somewhat complex' via overlap, grouping, and
    contention rather than realism — each exercises one methodology axis."""
    A, SUB = "add", "subtract"
    return {
        # 1 — sole curve: hardest straighten (square @16, octagon @32)
        "circle": lambda: ([Part(Circle(96, 96, 70), A)], (S, S)),
        # 2 — counter budget + force-merge (square ring @16, hole preserved)
        "annulus": lambda: ([Part(Circle(96, 96, 80), A),
                             Part(Circle(96, 96, 48), SUB)], (S, S)),
        # 3 — shave/bevel budget + bite growth: 4 corner-arc negates
        "bevel_rect": lambda: ([Part(Rectangle(36, 51, 120, 90), A),
                                Part(Circle(36, 51, 18), SUB),
                                Part(Circle(156, 51, 18), SUB),
                                Part(Circle(156, 141, 18), SUB),
                                Part(Circle(36, 141, 18), SUB)], (S, S)),
        # 4 — lobe role + sibling retention
        "lobe_union": lambda: ([Part(Rectangle(36, 61, 100, 70), A),
                                Part(Circle(146, 96, 35), A)], (S, S)),
        # 5 — arc-chord ladder + (a0, sweep) angle snap
        "wedge": lambda: ([Part(Wedge(96, 96, 80, 200.0, 290.0), A)], (S, S)),
        # 6 — shave at the edge (notch): rect stays one component
        "notch_rect": lambda: ([Part(Rectangle(36, 56, 120, 80), A),
                                Part(Circle(96, 56, 25), SUB)], (S, S)),
        # 7 — capsule (r = h/2): round 7 REPRESENTED-GEOMETRY — renders a
        #     rounded bar at EVERY size incl. 16 (chamfer-leg capped by the
        #     short dim so the flat middle survives: 0.339 sharp-rect →
        #     0.920 rounded @16). Supersedes the round-1 "degenerate to sharp
        #     rect @16" doctrine: identity curvature shows wherever the grid
        #     can represent a corner cut.
        "capsule": lambda: ([Part(RoundedRectangle(96, 96, 140, 60, 0.0, 30.0), A)], (S, S)),
        # 8 — negative-group counter (De Morgan): disc − (square moat − inner
        #     island) → all levels visible or 1:1-recorded drops
        "nested_counter": lambda: ([Part(Circle(96, 96, 80), A, group=0),
                                    Part(Rectangle(66, 66, 60, 60), A, group=-1),
                                    Part(Circle(96, 96, 18), SUB, group=-1)], (S, S)),
        # 9 — sibling competition: dominant rect + circle lobe + triangle accent
        #     + edge notch; the straightened lobe must not erase the accent
        "competition": lambda: ([Part(Rectangle(51, 41, 70, 110), A),
                                 Part(Circle(141, 76, 40), A),
                                 Part(Triangle((60.0, 165.0), (95.0, 165.0), (77.0, 135.0)), A),
                                 Part(Circle(51, 96, 15), SUB)], (S, S)),
        # round-3 cases: each exposes one critique mechanic
        # 10 — 1px-STEPPING: a V of two long ~63° bars (gmail's trap without
        #      the pillars) — must render as clean 2-px vertical runs, never a
        #      1-px 45° staircase; vertical axis must fire (sym 1.0)
        "long_diag": lambda: ([
            Part(RotatedRectangle(68.0, 101.0, 124.0, 20.0, 63.0), A),
            Part(RotatedRectangle(124.0, 101.0, 124.0, 20.0, 117.0), A)], (S, S)),
        # 11 — GLOBBING + repeat-regularity: hub + 6 uniform thin blades; gaps
        #      ~3 px at 32 (must stay open), ~1.5 px at 16 (decimation territory)
        "asterisk6": lambda: ([Part(Circle(96, 96, 18), A)] + [
            Part(RoundedRectangle(96.0 + 44.0 * math.cos(math.radians(60.0 * k)),
                                  96.0 + 44.0 * math.sin(math.radians(60.0 * k)),
                                  52.0, 10.0, 60.0 * k, 5.0), A)
            for k in range(6)], (S, S)),
        # 12 — PRONOUNCEDNESS + local pairs: thin X (±50° bars, mirror pair
        #      about the horizontal) + an off-axis accent that kills the global
        #      axis — the X tails must protrude, the quadrant pockets stay open
        "thin_x": lambda: ([
            Part(RotatedRectangle(96.0, 90.0, 130.0, 9.0, 50.0), A),
            Part(RotatedRectangle(96.0, 102.0, 130.0, 9.0, -50.0), A),
            Part(Rectangle(150, 44, 14, 14), A)], (S, S)),
        # 13 — PRONOUNCEDNESS (narrow+straighten): body + one thin protruding
        #      tab — must render narrowed/axis-snapped, never fat or absorbed
        "tab_body": lambda: ([
            Part(RoundedRectangle(86.0, 96.0, 120.0, 120.0, 0.0, 24.0), A),
            Part(RoundedRectangle(152.0, 96.0, 56.0, 10.0, 0.0, 5.0), A)], (S, S)),
        # round-4 cases: extrapolated symmetry
        # 14 — RADIAL EXTRAPOLATION (claude analogue): hub + 12 thin blades on
        #      a HAND-JITTERED organic comb (±6°-class, no RNG — H8). The
        #      per-gap evidence must decimate @16 and keep 12 @32; the
        #      rotational feasibility rule must reject organic renders, so an
        #      extrapolated uniform star/comb wins
        "asterisk12": lambda: ([Part(Circle(96, 96, 18), A)] + [
            Part(RoundedRectangle(
                96.0 + 44.0 * math.cos(math.radians(30.0 * k + j)),
                96.0 + 44.0 * math.sin(math.radians(30.0 * k + j)),
                52.0, 10.0, 30.0 * k + j, 5.0), A)
            for k, j in enumerate((6.0, -4.0, 2.0, -6.0, 4.0, -2.0,
                                   6.0, -4.0, 2.0, -6.0, 4.0, -2.0))], (S, S)),
        # 15 — TRANSLATIONAL REGULARITY (figma analogue): spine + pill + three
        #      near-equal circles with deliberate small perturbations (radii
        #      ±0.3, row-pair cy off by 0.8, organic drift) — the size/align
        #      detector must fire and the snapped variant must render the
        #      circles identical and aligned
        "row_stack": lambda: ([
            Part(Rectangle(64, 17, 32, 137), A),
            Part(RoundedRectangle(96.0, 42.8, 105.0, 52.8, 0.0, 26.4), A),
            Part(Circle(70.0, 96.0, 26.3), A),
            Part(Circle(122.6, 96.8, 26.0), A),
            Part(Circle(69.2, 149.4, 25.7), A)], (S, S)),
    }


# fillPoly half-pixel convention self-test


def _selftest() -> None:
    """Pins the raster convention the whole tier relies on: fillPoly is OUTLINE-
    INCLUSIVE with round-half-up vertex placement — an integer-corner box (a..b)
    fills pixels a..b exactly, and fractional vertices round to the nearest int."""
    cases = (
        # vertices (x0,y0)-(x1,y1) in pixel-center coords → expected filled bbox
        ((2.0, 2.0, 12.0, 12.0), (2, 2, 11, 11)),    # odd 11×11, integer corners
        ((6.0, 6.0, 9.0, 9.0), (6, 6, 4, 4)),        # even 4×4, integer corners
        ((5.7, 5.7, 9.3, 9.3), (6, 6, 4, 4)),        # fractional → nearest int
    )
    for (x0, y0, x1, y1), (ex, ey, ew, eh) in cases:
        poly = Polygon(pts=((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        m = _raster_part(poly, 16, 16)
        x, y, w, h = cv2.boundingRect(m)
        n = int(np.count_nonzero(m))
        assert (x, y, w, h) == (ex, ey, ew, eh) and n == ew * eh, (
            f"fillPoly convention drift: corners ({x0},{y0})-({x1},{y1}) filled "
            f"bbox ({x},{y},{w},{h}) n={n}, expected ({ex},{ey},{ew},{eh}) n={ew * eh}")


# panels


def _up(mask: np.ndarray, target: int) -> np.ndarray:
    k = max(1, PANEL_T // target)
    return cv2.resize(mask, (mask.shape[1] * k, mask.shape[0] * k),
                      interpolation=cv2.INTER_NEAREST)


def _mask_panel(mask: np.ndarray, target: int) -> np.ndarray:
    big = _up(mask, target)
    rgb = np.full((*big.shape, 3), 255, np.uint8)
    rgb[big > 0] = _INK
    return np.pad(rgb, ((22, 0), (0, 0), (0, 0)), constant_values=255)


def _diff_panel(render: np.ndarray, ref: np.ndarray, target: int) -> np.ndarray:
    r, f = _up(render, target), _up(ref, target)
    rgb = np.full((*r.shape, 3), 255, np.uint8)
    rgb[(f > 0) & (r == 0)] = _UNDER
    rgb[(r > 0) & (f == 0)] = _OVER
    return np.pad(rgb, ((22, 0), (0, 0), (0, 0)), constant_values=255)


def _degree_panels(parts: list[Part], canvas_hw: tuple[int, int],
                   target: int) -> list[np.ndarray]:
    """Uniform-degree ladder renders (no tweak search) — the H7 variant panels."""
    f = target / max(canvas_hw)
    out = []
    for d in range(DEG_TRUE + 1):
        forms = [straighten_part(_scale_part(p, f), d) for p in parts]
        m = _compound_mask(forms, target, target)
        out.append(labeled(_mask_panel(m, target), f"D{d} uniform"))
    return out


# fixture mode


def _fixture_constructions(stem: str):
    """Cached L2 -> constructions: the 3 raw pkl reads (debug-only pattern —
    production gets these objects from its own cached() chain) + the shared
    compress.render.build_constructions core. Returns (S, aligned, cons, src)."""
    cdir = CACHE_ROOT / stem
    fs = pickle.loads((cdir / "L2_feature_segment.pkl").read_bytes())
    aligned = pickle.loads((cdir / "align.pkl").read_bytes())
    sub_case = pickle.loads((cdir / "categorize.pkl").read_bytes()).routing
    S, cons, src = build_constructions(fs, sub_case, tag=stem)
    return S, aligned, cons, src


def _src_panel(aligned: np.ndarray) -> np.ndarray:
    img = cv2.resize(aligned, (PANEL_T, PANEL_T), interpolation=cv2.INTER_AREA)
    return np.pad(img, ((22, 0), (0, 0), (0, 0)), constant_values=255)


def _color_panel(layers: list[tuple[np.ndarray, tuple[int, int, int]]],
                 target: int) -> np.ndarray:
    """Composite per-construction masks in segment order (area-sorted upstream,
    so smaller features draw later = on top) over a light-grey field."""
    rgb = np.full((target, target, 3), _FX_BG, np.uint8)
    for m, c in layers:
        rgb[m > 0] = c
    k = max(1, PANEL_T // target)
    up = cv2.resize(rgb, (target * k, target * k), interpolation=cv2.INTER_NEAREST)
    # the fixture row hstacks this against the fixed-size source panel —
    # sizes where target*k < PANEL_T (e.g. 24 → 240) must pad onto a
    # constant PANEL_T field or the row heights diverge and hstack throws
    canvas = np.full((PANEL_T, PANEL_T, 3), _FX_BG, np.uint8)
    oy, ox = (PANEL_T - up.shape[0]) // 2, (PANEL_T - up.shape[1]) // 2
    canvas[oy:oy + up.shape[0], ox:ox + up.shape[1]] = up
    return np.pad(canvas, ((22, 0), (0, 0), (0, 0)), constant_values=255)


def _notch_px(ren: np.ndarray) -> int:
    """hull(ren)∖ren px in the TOP HALF of ren's bbox — the EXPECTED_NOTCH
    witness (vue's carved V opens through the top edge)."""
    nz = cv2.findNonZero((ren > 0).astype(np.uint8))
    if nz is None:
        return 0
    hm = np.zeros(ren.shape, np.uint8)
    cv2.fillPoly(hm, [cv2.convexHull(nz)], 255)
    diff = (hm > 0) & (ren == 0)
    ys = nz[:, 0, 1]
    diff[(int(ys.min()) + int(ys.max())) // 2 + 1:, :] = False
    return int(np.count_nonzero(diff))


def _sig_topology(mask: np.ndarray, min_feat: int) -> tuple[int, int]:
    """(components, SIGNIFICANT holes) — holes whose bounding box spans >= min_feat in
    BOTH dimensions. The cubify topology audit at the feed resolution uses this so a thin
    radial mark's resolution-dependent 1px inter-blade gaps (rasterization noise, not real
    topology — claude's sunburst) don't count as holes, while a representable counter
    (figma) still would. Holes via border flood-fill of the inverted mask."""
    m = (mask > 0).astype(np.uint8)
    comps = cv2.connectedComponents(m, connectivity=8)[0] - 1
    pad = cv2.copyMakeBorder(m * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(pad)
    ffm = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(inv, ffm, (0, 0), 0)          # border-connected bg → 0
    nh, _lbl, st, _c = cv2.connectedComponentsWithStats(inv[1:-1, 1:-1], connectivity=8)
    holes = sum(1 for i in range(1, nh)
                if min(int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])) >= min_feat)
    return comps, holes


def _process_fixture(stem: str, sizes: tuple[int, ...], twice: bool,
                     cleanup: bool = False, outline: bool = False) -> list[dict]:
    if not (CACHE_ROOT / stem / "L2_feature_segment.pkl").exists():
        print(f"{stem}: no cached L2 segmentation under .cache/stages/ — "
              f"run the testbed/build first")
        return []
    S, aligned, cons, src = _fixture_constructions(stem)
    if not cons:
        print(f"{stem}: nothing to render")
        return []
    # composited canonical-res topology — the reference degenerates at tiny T (a thin
    # ring's hole, a sunburst's inter-blade gaps), so matching the canonical topology is
    # as correct as matching the reference's (mirrors the synthetic path's ref-or-canon).
    canon_full = np.zeros((S, S), np.uint8)
    for parts, _c, _l in cons:
        canon_full |= _compound_mask(parts, S, S)
    topo_canon_comp = _topology(canon_full)
    stats: list[dict] = []
    rows: list[np.ndarray] = []
    for t in sizes:
        rrs = [r1_render(parts, (S, S), t) for parts, _c, _l in cons]
        if twice:
            again = [r1_render(parts, (S, S), t) for parts, _c, _l in cons]
            same = all(np.array_equal(a.mask, b.mask) for a, b in zip(rrs, again))
            print(f"  determinism({stem}@{t}): {'IDENTICAL' if same else 'MISMATCH'}")
        # cross-construction ASSEMBLY (the shared _assemble_masks chain: harmonize →
        # radial → divots → nested-rings → subrep-accents → bridge). The composite mask
        # + R1 panel use the assembled masks; per-construction r.* metrics stay from rrs.
        amasks, _accent, protected, n_dec = _assemble_masks(
            [r.mask for r in rrs], cons, S, t, src)
        if protected:                                     # szspread gate column (R1)
            hs = [int(np.count_nonzero(amasks[ci].any(axis=1))) for ci in sorted(protected)]
            ws = [int(np.count_nonzero(amasks[ci].any(axis=0))) for ci in sorted(protected)]
            szspread = (max(hs) - min(hs)) + (max(ws) - min(ws))
        else:
            szspread = 0
        if protected or szspread:
            print(f"  {stem:<14} T={t:<3} harmonized {len(protected)} stamps  "
                  f"szspread={szspread}")
        ren = np.zeros((t, t), np.uint8)
        ref = np.zeros((t, t), np.uint8)
        nai = np.zeros((t, t), np.uint8)
        viol = drops = strays = 0
        axis_fired = False
        reg_det_stem = None
        for idx, ((parts, _c, label), r) in enumerate(zip(cons, rrs)):
            ren |= amasks[idx]
            ref |= r.ref
            nai |= r.naive
            viol += len(r.violations)
            drops += len(r.drops)
            # round 6 — gate column: the drop record is the contract for
            # final-mask stray removals (no dedicated RenderResult field)
            strays += sum(1 for dr in r.drops
                          if "stray component removed" in dr)
            ratio_i = (r.jag_norm / r.jag_norm_naive) if r.jag_norm_naive > 0 else 1.0
            fid_i = min(r.area_iou, r.boundary_iou)
            # round-2 fields (getattr-guarded until render.py's brief is implemented)
            sym = getattr(r, "sym", None)
            axis = getattr(r, "axis", None)
            collapsed = getattr(r, "collapsed", None)
            fid_ref = getattr(r, "fid_ref", None)
            ref_kind = getattr(r, "ref_kind", None)
            removed_ink = getattr(r, "removed_ink", 0)
            reg_det_stem = reg_det_stem or getattr(r, "reg", None)
            axis_fired = axis_fired or axis is not None
            extra = ""
            if axis is not None:
                extra += f"  sym {sym:5.3f} axis {axis[0]}={axis[1]:.1f}"
            if collapsed:
                extra += f"  COLLAPSED({collapsed})"
            # round 4 per-cell truth: removal/extrapolation wins print BOTH
            # fidelity numbers
            if fid_ref is not None:
                kind_tag = "extr" if ref_kind == "extrapolate" else "live"
                extra += f"  {kind_tag} {fid_ref:.3f}"
                if removed_ink:
                    extra += f" removed {removed_ink}px"
            print(f"      {label:<22} T={t:<3} Ĵ {ratio_i:5.3f}"
                  f"{' LOSS' if ratio_i > 1.0 else '     '} fid {fid_i:.3f}"
                  f"  drops={len(r.drops)} viol={len(r.violations)}{extra}")
        # radial decimation is an honest per-cell drop (signal@16's 16->8 comb):
        # the topo/floor audits' drops>0 escape covers the member-count delta
        drops += n_dec
        jn_r, jn_n = jag_norm(ren), jag_norm(nai)
        ratio = (jn_r / jn_n) if jn_n > 0 else 1.0
        band = _band_width(ref)
        fid = min(_iou_bool(ren, ref), _boundary_iou(ren, ref, band))
        floor = floor_for(t)
        print(f"  {stem:<14} T={t:<3} silhouette J {jag_score(ren):6.1f} vs naive "
              f"{jag_score(nai):6.1f} (Ĵ ratio {ratio:5.3f}"
              f"{' LOSS' if ratio > 1.0 else ''})  fid {fid:.3f}/{floor:.2f}  "
              f"drops={drops} viol={viol}")
        row_panels = [
            labeled(_src_panel(aligned), f"{stem} aligned"),
            labeled(_color_panel([(r.ref, c) for r, (_p, c, _l) in zip(rrs, cons)], t),
                    f"reference@{t}"),
            labeled(_color_panel([(r.naive, c) for r, (_p, c, _l) in zip(rrs, cons)], t),
                    f"naive  J{jag_score(nai):.0f}"),
            # ASCII captions only — cv2.putText's Hershey fonts render non-ASCII as ??
            labeled(_color_panel([(dm, c) for dm, (_p, c, _l) in zip(amasks, cons)], t),
                    f"R1  J{jag_score(ren):.0f}  fid{fid:.2f}  "
                    f"Jratio {ratio:5.3f}{' LOSS' if ratio > 1.0 else ''}"),
        ]
        # stage-4 consistency cleanup column (per the layer-on-top design): the
        # cleanup pass corrects the produced per-construction masks; the no-op
        # contract means unchanged cells stay byte-identical to R1
        if cleanup:
            cl_masks = cleanup_scene(cons, [r.mask for r in rrs], (S, S), t)
            if twice:
                cl2 = cleanup_scene(cons, [r.mask for r in rrs], (S, S), t)
                cl_same = all(np.array_equal(a, b) for a, b in zip(cl_masks, cl2))
                print(f"  determinism(cleanup {stem}@{t}): "
                      f"{'IDENTICAL' if cl_same else 'MISMATCH'}")
            clr = np.zeros((t, t), np.uint8)
            for cm in cl_masks:
                clr |= cm
            changed = sum(1 for cm, r in zip(cl_masks, rrs)
                          if not np.array_equal(cm, r.mask))
            print(f"  cleanup({stem}@{t}): {changed}/{len(cons)} constructions "
                  f"changed  J {jag_score(clr):.1f} (R1 {jag_score(ren):.1f})  "
                  f"topo {_topology(clr)} (R1 {_topology(ren)})")
            row_panels.append(labeled(
                _color_panel([(cm, c) for cm, (_p, c, _l) in zip(cl_masks, cons)], t),
                f"cleanup J{jag_score(clr):.0f}  chg{changed}"))
        # stage-4 Phase-2 outline column: the fused composite reduced to a 1-px
        # connect/squeeze/L-corner-straightened medial-axis centerline
        if outline:
            sk = outline_scene(cons, [r.mask for r in rrs], (S, S), t)
            rs = refine_outline(cons, [r.mask for r in rrs], (S, S), t)   # re-stroked = shipped
            if twice:
                rs2 = refine_outline(cons, [r.mask for r in rrs], (S, S), t)
                print(f"  determinism(outline {stem}@{t}): "
                      f"{'IDENTICAL' if np.array_equal(rs, rs2) else 'MISMATCH'}")
            fid = _iou_bool(rs, ren)
            print(f"  outline({stem}@{t}): skel {int(np.count_nonzero(sk))}px  "
                  f"restroke IoU {fid:.3f}  topo {_topology(rs)} (R1 composite {_topology(ren)})")
            row_panels.append(labeled(
                _color_panel([(sk, (255, 255, 255))], t),
                f"skeleton {int(np.count_nonzero(sk))}px"))
            row_panels.append(labeled(
                _color_panel([(rs, (255, 255, 255))], t),
                f"restroke IoU{fid:.2f}"))
        rows.append(hstack(row_panels))
        exp_reg = EXPECTED_REG.get(stem)
        # cubify (T <= CUBIFY_MAX_T) renders at the pixel level and does not classify
        # radial/group regularity — the reg audit is a geometry-path check, skipped here.
        reg_audit_ok = (t <= CUBIFY_MAX_T or exp_reg is None
                        or (reg_det_stem or "none") == exp_reg)
        if not reg_audit_ok:
            print(f"  {stem:<14} T={t:<3} REG-AUDIT expected {exp_reg} "
                  f"got {reg_det_stem or 'none'}")
        # round 6 — gate column: a detached blob (gmail@24/32's sliver) is a
        # silhouette TOPOLOGY mismatch; drops escape like _feasible's
        # A deliberately-PLACED nest accent is carved from the audited mask (same doctrine
        # as count_tight_ref's protected clause): the reference cannot express it — its
        # natural downsample position COLLIDES with the rings (the very defect the
        # coordination pass fixes) and its ratified hollow form has a hole no solid
        # canonical carries. The audit judges the rings; the accent is the strip's truth.
        ren_audit = ren
        if _accent is not None:
            ren_audit = ren.copy()
            ren_audit[_accent > 0] = 0
        if t <= CUBIFY_MAX_T:
            # cubify represents labels at the feed resolution — pass on EITHER the
            # SIGNIFICANT topology matching (filters sub-min-feature rasterization holes —
            # claude's 1px blade gaps) OR the plain topology matching ref/canon (the
            # label-driven square ring + triangle matches the canonical structure).
            mft = _min_feature_px(t)
            topo_cell_ok = (_sig_topology(ren_audit, mft) == _sig_topology(ref, mft)
                            or _topology(ren_audit) in (_topology(ref), topo_canon_comp)
                            or drops > 0)
        else:
            topo_cell_ok = (_topology(ren_audit) in (_topology(ref), topo_canon_comp)
                            or drops > 0)
        # The cubify render (T <= CUBIFY_MAX_T) REPRESENTS labels, it does not pixel-fit
        # the round raster (user ruling) — a square ring barely overlaps a circle ring, so
        # area IoU is the WRONG floor there. A structurally-correct cubified render
        # (topology matches the reference / canonical truth) satisfies the floor regardless
        # of IoU; the geometry path (T > 24) keeps the fidelity floor.
        floor_ok = (fid >= floor or drops > 0
                    or (t <= CUBIFY_MAX_T and topo_cell_ok))
        exp_notch = EXPECTED_NOTCH.get(stem, {}).get(t)
        notch_v = _notch_px(ren)
        notch_ok = exp_notch is None or notch_v >= exp_notch
        if not notch_ok:
            print(f"  {stem:<14} T={t:<3} NOTCH-AUDIT expected >= {exp_notch} "
                  f"got {notch_v}")
        stats.append({"name": stem, "t": t, "ratio": ratio, "fid": fid,
                      "floor_ok": floor_ok,
                      "jag_win": ratio <= 1.0, "viol": viol, "drops": drops,
                      "strays": strays, "topo_ok": topo_cell_ok,
                      "axis_fired": axis_fired,
                      "reg_audit_ok": reg_audit_ok,
                      "notch": notch_v, "notch_ok": notch_ok})
    FX_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows), "RGB").save(
        FX_DIR / f"fx_{stem}.png", "PNG", optimize=True)
    return stats


# final render product (assembled deliverable; stage-4 refine's input)




def _final_product(stem: str, sizes: tuple[int, ...], twice: bool):
    """Deliver the FINAL render product for one fixture: a single clean row
    `aligned source | final@T ...` (no reference/naive/diff clutter — the product
    the user critiques), saved standalone to testbed/render/final/<stem>.png.
    Returns the row image (for the combined gallery) or None if uncached/empty."""
    if not (CACHE_ROOT / stem / "L2_feature_segment.pkl").exists():
        print(f"{stem}: no cached L2 segmentation under .cache/stages/ — "
              f"run the testbed/build first")
        return None
    S, aligned, cons, src = _fixture_constructions(stem)
    if not cons:
        print(f"{stem}: nothing to render")
        return None
    panels = [labeled(_src_panel(aligned), f"{stem} aligned")]
    for t in sizes:
        masks, _icon, _accent = assemble_render(cons, S, t, src)
        if twice:
            again, _i2, _a2 = assemble_render(cons, S, t, src)
            same = all(np.array_equal(a, b) for a, b in zip(masks, again))
            print(f"  determinism({stem}@{t}): "
                  f"{'IDENTICAL' if same else 'MISMATCH'}")
        union = np.zeros((t, t), np.uint8)
        for m in masks:
            union |= m
        panels.append(labeled(
            _color_panel([(m, c) for m, (_p, c, _l) in zip(masks, cons)], t),
            f"final@{t}  {len(cons)}pc"))
        print(f"  {stem:<14} final@{t:<3} {len(cons)} pieces composited  "
              f"topo {_topology(union)}")
    row = hstack(panels)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(row, "RGB").save(
        FINAL_DIR / f"{stem}.png", "PNG", optimize=True)
    return row


def _final_worker(stem: str, sizes: tuple[int, ...], twice: bool):
    """Pool worker for --final: assemble ONE fixture single-threaded (H8),
    capturing stdout so the parent prints fixtures in order. Returns
    (row | None, captured-log); a per-fixture failure is reported and yields
    None so the batch never dies on one bad fixture."""
    cv2.setNumThreads(1)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            row = _final_product(stem, sizes, twice)
    except Exception as e:  # noqa: BLE001 — debug harness: report and continue
        return None, f"{stem}: ERROR {type(e).__name__}: {e}\n"
    return row, buf.getvalue()


# per-case driver


def _process(name: str, builder, sizes: tuple[int, ...], degrees: bool,
             twice: bool) -> list[dict]:
    parts, canvas = builder()
    stats = []
    for t in sizes:
        res = r1_render(parts, canvas, t)
        if twice:
            again = r1_render(parts, canvas, t)
            same = np.array_equal(res.mask, again.mask)
            print(f"  determinism({name}@{t}): {'IDENTICAL' if same else 'MISMATCH'}")
        ratio = (res.jag_norm / res.jag_norm_naive) if res.jag_norm_naive > 0 else 1.0
        fid = min(res.area_iou, res.boundary_iou)
        floor_ok = fid >= res.floor or bool(res.drops)
        # the reference can degenerate at tiny T — matching the canonical-res
        # topology is as correct as matching the reference's
        topo_ok = (res.topo_render in (res.topo_ref, res.topo_canon)
                   or bool(res.drops))
        # 3 decimals + explicit LOSS tag: a 1.005 normalized loss must never
        # round into an invisible "1.00" tie (grader taxonomy:
        # acceptance-claim-vs-measured-mismatch)
        sym = getattr(res, "sym", None)
        axis = getattr(res, "axis", None)
        collapsed = getattr(res, "collapsed", None)
        pairs = getattr(res, "pairs", 0)
        stair = getattr(res, "staircase", 0.0)
        fid_ref = getattr(res, "fid_ref", None)
        ref_kind = getattr(res, "ref_kind", None)
        removed_ink = getattr(res, "removed_ink", 0)
        reg_det = getattr(res, "reg", None)
        extra = (f"  sym {sym:5.3f} axis {axis[0]}={axis[1]:.1f}"
                 if axis is not None else "")
        extra += f"  pairs={pairs}" if pairs and axis is None else ""
        extra += f"  STAIR {stair:.0f}" if stair > 0 else ""
        extra += f"  COLLAPSED({collapsed})" if collapsed else ""
        # round 4 per-cell truth: a removal/extrapolation win never hides
        # behind either fidelity number alone
        if fid_ref is not None:
            kind_tag = "extr" if ref_kind == "extrapolate" else "live"
            extra += f"  {kind_tag} {fid_ref:.3f}"
            if removed_ink:
                extra += f" removed {removed_ink}px"
        detected = ("axis" if axis is not None
                    else ("pairs" if pairs else "none"))
        exp = EXPECTED_SYM.get(name)
        sym_audit_ok = (exp is None or exp == "any" or detected == exp
                        # a recorded symmetrize→plain fallback is ACCOUNTED,
                        # like a recorded drop in the topology audit
                        or any("symmetrized render below floor" in v
                               for v in res.violations))
        if not sym_audit_ok:
            extra += f"  SYM-AUDIT expected {exp} got {detected}"
        exp_reg = EXPECTED_REG.get(name)
        # cubified sizes (T <= CUBIFY_MAX_T) don't classify regularity — skip the audit
        reg_audit_ok = (t <= CUBIFY_MAX_T or exp_reg is None
                        or (reg_det or "none") == exp_reg)
        if not reg_audit_ok:
            extra += f"  REG-AUDIT expected {exp_reg} got {reg_det or 'none'}"
        if reg_det:
            extra += f"  REG({reg_det})"
        print(f"  {name:<14} T={t:<3} J {res.jag:6.1f} vs naive {res.jag_naive:6.1f} "
              f"(Ĵ ratio {ratio:5.3f}{' LOSS' if ratio > 1.0 else ''})  "
              f"fid {fid:.3f}/{res.floor:.2f} "
              f"a{res.area_iou:.2f}/b{res.boundary_iou:.2f}  "
              f"topo {res.topo_render}vs{res.topo_ref}(canon{res.topo_canon})  "
              f"drops={len(res.drops)} viol={len(res.violations)}{extra}")
        for r in res.records:
            note = f" DROP({r.dropped})" if r.dropped else (
                f" VIOL" if r.violation else "")
            print(f"      #{r.index} {r.role:<8} ext{r.extent_px:6.1f} "
                  f"budget{r.budget} → D{r.degree} f({r.size_fx:.2f},{r.size_fy:.2f}) "
                  f"snap({r.dx:+.1f},{r.dy:+.1f}){note}")
        panels = [
            labeled(_mask_panel(res.ref, t), f"{name}@{t} reference"),
            labeled(_mask_panel(res.naive, t),
                    f"naive NN  J{res.jag_naive:.0f}"),
            labeled(_mask_panel(res.mask, t),
                    f"R1 render  J{res.jag:.0f}  fid{fid:.2f}"),
            labeled(_diff_panel(res.mask, res.ref, t), "diff vs ref"),
        ]
        if degrees:
            panels.extend(_degree_panels(parts, canvas, t))
        SYNTH_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(hstack(panels), "RGB").save(
            SYNTH_DIR / f"{name}_{t}.png", "PNG", optimize=True)
        stats.append({"name": name, "t": t, "ratio": ratio, "fid": fid,
                      "floor_ok": floor_ok, "jag_win": ratio <= 1.0,
                      "topo_ok": topo_ok, "sym_audit_ok": sym_audit_ok,
                      "reg_audit_ok": reg_audit_ok,
                      "viol": len(res.violations), "drops": len(res.drops)})
    return stats


def _gate_table(all_stats: list[dict]) -> None:
    """Round-6 shipping gate, per-cell truth: one row per (fixture, T).
    Green iff no LOSS, viol 0, strays 0, floor-or-drop, reg-audit ok, and the
    EXPECTED_NOTCH floor holds (identity-notch witness; report-only column for
    unlisted fixtures). Sym has no per-fixture expectation — the figma mirror
    guard prints separately. Visual strip sign-off stays a human step (H1)."""
    gsizes = sorted({s["t"] for s in all_stats})
    print(f"\ngate table (feed res — fixtures x {{{','.join(map(str, gsizes))}}}):")
    print(f"  {'fixture':<10} {'T':>3}  {'Jratio':>10}  {'viol':>4}  "
          f"{'strays':>6}  {'topo':>4}  {'floor':>5}  {'reg':>3}  {'notch':>5}  verdict")
    green = 0
    for s in all_stats:
        ok = (s["ratio"] <= 1.0 and s["viol"] == 0
              and s.get("strays", 0) == 0 and s.get("topo_ok", True)
              and s["floor_ok"] and s.get("reg_audit_ok", True)
              and s.get("notch_ok", True))
        green += ok
        print(f"  {s['name']:<10} {s['t']:>3}  {s['ratio']:>5.3f}"
              f"{' LOSS' if s['ratio'] > 1.0 else '     '}  "
              f"{s['viol']:>4}  {s.get('strays', 0):>6}  "
              f"{'ok' if s.get('topo_ok', True) else 'BAD':>4}  "
              f"{'ok' if s['floor_ok'] else 'MISS':>5}  "
              f"{'ok' if s.get('reg_audit_ok', True) else 'BAD':>3}  "
              f"{s.get('notch', 0):>5}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  gate: {green}/{len(all_stats)} cells green")


def _all_cached() -> list[str]:
    """Every fixture with a cached L2 segmentation under .cache/stages/ (the full H9
    dataset), sorted — the `--fixtures --all` target."""
    return sorted(p.name for p in CACHE_ROOT.iterdir()
                  if p.is_dir() and (p / "L2_feature_segment.pkl").exists())


def _half_cores() -> int:
    """Worker budget for the fixture pool: ICONOGRAPH_JOBS when set (the same
    env contract testbed._pool and refine_debug.main honor — the 25%-CPU sample
    batches run at JOBS=4), else half the host cores (a 16-core box → 8). GPU
    offload is unavailable on this box (RX 580 / no ROCm), so CPU-parallel
    across fixtures is the perf lever; each worker stays single-threaded (H8)."""
    env = os.environ.get("ICONOGRAPH_JOBS")
    if env:
        return max(1, int(env))
    return max(1, (os.cpu_count() or 2) // 2)


def _fixture_worker(stem: str, sizes: tuple[int, ...], twice: bool,
                    cleanup: bool, outline: bool) -> tuple[list[dict], str]:
    """Pool worker: process ONE fixture single-threaded (cv2 serial — inherited at fork,
    re-asserted here), capturing its stdout so the parent can print fixtures in order rather
    than interleaved. Returns (stats, captured-log). A per-fixture failure is reported in the
    log and yields empty stats (the batch never dies on one bad fixture). Determinism (H8):
    the worker is byte-identical to the serial path — single-threaded, no shared state, its
    own fx_<stem>.png."""
    cv2.setNumThreads(1)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            stats = _process_fixture(stem, sizes, twice, cleanup, outline)
    except Exception as e:  # noqa: BLE001 — debug harness: report and continue
        return [], f"{stem}: ERROR {type(e).__name__}: {e}\n"
    return stats, buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    twice = "--twice" in argv
    degrees = "--degrees" in argv
    fixtures = "--fixtures" in argv
    cleanup = "--cleanup" in argv
    outline = "--outline" in argv
    serial = "--serial" in argv
    want_all = "--all" in argv
    final = "--final" in argv
    argv = [a for a in argv
            if a not in ("--twice", "--degrees", "--fixtures", "--cleanup",
                         "--outline", "--serial", "--all", "--final")]
    # Feed-resolution lock: fixtures render + gate at {16, 24} only; the synthetic
    # sweep keeps {16, 24, 32, 64} as the geometry-path (T > 24) regression guard.
    sizes = GATE_SIZES if (fixtures or final) else DEFAULT_SIZES
    if "--sizes" in argv:
        i = argv.index("--sizes")
        sizes = tuple(int(x) for x in argv[i + 1].split(","))
        del argv[i:i + 2]
    cv2.setNumThreads(1)
    _selftest()
    all_stats: list[dict] = []
    if final:
        # Assemble + deliver the final render products: one clean
        # `source | final@T ...` row per fixture, plus a combined gallery.
        # Same fixture selection as --fixtures (stems / --all / default seeds).
        wanted = _all_cached() if want_all else (argv if argv else list(DEFAULT_FIXTURES))
        rows: list[np.ndarray] = []
        if serial or len(wanted) <= 1:
            for stem in wanted:
                try:
                    row = _final_product(stem, sizes, twice)
                except Exception as e:  # noqa: BLE001 — report and continue
                    print(f"{stem}: ERROR {type(e).__name__}: {e}")
                    row = None
                if row is not None:
                    rows.append(row)
        else:
            njobs = max(1, min(_half_cores(), len(wanted)))
            print(f"  [parallel: {njobs} workers (half of {os.cpu_count()} cores), "
                  f"each single-threaded for H8 determinism]")
            work = [(stem, sizes, twice) for stem in wanted]
            with mp.get_context("fork").Pool(njobs) as pool:
                for row, log in pool.starmap(_final_worker, work):
                    print(log, end="")          # starmap preserves order → in-order
                    if row is not None:
                        rows.append(row)
        if rows:
            FINAL_DIR.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.vstack(rows), "RGB").save(
                FINAL_DIR / "final_products.png", "PNG", optimize=True)
            print(f"\nfinal products → {FINAL_DIR.relative_to(REPO_ROOT)}/"
                  f"final_products.png  (+ per-fixture <stem>.png, "
                  f"{len(rows)} fixtures)")
        else:
            print("no final products produced")
        return 0
    if fixtures:
        wanted = _all_cached() if want_all else (argv if argv else list(DEFAULT_FIXTURES))
        if serial or len(wanted) <= 1:
            for stem in wanted:
                try:
                    all_stats.extend(_process_fixture(stem, sizes, twice, cleanup, outline))
                except Exception as e:  # noqa: BLE001 — debug harness: report and continue
                    print(f"{stem}: ERROR {type(e).__name__}: {e}")
        else:
            njobs = max(1, min(_half_cores(), len(wanted)))
            print(f"  [parallel: {njobs} workers (half of {os.cpu_count()} cores), "
                  f"each single-threaded for H8 determinism]")
            work = [(stem, sizes, twice, cleanup, outline) for stem in wanted]
            with mp.get_context("fork").Pool(njobs) as pool:
                for stats, log in pool.starmap(_fixture_worker, work):
                    print(log, end="")          # starmap preserves order → grouped, in-order
                    all_stats.extend(stats)
        if all_stats:
            ratios = [s["ratio"] for s in all_stats]
            wins = sum(s["jag_win"] for s in all_stats)
            floors = sum(s["floor_ok"] for s in all_stats)
            viols = sum(s["viol"] for s in all_stats)
            reg_ok = sum(s.get("reg_audit_ok", True) for s in all_stats)
            print(f"\nfixture summary: silhouette jag wins {wins}/{len(all_stats)}  "
                  f"median Ĵ ratio {statistics.median(ratios):.3f}  "
                  f"floor-or-drop {floors}/{len(all_stats)}  violations {viols}  "
                  f"reg-audit {reg_ok}/{len(all_stats)}")
            # round-2 guard, round-4 scope: figma is MIRROR-asymmetric — a
            # mirror AXIS fire is spurious; the round-4 regularity gate is
            # EXPECTED to fire on it (EXPECTED_REG above audits that side)
            figma = [s for s in all_stats if s["name"] == "figma"]
            if figma:
                fired = any(s.get("axis_fired") for s in figma)
                print("figma guard: " + ("MIRROR AXIS FIRED — investigate"
                                         if fired
                                         else "mirror axis did not fire (OK)"))
            _gate_table(all_stats)
        print(f"strips → {FX_DIR.relative_to(REPO_ROOT)}/fx_<stem>.png")
        return 0
    shapes = _shapes()
    wanted = argv if argv else list(shapes)
    unknown = [n for n in wanted if n not in shapes]
    if unknown:
        print(f"unknown shapes {unknown}; available: {', '.join(shapes)}")
        return 2
    for name in wanted:
        try:
            all_stats.extend(_process(name, shapes[name], sizes, degrees, twice))
        except Exception as e:  # noqa: BLE001 — debug harness: report and continue
            print(f"{name}: ERROR {type(e).__name__}: {e}")
    if all_stats:
        ratios = [s["ratio"] for s in all_stats]
        wins = sum(s["jag_win"] for s in all_stats)
        floors = sum(s["floor_ok"] for s in all_stats)
        topos = sum(s["topo_ok"] for s in all_stats)
        viols = sum(s["viol"] for s in all_stats)
        sym_ok = sum(s.get("sym_audit_ok", True) for s in all_stats)
        reg_ok = sum(s.get("reg_audit_ok", True) for s in all_stats)
        print(f"\nacceptance: jag wins {wins}/{len(all_stats)}  "
              f"median Ĵ ratio {statistics.median(ratios):.3f}  "
              f"floor-or-drop {floors}/{len(all_stats)}  "
              f"topo-or-drop {topos}/{len(all_stats)}  violations {viols}  "
              f"sym-audit {sym_ok}/{len(all_stats)}  "
              f"reg-audit {reg_ok}/{len(all_stats)}")
    print(f"strips → {SYNTH_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
