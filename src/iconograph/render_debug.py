"""Lightweight, single-process debug harness for the stage-3 R1 render tier.

Two modes:

SYNTHETIC (default): hand-built Part compositions spanning the methodology axes
— sole-curve hard straightening, counters vs shaves, lobe/accent sibling
competition, negative-group counters, auto-degenerate capsules — rendered at
small hicolor grids and compared against the supersampled reference and the
naive nearest-neighbour baseline the renderer must beat.

FIXTURE (--fixtures): the real pipeline — loads the cached L2 segmentation (raw
pickle reads, diff_debug's pattern), runs decompose_shape per Feature in-process
(decompositions are computed once and reused across all target sizes), offsets
each construction to full-canvas coords, renders every construction
INDEPENDENTLY on the shared global grid (per-feature rendering avoids
cross-feature CSG group collisions), and composites by segment color. L1
Classified segments render as 1-part add constructions. Features whose
decomposition returns no parts are deferred (R3 residual territory, reported).

    python -m iconograph.render_debug                    # all shapes, sizes 16/24/32/64
    python -m iconograph.render_debug circle annulus     # specific shapes
    python -m iconograph.render_debug --fixtures         # default fixtures (discord, gmail, claude)
    python -m iconograph.render_debug --fixtures vscode  # specific fixture stems
    python -m iconograph.render_debug --final            # assemble + deliver the
                                                         #   FINAL render products
                                                         #   (source | final@16 |
                                                         #   final@24), one clean row
                                                         #   per fixture + a combined
                                                         #   gallery → testbed/render/
                                                         #   final/ . --final --all for
                                                         #   the full dataset.
    python -m iconograph.render_debug --sizes 16,32      # specific target sizes
    python -m iconograph.render_debug --degrees          # + per-degree ladder panels (H7,
                                                         #   synthetic mode only)
    python -m iconograph.render_debug --twice            # render twice, assert byte-
                                                         #   identical masks (H8)

Synthetic strip panels per (shape, size): reference | naive NN | R1 render |
diff (red = missed reference, blue = spill), all nearest-upscaled. Fixture
strips: one PNG per fixture, a row per size: aligned source | reference
composite | naive composite | R1 composite (segment colors, light-grey bg so
white-on-white features stay visible) → testbed/render/fx_<stem>.png.
Startup self-test pins the fillPoly raster convention (outline-inclusive,
round-half-up: an integer-corner box (a..b) fills pixels a..b exactly) — it
silently corrupts every downstream number if wrong.

────────────────────────────────────────────────────────────────────────────
ROUND-4 BRIEF (H10: you implement) — harness side of removal-on-globbing
(render.py's round-4 brief is the renderer side; land that first).

(a) New synthetic "asterisk12" in _shapes() — the claude-geometry analogue
    the decimation evidence gate is tuned against:
        hub Circle(96, 96, 18) + 12 RotatedRectangle blades ~52×7, centers
        at radius ~44 from the hub, angles = 30° pitch + a HAND-WRITTEN
        deterministic jitter list (e.g. +8,-6,+4,-8,+6,-4 repeating — no
        RNG, H8), all "add", canvas (S, S).
    EXPECTED_SYM["asterisk12"] = "none" (organic jitter must not fire an
    axis — guard). Tune blade width/jitter so @16 is majority-sub-
    representable (decimates; ladder picks 6 vs 4) and @32 is not (12
    blades survive). Probe with _subrep_pairs' printed {sub}/{n} evidence
    in the absorbed reason.
(b) Per-cell truth columns: both print sites (the fixture per-construction
    line and the synthetic per-cell line) append, getattr-guarded like the
    round-2/3 fields:
        f"  {kind} {fid_ref:.3f} removed {removed_ink}px"  # only when
        # fid_ref is not None; kind = "live" (removal) | "extr"
        # (extrapolated) — a removal/decimation/idealization win must
        # never hide behind the full-ref fid alone (per-cell-truth
        # taxonomy); removed px only for removal wins
(c) Optional, TIMEBOXED: "pocket_dot" — body whose pocket lodges a dot
    accent such that every candidate (incl. mode-0) fills the pocket →
    absolute pocket failure attributable to the accent → phase-B removal
    fires and wins with battery TRUE. The search avoids avoidable globs by
    design, so this may be untunable — if so, record an open item and gate
    the accent path on the claude fixture + unit-level attribution instead.

AMENDMENT (extrapolated symmetry — render.py brief R4-6/7/8):
(d) New synthetic "row_stack" in _shapes() — the figma-geometry analogue:
        Rectangle spine + RoundedRectangle pill (h ≈ 2r) + 3 Circles with
        DELIBERATE small perturbations (radii e.g. 26.0/26.6/25.4, two cy
        equal within ~1 px, row pitch off-uniform by ~1 %), all "add",
        S = 192, hand-written values (no RNG, H8). The regularity detector
        must fire (margins mirror the figma probe: spread 0.020, CV 0.003)
        and the extrapolated variant must snap radii/alignment/pitch
        exactly. EXPECTED_SYM["row_stack"] = "none".
(e) EXPECTED_SYM amendments: "claude" and "asterisk12" become "any" — a
    phase-snapped extrapolated star may legitimately fire a mirror axis;
    neither a fire nor a non-fire is an audit failure for these two.
(f) New EXPECTED_REG audit table (same gate shape as EXPECTED_SYM — zero
    expected-but-missed AND zero spurious):
        "radial": claude, asterisk6, asterisk12     # _radial_family fires
        "group":  figma, row_stack                  # _regularity_groups fires
        "none":   everything else (wedge, competition, discord, gmail,
                  vscode, thin_x, long_diag, tab_body, ... — discord's two
                  eyes are a 2-member set: mirror-twin territory, size
                  groups need ≥ 3)
    Harness prints detected-vs-expected per cell; REG constants loosen only
    against a named miss, one constant per change, strips before/after.

Acceptance gates (restated for round 4 + amendment): full grid no
regressions (jag wins, floor-or-drop, topo, sym-audit at parity or better;
viol budget re-measured and reported per cell); asterisk6@16/@32 strips
BYTE-IDENTICAL (its winners are already regular stars — the feasibility
rule must not shift them); thin_x@16 unchanged and phase B not constructed
(its violations classify "floor"); claude@16 = uniform 6-spike extrapolated
star, claude@32 = uniform 12-spike extrapolated star (fid_ref ≥ floor,
full-ref fid reported beside it either way, viol < 7); figma@32 = three
EQUAL aligned circles on a uniform pitch with the circle at ≥ octagon
degree, figma@16 = regular aligned stack (recognizability strips judged
under H1); asterisk12 @16 decimated-star / @32 12-spike per the same
evidence; row_stack per (d); EXPECTED_REG audit clean; --twice
byte-identical on the grid + claude + figma ×2; vscode/discord/gmail cells
unchanged.

════════════════════════════════════════════════════════════════════════════
ROUND-6 BRIEF — the gate surface (step 1 of the approved plan)

TASK 6-H1
  Make the 5-fixture × {16,24,32} gate visible: defaults + per-cell table.

WHAT YOU ARE WRITING
  File:      this file (render_debug.py)
  Position:  (a) DEFAULT_FIXTURES (~line 165) → ("discord", "gmail",
             "vscode", "claude", "figma");
             (b) a `strays` count in _process_fixture's stats dict
             (~line 503) — count of r.drops entries containing
             "stray component removed", summed across features per cell;
             (c) a gate table in main()'s fixtures branch, printed after
             the existing fixture summary (~line 637);
             (d) module docstring: record the ROUND-6 GATE — every fixture
             cell at T ∈ {16, 24, 32}: no LOSS, viol 0, strays 0,
             reg/sym audits pass, visual strip sign-off (user's ruling:
             "16, 24, 32 exclusively").
  Imports:   none new.
  Signature (table emitter, module level beside the summaries):
      def _gate_table(all_stats: list[dict]) -> None:
          # one row per (stem, T): Jratio (LOSS tag), viol, strays,
          # reg-audit, floor-or-drop, verdict PASS/FAIL; then a one-line
          # tally "gate: N/15 cells green"
  Called by: main(), fixtures branch only.

CONCEPT
  - The gate is a TABLE, not a paragraph: 15 cells, each green or named —
    the per-cell-truth doctrine (3-decimal ratios, LOSS tags) applied to
    the shipping decision. A cell is green iff: ratio ≤ 1.0, viol == 0,
    strays == 0, floor_ok, reg_audit_ok (sym has no per-fixture
    expectation — the figma mirror guard already prints separately).
  - Synthetic DEFAULT_SIZES stays (16, 24, 32, 64) — the 60-cell sweep is
    the regression guard; only the FIXTURE gate is 16/24/32.

GOTCHAS
  - Print-only change: stats math and strips must be byte-identical —
    --twice before/after is the step-1 check.
  - r.drops strings are the source of truth for strays (RenderResult has
    no dedicated field; do not add one — the drop record IS the contract).
  - ASCII only in cv2 panel captions (Hershey renders non-ASCII as ??);
    the table goes to stdout, so Ĵ etc. are fine THERE.
────────────────────────────────────────────────────────────────────────────
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
    _topology,
    floor_for,
    jag_norm,
    jag_score,
)
from iconograph.application.compress.render import render_debug as r1_render  # noqa: E402
from iconograph.application.compress.straighten import DEG_TRUE, straighten_part  # noqa: E402
from iconograph.cache import CACHE_ROOT  # noqa: E402
from iconograph.debug_common import hstack, labeled  # noqa: E402
from iconograph.application.refine.common import (  # noqa: E402
    cleanup_scene, outline_scene, refine_outline)

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMP_DIR = REPO_ROOT / "testbed" / "render"
FINAL_DIR = DUMP_DIR / "final"     # --final: assembled render products (the
#                                    deliverable + stage-4 refine's input)

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
        # ── round-3 cases: each exposes one critique mechanic ──
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
        # ── round-4 cases: extrapolated symmetry ──
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


# ── fillPoly half-pixel convention self-test ─────────────────────────────


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


# ── panels ───────────────────────────────────────────────────────────────


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


# ── fixture mode ─────────────────────────────────────────────────────────


def _fixture_constructions(stem: str):
    """Cached L2 → per-segment constructions in FULL-canvas coords.
    Classified → 1-part add; Feature → decompose_shape (computed once, reused
    across every target size) offset by (bbox − pad). Returns
    (S, aligned_rgb, [(parts, color_rgb, label), ...])."""
    cdir = CACHE_ROOT / stem
    fs = pickle.loads((cdir / "L2_feature_segment.pkl").read_bytes())
    aligned = pickle.loads((cdir / "align.pkl").read_bytes())
    # stage-1 routing ('standard'|'inverted'|'burst') — the label-cleaner's sub-case
    # signal (WS2). seg.shape_hint is the per-segment refinement, passed below.
    sub_case = pickle.loads((cdir / "categorize.pkl").read_bytes()).routing
    S = int(fs.image_shape[0])
    min_feat = _min_feature_px(S)
    cons: list[tuple[list[Part], tuple[int, int, int], str]] = []
    n_residual = 0
    for si, seg in enumerate(fs.segments):
        if isinstance(seg, Classified):
            cons.append(([Part(seg.primitive, "add")], seg.color_rgb,
                         f"#{si} L1 {type(seg.primitive).__name__}"))
            continue
        if not isinstance(seg, Feature):
            continue
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
            print(f"  {stem}#{si}: decompose returned no parts — deferred "
                  f"(R3 residual territory) [{dt:.1f}s]")
            continue
        ox, oy = seg.bbox[0] - d.pad, seg.bbox[1] - d.pad
        parts = [_offset_part(p, ox, oy) for p in d.parts]
        cons.append((parts, seg.color_rgb, f"#{si} D2 n={len(parts)}"))
        print(f"  {stem}#{si} decomposed: {len(parts)} parts "
              f"cov a{d.coverage:.2f}/b{d.boundary_cov:.2f} [{dt:.1f}s]")
    if n_residual:
        print(f"  {stem}: {n_residual} feature(s) carry a residual "
              f"(R3 tier, not rendered)")
    return S, aligned, cons


_FX_BG = (238, 238, 238)   # light grey: white-on-white features stay visible


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
    S, aligned, cons = _fixture_constructions(stem)
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
        # cross-construction separation: re-open 1px divots between source-separate
        # constructions that fused at the feed res (slack's pinwheel). The composite mask
        # + R1 panel use the divoted masks; per-construction r.* metrics stay from rrs.
        dmasks = _separate_divots([r.mask for r in rrs], cons, S)
        ren = np.zeros((t, t), np.uint8)
        ref = np.zeros((t, t), np.uint8)
        nai = np.zeros((t, t), np.uint8)
        viol = drops = strays = 0
        axis_fired = False
        reg_det_stem = None
        for idx, ((parts, _c, label), r) in enumerate(zip(cons, rrs)):
            ren |= dmasks[idx]
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
            labeled(_color_panel([(dm, c) for dm, (_p, c, _l) in zip(dmasks, cons)], t),
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
        if t <= CUBIFY_MAX_T:
            # cubify represents labels at the feed resolution — pass on EITHER the
            # SIGNIFICANT topology matching (filters sub-min-feature rasterization holes —
            # claude's 1px blade gaps) OR the plain topology matching ref/canon (the
            # label-driven square ring + triangle matches the canonical structure).
            mft = _min_feature_px(t)
            topo_cell_ok = (_sig_topology(ren, mft) == _sig_topology(ref, mft)
                            or _topology(ren) in (_topology(ref), topo_canon_comp)
                            or drops > 0)
        else:
            topo_cell_ok = (_topology(ren) in (_topology(ref), topo_canon_comp)
                            or drops > 0)
        # The cubify render (T <= CUBIFY_MAX_T) REPRESENTS labels, it does not pixel-fit
        # the round raster (user ruling) — a square ring barely overlaps a circle ring, so
        # area IoU is the WRONG floor there. A structurally-correct cubified render
        # (topology matches the reference / canonical truth) satisfies the floor regardless
        # of IoU; the geometry path (T > 24) keeps the fidelity floor.
        floor_ok = (fid >= floor or drops > 0
                    or (t <= CUBIFY_MAX_T and topo_cell_ok))
        stats.append({"name": stem, "t": t, "ratio": ratio, "fid": fid,
                      "floor_ok": floor_ok,
                      "jag_win": ratio <= 1.0, "viol": viol, "drops": drops,
                      "strays": strays, "topo_ok": topo_cell_ok,
                      "axis_fired": axis_fired,
                      "reg_audit_ok": reg_audit_ok})
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows), "RGB").save(
        DUMP_DIR / f"fx_{stem}.png", "PNG", optimize=True)
    return stats


# ── final render product (assembled deliverable; stage-4 refine's input) ──


SEP_MIN_CONS = 4   # divots fire only on OVERREPRESENTED multi-element marks (>= this many
#                    constructions — slack's 8-pill pinwheel). A 2-3 piece glyph (adobe's
#                    A + i-stem + i-dot) is NOT force-separated: the i-dot stays attached to
#                    its stem (user: 'dot attached to stem'). A size guard can't do this —
#                    at 16px slack's short pills are the SAME size as the i-dot — so the
#                    discriminator is piece COUNT (the user's 'overrepresented' principle).


def _separate_divots(masks: list[np.ndarray], cons, S: int) -> list[np.ndarray]:
    """Re-open 1px divots between constructions that are SEPARATE in the source but whose
    rendered masks fused at the feed resolution (slack's 8 pinwheel pills 'glob together
    for no good reason'). Gated to OVERREPRESENTED marks (>= SEP_MIN_CONS constructions);
    a 2-3 piece glyph is left as rendered. Source-separate iff CANONICAL footprints are
    8-disconnected at canvas res (unambiguous, before the feed-res downsample globs the
    gaps — slack's gaps are sub-pixel in the reference at 16, so a reference-based test
    under-separates). Where such a pair's rendered masks touch, the contact pixels are
    cleared from the later (area-sorted smaller, drawn on top) construction so each label
    keeps a >=1px gap; pairs that genuinely abut in the source are left fused. (H8)."""
    if len(masks) < SEP_MIN_CONS:
        return masks
    k = np.ones((3, 3), np.uint8)
    canon = [_compound_mask(parts, S, S) for parts, _c, _l in cons]
    out = [m.copy() for m in masks]
    for i in range(len(masks)):
        di = cv2.dilate(masks[i], k)
        ci = cv2.dilate(canon[i], k)
        for j in range(i + 1, len(masks)):
            if np.any((ci > 0) & (canon[j] > 0)):
                continue                              # abut in source — preserve the join
            contact = (di > 0) & (masks[j] > 0)
            if np.any(contact):
                out[j] = out[j].copy()
                out[j][contact] = 0
    return out


def assemble_render(cons, S: int, target: int):
    """Add all the render pieces together for ONE target size.

    cons: [(parts, color_rgb, label), ...] from _fixture_constructions (decomposed
    + label-cleaned once, target-independent). Renders every construction
    INDEPENDENTLY on the shared (S, S) grid (per-feature, no cross-feature CSG
    group collisions — the strip path's contract) and composites the
    per-construction R1 masks in segment order (area-sorted upstream → smaller
    features draw on top) into one final color icon.

    Returns (masks, icon): masks = per-construction (target, target) uint8 {0,255}
    R1 masks; icon = (target, target, 3) uint8 composited render. The tuple
    (cons, masks, (S, S), target) is exactly what stage-4
    refine.common.refine_outline consumes — this function IS the seam refine
    touches up next."""
    rrs = [r1_render(parts, (S, S), target) for parts, _c, _l in cons]
    masks = _separate_divots([r.mask for r in rrs], cons, S)
    icon = np.full((target, target, 3), _FX_BG, np.uint8)
    for m, (_p, c, _l) in zip(masks, cons):
        icon[m > 0] = c
    return masks, icon


def _final_product(stem: str, sizes: tuple[int, ...], twice: bool):
    """Deliver the FINAL render product for one fixture: a single clean row
    `aligned source | final@T ...` (no reference/naive/diff clutter — the product
    the user critiques), saved standalone to testbed/render/final/<stem>.png.
    Returns the row image (for the combined gallery) or None if uncached/empty."""
    if not (CACHE_ROOT / stem / "L2_feature_segment.pkl").exists():
        print(f"{stem}: no cached L2 segmentation under .cache/stages/ — "
              f"run the testbed/build first")
        return None
    S, aligned, cons = _fixture_constructions(stem)
    if not cons:
        print(f"{stem}: nothing to render")
        return None
    panels = [labeled(_src_panel(aligned), f"{stem} aligned")]
    for t in sizes:
        masks, _icon = assemble_render(cons, S, t)
        if twice:
            again, _ = assemble_render(cons, S, t)
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


# ── per-case driver ──────────────────────────────────────────────────────


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
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(hstack(panels), "RGB").save(
            DUMP_DIR / f"{name}_{t}.png", "PNG", optimize=True)
        stats.append({"name": name, "t": t, "ratio": ratio, "fid": fid,
                      "floor_ok": floor_ok, "jag_win": ratio <= 1.0,
                      "topo_ok": topo_ok, "sym_audit_ok": sym_audit_ok,
                      "reg_audit_ok": reg_audit_ok,
                      "viol": len(res.violations), "drops": len(res.drops)})
    return stats


def _gate_table(all_stats: list[dict]) -> None:
    """Round-6 shipping gate, per-cell truth: one row per (fixture, T).
    Green iff no LOSS, viol 0, strays 0, floor-or-drop, reg-audit ok.
    Sym has no per-fixture expectation — the figma mirror guard prints
    separately. Visual strip sign-off stays a human step (H1)."""
    gsizes = sorted({s["t"] for s in all_stats})
    print(f"\ngate table (feed res — fixtures x {{{','.join(map(str, gsizes))}}}):")
    print(f"  {'fixture':<10} {'T':>3}  {'Jratio':>10}  {'viol':>4}  "
          f"{'strays':>6}  {'topo':>4}  {'floor':>5}  {'reg':>3}  verdict")
    green = 0
    for s in all_stats:
        ok = (s["ratio"] <= 1.0 and s["viol"] == 0
              and s.get("strays", 0) == 0 and s.get("topo_ok", True)
              and s["floor_ok"] and s.get("reg_audit_ok", True))
        green += ok
        print(f"  {s['name']:<10} {s['t']:>3}  {s['ratio']:>5.3f}"
              f"{' LOSS' if s['ratio'] > 1.0 else '     '}  "
              f"{s['viol']:>4}  {s.get('strays', 0):>6}  "
              f"{'ok' if s.get('topo_ok', True) else 'BAD':>4}  "
              f"{'ok' if s['floor_ok'] else 'MISS':>5}  "
              f"{'ok' if s.get('reg_audit_ok', True) else 'BAD':>3}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  gate: {green}/{len(all_stats)} cells green")


def _all_cached() -> list[str]:
    """Every fixture with a cached L2 segmentation under .cache/stages/ (the full H9
    dataset), sorted — the `--fixtures --all` target."""
    return sorted(p.name for p in CACHE_ROOT.iterdir()
                  if p.is_dir() and (p / "L2_feature_segment.pkl").exists())


def _half_cores() -> int:
    """Half the host cores — the bounded thread/CPU budget for the fixture pool (a 16-core
    box → 8). GPU offload is unavailable on this box (RX 580 / no ROCm), so CPU-parallel
    across fixtures is the perf lever; each worker stays single-threaded for H8."""
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
        print(f"strips → {DUMP_DIR.relative_to(REPO_ROOT)}/fx_<stem>.png")
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
    print(f"strips → {DUMP_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
