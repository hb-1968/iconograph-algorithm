"""Inspect monogram composition using the production outline path.

Write layout strips and use --twice to check repeatability.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402

# Pin cv2 to one thread BEFORE the iconograph imports: refine.common's module-level
# cv2 ops otherwise warm the pool multi-threaded, making warpPolar nondeterministic
# (H8) — a later pin can't shrink a warm pool.
cv2.setNumThreads(1)

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from iconograph.application.common import masks_8adjacent  # noqa: E402
from iconograph.application.monogram.compose import compose_icon  # noqa: E402
from iconograph.application.monogram.connect import connect_snake, join_report  # noqa: E402
from iconograph.application.monogram.common import select_monogram_letters  # noqa: E402
from iconograph.application.pipeline import build_outlines  # noqa: E402
from iconograph.debug_common import hstack, labeled  # noqa: E402
from iconograph.render_debug import _fixture_constructions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
from iconograph.cache import FIXTURES_ROOT, TESTBED_ROOT  # noqa: E402

FIXTURES = FIXTURES_ROOT   # env-overridable basis (ICONOGRAPH_FIXTURES)
NAMES_JSON = FIXTURES / "names.json"
DUMP_DIR = TESTBED_ROOT / "monogram"
OCACHE = REPO_ROOT / ".cache" / "monogram_outlines"

DEFAULT_FIXTURES = ("discord", "gmail", "vscode", "claude", "figma")
CONV = ((16, 24), (24, 32))   # (T_logo, N_canvas): logo native size, +8 canvas
PANEL_H = 384                 # strip panel height (NN upscale of the tiny icons)


def _load_names() -> dict[str, str]:
    return json.loads(NAMES_JSON.read_text())


def _compute_outlines(stem: str) -> dict[int, np.ndarray]:
    """{T: (T,T) uint8 {0,255} stage-4 refined 1-px outline} for T in {16,24}.
    Slow — decompose + render + refine. Delegates to the PRODUCTION path
    (application.pipeline.build_outlines — one implementation, harness consumes)."""
    S, _aligned, cons, src = _fixture_constructions(stem)
    return build_outlines(cons, S, src)


def _outlines(stem: str, fresh: bool) -> dict[int, np.ndarray]:
    OCACHE.mkdir(parents=True, exist_ok=True)
    p = OCACHE / f"{stem}.pkl"
    if p.exists() and not fresh:
        return pickle.loads(p.read_bytes())
    outs = _compute_outlines(stem)
    p.write_bytes(pickle.dumps(outs))
    return outs


def _up(img: np.ndarray) -> np.ndarray:
    """Nearest-neighbour upscale to exactly PANEL_H tall (logos vary in size now, so
    a fixed integer factor won't align panel heights for hstack)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (round(w * PANEL_H / h), PANEL_H), interpolation=cv2.INTER_NEAREST)


def _vis(mask: np.ndarray) -> np.ndarray:
    """(H,W) {0,255} mask -> (H,W,3) ink-black-on-white, matching the icon visual."""
    v = np.full((*mask.shape, 3), 255, np.uint8)
    v[mask > 0] = 0
    return v


def _letters_separated(res, N: int) -> bool:
    """Letter<->letter clause of the separation invariant (letters<->logo is
    res.letters vs res.logo). A missing letter -> vacuously True."""
    if res.l1 is None or res.l2 is None:
        return True
    return not masks_8adjacent(res.l1.mask, res.l2.mask)


def _onepx(mask: np.ndarray) -> bool:
    """True if no pixel is fully 8-surrounded — i.e. every stroke is <=1px wide."""
    u = (mask > 0).astype(np.uint8)
    nb = cv2.filter2D(u, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    return not bool(np.any((nb >= 9) & (u > 0)))


def _singly_connected(mask: np.ndarray) -> bool:
    """True if the mask is a single FACE-connected (4-connectivity) component (or empty)
    — i.e. one continuous stroke with no diagonal-only cuts."""
    return (cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=4)[0] - 1) <= 1


def _process(stem: str, names: dict[str, str], fresh: bool, twice: bool) -> list[tuple]:
    mono = select_monogram_letters(names.get(stem, stem))
    outs = _outlines(stem, fresh)
    panels, stats = [], []
    for T, N in CONV:
        outline = outs[T]
        res = compose_icon(outline, mono, N)
        sep_gl = not masks_8adjacent(res.letters, res.logo)   # letters <-> logo
        sep_ll = _letters_separated(res, N)                   # letter  <-> letter
        both = res.l1 is not None and res.l2 is not None
        snake = connect_snake(res.union)                       # phase 5 op-1: connect the pieces
        snake_conn = _singly_connected(snake)                  # ONE 4-connected stroke = success
        nj, nc = join_report(res.union)                        # report-only: T-free-join rate
        joins = f"{nc}/{nj}" if nj else "-"                    # (connectivity gate can't see this)
        det_ok = True
        if twice:
            det_ok = bool(np.array_equal(res.canvas, compose_icon(outline, mono, N).canvas)
                          and np.array_equal(snake, connect_snake(res.union)))
            print(f"  determinism({stem}@{N}): {'IDENTICAL' if det_ok else 'MISMATCH'}")
        onepx = _onepx(res.letters)                            # LETTERS <=1px (I render them;
        #                            the native logo's own 1px-ness is stage-4 refine's concern)
        # Each LETTER must be one continuous stroke; the logo is placed native and keeps
        # its own component count (face ring + eye/mouth counters) — not checked.
        conn = ((res.l1 is None or _singly_connected(res.l1.mask))
                and (res.l2 is None or _singly_connected(res.l2.mask)))
        sz = f"{res.l1.w}x{res.l1.h}/{res.l2.w}x{res.l2.h}" if both else "-"
        stats.append((stem, T, N, mono, res.layout, sep_gl, sep_ll, both, det_ok, onepx, conn, snake_conn, joins, sz))
        panels.append(labeled(_up(_vis(outline)), f"outline@{T}"))
        panels.append(labeled(_up(res.canvas), f"{mono} @{N}"))
        panels.append(labeled(_up(_vis(snake)), f"snake @{N}"))
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hstack(panels), "RGB").save(DUMP_DIR / f"{stem}.png", "PNG", optimize=True)
    return stats


def main(argv: list[str]) -> int:
    twice = "--twice" in argv
    fresh = "--fresh" in argv
    allf = "--all" in argv                                  # every fixture in names.json
    argv = [a for a in argv if a not in ("--twice", "--fresh", "--all")]
    cv2.setNumThreads(1)
    names = _load_names()
    wanted = sorted(names) if allf else (argv or list(DEFAULT_FIXTURES))
    all_stats: list[tuple] = []
    for stem in wanted:
        try:
            all_stats.extend(_process(stem, names, fresh, twice))
        except Exception as e:  # noqa: BLE001 — debug harness: report and continue
            print(f"{stem}: ERROR {type(e).__name__}: {e}")

    print(f"\n{'fixture':<12} {'T→N':<7} {'mono':<5} {'layout':<9} {'sep':<4} {'both':<5} "
          f"{'det':<4} {'1px':<4} {'conn':<5} {'snake':<6} {'joins':<6} box(w×h)")
    ok = True
    for stem, T, N, mono, layout, sep_gl, sep_ll, both, det_ok, onepx, conn, snake_conn, joins, sz in all_stats:
        sep = "OK" if (sep_gl and sep_ll) else "BAD"
        print(f"{stem:<12} {f'{T}→{N}':<7} {mono:<5} {layout:<9} {sep:<4} {str(both):<5} "
              f"{'OK' if det_ok else 'BAD':<4} {'OK' if onepx else 'BAD':<4} "
              f"{'OK' if conn else 'BAD':<5} {'OK' if snake_conn else 'BAD':<6} {joins:<6} {sz}")
        ok = ok and sep_gl and sep_ll and both and det_ok and onepx and conn and snake_conn
    print(f"\n{'ALL CLAUSES OK — separation (L↔logo + L↔L) + both letters + determinism'
          if ok else '*** FAILURES PRESENT ***'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
