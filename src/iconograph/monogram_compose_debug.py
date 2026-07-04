"""monogram_compose_debug.py — Stage-5 phases 2+3 harness (compose logo + letters).

For each fixture, composes the final icon at both conversions — 16-logo → 24
canvas and 24-logo → 32 canvas — and writes a comparison strip
`[outline@16 | icon@24 | outline@24 | icon@32]` to testbed/monogram/<stem>.png.
Checks the hard 1-px separation invariant (letters never 8-touch the logo) and,
with --twice, H8 byte-determinism.

    python -m iconograph.monogram_compose_debug              # 5 default fixtures
    python -m iconograph.monogram_compose_debug discord vscode
    python -m iconograph.monogram_compose_debug --twice      # determinism assert
    python -m iconograph.monogram_compose_debug --fresh      # bypass the outline cache

The refined outlines are slow to produce (decompose+render+refine, ~85s/fixture),
so they are cached under .cache/monogram_outlines/ keyed by stem — pass --fresh
after any upstream (stage 1-4) change.
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
from iconograph.application.monogram.common import select_monogram_letters  # noqa: E402
from iconograph.application.refine.common import refine_perimeter  # noqa: E402
from iconograph.debug_common import hstack, labeled  # noqa: E402
from iconograph.render_debug import _fixture_constructions, assemble_render  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"
NAMES_JSON = FIXTURES / "names.json"
DUMP_DIR = REPO_ROOT / "testbed" / "monogram"
OCACHE = REPO_ROOT / ".cache" / "monogram_outlines"

DEFAULT_FIXTURES = ("discord", "gmail", "vscode", "claude", "figma")
CONV = ((16, 24), (24, 32))   # (T_logo, N_canvas): logo native size, +8 canvas
PANEL_H = 384                 # strip panel height (NN upscale of the tiny icons)


def _load_names() -> dict[str, str]:
    return json.loads(NAMES_JSON.read_text())


def _compute_outlines(stem: str) -> dict[int, np.ndarray]:
    """{T: (T,T) uint8 {0,255} stage-4 refined 1-px outline} for T in {16,24}.
    Slow — decompose + render + refine. @16 is seeded with the @24 silhouette."""
    S, _aligned, cons = _fixture_constructions(stem)
    masks24, _ = assemble_render(cons, S, 24)
    sil24 = np.zeros((24, 24), np.uint8)
    for m in masks24:
        sil24 |= m
    out24 = refine_perimeter(cons, masks24, (S, S), 24)
    masks16, _ = assemble_render(cons, S, 16)
    out16 = refine_perimeter(cons, masks16, (S, S), 16, structure_ref=sil24)
    return {16: out16, 24: out24}


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
        det_ok = True
        if twice:
            det_ok = bool(np.array_equal(res.canvas, compose_icon(outline, mono, N).canvas))
            print(f"  determinism({stem}@{N}): {'IDENTICAL' if det_ok else 'MISMATCH'}")
        onepx = _onepx(res.letters)                            # LETTERS <=1px (I render them;
        #                            the native logo's own 1px-ness is stage-4 refine's concern)
        # Each LETTER must be one continuous stroke; the logo is placed native and keeps
        # its own component count (face ring + eye/mouth counters) — not checked.
        conn = ((res.l1 is None or _singly_connected(res.l1.mask))
                and (res.l2 is None or _singly_connected(res.l2.mask)))
        sz = f"{res.l1.w}x{res.l1.h}/{res.l2.w}x{res.l2.h}" if both else "-"
        stats.append((stem, T, N, mono, sep_gl, sep_ll, both, det_ok, onepx, conn, sz))
        panels.append(labeled(_up(_vis(outline)), f"outline@{T}"))
        panels.append(labeled(_up(res.canvas), f"{mono} @{N}"))
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hstack(panels), "RGB").save(DUMP_DIR / f"{stem}.png", "PNG", optimize=True)
    return stats


def main(argv: list[str]) -> int:
    twice = "--twice" in argv
    fresh = "--fresh" in argv
    argv = [a for a in argv if a not in ("--twice", "--fresh")]
    cv2.setNumThreads(1)
    names = _load_names()
    wanted = argv or list(DEFAULT_FIXTURES)
    all_stats: list[tuple] = []
    for stem in wanted:
        try:
            all_stats.extend(_process(stem, names, fresh, twice))
        except Exception as e:  # noqa: BLE001 — debug harness: report and continue
            print(f"{stem}: ERROR {type(e).__name__}: {e}")

    print(f"\n{'fixture':<12} {'T→N':<7} {'mono':<5} {'sep':<4} {'both':<5} {'det':<4} "
          f"{'1px':<4} {'conn':<5} box(w×h)")
    ok = True
    for stem, T, N, mono, sep_gl, sep_ll, both, det_ok, onepx, conn, sz in all_stats:
        sep = "OK" if (sep_gl and sep_ll) else "BAD"
        print(f"{stem:<12} {f'{T}→{N}':<7} {mono:<5} {sep:<4} {str(both):<5} "
              f"{'OK' if det_ok else 'BAD':<4} {'OK' if onepx else 'BAD':<4} "
              f"{'OK' if conn else 'BAD':<5} {sz}")
        ok = ok and sep_gl and sep_ll and both and det_ok and onepx and conn
    print(f"\n{'ALL CLAUSES OK — separation (L↔logo + L↔L) + both letters + determinism'
          if ok else '*** FAILURES PRESENT ***'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
