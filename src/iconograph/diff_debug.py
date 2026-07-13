"""Lightweight, single-process debug harness for the stage-3 diff_fit refiner.

No fork pool, no full testbed wiring — loads the cached L0 align + L2 segmentation
(raw pickle reads, same as the decompose testbed: no cached() code-version stampede),
runs decompose_shape per feature as the deterministic "before", refines its winning
parts with diff_fit.refine_candidates, hard-scores everything with _compound_score,
and emits a per-fixture H7 strip plus a numeric report.

    python -m iconograph.diff_debug                  # default: homebrew, mastodon, vscode
    python -m iconograph.diff_debug <stem> [stem …]  # specific fixtures
    python -m iconograph.diff_debug --twice [stem …] # run the refiner twice, assert
                                                     # byte-identical decodes (H8)

Strip panels per feature: target | before (deterministic winner) | refined (best
variant by hard min(area, boundary)) | refined disagreement (under=red, over=blue).
BLAS/OpenMP pinned to one thread BEFORE imports — torch's pool initializes at import
time, and the determinism guarantee depends on it.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from iconograph.application.compress.common import (  # noqa: E402
    Part,
    _compound_mask,
    _compound_score,
    _min_feature_px,
    _raster_part,
)
from iconograph.application.compress.decompose import decompose_shape  # noqa: E402
from iconograph.application.compress.diff_fit import refine_candidates  # noqa: E402
from iconograph.application.compress.L2_feature_segment import Feature  # noqa: E402
from iconograph.cache import CACHE_ROOT, FIXTURES_ROOT, TESTBED_ROOT  # noqa: E402
from iconograph.debug_common import hstack as _hstack  # noqa: E402
from iconograph.debug_common import labeled as _labeled  # noqa: E402
from iconograph.debug_common import shrink  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = FIXTURES_ROOT   # env-overridable basis (ICONOGRAPH_FIXTURES)
DUMP_DIR = TESTBED_ROOT / "diff_fit"

DEFAULT_FIXTURES = ("homebrew", "mastodon", "vscode")
PANEL_H = 420

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0x3C, 0xB4, 0x4B), (0x43, 0x63, 0xD8), (0xF5, 0x82, 0x31),
    (0x91, 0x1E, 0xB4), (0x46, 0x99, 0x90), (0x9A, 0x63, 0x24),
)
_UNDER = (230, 25, 75)     # missed target — red
_OVER = (67, 99, 216)      # construction spill — blue


def _shrink(img: np.ndarray) -> np.ndarray:
    return shrink(img, PANEL_H)


def _target_panel(target: np.ndarray) -> np.ndarray:
    rgb = np.full((*target.shape, 3), 255, np.uint8)
    rgb[target > 0] = (40, 40, 40)
    return rgb


def _construction_panel(parts: list[Part], target: np.ndarray,
                        h: int, w: int) -> np.ndarray:
    recon = _compound_mask(parts, h, w)
    rgb = np.full((h, w, 3), 255, np.uint8)
    rgb[(recon > 0) & (target > 0)] = (185, 220, 190)            # agreement fill
    rgb[(target > 0) & (recon == 0)] = _UNDER                    # missed
    rgb[(recon > 0) & (target == 0)] = _OVER                     # spill
    for i, pt in enumerate(parts):
        cnts, _ = cv2.findContours(_raster_part(pt.primitive, h, w),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = _PALETTE[i % len(_PALETTE)]
        cv2.drawContours(rgb, cnts, -1, col, 2 if pt.op == "add" else 1)
    return rgb


def _disagreement_panel(parts: list[Part], target: np.ndarray,
                        h: int, w: int) -> np.ndarray:
    recon = _compound_mask(parts, h, w)
    rgb = np.full((h, w, 3), 255, np.uint8)
    rgb[(target > 0) & (recon == 0)] = _UNDER
    rgb[(recon > 0) & (target == 0)] = _OVER
    return rgb


def _chip(a: float, b: float, n: int) -> str:
    return f"a{a:.2f}/b{b:.2f} n={n}"


def _process(p: Path, twice: bool) -> None:
    cdir = CACHE_ROOT / p.stem
    fs = pickle.loads((cdir / "L2_feature_segment.pkl").read_bytes())
    min_feat = _min_feature_px(int(fs.image_shape[0]))
    rows: list[np.ndarray] = []
    for fi, seg in enumerate(s for s in fs.segments if isinstance(s, Feature)):
        t0 = time.time()
        before = decompose_shape(seg.mask, min_feat, seg.shape_hint,
                                 ring=seg.ring_geom, bbox_xy=(seg.bbox[0], seg.bbox[1]))
        h, w = before.shape
        pad = before.pad
        mh, mw = seg.mask.shape[:2]
        target = np.zeros((h, w), np.uint8)
        target[pad:pad + mh, pad:pad + mw] = seg.mask

        refined = refine_candidates([before.parts], target, min_feat, h, w)
        if twice:
            again = refine_candidates([before.parts], target, min_feat, h, w)
            same = repr(refined) == repr(again)
            print(f"  determinism({p.stem}#{fi}): {'IDENTICAL' if same else 'MISMATCH'}")
        scored = [(parts, *_compound_score(parts, target, h, w)) for parts in refined]
        ba, bb, _bm = _compound_score(before.parts, target, h, w) if before.parts else (0.0, 0.0, 0.0)
        for vi, (_parts, a, b, mn) in enumerate(scored):
            tag = "sym" if vi == len(scored) - 1 and len(scored) > 1 else f"seed{vi}"
            print(f"  {p.stem}#{fi} {tag:6} a={a:.3f} b={b:.3f} min={mn:.3f} parts={len(_parts)}")
        best = max(scored, key=lambda t: t[3]) if scored else None
        secs = time.time() - t0

        panels = [
            _labeled(_shrink(_target_panel(target)), f"{p.stem}#{fi} target"),
            _labeled(_shrink(_construction_panel(before.parts, target, h, w)),
                     f"before {_chip(ba, bb, len(before.parts))}"),
        ]
        if best is not None:
            bparts, a, b, mn = best
            panels.append(_labeled(_shrink(_construction_panel(bparts, target, h, w)),
                                   f"refined {_chip(a, b, len(bparts))}"))
            panels.append(_labeled(_shrink(_disagreement_panel(bparts, target, h, w)),
                                   "refined disagreement"))
            delta = mn - min(ba, bb)
        else:
            delta = 0.0
        rows.append(_hstack(panels))
        print(f"{p.stem}#{fi}: before min={min(ba, bb):.3f} -> "
              f"refined best min={(best[3] if best else 0.0):.3f} "
              f"(Δ{delta:+.3f})  [{secs:.1f}s]")

    if rows:
        wmax = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)),
                       constant_values=255) for r in rows]
        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.vstack(rows), "RGB").save(DUMP_DIR / f"{p.stem}.png",
                                                     "PNG", optimize=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    twice = "--twice" in argv
    argv = [a for a in argv if a != "--twice"]
    cv2.setNumThreads(1)
    wanted = argv if argv else list(DEFAULT_FIXTURES)
    fixtures = [FIXTURES / f"{stem}.png" for stem in wanted]
    fixtures = [p for p in fixtures if p.exists()]
    if not fixtures:
        print(f"no matching fixtures under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2
    for p in fixtures:
        try:
            _process(p, twice)
        except Exception as e:  # noqa: BLE001 — debug harness: report and continue
            print(f"{p.stem}: ERROR {type(e).__name__}: {e}")
    print(f"strips → {DUMP_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
