"""Compare fixtures against the legacy L3 beam-search fitting baseline."""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from iconograph.application.categorize import categorize, load_rgba  # noqa: E402
from iconograph.application.preprocess import preprocess  # noqa: E402
from iconograph.application.compress import align, segment_prim_classify  # noqa: E402
from iconograph.application.compress.common import (  # noqa: E402
    _compound_mask,
    _min_feature_px,
)
from iconograph.application.compress.L2_feature_segment import (  # noqa: E402
    Feature,
    draw_feature_overlay,
    segment_features,
)
from iconograph.application.compress.L3_fit import (  # noqa: E402
    ADEQUATE_IOU,
    Deferral,
    FittedShape,
    _fit_one,
    draw_fit_overlay,
    fit_features,
)
from iconograph.cache import FIXTURES_ROOT, TESTBED_ROOT, cached  # noqa: E402
from iconograph.debug_common import hstack as _hstack  # noqa: E402
from iconograph.debug_common import labeled as _labeled  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = FIXTURES_ROOT   # env-overridable basis (ICONOGRAPH_FIXTURES)
DUMP_DIR = TESTBED_ROOT / "L3_fit"
CACHE = REPO_ROOT / ".cache" / "stages"

DEFAULT_FIXTURES = ("android-head_flat", "gmail", "obsidian")

# pass-1 panel palette (pass-2 reuses L3_fit's own palette via draw_fit_overlay)
_PASS1_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0x3C, 0xB4, 0x4B), (0x43, 0x63, 0xD8), (0xF5, 0x82, 0x31),
    (0x91, 0x1E, 0xB4), (0x46, 0x99, 0x90), (0x9A, 0x63, 0x24),
)


def _do_categorize(p):
    return categorize(p)


def _do_preprocess(rgba, cat):
    return preprocess(cat, rgba)


def _do_align(pre):
    return align(pre)


def _do_spc(aligned):
    return segment_prim_classify(aligned)


def _load(p: Path):
    """Direct-load the cached upstream outputs (fast — bypasses the code-version check that a
    downstream-module edit trips); fall back to a serial recompute on miss/corruption."""
    cdir = CACHE / p.stem
    try:
        sc = pickle.loads((cdir / "segment_prim_classify.pkl").read_bytes())
        cat = pickle.loads((cdir / "categorize.pkl").read_bytes())
        aligned = pickle.loads((cdir / "align.pkl").read_bytes())
        return sc, cat, aligned
    except Exception:
        cat = cached(p, "categorize", _do_categorize, p)
        rgba = load_rgba(p)
        pre = cached(p, "preprocess", _do_preprocess, rgba, cat)
        aligned = cached(p, "align", _do_align, pre)
        sc = cached(p, "segment_prim_classify", _do_spc, aligned)
        return sc, cat, aligned


def _pass1_overlay(aligned: np.ndarray, fs, min_feat: int) -> tuple[np.ndarray, int]:
    """Render the pass-1 (pre-cleanup) construction of every Feature; returns
    (overlay image, total pass-1 part count)."""
    h_img, w_img = aligned.shape[:2]
    overlay = np.full_like(aligned, 255)
    ci, n1 = 0, 0
    for seg in fs.segments:
        if not isinstance(seg, Feature):
            continue
        pass1, _pass2, _iou = _fit_one(seg, min_feat)
        n1 += len(pass1)
        mask = _compound_mask(pass1, h_img, w_img)
        overlay[mask > 0] = _PASS1_PALETTE[ci % len(_PASS1_PALETTE)]
        ci += 1
    return (0.40 * aligned + 0.60 * overlay).astype(np.uint8), n1


def _process(p: Path) -> tuple[int, int]:
    sc, cat, aligned = _load(p)
    fs = segment_features(sc, cat)
    min_feat = _min_feature_px(int(fs.image_shape[0]))
    decomp = fit_features(fs, cat)

    pass1_img, n1 = _pass1_overlay(aligned, fs, min_feat)
    strip = _hstack([
        _labeled(aligned, "aligned (target)"),
        _labeled(draw_feature_overlay(aligned, fs), "L2 features"),
        _labeled(pass1_img, f"L3 pass-1  ({n1}p)"),
        _labeled(draw_fit_overlay(aligned, decomp), "L3 pass-2  (cleaned)"),
    ])
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip, "RGB").save(DUMP_DIR / f"{p.stem}.png", "PNG", optimize=True)

    fitted = [s for s in decomp.shapes if isinstance(s, FittedShape)]
    deferred = [s for s in decomp.shapes if isinstance(s, Deferral)]
    n2 = sum(len(s.parts) for s in fitted)
    per = " ".join(
        (f"F:{s.shape_hint or '·'}:{s.iou:.2f}" if isinstance(s, FittedShape)
         else f"D:{s.shape_hint or '·'}:{s.iou:.2f}")
        for s in decomp.shapes if isinstance(s, (FittedShape, Deferral))
    )
    print(f"{p.stem:<18} feats={len(fitted) + len(deferred):<2} "
          f"FIT {len(fitted)} DEFER {len(deferred)}  parts p1={n1} p2={n2}  | {per}")
    return len(fitted), len(deferred)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cv2.setNumThreads(1)
    wanted = argv if argv else list(DEFAULT_FIXTURES)
    fixtures = [FIXTURES / f"{stem}.png" for stem in wanted]
    fixtures = [p for p in fixtures if p.exists()]
    if not fixtures:
        print(f"no matching fixtures under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2

    tot_fit, tot_def = 0, 0
    for p in fixtures:
        try:
            f, d = _process(p)
            tot_fit += f
            tot_def += d
        except Exception as e:  # noqa: BLE001 — debug harness: report and continue
            print(f"{p.stem:<18} ERROR {type(e).__name__}: {e}")
    print("-" * 72)
    print(f"TOTAL  FIT {tot_fit}  DEFER {tot_def}   (ADEQUATE_IOU={ADEQUATE_IOU})")
    print(f"strips → {DUMP_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
