"""D2 decomposer debug harness — contour-driven decomposition vs the old L3 beam.

    python -m iconograph.d2_debug                 # default: docker, apple-rainbow, obsidian, android, gmail
    python -m iconograph.d2_debug <stem> [stem …]

Per-fixture 4-panel strip: aligned (target) | D1 contour | D2 decomposition (filled recon + outlined
primitives, residual red, junctions cyan) | old L3 beam. Reads cached upstream through L2; BLAS pinned.
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
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from iconograph.application.categorize import categorize, load_rgba  # noqa: E402
from iconograph.application.preprocess import preprocess  # noqa: E402
from iconograph.application.compress import align, segment_prim_classify  # noqa: E402
from iconograph.application.compress.common import _min_feature_px  # noqa: E402
from iconograph.application.compress.L2_feature_segment import Feature, segment_features  # noqa: E402
from iconograph.application.compress.contour_parse import draw_shape_overlay, parse_shape  # noqa: E402
from iconograph.application.compress.decompose import decompose_shape, draw_decomposition_overlay  # noqa: E402
from iconograph.application.compress.L3_fit import draw_fit_overlay, fit_features  # noqa: E402
from iconograph.cache import cached  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"
DUMP_DIR = REPO_ROOT / "testbed" / "D2_decompose"
CACHE = REPO_ROOT / ".cache" / "stages"
DEFAULT_FIXTURES = ("docker-mark-black", "apple-rainbow", "obsidian", "android-head_flat", "gmail")


def _do_categorize(p):
    return categorize(p)


def _do_preprocess(rgba, cat):
    return preprocess(cat, rgba)


def _do_align(pre):
    return align(pre)


def _do_spc(aligned):
    return segment_prim_classify(aligned)


def _load(p: Path):
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


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 20), (255, 255, 255), -1)
    cv2.putText(out, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def _process(p: Path) -> None:
    sc, cat, aligned = _load(p)
    fs = segment_features(sc, cat)
    min_feat = _min_feature_px(int(fs.image_shape[0]))

    shapes, decomps, summary = [], [], []
    for seg in fs.segments:
        if not isinstance(seg, Feature):
            continue
        off = (seg.bbox[0], seg.bbox[1])
        shapes.append((parse_shape(seg.mask), off))
        d = decompose_shape(seg.mask, min_feat, seg.shape_hint)
        decomps.append((d, off))
        summary.append(
            f"{seg.shape_hint or '·'}:cov{d.coverage:.2f}+{d.n_add}-{d.n_sub}J{len(d.junctions)}"
            f"R{0 if d.residual is None else 1}"
        )

    d1 = draw_shape_overlay(aligned, shapes)
    d2 = draw_decomposition_overlay(aligned, decomps)
    beam = draw_fit_overlay(aligned, fit_features(fs, cat))
    sep = np.full((aligned.shape[0], 3, 3), 0xCC, np.uint8)
    strip = np.hstack([
        _label(aligned, "target"), sep,
        _label(d1, "D1 contour"), sep,
        _label(d2, "D2 decompose (fill+outlines, residual=red, junc=cyan)"), sep,
        _label(beam, "old L3 beam"),
    ])
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip, "RGB").save(DUMP_DIR / f"{p.stem}.png", "PNG", optimize=True)
    print(f"{p.stem:<18} | " + "   ".join(summary))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cv2.setNumThreads(1)
    wanted = argv if argv else list(DEFAULT_FIXTURES)
    fixtures = [FIXTURES / f"{stem}.png" for stem in wanted]
    fixtures = [p for p in fixtures if p.exists()]
    if not fixtures:
        print(f"no matching fixtures under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2
    for p in fixtures:
        try:
            _process(p)
        except Exception as e:  # noqa: BLE001 — debug harness: report and continue
            import traceback
            traceback.print_exc()
            print(f"{p.stem:<18} ERROR {type(e).__name__}: {e}")
    print(f"strips → {DUMP_DIR.relative_to(REPO_ROOT)}/   (cov=coverage +adds -subs J junctions R residual)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
