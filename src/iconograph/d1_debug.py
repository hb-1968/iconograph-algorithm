"""D1 contour-parse debug harness — typed-contour strips on select fixtures.

    python -m iconograph.d1_debug                 # default: android-head_flat, docker-mark-black, obsidian, gmail
    python -m iconograph.d1_debug <stem> [stem …]

Per-fixture strip: aligned (target) | parsed contour (straight=blue, arc=red, corner=green dot,
smooth=orange dot). Reads cached upstream through L2; computes D1 fresh; BLAS pinned to one thread.
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
from iconograph.application.compress.L2_feature_segment import Feature, segment_features  # noqa: E402
from iconograph.application.compress.contour_parse import draw_shape_overlay, parse_shape  # noqa: E402
from iconograph.cache import FIXTURES_ROOT, TESTBED_ROOT, cached  # noqa: E402
from iconograph.debug_common import labeled as _label  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = FIXTURES_ROOT   # env-overridable basis (ICONOGRAPH_FIXTURES)
DUMP_DIR = TESTBED_ROOT / "D1_contour"
CACHE = REPO_ROOT / ".cache" / "stages"
DEFAULT_FIXTURES = ("android-head_flat", "docker-mark-black", "obsidian", "gmail")


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


def _process(p: Path) -> None:
    sc, cat, aligned = _load(p)
    fs = segment_features(sc, cat)
    shapes_by_feature, summary = [], []
    for seg in fs.segments:
        if not isinstance(seg, Feature):
            continue
        shape = parse_shape(seg.mask)
        shapes_by_feature.append((shape, (seg.bbox[0], seg.bbox[1])))
        if shape is not None:
            o = shape.outer
            ns = sum(1 for s in o.segments if s.kind == "straight")
            ac = sum(1 for s in o.segments if s.kind == "arc" and s.convex)
            av = sum(1 for s in o.segments if s.kind == "arc" and not s.convex)
            nc = sum(1 for c in o.corners if c.kind == "corner")
            summary.append(f"{seg.shape_hint or '·'}:S{ns}·cvxA{ac}·ccvA{av}·C{nc}·H{len(shape.holes)}")
        else:
            summary.append(f"{seg.shape_hint or '·'}:∅")

    overlay = draw_shape_overlay(aligned, shapes_by_feature)
    sep = np.full((aligned.shape[0], 3, 3), 0xCC, np.uint8)
    strip = np.hstack([
        _label(aligned, "aligned (target)"), sep,
        _label(overlay, "D1 v2  straight=blu cvxArc=red ccvArc=ppl hole=amb peak=grn valley=cyn"),
    ])
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip, "RGB").save(DUMP_DIR / f"{p.stem}.png", "PNG", optimize=True)
    print(f"{p.stem:<18} feats={len(shapes_by_feature)}  | " + "  ".join(summary))


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
            print(f"{p.stem:<18} ERROR {type(e).__name__}: {e}")
    print(f"strips → {DUMP_DIR.relative_to(REPO_ROOT)}/   (S=straight A=arc C=corner runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
