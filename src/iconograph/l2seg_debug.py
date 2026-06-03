"""Lightweight, single-process debug harness for stage-3 L2 feature segmentation.

No fork pool, no grid composite — runs segment_features over cached L1 outputs and
prints a numeric report, for fast iteration without spiking every core.

    python -m iconograph.l2seg_debug [stem ...]      # default: all fixtures
    python -m iconograph.l2seg_debug --dump <stem>   # colour-coded part split PNG

Loads each stage through the project cache (self-heals crash-truncated caches by
recomputing serially). BLAS pinned to one thread.
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
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402

from iconograph.application.categorize import categorize, load_rgba  # noqa: E402
from iconograph.application.preprocess import preprocess  # noqa: E402
from iconograph.application.compress import align, segment_prim_classify  # noqa: E402
from iconograph.application.compress.L2_feature_segment import (  # noqa: E402
    SEG_FIT_IOU,
    Feature,
    draw_feature_overlay,
    segment_features,
)
from iconograph.application.compress.segment_prim_classify import Classified, L1Miss  # noqa: E402
from iconograph.cache import cached  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"
DUMP_DIR = REPO_ROOT / "testbed" / "L2_feature_segment"
CACHE = REPO_ROOT / ".cache" / "stages"


def _do_categorize(p):
    return categorize(p)


def _do_preprocess(rgba, cat):
    return preprocess(cat, rgba)


def _do_align(pre):
    return align(pre)


def _do_spc(aligned):
    return segment_prim_classify(aligned)


def _load(p: Path):
    """Direct-load the cached L1 outputs (fast — bypasses the code-version check that
    a downstream-module edit trips); fall back to a serial recompute on miss/corruption.
    Valid while upstream stages (categorize→L1) are unchanged; clear .cache if you edit them."""
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


def report(fixtures: list[Path]) -> None:
    cv2.setNumThreads(1)
    totals: Counter[str] = Counter()
    hints: Counter[str] = Counter()
    sum_local, n_local = 0.0, 0
    near: list[tuple[str, float]] = []
    fixtures_with_defer = 0

    for p in fixtures:
        try:
            sc, cat, _ = _load(p)
        except Exception as e:
            print(f"{p.stem:<20} ERROR {type(e).__name__}: {e}")
            continue
        fs = segment_features(sc, cat)
        feats = [x for x in fs.segments if isinstance(x, Feature)]
        n_miss = sum(1 for x in sc.segments if isinstance(x, L1Miss))
        kc = Counter(f.kind for f in feats)
        totals.update(kc)
        hints.update(f.shape_hint or "none" for f in feats)
        for f in feats:
            if f.kind == "local":
                sum_local += f.best_fit_iou
                n_local += 1
            elif f.kind == "defer" and 0.80 <= f.best_fit_iou < SEG_FIT_IOU:
                near.append((p.stem, f.best_fit_iou))
        if kc.get("defer", 0):
            fixtures_with_defer += 1
        ious = " ".join(f"{f.kind[0]}:{f.shape_hint or '·'}:{f.best_fit_iou:.2f}" for f in feats)
        print(f"{p.stem:<20} {cat.routing:<9} miss={n_miss:<2} → "
              f"L{kc.get('local',0)} G{kc.get('global',0)} D{kc.get('defer',0)} | {ious}")

    print("-" * 76)
    mean_local = sum_local / n_local if n_local else 0.0
    print(f"TOTAL features: local={totals['local']} global={totals['global']} "
          f"defer={totals['defer']}  mean_local_iou={mean_local:.3f}  "
          f"fixtures_with_defer={fixtures_with_defer}")
    print(f"shape hints: {dict(hints)}")
    if near:
        near.sort(key=lambda t: -t[1])
        print(f"near-defer [0.80, {SEG_FIT_IOU:.2f}): " + ", ".join(f"{s}={i:.2f}" for s, i in near))


def dump(stem: str) -> None:
    """Colour-code each Feature onto the aligned image for one fixture."""
    from PIL import Image

    cv2.setNumThreads(1)
    p = FIXTURES / f"{stem}.png"
    sc, cat, aligned = _load(p)
    fs = segment_features(sc, cat)
    blended = draw_feature_overlay(aligned, fs)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    out = DUMP_DIR / f"{stem}.seg.png"
    Image.fromarray(blended, "RGB").save(out, "PNG", optimize=True)
    n = sum(1 for f in fs.segments if isinstance(f, Feature))
    print(f"wrote {out.relative_to(REPO_ROOT)}  ({n} feature(s))")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--dump":
        if len(argv) < 2:
            print("usage: python -m iconograph.l2seg_debug --dump <stem>")
            return 2
        dump(argv[1])
        return 0
    fixtures = sorted(FIXTURES.glob("*.png"))
    if argv:
        wanted = set(argv)
        fixtures = [p for p in fixtures if p.stem in wanted]
    if not fixtures:
        print(f"no fixtures under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2
    report(fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
