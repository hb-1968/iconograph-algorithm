"""Visualize the classification and decomposition inputs to rendering.

Show fitted primitives, segmented features and their labels per fixture.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from iconograph.application.categorize import categorize, load_rgba  # noqa: E402
from iconograph.application.preprocess import preprocess  # noqa: E402
from iconograph.application.compress import align, segment_prim_classify  # noqa: E402
from iconograph.application.compress.segment_prim_classify import (  # noqa: E402
    Classified, L1Miss, draw_overlay)
from iconograph.application.compress.L2_feature_segment import (  # noqa: E402
    Feature, draw_feature_overlay, segment_features)
from iconograph.application.compress.render import (  # noqa: E402
    _FX_BG, build_constructions, reference_mask)
from iconograph.cache import FIXTURES_ROOT, TESTBED_ROOT, cached  # noqa: E402
from iconograph.debug_common import hstack, labeled, shrink  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = FIXTURES_ROOT
DUMP_DIR = TESTBED_ROOT / "classify_map"
CACHE = REPO_ROOT / ".cache" / "stages"


def _do_categorize(p):
    return categorize(p)


def _do_preprocess(rgba, cat):
    return preprocess(cat, rgba)


def _do_align(pre):
    return align(pre)


def _do_spc(aligned):
    return segment_prim_classify(aligned)


def _do_constructions(fs, routing):
    S, cons, _src = build_constructions(fs, routing)
    return S, cons


def _load(p: Path, decompose: bool = False):
    """Direct-load cached categorize/align/L1 (fast); serial-recompute on miss, then
    L2 features. The DECOMPOSE (build_constructions, D2/D3 — 60-120s/logo) is the
    slow stage-3 step and is run ONLY when `decompose=True` (then cached, so a later
    strip is instant). Default skips it: run every icon through feature CLASSIFICATION
    without paying for a decomposition that misclassified features can't make good.
    Returns (cat, aligned, sc, fs, S|None, cons|None). Uses the code-version-aware
    cache (not a raw pickle load) so a map always reflects the CURRENT classifier —
    stale caches from an older categorize/L1 recompute automatically."""
    cat = cached(p, "categorize", _do_categorize, p)
    rgba = load_rgba(p)
    pre = cached(p, "preprocess", _do_preprocess, rgba, cat)
    aligned = cached(p, "align", _do_align, pre)
    sc = cached(p, "segment_prim_classify", _do_spc, aligned)
    fs = segment_features(sc, cat)
    if decompose:
        S, cons = cached(p, "classify_constructions", _do_constructions, fs, cat.routing)
    else:
        S, cons = None, None
    return cat, aligned, sc, fs, S, cons


# label summaries

def _cat_str(cat) -> str:
    flags = [cat.routing]
    if cat.color_essential:
        flags.append("CE")
    if cat.is_compound:
        flags.append(f"CMP/{cat.precedence}")
    if cat.color_carried:
        flags.append("color_carried")
    return " ".join(flags)


def _l1_str(sc) -> str:
    out = []
    for s in sc.segments:
        if isinstance(s, Classified):
            out.append(f"{type(s.primitive).__name__}:{s.iou:.2f}")
        else:  # L1Miss
            k, v = s.best_attempt
            out.append(f"MISS({k}:{v:.2f})")
    return " ".join(out) if out else "(none)"


# Hints whose best_fit_iou is a HARDCODED 0.0 sentinel — L2 defers these to the D2/D3
# annular decomposer (with a RingGeom), so a single-primitive IoU is meaningless. Show
# "→D3" not "0.00" so the map doesn't false-alarm; run --decompose for the real coverage.
_DEFERRED_HINTS = {"annular_band", "annular_sector", "gapped_annulus"}


def _l2_str(fs) -> str:
    kinds: Counter[str] = Counter()
    hints = []
    for s in fs.segments:
        if isinstance(s, Feature):
            kinds[s.kind] += 1
            if s.shape_hint in _DEFERRED_HINTS:
                hints.append(f"{s.kind[0]}:{s.shape_hint}→D3")   # deferred, not a fit score
            else:
                hints.append(f"{s.kind[0]}:{s.shape_hint or '·'}:{s.best_fit_iou:.2f}")
        else:  # Classified passthrough
            kinds["cls"] += 1
            hints.append(f"c:{type(s.primitive).__name__}:{s.iou:.2f}")
    head = (f"L{kinds.get('local', 0)} G{kinds.get('global', 0)} "
            f"D{kinds.get('defer', 0)} cls{kinds.get('cls', 0)}")
    return f"{head} | {' '.join(hints)}" if hints else head


def _decomp_str(cons) -> str:
    n_parts = sum(len(parts) for parts, _c, _l in cons)
    per = []
    for parts, _c, label in cons:
        kinds = "+".join(type(p.primitive).__name__[:4] for p in parts) or "∅"
        per.append(f"{label}[{kinds}]")
    return f"{len(cons)} cons / {n_parts} parts: {' '.join(per)}" if cons else "(no constructions)"


# panels

def _rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def _decompose_panel(cons, S: int, target: int = 220) -> np.ndarray:
    rgb = np.full((target, target, 3), _FX_BG, np.uint8)
    for parts, color, _label in cons:
        if not parts:
            continue
        m = reference_mask(parts, (S, S), target)
        rgb[m > 0] = color
    return rgb


def report(fixtures: list[Path], decompose: bool = False) -> None:
    cv2.setNumThreads(1)
    for p in fixtures:
        try:
            cat, _al, sc, fs, _S, cons = _load(p, decompose)
        except Exception as e:                              # noqa: BLE001 — batch
            print(f"{p.stem:<22} ERROR {type(e).__name__}: {e}")
            continue
        print(f"\n{p.stem}")
        print(f"  categorize : {_cat_str(cat)}")
        print(f"  L1 prims   : {_l1_str(sc)}")
        print(f"  L2 features: {_l2_str(fs)}")
        if cons is not None:
            print(f"  decompose  : {_decomp_str(cons)}")


def dump(stem: str, decompose: bool = False) -> None:
    from PIL import Image

    cv2.setNumThreads(1)
    p = FIXTURES / f"{stem}.png"
    cat, aligned, sc, fs, S, cons = _load(p, decompose)
    panels = [
        labeled(shrink(_rgb(aligned)), f"{stem}  [{_cat_str(cat)}]"),
        labeled(shrink(_rgb(draw_overlay(aligned, sc))), f"L1  {_l1_str(sc)}"),
        labeled(shrink(_rgb(draw_feature_overlay(aligned, fs))), f"L2  {_l2_str(fs)}"),
    ]
    if cons is not None:
        panels.append(labeled(shrink(_decompose_panel(cons, S)),
                              f"decompose  {_decomp_str(cons)}"))
    strip = hstack(panels)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    out = DUMP_DIR / f"{stem}.png"
    Image.fromarray(strip, "RGB").save(out, "PNG", optimize=True)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # decompose (opt-in): also run the slow D2/D3 decompose + panel. Default OFF —
    # classification (categorize/L1/L2) only, so the whole set runs fast.
    decompose = "--decompose" in argv
    argv = [a for a in argv if a != "--decompose"]
    if argv and argv[0] == "--dump":
        if len(argv) < 2:
            print("usage: python -m iconograph.classify_map_debug [--decompose] --dump <stem>")
            return 2
        dump(argv[1], decompose)
        return 0
    if argv and argv[0] == "--dump-all":
        for p in sorted(FIXTURES.glob("*.png")):
            dump(p.stem, decompose)
        return 0
    fixtures = sorted(FIXTURES.glob("*.png"))
    if argv:
        wanted = set(argv)
        fixtures = [p for p in fixtures if p.stem in wanted]
    if not fixtures:
        print(f"no fixtures under {FIXTURES}")
        return 2
    report(fixtures, decompose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
