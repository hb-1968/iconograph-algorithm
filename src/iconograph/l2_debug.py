"""Lightweight, single-process debug harness for stage-3 L2.

No multiprocessing fork pool, no per-fixture PIL strips, no grid composite — it
runs L2_compound_classify over the cached L1 outputs and prints a numeric report,
for fast methodology iteration without spiking every core.

    python -m iconograph.l2_debug [stem ...]       # default: all fixtures
    python -m iconograph.l2_debug --dump <stem>    # also write one spot-check PNG

Loads each stage through the project cache (iconograph.cache.cached): a valid
cache loads instantly; a missing or crash-truncated one is recomputed serially
in this single process (gentle on the CPU). BLAS is pinned to one thread.
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP to 1 thread before numpy is imported, so a serial cache
# rebuild stays single-threaded (this harness never forks a pool).
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
import numpy as np  # noqa: E402

from iconograph.application.categorize import categorize, load_rgba  # noqa: E402
from iconograph.application.preprocess import preprocess  # noqa: E402
from iconograph.application.compress import align, segment_prim_classify  # noqa: E402
from iconograph.application.compress.L2_compound_classify import (  # noqa: E402
    ACCEPT_IOU,
    L2Compound,
    L2Defer,
    L2_compound_classify,
    _offset_part,
    render_compound,
)
from iconograph.application.compress.segment_prim_classify import Classified  # noqa: E402
from iconograph.cache import cached  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"
DUMP_DIR = REPO_ROOT / "testbed" / "L2_compound_classify"


def _do_categorize(p: Path):
    return categorize(p)


def _do_preprocess(rgba, cat):
    return preprocess(cat, rgba)


def _do_align(pre):
    return align(pre)


def _do_spc(aligned):
    return segment_prim_classify(aligned)


def _load(p: Path):
    """Cache-first; recompute serially on miss/corruption. Returns (sc, cat, aligned)."""
    cat = cached(p, "categorize", _do_categorize, p)
    rgba = load_rgba(p)
    pre = cached(p, "preprocess", _do_preprocess, rgba, cat)
    aligned = cached(p, "align", _do_align, pre)
    sc = cached(p, "segment_prim_classify", _do_spc, aligned)
    return sc, cat, aligned


def _seg_line(seg) -> str:
    if isinstance(seg, L2Compound):
        n_sub = sum(1 for p in seg.parts if p.op == "subtract")
        n_add = len(seg.parts) - n_sub
        return f"{seg.structure}:iou={seg.iou:.3f},{n_add}+{n_sub}p"
    return f"defer({seg.reason}):iou={seg.iou:.3f},holes={seg.n_holes}"


def report(fixtures: list[Path]) -> None:
    cv2.setNumThreads(1)
    tot_miss = tot_comp = tot_ring = tot_defer = 0
    sum_iou = 0.0
    near: list[tuple[str, float]] = []
    reasons: Counter[str] = Counter()

    for p in fixtures:
        try:
            sc, cat, _aligned = _load(p)
        except Exception as e:
            print(f"{p.stem:<20} ERROR {type(e).__name__}: {e}")
            continue
        cc = L2_compound_classify(sc, cat)
        misses = sum(1 for s in sc.segments if not isinstance(s, Classified))
        comps = [s for s in cc.segments if isinstance(s, L2Compound)]
        defers = [s for s in cc.segments if isinstance(s, L2Defer)]
        rings = sum(1 for s in comps if s.structure == "global_ring")

        tot_miss += misses
        tot_comp += len(comps)
        tot_ring += rings
        tot_defer += len(defers)
        sum_iou += sum(s.iou for s in comps)
        for d in defers:
            reasons[d.reason] += 1
            if 0.80 <= d.iou < ACCEPT_IOU:
                near.append((p.stem, d.iou))

        segs = "  ".join(_seg_line(s) for s in cc.segments if not isinstance(s, Classified))
        print(f"{p.stem:<20} {cat.routing:<9} miss={misses:<2} | {segs}")

    print("-" * 72)
    mean = sum_iou / tot_comp if tot_comp else 0.0
    print(
        f"TOTAL  misses={tot_miss}  compounds={tot_comp} (global_ring={tot_ring})  "
        f"defers={tot_defer}  mean_compound_iou={mean:.3f}"
    )
    print(f"defer reasons: {dict(reasons)}")
    if near:
        near.sort(key=lambda t: -t[1])
        print(f"near-misses [0.80, {ACCEPT_IOU:.2f}): " + ", ".join(f"{s}={i:.3f}" for s, i in near))


def dump(stem: str) -> None:
    """Spot-check render of ONE fixture's L2 segments → a single stacked PNG
    (aligned crop | target mask | reconstruction). No grid, no fork."""
    from PIL import Image

    cv2.setNumThreads(1)
    p = FIXTURES / f"{stem}.png"
    sc, cat, aligned = _load(p)
    cc = L2_compound_classify(sc, cat)
    segs = [s for s in cc.segments if not isinstance(s, Classified)]
    if not segs:
        print(f"{stem}: no L2 segments to dump")
        return

    panel = 240
    canvas = Image.new("RGB", (panel * 3, panel * len(segs)), (255, 255, 255))
    for i, seg in enumerate(segs):
        x, y, w, h = seg.bbox
        crop = aligned[y:y + h, x:x + w]
        target = np.full((h, w, 3), 255, np.uint8)
        target[np.any(crop < 250, axis=2)] = (60, 60, 60)
        if isinstance(seg, L2Compound):
            local = tuple(_offset_part(pt, -x, -y) for pt in seg.parts)
            recon = render_compound(local, h, w, color=tuple(int(c) for c in seg.color_rgb))
        else:
            recon = np.full((h, w, 3), 255, np.uint8)
            recon[seg.mask > 0] = (200, 60, 60)
        for col, arr in enumerate((crop, target, recon)):
            im = Image.fromarray(arr, "RGB").copy()
            im.thumbnail((panel, panel))
            canvas.paste(im, (col * panel + (panel - im.width) // 2, i * panel + (panel - im.height) // 2))
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    out = DUMP_DIR / f"{stem}.dump.png"
    canvas.save(out, "PNG", optimize=True)
    print(f"wrote {out.relative_to(REPO_ROOT)}  ({len(segs)} segment(s))")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--dump":
        if len(argv) < 2:
            print("usage: python -m iconograph.l2_debug --dump <stem>")
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
