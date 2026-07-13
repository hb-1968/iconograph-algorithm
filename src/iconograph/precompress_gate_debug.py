"""Single-process debug harness for the PRE-COMPRESSION categorization gate.

Runs application.precompress_gate.evaluate_precompress over cached stage-1/2 outputs
and prints a per-fixture PASS/FAIL table + a `suspects` list — the validation surface
for categorize._color_carried. A logo is colour-carried when its alpha silhouette
collapses to one simple primitive AND holds enclosed interior colour-contrast marks
(EasyEffects' EQ knobs on a disc, Ptyxis' prompt on a square). Report-only.

    python -m iconograph.precompress_gate_debug [stem ...]     # default: all fixtures
    python -m iconograph.precompress_gate_debug --dump <stem>  # source|silhouette|marks strip

Basis = FIXTURES_ROOT (ICONOGRAPH_FIXTURES-overridable). Loads cached categorize +
preprocess directly; serial-recomputes on miss. BLAS pinned to one thread.
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
from skimage.color import rgb2lab  # noqa: E402

from iconograph.application.categorize import (  # noqa: E402
    ALPHA_OPAQUE_THRESHOLD, categorize, load_rgba)
from iconograph.application.common import interior_mark_mask  # noqa: E402
from iconograph.application.precompress_gate import evaluate_precompress  # noqa: E402
from iconograph.application.preprocess import preprocess  # noqa: E402
from iconograph.cache import FIXTURES_ROOT, TESTBED_ROOT, cached  # noqa: E402
from iconograph.debug_common import hstack, labeled, shrink  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = FIXTURES_ROOT
DUMP_DIR = TESTBED_ROOT / "precompress_gate"
CACHE = REPO_ROOT / ".cache" / "stages"

# Expectation pins (the EXPECTED_SYM/REG idiom): "color" = MUST read suspect
# (colour-carried); "mono" = MUST read clean; "any"/unlisted = report-only. Audit:
# zero expected-but-missed AND zero spurious; the tunables in application/common.py
# loosen only against a NAMED miss here (H1).
EXPECTED_STRUCTURE: dict[str, str] = {
    "easyeffects": "color", "ptyxis": "color", "electron": "color",
    "discord": "mono", "gmail": "mono", "claude": "mono",
    "chrome": "mono", "firefox": "mono", "figma": "mono", "vscode": "mono",
}


def _do_categorize(p):
    return categorize(p)


def _do_preprocess(rgba, cat):
    return preprocess(cat, rgba)


def _load(p: Path):
    """Direct-load cached categorize + preprocess; serial-recompute on miss.
    Returns (cat, preprocessed, rgba)."""
    cdir = CACHE / p.stem
    rgba = load_rgba(p)
    try:
        cat = pickle.loads((cdir / "categorize.pkl").read_bytes())
        pre = pickle.loads((cdir / "preprocess.pkl").read_bytes())
    except Exception:
        cat = cached(p, "categorize", _do_categorize, p)
        pre = cached(p, "preprocess", _do_preprocess, rgba, cat)
    return cat, pre, rgba


def _gate_table(stats: list[dict]) -> None:
    print(f"\npre-compression gate ({len(stats)} fixtures, basis {FIXTURES.name}):")
    print(f"  {'fixture':<22} {'routing':<9} {'collapse':<9} {'iou':>5} "
          f"{'marks':>6} {'suspect':>7} {'expect':>6}  verdict")
    green = 0
    suspects: list[str] = []
    for s in stats:
        exp = s["expect"]
        want = {"color": True, "mono": False}.get(exp)   # None = report-only
        audit_ok = want is None or s["suspect"] == want
        green += audit_ok
        if s["suspect"]:
            suspects.append(s["name"])
        print(f"  {s['name']:<22} {s['routing']:<9} {s['kind']:<9} "
              f"{s['iou']:>5.3f} {s['marks']:>6.3f} "
              f"{'YES' if s['suspect'] else 'no':>7} {exp:>6}  "
              f"{'PASS' if audit_ok else 'FAIL'}")
    print(f"  gate: {green}/{len(stats)} audits ok")
    print(f"  suspects (recommend color-difference): "
          f"{', '.join(suspects) if suspects else '(none)'}")


def report(fixtures: list[Path]) -> None:
    cv2.setNumThreads(1)
    stats: list[dict] = []
    for p in fixtures:
        try:
            cat, pre, rgba = _load(p)
            g = evaluate_precompress(cat, pre, rgba)
        except Exception as e:                              # noqa: BLE001 — batch
            print(f"{p.stem:<22} ERROR {type(e).__name__}: {e}")
            continue
        stats.append({
            "name": p.stem, "routing": cat.routing, "kind": g.collapse_kind,
            "iou": g.collapse_iou, "marks": g.mark_frac, "suspect": g.suspect,
            "expect": EXPECTED_STRUCTURE.get(p.stem, "-"),
        })
    if stats:
        _gate_table(stats)


def _mark_overlay(rgba: np.ndarray) -> np.ndarray:
    """Source with the detected interior marks painted red — a picture of what the
    cutout carves as holes. Transparent bg → white."""
    h, w = rgba.shape[:2]
    out = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB) if rgba.shape[2] == 4 else rgba.copy()
    out = out.copy()
    opaque = rgba[..., 3] > ALPHA_OPAQUE_THRESHOLD
    lab = rgb2lab(rgba[..., :3].astype(np.float32) / 255.0)
    marks = interior_mark_mask(lab, opaque)
    out[~opaque] = 255
    out[marks] = (255, 0, 0)
    return out


def dump(stem: str) -> None:
    """Strip: source | preprocess badge-with-holes | interior-mark overlay."""
    from PIL import Image

    cv2.setNumThreads(1)
    p = FIXTURES / f"{stem}.png"
    cat, pre, rgba = _load(p)
    g = evaluate_precompress(cat, pre, rgba)
    src = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB) if rgba.shape[2] == 4 else rgba
    panels = [
        labeled(shrink(src), f"source  {stem}"),
        labeled(shrink(pre), f"preprocess  ({'carved' if cat.color_carried else 'standard'})"),
        labeled(shrink(_mark_overlay(rgba)),
                f"marks {g.mark_frac:.3f}  collapse {g.collapse_kind}:{g.collapse_iou:.2f}  "
                f"{'SUSPECT' if g.suspect else 'clean'}"),
    ]
    strip = hstack(panels)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    out = DUMP_DIR / f"{stem}.png"
    Image.fromarray(strip, "RGB").save(out, "PNG", optimize=True)
    print(f"wrote {out.relative_to(REPO_ROOT)}  "
          f"({'SUSPECT' if g.suspect else 'clean'}, marks={g.mark_frac:.3f})")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--dump":
        if len(argv) < 2:
            print("usage: python -m iconograph.precompress_gate_debug --dump <stem>")
            return 2
        dump(argv[1])
        return 0
    fixtures = sorted(FIXTURES.glob("*.png"))
    if argv:
        wanted = set(argv)
        fixtures = [p for p in fixtures if p.stem in wanted]
    if not fixtures:
        print(f"no fixtures under {FIXTURES}")
        return 2
    report(fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
