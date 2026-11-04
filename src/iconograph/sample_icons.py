"""Sample installed-app icons for fixture testing.

Use the shared app census and resolver, exclude curated matches, and sample
with a fixed seed. --dry lists the selection without writing files. The
builder processes installed apps directly; sampling is a development tool.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cairosvg

from iconograph.appicons import (
    SVG_RASTER_W,
    installed_apps,
    materialize,
    norm,
    resolve_icon,
    slug,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "fixtures-sample"
CURATED = REPO_ROOT / "fixtures"

SEED = 20260711
N_SAMPLE = 10


def main(argv: list[str]) -> int:
    dry = "--dry" in argv
    n = N_SAMPLE
    if "--n" in argv:
        n = int(argv[argv.index("--n") + 1])
    apps = installed_apps()
    curated = [norm(p.stem) for p in sorted(CURATED.glob("*.png"))]
    eligible: dict[str, tuple] = {}
    for lookup, entry in apps.items():
        nn = norm(lookup)
        if any(c and (c in nn or nn in c) for c in curated):
            continue                        # never duplicate the curated H9 set
        src = resolve_icon(entry.icon)
        if src is None:
            continue
        eligible[lookup] = (entry, src)
    picks = random.Random(SEED).sample(sorted(eligible), min(n, len(eligible)))
    print(f"installed apps: {len(apps)}, eligible after curated-exclusion + "
          f"resolution: {len(eligible)}, seed={SEED}, n={len(picks)}")
    rows = []
    names_json: dict[str, str] = {}
    for lookup in picks:
        entry, src = eligible[lookup]
        sl = slug(lookup)
        names_json[sl] = entry.name
        rows.append((sl, lookup, str(src.path), src.fmt,
                     src.size if src.fmt == "png" else f"svg->{SVG_RASTER_W}",
                     entry.name, src.tier))
        print(f"  {sl:<36} {src.fmt:>3} [{src.tier}] {entry.name:<28} {src.path}")
        if not dry:
            OUT_DIR.mkdir(exist_ok=True)
            materialize(src, OUT_DIR / f"{sl}.png")
    if dry:
        return 0
    (OUT_DIR / "names.json").write_text(
        json.dumps(names_json, indent=2, sort_keys=True) + "\n")
    man = [
        "# fixtures-sample — random installed-app-icon basis (UNCURATED)",
        "",
        f"Deterministic sample: seed {SEED}, n {len(picks)}, eligible pool "
        f"{len(eligible)} installed apps (census+resolution via appicons.py — the "
        f"production `build` chain; arcticons/breeze restyles are not sources). SVGs "
        f"rasterized by cairosvg {cairosvg.__version__} at width {SVG_RASTER_W}. "
        "Sub-case labels TBD until triage (H9 recurrence path). Run the harnesses "
        "with `ICONOGRAPH_FIXTURES=fixtures-sample`; outputs land under "
        "`testbed/fixtures-sample/` (basis-aware TESTBED_ROOT). The stage cache "
        "namespace is shared — sample runs use explicit stem lists.",
        "",
        "| slug | icon name | source | fmt | size | app name | tier |",
        "|---|---|---|---|---|---|---|",
    ]
    man += [f"| {s} | {lk} | {src} | {fmt} | {sz} | {app} | {tier} |"
            for s, lk, src, fmt, sz, app, tier in rows]
    (OUT_DIR / "README.md").write_text("\n".join(man) + "\n")
    print(f"wrote {len(picks)} PNGs + names.json + README.md -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
