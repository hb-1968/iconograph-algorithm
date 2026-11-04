"""Build application icons from source PNGs and names.

Cache categorization through feature segmentation, then construct, render and
refine 16/24-pixel outlines. Compose them on 24/32-pixel monogram canvases and
connect the pieces. Debug harnesses reuse this production path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from iconograph.application.categorize import categorize, load_rgba
from iconograph.application.compress import (
    align,
    segment_features,
    segment_prim_classify,
)
from iconograph.application.compress.render import (
    assemble_render,
    build_constructions,
)
from iconograph.application.monogram.common import select_monogram_letters
from iconograph.application.monogram.compose import compose_icon
from iconograph.application.monogram.connect import connect_snake
from iconograph.application.precompress_gate import GateResult, evaluate_precompress
from iconograph.application.preprocess import preprocess
from iconograph.application.refine.common import refine_perimeter
from iconograph.cache import cached

CONV: tuple[tuple[int, int], ...] = ((16, 24), (24, 32))   # (T_logo, N_canvas)


class BuildError(RuntimeError):
    """A per-app pipeline failure with a report-friendly reason."""


@dataclass
class BuiltIcon:
    """One app's finished pipeline products."""
    outlines: dict[int, np.ndarray]   # {16, 24}: (T,T) uint8 {0,255} refined outline
    snakes: dict[int, np.ndarray]     # {24, 32}: (N,N) uint8 {0,255} composed snake
    monogram: str                     # the two chosen letters
    layouts: dict[int, str]           # {N: compose layout name}
    gate: GateResult | None = None    # pre-compression categorization gate (report-only)


def build_outlines(cons, S: int, src: np.ndarray | None) -> dict[int, np.ndarray]:
    """Stage 3 render + stage 4 refine for one construction set: the refined
    1-px outlines at both feed sizes. @16 is seeded with the @24 silhouette
    (accent-CARVED — widen must never chase a deliberately-placed accent's
    footprint) and both calls carry their protected-accent masks. The single
    shared implementation for production build and the monogram harness."""
    masks24, _i24, prot24 = assemble_render(cons, S, 24, src)
    sil24 = np.zeros((24, 24), np.uint8)
    for m in masks24:
        sil24 |= m
    out24 = refine_perimeter(cons, masks24, (S, S), 24, protected=prot24)
    masks16, _i16, prot16 = assemble_render(cons, S, 16, src)
    if prot24 is not None:
        sil24 = sil24.copy()
        sil24[prot24 > 0] = 0
    out16 = refine_perimeter(cons, masks16, (S, S), 16, structure_ref=sil24,
                             protected=prot16)
    return {16: out16, 24: out24}


def build_icon(source_png: Path, app_name: str) -> BuiltIcon:
    """The full stage 1→5 pipeline for ONE app icon. `source_png` is the staged
    source (appicons.materialize output); its stem is the cache namespace.
    Raises BuildError with a report-friendly reason on any per-app dead end —
    the build command records it in the manifest and moves on."""
    try:
        cat = cached(source_png, "categorize", categorize, source_png)
        rgba = load_rgba(source_png)
        pre = cached(source_png, "preprocess", preprocess, cat, rgba)
        aligned = cached(source_png, "align", align, pre)
        sc = cached(source_png, "segment_prim_classify",
                    segment_prim_classify, aligned)
        fs = cached(source_png, "L2_feature_segment", segment_features, sc, cat)
    except Exception as e:                                   # noqa: BLE001 — per-app
        #                                                       capture, batch survives
        raise BuildError(f"stage1-2 ({type(e).__name__}: {e})") from e
    try:                                                     # report-only diagnostic —
        gate = evaluate_precompress(cat, pre, rgba)          # must NEVER break the build
    except Exception:                                        # noqa: BLE001
        gate = None
    S, cons, src = build_constructions(fs, cat.routing)
    if not cons:
        raise BuildError("no constructions (all features deferred)")
    outlines = build_outlines(cons, S, src)
    mono = select_monogram_letters(app_name)
    snakes: dict[int, np.ndarray] = {}
    layouts: dict[int, str] = {}
    for t_logo, n_canvas in CONV:
        res = compose_icon(outlines[t_logo], mono, n_canvas)
        snakes[n_canvas] = connect_snake(res.union)
        layouts[n_canvas] = res.layout
    return BuiltIcon(outlines=outlines, snakes=snakes, monogram=mono,
                     layouts=layouts, gate=gate)
