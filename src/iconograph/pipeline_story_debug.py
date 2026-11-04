"""Write per-fixture storyboards of the application-icon pipeline.

Panels cover source, categorization, segmentation, decomposition, rendering,
refinement, monogram composition and connection. Use the production stage
helpers so the storyboard follows the build path.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2  # noqa: E402

# Pin cv2 to one thread BEFORE the iconograph imports (refine.common warms the pool at
# import) so warpPolar-class ops stay deterministic (H8) — a later pin can't shrink a warm pool.
cv2.setNumThreads(1)

import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from iconograph.application.compress import (  # noqa: E402
    Feature, draw_feature_overlay, draw_overlay)
from iconograph.application.compress.common import (  # noqa: E402
    _compound_mask, _raster_part)
from iconograph.application.compress.contour_parse import (  # noqa: E402
    draw_shape_overlay, parse_shape)
from iconograph.application.monogram.common import (  # noqa: E402
    select_monogram_letters, split_monogram)
from iconograph.application.monogram.compose import compose_icon  # noqa: E402
from iconograph.application.monogram.connect import connect_snake, join_report  # noqa: E402
from iconograph.application.monogram.glyphs import render_letter  # noqa: E402
from iconograph.application.refine.common import refine_perimeter  # noqa: E402
from iconograph.cache import CACHE_ROOT, FIXTURES_ROOT, TESTBED_ROOT  # noqa: E402
from iconograph.render_debug import _fixture_constructions, assemble_render  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = FIXTURES_ROOT   # env-overridable basis (ICONOGRAPH_FIXTURES)
DUMP_DIR = TESTBED_ROOT / "story"

DEFAULT_FIXTURES = ("discord", "figma", "gmail")   # 3 strong, non-radial cases:
#   discord = side-by-side (multi-piece logo → real snake joins), figma = frame-weave
#   (portrait logo), gmail = side-by-side w/ the envelope V-carve decompose. Frame + side
#   layouts; every stage fires on all three (vscode dropped — weak example).
CONV = (24, 32)                # (T_logo, N_canvas): 24-logo native → 32 canvas (+8 band)
BOX = 256                      # each panel's square content box (NN upscale of the tiny icons)
TITLE_H, BLURB_H = 30, 104     # caption / methodology-callout bar heights
_INK = (30, 30, 30)
_GREY = (110, 110, 110)
_BG = (255, 255, 255)
_FX_BG = (238, 238, 238)       # render composite background (white-on-white stays visible)


# content builders (each returns a BOX×BOX×3 RGB visual)


def _square(img: np.ndarray, nearest: bool) -> np.ndarray:
    """Pad `img` to square (white), then resize to BOX×BOX. INTER_NEAREST keeps cubified
    pixel art crisp; INTER_AREA is for the continuous-tone source/aligned photos."""
    h, w = img.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 255, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(sq, (BOX, BOX), interpolation=interp)


def _mask_rgb(mask: np.ndarray, ink=_INK, bg=(255, 255, 255)) -> np.ndarray:
    """(H,W) {0,255}/bool → (H,W,3) ink-on-bg, matching the icon visual."""
    rgb = np.full((*mask.shape, 3), bg, np.uint8)
    rgb[mask > 0] = ink
    return rgb


def _source_rgb(stem: str) -> np.ndarray:
    """The raw fixture PNG composited over white (alpha honored), BGR→RGB."""
    raw = cv2.imread(str(FIXTURES / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    if raw.shape[2] == 4:
        bgr = raw[..., :3].astype(np.float32)
        a = (raw[..., 3:4].astype(np.float32)) / 255.0
        comp = bgr * a + 255.0 * (1.0 - a)
        return cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def _glyphs_rgb(ch1: str, ch2: str) -> np.ndarray:
    """The two chosen monogram letters as 1-px rectilinear FRAME glyphs, side by side."""
    gw, gh = 96, 168
    canvas = np.full((BOX, BOX, 3), 255, np.uint8)
    y = (BOX - gh) // 2
    for k, ch in enumerate((ch1, ch2)):
        g = render_letter(ch, gw, gh)
        x = 16 + k * (gw + 32)
        canvas[y:y + gh, x:x + gw][g] = _INK
    return canvas


_ADD_COL = (40, 90, 205)       # add-primitive outline — blue (union)
_SUB_COL = (205, 45, 45)       # subtract-primitive outline — red (carve / shave)


def _csg_rgb(cons, S: int) -> np.ndarray:
    """The D2/D3 CSG construction at full S res: the resulting shape (grey fill) with every
    primitive's outline drawn — ADD in blue (union), SUBTRACT in red (the shaved-off negates)
    — so the union/negate boolean algebra is legible, not just the net silhouette."""
    img = np.full((S, S, 3), 255, np.uint8)
    for parts, _color, _label in cons:                       # the resulting shape (net CSG)
        img[_compound_mask(parts, S, S) > 0] = (216, 216, 216)
    for parts, _color, _label in cons:                       # every primitive's own outline
        for p in parts:
            pm = (_raster_part(p.primitive, S, S) > 0).astype(np.uint8)
            cnts, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1,
                             _SUB_COL if p.op == "subtract" else _ADD_COL, 1, cv2.LINE_AA)
    return img


def _d1_shapes(fs):
    """[(ParsedShape, (bbox_x, bbox_y)), ...] over the L2 Features — the D1 draw input."""
    return [(parse_shape(seg.mask), (seg.bbox[0], seg.bbox[1]))
            for seg in fs.segments if isinstance(seg, Feature)]


# slow-bundle cache: only _fixture_constructions (decompose ~75s/fixture) is slow;
#    everything downstream is fast, so presentation tweaks re-run instantly. --fresh bypasses.
STORY_CACHE = REPO_ROOT / ".cache" / "story"


def _bundle(stem: str, fresh: bool):
    """(S, aligned, cons, src) for `stem`, cached — the expensive decompose is computed
    once. Arity guard: a stale 3-tuple cache (pre-source-silhouette) self-heals by
    recomputing and rewriting."""
    STORY_CACHE.mkdir(parents=True, exist_ok=True)
    p = STORY_CACHE / f"{stem}.pkl"
    if p.exists() and not fresh:
        bundle = pickle.loads(p.read_bytes())
        if len(bundle) == 4:
            return bundle
    bundle = _fixture_constructions(stem)
    p.write_bytes(pickle.dumps(bundle))
    return bundle


# annotated panel + strip assembly


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word-wrap to <= `width` chars per line (ASCII; Hershey renders non-ASCII ??)."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def _panel(content: np.ndarray, title: str, blurb: str) -> np.ndarray:
    """A BOX-wide annotated panel: [title bar | content | wrapped methodology callout]."""
    panel = np.full((TITLE_H + BOX + BLURB_H, BOX, 3), _BG, np.uint8)
    cv2.putText(panel, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _INK, 1, cv2.LINE_AA)
    panel[TITLE_H:TITLE_H + BOX] = content
    y = TITLE_H + BOX + 18
    for line in _wrap(blurb, 33)[:6]:   # ~33 chars = the most that fit a BOX-wide panel at 0.4
        cv2.putText(panel, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GREY, 1, cv2.LINE_AA)
        y += 15
    return panel


def _hstack(panels: list[np.ndarray], sep: int = 4) -> np.ndarray:
    bar = np.full((panels[0].shape[0], sep, 3), 0xCC, np.uint8)
    out: list[np.ndarray] = []
    for i, p in enumerate(panels):
        out.append(p)
        if i < len(panels) - 1:
            out.append(bar)
    return np.hstack(out)


def _banner(text: str, width: int) -> np.ndarray:
    b = np.full((34, width, 3), _BG, np.uint8)
    cv2.putText(b, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, _INK, 1, cv2.LINE_AA)
    return b


def _process(stem: str, name: str, fresh: bool) -> None:
    T, N = CONV
    cdir = CACHE_ROOT / stem
    sc = pickle.loads((cdir / "segment_prim_classify.pkl").read_bytes())      # L1
    fs = pickle.loads((cdir / "L2_feature_segment.pkl").read_bytes())         # L2
    routing = pickle.loads((cdir / "categorize.pkl").read_bytes()).routing    # stage 1

    # stage 3 sub-pipeline: aligned (L0) + CSG constructions (D2/D3 decompose runs here — slow,
    # cached). D1 contour parse is re-run per Feature (cheap) for its own panel.
    S, aligned, cons, src = _bundle(stem, fresh)
    d1_shapes = _d1_shapes(fs)
    n_parts = sum(len(parts) for parts, _c, _l in cons)

    # stage 3 render (R1) + stage 4 refine on the 24→32 path
    masks, icon, prot = assemble_render(cons, S, T, src)
    outline = refine_perimeter(cons, masks, (S, S), T, protected=prot)

    # stage 5: select letters → compose → connect
    mono = select_monogram_letters(name)
    ch1, ch2 = split_monogram(mono)
    res = compose_icon(outline, mono, N)
    snake = connect_snake(res.union)
    nj, nc = join_report(res.union)

    panels = [
        _panel(_square(_source_rgb(stem), nearest=False), "0  source",
               "Raw logo PNG (inputs/). Alpha composited over white; the pipeline input."),
        _panel(_square(aligned, nearest=False), "1  categorize + align",
               f"Sub-case detect, alpha/bg fixup, canvas trim, then axis-align (L0). "
               f"routing = {routing}."),
        # stage 3 geometric decomposition, sub-layer by sub-layer
        _panel(_square(draw_overlay(aligned, sc), nearest=False), "3a  L1 classify",
               "Label 8-connected components; fit ONE primitive each (circle / rect / "
               "triangle / ellipse). Unfittable region -> L1Miss, carried on."),
        _panel(_square(draw_feature_overlay(aligned, fs), nearest=False), "3b  L2 features",
               "Split each miss into typed features (body + appendages); rings are "
               "polar-unrolled into annular sub-features."),
        _panel(_square(draw_shape_overlay(aligned, d1_shapes), nearest=False), "3c  D1 contour-parse",
               "Trace the boundary into typed runs: straight (blue), arc (red=convex, "
               "purple=concave); mark corners + interior holes."),
        _panel(_square(_csg_rgb(cons, S), nearest=True), "3d  D2/D3 CSG",
               "Anchor primitives so their arcs COINCIDE with the runs: convex arc -> ADD "
               "(blue), concave -> SUBTRACT (red). Grouped union / negate."),
        _panel(_square(icon, nearest=True), f"3e  render R1  @{T}",
               "Idealize onto the hicolor grid: pixel-cubify at T<=24 (round-identity "
               "cubes), composite by segment color."),
        # stage 4 + stage 5
        _panel(_square(_mask_rgb(outline), nearest=True), f"4  refine  @{T}",
               "1-px perimeter outline: connect disjoint, normalize stroke width, "
               "mirror-symmetrize the silhouette."),
        _panel(_glyphs_rgb(ch1, ch2), f"5  glyphs  {mono}",
               f"Name -> 2 defining letters ({mono}); each a scalable 1-px rectilinear "
               f"FRAME glyph that hosts the logo."),
        _panel(_square(res.canvas, nearest=True), f"5  compose  @{N}",
               f"Weave native logo + letters, kept 1-px separated. Layout = {res.layout} "
               f"(of frame / side / fallback)."),
        _panel(_square(_mask_rgb(snake), nearest=True), f"5  connect  @{N}",
               f"MST single-point bridges + open the T-joins -> one continuous stroke. "
               f"T-free joins {nc}/{nj}."),
    ]
    strip = _hstack(panels)
    banner = _banner(f"{name}  ({stem})   -   full pipeline transformation   -   "
                     f"monogram {mono}   -   layout {res.layout}   -   {T}-logo -> {N} canvas",
                     strip.shape[1])
    out = np.vstack([banner, np.full((4, strip.shape[1], 3), 0xCC, np.uint8), strip])
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    path = DUMP_DIR / f"{stem}.png"
    Image.fromarray(out, "RGB").save(path, "PNG", optimize=True)
    print(f"  {stem:<10} routing={routing:<9} feats={len(d1_shapes):<2} "
          f"parts={n_parts:<2} mono={mono} layout={res.layout:<8} joins={nc}/{nj} "
          f"-> {path.relative_to(REPO_ROOT)}")


def main(argv: list[str]) -> int:
    import json
    cv2.setNumThreads(1)
    fresh = "--fresh" in argv                             # bypass the decompose bundle cache
    argv = [a for a in argv if a != "--fresh"]
    names = json.loads((FIXTURES / "names.json").read_text())
    wanted = argv if argv else list(DEFAULT_FIXTURES)
    for stem in wanted:
        try:
            _process(stem, names.get(stem, stem), fresh)
        except Exception as e:  # noqa: BLE001 — debug harness: report and continue
            print(f"{stem}: ERROR {type(e).__name__}: {e}")
    print(f"\nstoryboards → {DUMP_DIR.relative_to(REPO_ROOT)}/<stem>.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
