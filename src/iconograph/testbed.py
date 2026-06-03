"""Testbed driver — per-stage in-development runners, plus the `bench`
harness. Each runner maps a per-fixture worker across a fork()-mode
multiprocessing.Pool and writes the stage's H7 artifact. Upstream stages
flow through iconograph.cache so reruns and downstream testbeds skip
recompute.

Performance: the master process pre-decodes every fixture PNG into
_FIXTURE_RGBA before forking the pool, and pre-warms the cache RAM tier
from existing on-disk pkls. Both dicts are COW-inherited by every worker,
so workers do pure-RAM dict lookups for upstream stages instead of disk
reads. Per H6 the disk pickle store remains the source of truth.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iconograph.application.categorize import Categorization, categorize, load_rgba
from iconograph.application.compress import (
    Circle,
    Classified,
    Ellipse,
    Feature,
    FeatureSegmentation,
    L1Miss,
    Rectangle,
    SegmentClassification,
    Triangle,
    align,
    draw_feature_overlay,
    draw_overlay,
    segment_features,
    segment_prim_classify,
)
from iconograph.application.preprocess import preprocess
from iconograph.cache import cached, prewarm

ROUTING_LABELS = ("standard", "inverted", "burst")

ROUTING_COLORS = {
    "standard": (0x88, 0x88, 0x88),
    "inverted": (0x1A, 0x1A, 0x1A),
    "burst": (0xFF, 0x7F, 0x00),
}
BOOL_COLORS = {True: (0x2E, 0x7D, 0x32), False: (0xC6, 0x28, 0x28)}
PRECEDENCE_COLORS = {
    "outer": (0x19, 0x76, 0xD2),
    "inner": (0xFB, 0xC0, 0x2D),
    None: (0x9E, 0x9E, 0x9E),
}
MATCH = (0x2E, 0x7D, 0x32)
MISMATCH = (0xC6, 0x28, 0x28)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"
README = FIXTURES / "README.md"
TESTBED_DIR = REPO_ROOT / "testbed" / "categorize"
GRID_PATH = TESTBED_DIR / "grid.png"
PREPROCESS_DIR = REPO_ROOT / "testbed" / "preprocess"
PREPROCESS_GRID_PATH = PREPROCESS_DIR / "grid.png"
ALIGN_DIR = REPO_ROOT / "testbed" / "align"
ALIGN_GRID_PATH = ALIGN_DIR / "grid.png"
SPC_DIR = REPO_ROOT / "testbed" / "segment_prim_classify"
SPC_GRID_PATH = SPC_DIR / "grid.png"
L2FS_DIR = REPO_ROOT / "testbed" / "L2_feature_segment"
L2FS_GRID_PATH = L2FS_DIR / "grid.png"


# ─── Parallel-fixture runner infrastructure ───────────────────────────
#
# Each testbed runner maps a per-fixture worker across a multiprocessing.Pool
# (fork context — workers inherit the parent's already-imported modules and
# the OMP/BLAS=1 env vars set in __main__.py, so we avoid thread oversubscrip-
# tion without re-importing in each worker). Per-stage outputs go through the
# H6 cache in iconograph.cache so reruns and downstream-stage testbeds skip
# recomputing upstream stages.


def _worker_init() -> None:
    cv2.setNumThreads(1)


def _pool(n_items: int):
    nproc = max(1, min(os.cpu_count() or 1, n_items))
    return mp.get_context("fork").Pool(processes=nproc, initializer=_worker_init)


# Master-only RGBA cache. Populated by _prewarm_fixtures before _pool()
# forks; workers see it via copy-on-write. KeyError on lookup means the
# caller forgot to call _setup_run() before forking — that's a bug.
_FIXTURE_RGBA: dict[str, np.ndarray] = {}

_PREWARM_STAGES: tuple[str, ...] = (
    "categorize", "preprocess", "align", "segment_prim_classify", "L2_feature_segment",
)


def _prewarm_fixtures(fixtures: list[Path]) -> None:
    """Decode every fixture PNG into _FIXTURE_RGBA[stem]. Idempotent.
    Byte-identical to the per-stage load_rgba() it replaces."""
    for p in fixtures:
        if p.stem in _FIXTURE_RGBA:
            continue
        _FIXTURE_RGBA[p.stem] = load_rgba(p)


def _setup_run(fixtures: list[Path]) -> None:
    """Master-side prep before _pool() fork: pre-decode fixtures and
    pre-warm the cache RAM tier for every implemented stage."""
    _prewarm_fixtures(fixtures)
    prewarm(fixtures, _PREWARM_STAGES)


def _do_categorize(p: Path) -> Categorization:
    return categorize(p)


def _do_preprocess(rgba: np.ndarray, cat: Categorization) -> np.ndarray:
    return preprocess(cat, rgba)


def _do_align(pre_out: np.ndarray) -> np.ndarray:
    return align(pre_out)


def _do_segment_prim_classify(aligned: np.ndarray) -> SegmentClassification:
    return segment_prim_classify(aligned)


def _do_l2_feature_segment(
    sc: SegmentClassification, cat: Categorization
) -> FeatureSegmentation:
    return segment_features(sc, cat)


def _worker_categorize(p: Path) -> tuple[Path, Categorization]:
    pred = cached(p, "categorize", _do_categorize, p)
    return p, pred


def _worker_preprocess(p: Path) -> "PreprocessResult":
    try:
        cat = cached(p, "categorize", _do_categorize, p)
    except Exception as e:
        r = PreprocessResult(p, None, None, f"categorize: {type(e).__name__}: {e}")
        _write_preprocess_strip(r)
        return r
    try:
        out = cached(p, "preprocess", _do_preprocess, _FIXTURE_RGBA[p.stem], cat)
    except Exception as e:
        r = PreprocessResult(p, cat.routing, None, f"preprocess: {type(e).__name__}: {e}")
        _write_preprocess_strip(r)
        return r
    contract_err = _check_preprocess_contract(out)
    if contract_err is not None:
        r = PreprocessResult(p, cat.routing, None, contract_err)
        _write_preprocess_strip(r)
        return r
    r = PreprocessResult(p, cat.routing, out, None)
    _write_preprocess_strip(r)
    return r


def _worker_align(p: Path) -> "AlignResult":
    try:
        cat = cached(p, "categorize", _do_categorize, p)
    except Exception as e:
        r = AlignResult(p, None, None, None, f"categorize: {type(e).__name__}: {e}")
        _write_align_strip(r)
        return r
    try:
        pre_out = cached(p, "preprocess", _do_preprocess, _FIXTURE_RGBA[p.stem], cat)
    except Exception as e:
        r = AlignResult(p, cat.routing, None, None, f"preprocess: {type(e).__name__}: {e}")
        _write_align_strip(r)
        return r
    try:
        aligned = cached(p, "align", _do_align, pre_out)
    except Exception as e:
        r = AlignResult(p, cat.routing, pre_out, None, f"align: {type(e).__name__}: {e}")
        _write_align_strip(r)
        return r
    contract_err = _check_align_contract(aligned)
    if contract_err is not None:
        r = AlignResult(p, cat.routing, pre_out, None, contract_err)
        _write_align_strip(r)
        return r
    r = AlignResult(p, cat.routing, pre_out, aligned, None)
    _write_align_strip(r)
    return r


def _worker_segment_prim_classify(p: Path) -> "SegmentPrimClassifyResult":
    try:
        cat = cached(p, "categorize", _do_categorize, p)
    except Exception as e:
        r = SegmentPrimClassifyResult(p, None, None, None, f"categorize: {type(e).__name__}: {e}")
        _write_segment_prim_classify_strip(r)
        return r
    try:
        pre_out = cached(p, "preprocess", _do_preprocess, _FIXTURE_RGBA[p.stem], cat)
    except Exception as e:
        r = SegmentPrimClassifyResult(p, cat.routing, None, None, f"preprocess: {type(e).__name__}: {e}")
        _write_segment_prim_classify_strip(r)
        return r
    try:
        aligned = cached(p, "align", _do_align, pre_out)
    except Exception as e:
        r = SegmentPrimClassifyResult(p, cat.routing, None, None, f"align: {type(e).__name__}: {e}")
        _write_segment_prim_classify_strip(r)
        return r
    try:
        out = cached(p, "segment_prim_classify", _do_segment_prim_classify, aligned)
    except Exception as e:
        r = SegmentPrimClassifyResult(p, cat.routing, aligned, None, f"segment_prim_classify: {type(e).__name__}: {e}")
        _write_segment_prim_classify_strip(r)
        return r
    contract_err = _check_segment_prim_classify_contract(out)
    if contract_err is not None:
        r = SegmentPrimClassifyResult(p, cat.routing, aligned, None, contract_err)
        _write_segment_prim_classify_strip(r)
        return r
    r = SegmentPrimClassifyResult(p, cat.routing, aligned, out, None)
    _write_segment_prim_classify_strip(r)
    return r


def _worker_l2_feature_segment(p: Path) -> "L2FeatureSegmentResult":
    try:
        cat = cached(p, "categorize", _do_categorize, p)
    except Exception as e:
        r = L2FeatureSegmentResult(p, None, None, None, None, f"categorize: {type(e).__name__}: {e}")
        _write_l2_feature_segment_strip(r)
        return r
    try:
        pre_out = cached(p, "preprocess", _do_preprocess, _FIXTURE_RGBA[p.stem], cat)
        aligned = cached(p, "align", _do_align, pre_out)
        sc = cached(p, "segment_prim_classify", _do_segment_prim_classify, aligned)
    except Exception as e:
        r = L2FeatureSegmentResult(p, cat.routing, None, None, None, f"upstream: {type(e).__name__}: {e}")
        _write_l2_feature_segment_strip(r)
        return r
    try:
        out = cached(p, "L2_feature_segment", _do_l2_feature_segment, sc, cat)
    except Exception as e:
        r = L2FeatureSegmentResult(p, cat.routing, aligned, sc, None, f"L2_feature_segment: {type(e).__name__}: {e}")
        _write_l2_feature_segment_strip(r)
        return r
    contract_err = _check_l2_feature_segment_contract(out)
    if contract_err is not None:
        r = L2FeatureSegmentResult(p, cat.routing, aligned, sc, None, contract_err)
        _write_l2_feature_segment_strip(r)
        return r
    r = L2FeatureSegmentResult(p, cat.routing, aligned, sc, out, None)
    _write_l2_feature_segment_strip(r)
    return r


@dataclass(frozen=True)
class GroundTruth:
    color_essential: bool
    is_compound: bool
    precedence: Literal["outer", "inner"] | None
    routing: Literal["standard", "inverted", "burst"]


_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    r"\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
)


def _parse_bool(v: str) -> bool:
    s = v.strip()
    if s == "T":
        return True
    if s == "F":
        return False
    raise ValueError(f"bool expected T/F, got {v!r}")


def _parse_prec(v: str) -> Literal["outer", "inner"] | None:
    s = v.strip().lower()
    if s == "n/a":
        return None
    if s in ("outer", "inner"):
        return s  # type: ignore[return-value]
    raise ValueError(f"precedence expected outer/inner/n/a, got {v!r}")


def _parse_routing(v: str) -> Literal["standard", "inverted", "burst"]:
    s = v.strip().split()[0].strip("*")
    if s not in ROUTING_LABELS:
        raise ValueError(f"routing expected one of {ROUTING_LABELS}, got {v!r}")
    return s  # type: ignore[return-value]


def _parse_readme_truth(readme: Path) -> dict[str, GroundTruth]:
    rows: dict[str, GroundTruth] = {}
    text = readme.read_text()
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        filename = m.group(1).strip()
        if not filename or filename in {"filename"} or filename.startswith("---"):
            continue
        try:
            rows[filename] = GroundTruth(
                color_essential=_parse_bool(m.group(3)),
                is_compound=_parse_bool(m.group(4)),
                precedence=_parse_prec(m.group(5)),
                routing=_parse_routing(m.group(6)),
            )
        except ValueError:
            continue
    return rows


_FONT_CANDIDATES = (
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise FileNotFoundError(
        f"no TTF font found in {_FONT_CANDIDATES!r}"
    )


def run_categorize_testbed() -> int:
    truth = _parse_readme_truth(README)
    if not truth:
        print("error: no rows parsed from fixtures/README.md")
        return 2
    fixtures = sorted(FIXTURES.glob("*.png"))
    if not fixtures:
        print(f"error: no PNG fixtures found under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2
    _setup_run(fixtures)
    with _pool(len(fixtures)) as pool:
        raw = list(pool.imap_unordered(_worker_categorize, fixtures))
    raw.sort(key=lambda x: x[0].name)
    results: list[tuple[Path, GroundTruth, Categorization]] = []
    skipped: list[str] = []
    for p, pred in raw:
        gt = truth.get(p.name)
        if gt is None:
            skipped.append(p.name)
            continue
        results.append((p, gt, pred))

    matrix = np.zeros((3, 3), dtype=int)
    for _, gt, pred in results:
        i = ROUTING_LABELS.index(gt.routing)
        j = ROUTING_LABELS.index(pred.routing)
        matrix[i, j] += 1

    fields = ("color_essential", "is_compound", "precedence", "routing")
    n = len(results)
    acc = {
        f: sum(getattr(gt, f) == getattr(pred, f) for _, gt, pred in results)
        for f in fields
    }

    print("Routing confusion matrix (rows=truth, cols=predicted):")
    print(f"  {'':12s}" + "".join(f"{lbl:>10s}" for lbl in ROUTING_LABELS))
    for i, lbl in enumerate(ROUTING_LABELS):
        print(f"  {lbl:12s}" + "".join(f"{matrix[i, j]:>10d}" for j in range(3)))
    print()
    for f in fields:
        print(f"  {f:>16s}: {acc[f]:2d}/{n}")
    print()
    mismatches = []
    for p, gt, pred in results:
        diffs = []
        for f in fields:
            gv = getattr(gt, f)
            pv = getattr(pred, f)
            if gv != pv:
                diffs.append(f"{f}={pv!r} (want {gv!r})")
        if diffs:
            mismatches.append((p.name, diffs))
    if mismatches:
        print(f"Mismatches ({len(mismatches)}):")
        for name, diffs in mismatches:
            print(f"  {name}: " + "; ".join(diffs))
    else:
        print(f"All {n} fixtures pass.")
    if skipped:
        print(f"Skipped (no truth row): {', '.join(skipped)}")

    _compose_grid(results, matrix, acc, n)
    return 0 if not mismatches else 1


def _compose_grid(
    results: list[tuple[Path, GroundTruth, Categorization]],
    matrix: np.ndarray,
    acc: dict[str, int],
    n: int,
) -> None:
    THUMB = 96
    CHIP_W = 110
    BAND_W = 100
    CHECK_W = 36
    PAD = 8
    ROW_H = THUMB

    # Column layout: thumb | name | ce | ic | prec | routing | check
    NAME_W = 180
    cols = [THUMB, NAME_W, CHIP_W, CHIP_W, CHIP_W, BAND_W, CHECK_W]
    col_x = [PAD]
    for w in cols[:-1]:
        col_x.append(col_x[-1] + w + PAD)
    total_w = col_x[-1] + cols[-1] + PAD

    header_h = 260
    body_h = (ROW_H + PAD) * len(results) + PAD
    total_h = header_h + body_h

    canvas = Image.new("RGB", (total_w, total_h), (0xF4, 0xF4, 0xF4))
    draw = ImageDraw.Draw(canvas)
    f_title = _load_font(18)
    f_body = _load_font(13)
    f_small = _load_font(11)
    f_big = _load_font(22)

    draw.text((PAD, PAD), "Stage 1 categorize — testbed grid", font=f_title, fill=(0, 0, 0))

    # Confusion matrix
    cm_x = PAD
    cm_y = PAD + 32
    draw.text(
        (cm_x, cm_y),
        "Routing confusion matrix (rows=truth, cols=predicted):",
        font=f_body,
        fill=(0, 0, 0),
    )
    cell_w = 80
    cell_h = 24
    table_y = cm_y + 22
    for j, lbl in enumerate(ROUTING_LABELS):
        draw.text(
            (cm_x + 110 + cell_w * j, table_y),
            lbl,
            font=f_body,
            fill=(0, 0, 0),
        )
    for i, lbl in enumerate(ROUTING_LABELS):
        ry = table_y + cell_h * (i + 1)
        draw.text((cm_x, ry), lbl, font=f_body, fill=(0, 0, 0))
        for j in range(3):
            v = int(matrix[i, j])
            fill = (0, 0, 0)
            color = (0x2E, 0x7D, 0x32) if (i == j and v > 0) else (
                (0xC6, 0x28, 0x28) if (i != j and v > 0) else (0x55, 0x55, 0x55)
            )
            draw.text(
                (cm_x + 130 + cell_w * j, ry),
                str(v),
                font=f_body,
                fill=color,
            )

    # Per-field accuracies
    ax = cm_x + 110 + cell_w * 3 + 40
    ay = table_y
    draw.text((ax, ay), "Per-field accuracy (of N=%d):" % n, font=f_body, fill=(0, 0, 0))
    for k, f in enumerate(("color_essential", "is_compound", "precedence", "routing")):
        v = acc[f]
        color = MATCH if v == n else MISMATCH
        draw.text((ax, ay + 22 + 22 * k), f"{f:>16s}: {v:2d}/{n}", font=f_body, fill=color)

    # Body
    for r, (p, gt, pred) in enumerate(results):
        y = header_h + r * (ROW_H + PAD)
        # thumb cell — white bg so transparent PNGs show
        draw.rectangle(
            (col_x[0], y, col_x[0] + THUMB, y + ROW_H), fill=(255, 255, 255), outline=(0xAA, 0xAA, 0xAA)
        )
        thumb = Image.open(p).convert("RGBA")
        thumb.thumbnail((THUMB - 6, ROW_H - 6))
        tx = col_x[0] + (THUMB - thumb.width) // 2
        ty = y + (ROW_H - thumb.height) // 2
        canvas.paste(thumb, (tx, ty), thumb)

        # filename cell
        draw.rectangle(
            (col_x[1], y, col_x[1] + NAME_W, y + ROW_H), fill=(0xFA, 0xFA, 0xFA)
        )
        draw.text((col_x[1] + 6, y + ROW_H // 2 - 8), p.name, font=f_small, fill=(0, 0, 0))

        # color_essential chip
        _draw_pair_chip(
            draw, col_x[2], y, CHIP_W, ROW_H,
            label="color_essential",
            pred_text=("T" if pred.color_essential else "F"),
            truth_text=("T" if gt.color_essential else "F"),
            pred_fill=BOOL_COLORS[pred.color_essential],
            truth_fill=BOOL_COLORS[gt.color_essential],
            ok=(pred.color_essential == gt.color_essential),
            font=f_small,
        )
        # is_compound chip
        _draw_pair_chip(
            draw, col_x[3], y, CHIP_W, ROW_H,
            label="is_compound",
            pred_text=("T" if pred.is_compound else "F"),
            truth_text=("T" if gt.is_compound else "F"),
            pred_fill=BOOL_COLORS[pred.is_compound],
            truth_fill=BOOL_COLORS[gt.is_compound],
            ok=(pred.is_compound == gt.is_compound),
            font=f_small,
        )
        # precedence chip
        _draw_pair_chip(
            draw, col_x[4], y, CHIP_W, ROW_H,
            label="precedence",
            pred_text=("n/a" if pred.precedence is None else pred.precedence),
            truth_text=("n/a" if gt.precedence is None else gt.precedence),
            pred_fill=PRECEDENCE_COLORS[pred.precedence],
            truth_fill=PRECEDENCE_COLORS[gt.precedence],
            ok=(pred.precedence == gt.precedence),
            font=f_small,
        )
        # routing band (pred over truth)
        _draw_pair_chip(
            draw, col_x[5], y, BAND_W, ROW_H,
            label="routing",
            pred_text=pred.routing,
            truth_text=gt.routing,
            pred_fill=ROUTING_COLORS[pred.routing],
            truth_fill=ROUTING_COLORS[gt.routing],
            ok=(pred.routing == gt.routing),
            font=f_small,
        )
        # check/cross
        all_ok = (
            pred.routing == gt.routing
            and pred.color_essential == gt.color_essential
            and pred.is_compound == gt.is_compound
            and pred.precedence == gt.precedence
        )
        draw.rectangle(
            (col_x[6], y, col_x[6] + CHECK_W, y + ROW_H),
            fill=(MATCH if all_ok else MISMATCH),
        )
        mark = "✓" if all_ok else "✗"
        bbox = draw.textbbox((0, 0), mark, font=f_big)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (col_x[6] + (CHECK_W - tw) // 2, y + (ROW_H - th) // 2 - 2),
            mark,
            font=f_big,
            fill=(255, 255, 255),
        )

    TESTBED_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(GRID_PATH, "PNG", optimize=True)
    print(f"Saved {GRID_PATH.relative_to(REPO_ROOT)}")


def _draw_pair_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    label: str,
    pred_text: str,
    truth_text: str,
    pred_fill: tuple[int, int, int],
    truth_fill: tuple[int, int, int],
    ok: bool,
    font: ImageFont.FreeTypeFont,
) -> None:
    border = MATCH if ok else MISMATCH
    draw.rectangle((x, y, x + w, y + h), fill=(0xF8, 0xF8, 0xF8), outline=border, width=2)
    half_h = h // 2
    # Predicted top half
    draw.rectangle((x + 2, y + 2, x + w - 2, y + half_h - 1), fill=pred_fill)
    _draw_centered_text(draw, x + 2, y + 2, w - 4, half_h - 3, f"P {pred_text}", font, (255, 255, 255))
    # Truth bottom half
    draw.rectangle((x + 2, y + half_h + 1, x + w - 2, y + h - 2), fill=truth_fill)
    _draw_centered_text(
        draw, x + 2, y + half_h + 1, w - 4, h - half_h - 3, f"T {truth_text}", font, (255, 255, 255)
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2 - 1), text, font=font, fill=fill)


# ─── Stage 2 preprocess testbed ───────────────────────────────────────


@dataclass(frozen=True)
class PreprocessResult:
    path: Path
    routing: Literal["standard", "inverted", "burst"] | None
    output: np.ndarray | None
    error: str | None


def run_preprocess_testbed() -> int:
    fixtures = sorted(FIXTURES.glob("*.png"))
    if not fixtures:
        print(f"error: no PNG fixtures found under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2

    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    _setup_run(fixtures)
    with _pool(len(fixtures)) as pool:
        results = list(pool.imap_unordered(_worker_preprocess, fixtures))
    results.sort(key=lambda r: r.path.name)

    n = len(results)
    hist = {lbl: sum(1 for r in results if r.routing == lbl) for lbl in ROUTING_LABELS}
    passed = sum(1 for r in results if r.error is None)
    failures = [r for r in results if r.error is not None]

    print(
        "Routing histogram: "
        + " ".join(f"{lbl}={hist[lbl]}" for lbl in ROUTING_LABELS)
    )
    print(f"{passed}/{n} fixtures passed (no exception, contract OK)")
    if failures:
        print(f"Failures ({len(failures)}):")
        for r in failures:
            print(f"  {r.path.name}: {r.error}")

    _compose_preprocess_grid(results, hist, passed, n)

    return 0 if not failures else 1


def _check_preprocess_contract(out: object) -> str | None:
    """Verify preprocess() honors its (S, S, 3) uint8 RGB contract.

    Returns a human-readable violation string, or None if the output is
    well-formed.
    """
    if not isinstance(out, np.ndarray):
        return f"output type {type(out).__name__}, expected np.ndarray"
    if out.dtype != np.uint8:
        return f"dtype {out.dtype}, expected uint8"
    if out.ndim != 3 or out.shape[2] != 3:
        return f"shape {out.shape}, expected (S, S, 3)"
    if out.shape[0] != out.shape[1]:
        return f"shape {out.shape}, not square"
    return None


def _write_preprocess_strip(r: PreprocessResult) -> None:
    """Per-fixture H7 strip: source | preprocess-output + caption bar."""
    PANEL = 512
    CAPTION_H = 36
    total_w = PANEL * 2
    total_h = PANEL + CAPTION_H

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    src = Image.open(r.path).convert("RGBA")
    src_panel = _fit_to_panel(src, PANEL)
    canvas.paste(src_panel, (0, 0))

    if r.output is not None:
        out_img = Image.fromarray(r.output, "RGB")
        out_panel = _fit_to_panel(out_img, PANEL)
        canvas.paste(out_panel, (PANEL, 0))
    else:
        f_fail = _load_font(48)
        _draw_centered_text(draw, PANEL, 0, PANEL, PANEL, "FAIL", f_fail, MISMATCH)

    draw.line([(PANEL, 0), (PANEL, PANEL)], fill=(0xCC, 0xCC, 0xCC), width=1)

    draw.rectangle((0, PANEL, total_w, total_h), fill=(0xF4, 0xF4, 0xF4))
    f = _load_font(13)
    if r.routing is not None:
        chip_w = 80
        chip_h = 20
        chip_y = PANEL + (CAPTION_H - chip_h) // 2
        draw.rectangle(
            (8, chip_y, 8 + chip_w, chip_y + chip_h), fill=ROUTING_COLORS[r.routing]
        )
        _draw_centered_text(draw, 8, chip_y, chip_w, chip_h, r.routing, f, (255, 255, 255))
        text_x = 8 + chip_w + 12
    else:
        text_x = 8
    caption = r.path.name + (f"  •  ERROR: {r.error}" if r.error else "")
    draw.text(
        (text_x, PANEL + (CAPTION_H - 14) // 2),
        caption,
        font=f,
        fill=(0, 0, 0) if r.error is None else MISMATCH,
    )

    canvas.save(PREPROCESS_DIR / r.path.name, "PNG", optimize=True)


def _fit_to_panel(img: Image.Image, size: int) -> Image.Image:
    """Center `img` on a `size`x`size` white panel, preserving aspect."""
    panel = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    src = img.copy()
    src.thumbnail((size, size))
    x = (size - src.width) // 2
    y = (size - src.height) // 2
    if src.mode in ("RGBA", "LA"):
        panel.paste(src, (x, y), src)
    else:
        panel.paste(src, (x, y))
    return panel.convert("RGB")


def _compose_preprocess_grid(
    results: list[PreprocessResult],
    hist: dict[str, int],
    passed: int,
    n: int,
) -> None:
    THUMB = 96
    OUT_W = 128
    BAND_W = 100
    CHECK_W = 36
    NAME_W = 180
    PAD = 8
    ROW_H = THUMB

    cols = [THUMB, NAME_W, OUT_W, BAND_W, CHECK_W]
    col_x = [PAD]
    for w in cols[:-1]:
        col_x.append(col_x[-1] + w + PAD)
    total_w = col_x[-1] + cols[-1] + PAD

    header_h = 100
    body_h = (ROW_H + PAD) * len(results) + PAD
    total_h = header_h + body_h

    canvas = Image.new("RGB", (total_w, total_h), (0xF4, 0xF4, 0xF4))
    draw = ImageDraw.Draw(canvas)
    f_title = _load_font(18)
    f_body = _load_font(13)
    f_small = _load_font(11)
    f_big = _load_font(22)
    f_fail = _load_font(14)

    draw.text((PAD, PAD), "Stage 2 preprocess — testbed grid", font=f_title, fill=(0, 0, 0))
    hist_text = " ".join(f"{lbl}={hist[lbl]}" for lbl in ROUTING_LABELS)
    draw.text((PAD, PAD + 28), f"Routing histogram: {hist_text}", font=f_body, fill=(0, 0, 0))
    pass_color = MATCH if passed == n else MISMATCH
    draw.text(
        (PAD, PAD + 50),
        f"{passed}/{n} fixtures passed (no exception, contract OK)",
        font=f_body,
        fill=pass_color,
    )

    for r_idx, r in enumerate(results):
        y = header_h + r_idx * (ROW_H + PAD)

        draw.rectangle(
            (col_x[0], y, col_x[0] + THUMB, y + ROW_H),
            fill=(255, 255, 255),
            outline=(0xAA, 0xAA, 0xAA),
        )
        thumb = Image.open(r.path).convert("RGBA")
        thumb.thumbnail((THUMB - 6, ROW_H - 6))
        tx = col_x[0] + (THUMB - thumb.width) // 2
        ty = y + (ROW_H - thumb.height) // 2
        canvas.paste(thumb, (tx, ty), thumb)

        draw.rectangle(
            (col_x[1], y, col_x[1] + NAME_W, y + ROW_H), fill=(0xFA, 0xFA, 0xFA)
        )
        draw.text(
            (col_x[1] + 6, y + ROW_H // 2 - 8), r.path.name, font=f_small, fill=(0, 0, 0)
        )

        draw.rectangle(
            (col_x[2], y, col_x[2] + OUT_W, y + ROW_H),
            fill=(255, 255, 255),
            outline=(0xAA, 0xAA, 0xAA),
        )
        if r.output is not None:
            out_img = Image.fromarray(r.output, "RGB")
            out_img.thumbnail((OUT_W - 6, ROW_H - 6))
            ox = col_x[2] + (OUT_W - out_img.width) // 2
            oy = y + (ROW_H - out_img.height) // 2
            canvas.paste(out_img, (ox, oy))
        else:
            _draw_centered_text(draw, col_x[2], y, OUT_W, ROW_H, "FAIL", f_fail, MISMATCH)

        if r.routing is not None:
            draw.rectangle(
                (col_x[3], y, col_x[3] + BAND_W, y + ROW_H),
                fill=ROUTING_COLORS[r.routing],
            )
            _draw_centered_text(
                draw, col_x[3], y, BAND_W, ROW_H, r.routing, f_body, (255, 255, 255)
            )
        else:
            draw.rectangle(
                (col_x[3], y, col_x[3] + BAND_W, y + ROW_H), fill=(0x9E, 0x9E, 0x9E)
            )
            _draw_centered_text(
                draw, col_x[3], y, BAND_W, ROW_H, "n/a", f_body, (255, 255, 255)
            )

        ok = r.error is None
        draw.rectangle(
            (col_x[4], y, col_x[4] + CHECK_W, y + ROW_H),
            fill=(MATCH if ok else MISMATCH),
        )
        mark = "✓" if ok else "✗"
        bbox = draw.textbbox((0, 0), mark, font=f_big)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (col_x[4] + (CHECK_W - tw) // 2, y + (ROW_H - th) // 2 - 2),
            mark,
            font=f_big,
            fill=(255, 255, 255),
        )

    canvas.save(PREPROCESS_GRID_PATH, "PNG", optimize=True)
    print(f"Saved {PREPROCESS_GRID_PATH.relative_to(REPO_ROOT)}")


# ─── Stage 3 align testbed ────────────────────────────────────────────


@dataclass(frozen=True)
class AlignResult:
    path: Path
    routing: Literal["standard", "inverted", "burst"] | None
    preprocess_output: np.ndarray | None
    aligned_output: np.ndarray | None
    error: str | None


def run_align_testbed() -> int:
    fixtures = sorted(FIXTURES.glob("*.png"))
    if not fixtures:
        print(f"error: no PNG fixtures found under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2

    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    _setup_run(fixtures)
    with _pool(len(fixtures)) as pool:
        results = list(pool.imap_unordered(_worker_align, fixtures))
    results.sort(key=lambda r: r.path.name)

    n = len(results)
    hist = {lbl: sum(1 for r in results if r.routing == lbl) for lbl in ROUTING_LABELS}
    passed = sum(1 for r in results if r.error is None)
    failures = [r for r in results if r.error is not None]

    print(
        "Routing histogram: "
        + " ".join(f"{lbl}={hist[lbl]}" for lbl in ROUTING_LABELS)
    )
    print(f"{passed}/{n} fixtures passed (no exception, contract OK)")
    if failures:
        print(f"Failures ({len(failures)}):")
        for r in failures:
            print(f"  {r.path.name}: {r.error}")

    _compose_align_grid(results, hist, passed, n)

    return 0 if not failures else 1


def _check_align_contract(out: object) -> str | None:
    """Verify align() honors its (S, S, 3) uint8 RGB contract."""
    if not isinstance(out, np.ndarray):
        return f"output type {type(out).__name__}, expected np.ndarray"
    if out.dtype != np.uint8:
        return f"dtype {out.dtype}, expected uint8"
    if out.ndim != 3 or out.shape[2] != 3:
        return f"shape {out.shape}, expected (S, S, 3)"
    if out.shape[0] != out.shape[1]:
        return f"shape {out.shape}, not square"
    return None


def _write_align_strip(r: AlignResult) -> None:
    """Per-fixture H7 strip: preprocess output | aligned output + caption."""
    PANEL = 512
    CAPTION_H = 36
    total_w = PANEL * 2
    total_h = PANEL + CAPTION_H

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if r.preprocess_output is not None:
        pre_img = Image.fromarray(r.preprocess_output, "RGB")
        pre_panel = _fit_to_panel(pre_img, PANEL)
        canvas.paste(pre_panel, (0, 0))
    else:
        f_dash = _load_font(48)
        _draw_centered_text(draw, 0, 0, PANEL, PANEL, "—", f_dash, (0xAA, 0xAA, 0xAA))

    if r.aligned_output is not None:
        out_img = Image.fromarray(r.aligned_output, "RGB")
        out_panel = _fit_to_panel(out_img, PANEL)
        canvas.paste(out_panel, (PANEL, 0))
    else:
        f_fail = _load_font(48)
        _draw_centered_text(draw, PANEL, 0, PANEL, PANEL, "FAIL", f_fail, MISMATCH)

    draw.line([(PANEL, 0), (PANEL, PANEL)], fill=(0xCC, 0xCC, 0xCC), width=1)

    draw.rectangle((0, PANEL, total_w, total_h), fill=(0xF4, 0xF4, 0xF4))
    f = _load_font(13)
    if r.routing is not None:
        chip_w = 80
        chip_h = 20
        chip_y = PANEL + (CAPTION_H - chip_h) // 2
        draw.rectangle(
            (8, chip_y, 8 + chip_w, chip_y + chip_h), fill=ROUTING_COLORS[r.routing]
        )
        _draw_centered_text(draw, 8, chip_y, chip_w, chip_h, r.routing, f, (255, 255, 255))
        text_x = 8 + chip_w + 12
    else:
        text_x = 8
    caption = r.path.name + (f"  •  ERROR: {r.error}" if r.error else "")
    draw.text(
        (text_x, PANEL + (CAPTION_H - 14) // 2),
        caption,
        font=f,
        fill=(0, 0, 0) if r.error is None else MISMATCH,
    )

    canvas.save(ALIGN_DIR / r.path.name, "PNG", optimize=True)


def _compose_align_grid(
    results: list[AlignResult],
    hist: dict[str, int],
    passed: int,
    n: int,
) -> None:
    THUMB = 96
    OUT_W = 128
    BAND_W = 100
    CHECK_W = 36
    NAME_W = 180
    PAD = 8
    ROW_H = THUMB

    cols = [THUMB, NAME_W, OUT_W, BAND_W, CHECK_W]
    col_x = [PAD]
    for w in cols[:-1]:
        col_x.append(col_x[-1] + w + PAD)
    total_w = col_x[-1] + cols[-1] + PAD

    header_h = 100
    body_h = (ROW_H + PAD) * len(results) + PAD
    total_h = header_h + body_h

    canvas = Image.new("RGB", (total_w, total_h), (0xF4, 0xF4, 0xF4))
    draw = ImageDraw.Draw(canvas)
    f_title = _load_font(18)
    f_body = _load_font(13)
    f_small = _load_font(11)
    f_big = _load_font(22)
    f_fail = _load_font(14)

    draw.text((PAD, PAD), "Stage 3 align — testbed grid", font=f_title, fill=(0, 0, 0))
    hist_text = " ".join(f"{lbl}={hist[lbl]}" for lbl in ROUTING_LABELS)
    draw.text((PAD, PAD + 28), f"Routing histogram: {hist_text}", font=f_body, fill=(0, 0, 0))
    pass_color = MATCH if passed == n else MISMATCH
    draw.text(
        (PAD, PAD + 50),
        f"{passed}/{n} fixtures passed (no exception, contract OK)",
        font=f_body,
        fill=pass_color,
    )

    for r_idx, r in enumerate(results):
        y = header_h + r_idx * (ROW_H + PAD)

        draw.rectangle(
            (col_x[0], y, col_x[0] + THUMB, y + ROW_H),
            fill=(255, 255, 255),
            outline=(0xAA, 0xAA, 0xAA),
        )
        if r.preprocess_output is not None:
            thumb = Image.fromarray(r.preprocess_output, "RGB")
            thumb.thumbnail((THUMB - 6, ROW_H - 6))
            tx = col_x[0] + (THUMB - thumb.width) // 2
            ty = y + (ROW_H - thumb.height) // 2
            canvas.paste(thumb, (tx, ty))

        draw.rectangle(
            (col_x[1], y, col_x[1] + NAME_W, y + ROW_H), fill=(0xFA, 0xFA, 0xFA)
        )
        draw.text(
            (col_x[1] + 6, y + ROW_H // 2 - 8), r.path.name, font=f_small, fill=(0, 0, 0)
        )

        draw.rectangle(
            (col_x[2], y, col_x[2] + OUT_W, y + ROW_H),
            fill=(255, 255, 255),
            outline=(0xAA, 0xAA, 0xAA),
        )
        if r.aligned_output is not None:
            out_img = Image.fromarray(r.aligned_output, "RGB")
            out_img.thumbnail((OUT_W - 6, ROW_H - 6))
            ox = col_x[2] + (OUT_W - out_img.width) // 2
            oy = y + (ROW_H - out_img.height) // 2
            canvas.paste(out_img, (ox, oy))
        else:
            _draw_centered_text(draw, col_x[2], y, OUT_W, ROW_H, "FAIL", f_fail, MISMATCH)

        if r.routing is not None:
            draw.rectangle(
                (col_x[3], y, col_x[3] + BAND_W, y + ROW_H),
                fill=ROUTING_COLORS[r.routing],
            )
            _draw_centered_text(
                draw, col_x[3], y, BAND_W, ROW_H, r.routing, f_body, (255, 255, 255)
            )
        else:
            draw.rectangle(
                (col_x[3], y, col_x[3] + BAND_W, y + ROW_H), fill=(0x9E, 0x9E, 0x9E)
            )
            _draw_centered_text(
                draw, col_x[3], y, BAND_W, ROW_H, "n/a", f_body, (255, 255, 255)
            )

        ok = r.error is None
        draw.rectangle(
            (col_x[4], y, col_x[4] + CHECK_W, y + ROW_H),
            fill=(MATCH if ok else MISMATCH),
        )
        mark = "✓" if ok else "✗"
        bbox = draw.textbbox((0, 0), mark, font=f_big)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (col_x[4] + (CHECK_W - tw) // 2, y + (ROW_H - th) // 2 - 2),
            mark,
            font=f_big,
            fill=(255, 255, 255),
        )

    canvas.save(ALIGN_GRID_PATH, "PNG", optimize=True)
    print(f"Saved {ALIGN_GRID_PATH.relative_to(REPO_ROOT)}")


# ─── Stage 3 L1 segment_prim_classify testbed ─────────────────────────


@dataclass(frozen=True)
class SegmentPrimClassifyResult:
    path: Path
    routing: Literal["standard", "inverted", "burst"] | None
    aligned_input: np.ndarray | None
    output: SegmentClassification | None
    error: str | None


def run_segment_prim_classify_testbed() -> int:
    fixtures = sorted(FIXTURES.glob("*.png"))
    if not fixtures:
        print(f"error: no PNG fixtures found under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2

    SPC_DIR.mkdir(parents=True, exist_ok=True)
    _setup_run(fixtures)
    with _pool(len(fixtures)) as pool:
        results = list(pool.imap_unordered(_worker_segment_prim_classify, fixtures))
    results.sort(key=lambda r: r.path.name)

    n = len(results)
    hist = {lbl: sum(1 for r in results if r.routing == lbl) for lbl in ROUTING_LABELS}
    passed = sum(1 for r in results if r.error is None)
    failures = [r for r in results if r.error is not None]

    n_classified = 0
    n_l1miss = 0
    for r in results:
        if r.output is None:
            continue
        for seg in r.output.segments:
            if isinstance(seg, Classified):
                n_classified += 1
            else:
                n_l1miss += 1

    print(
        "Routing histogram: "
        + " ".join(f"{lbl}={hist[lbl]}" for lbl in ROUTING_LABELS)
    )
    print(f"{passed}/{n} fixtures passed (no exception, contract OK)")
    print(f"Total segments: classified={n_classified}, L1miss={n_l1miss}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for r in failures:
            print(f"  {r.path.name}: {r.error}")

    _compose_segment_prim_classify_grid(results, hist, passed, n)

    return 0 if not failures else 1


def _check_segment_prim_classify_contract(out: object) -> str | None:
    if not isinstance(out, SegmentClassification):
        return f"output type {type(out).__name__}, expected SegmentClassification"
    if not isinstance(out.segments, list):
        return f"segments type {type(out.segments).__name__}, expected list"
    for i, seg in enumerate(out.segments):
        if not isinstance(seg, (Classified, L1Miss)):
            return f"segments[{i}] type {type(seg).__name__}, expected Classified|L1Miss"
    if not isinstance(out.binarized, np.ndarray) or out.binarized.dtype != np.uint8:
        return f"binarized dtype {getattr(out.binarized, 'dtype', None)}, expected uint8 ndarray"
    if not isinstance(out.labels, np.ndarray) or out.labels.dtype != np.int32:
        return f"labels dtype {getattr(out.labels, 'dtype', None)}, expected int32 ndarray"
    return None


def _write_segment_prim_classify_strip(r: SegmentPrimClassifyResult) -> None:
    PANEL = 512
    CAPTION_H = 36
    total_w = PANEL * 2
    total_h = PANEL + CAPTION_H

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if r.aligned_input is not None:
        aligned_img = Image.fromarray(r.aligned_input, "RGB")
        aligned_panel = _fit_to_panel(aligned_img, PANEL)
        canvas.paste(aligned_panel, (0, 0))
    else:
        f_dash = _load_font(48)
        _draw_centered_text(draw, 0, 0, PANEL, PANEL, "—", f_dash, (0xAA, 0xAA, 0xAA))

    if r.output is not None and r.aligned_input is not None:
        overlay_arr = draw_overlay(r.aligned_input, r.output)
        overlay_img = Image.fromarray(overlay_arr, "RGB")
        overlay_panel = _fit_to_panel(overlay_img, PANEL)
        canvas.paste(overlay_panel, (PANEL, 0))
    else:
        f_fail = _load_font(48)
        _draw_centered_text(draw, PANEL, 0, PANEL, PANEL, "FAIL", f_fail, MISMATCH)

    draw.line([(PANEL, 0), (PANEL, PANEL)], fill=(0xCC, 0xCC, 0xCC), width=1)

    draw.rectangle((0, PANEL, total_w, total_h), fill=(0xF4, 0xF4, 0xF4))
    f = _load_font(13)
    if r.routing is not None:
        chip_w = 80
        chip_h = 20
        chip_y = PANEL + (CAPTION_H - chip_h) // 2
        draw.rectangle(
            (8, chip_y, 8 + chip_w, chip_y + chip_h), fill=ROUTING_COLORS[r.routing]
        )
        _draw_centered_text(draw, 8, chip_y, chip_w, chip_h, r.routing, f, (255, 255, 255))
        text_x = 8 + chip_w + 12
    else:
        text_x = 8

    if r.output is not None:
        c_circ = sum(1 for s in r.output.segments
                     if isinstance(s, Classified) and isinstance(s.primitive, Circle))
        c_rect = sum(1 for s in r.output.segments
                     if isinstance(s, Classified) and isinstance(s.primitive, Rectangle))
        c_tri = sum(1 for s in r.output.segments
                    if isinstance(s, Classified) and isinstance(s.primitive, Triangle))
        c_ell = sum(1 for s in r.output.segments
                    if isinstance(s, Classified) and isinstance(s.primitive, Ellipse))
        n_l1 = sum(1 for s in r.output.segments if isinstance(s, L1Miss))
        summary = (
            f"  •  classified={c_circ + c_rect + c_tri + c_ell} "
            f"(circ={c_circ} rect={c_rect} tri={c_tri} ell={c_ell})  "
            f"•  L1miss={n_l1}"
        )
    elif r.error:
        summary = f"  •  ERROR: {r.error}"
    else:
        summary = ""

    caption = r.path.name + summary
    draw.text(
        (text_x, PANEL + (CAPTION_H - 14) // 2),
        caption,
        font=f,
        fill=(0, 0, 0) if r.error is None else MISMATCH,
    )

    canvas.save(SPC_DIR / r.path.name, "PNG", optimize=True)


def _compose_segment_prim_classify_grid(
    results: list[SegmentPrimClassifyResult],
    hist: dict[str, int],
    passed: int,
    n: int,
) -> None:
    THUMB = 96
    OUT_W = 128
    BAND_W = 100
    CHECK_W = 36
    NAME_W = 180
    PAD = 8
    ROW_H = THUMB

    cols = [THUMB, NAME_W, OUT_W, BAND_W, CHECK_W]
    col_x = [PAD]
    for w in cols[:-1]:
        col_x.append(col_x[-1] + w + PAD)
    total_w = col_x[-1] + cols[-1] + PAD

    header_h = 120
    body_h = (ROW_H + PAD) * len(results) + PAD
    total_h = header_h + body_h

    canvas = Image.new("RGB", (total_w, total_h), (0xF4, 0xF4, 0xF4))
    draw = ImageDraw.Draw(canvas)
    f_title = _load_font(18)
    f_body = _load_font(13)
    f_small = _load_font(11)
    f_big = _load_font(22)
    f_fail = _load_font(14)

    draw.text((PAD, PAD), "Stage 3 L1 segment_prim_classify — testbed grid", font=f_title, fill=(0, 0, 0))
    hist_text = " ".join(f"{lbl}={hist[lbl]}" for lbl in ROUTING_LABELS)
    draw.text((PAD, PAD + 28), f"Routing histogram: {hist_text}", font=f_body, fill=(0, 0, 0))
    pass_color = MATCH if passed == n else MISMATCH
    draw.text(
        (PAD, PAD + 50),
        f"{passed}/{n} fixtures passed (no exception, contract OK)",
        font=f_body,
        fill=pass_color,
    )
    total_classified = sum(
        sum(1 for s in r.output.segments if isinstance(s, Classified))
        for r in results if r.output is not None
    )
    total_l1miss = sum(
        sum(1 for s in r.output.segments if isinstance(s, L1Miss))
        for r in results if r.output is not None
    )
    draw.text(
        (PAD, PAD + 72),
        f"Total segments: classified={total_classified}  •  L1miss={total_l1miss}",
        font=f_body,
        fill=(0, 0, 0),
    )

    for r_idx, r in enumerate(results):
        y = header_h + r_idx * (ROW_H + PAD)

        draw.rectangle(
            (col_x[0], y, col_x[0] + THUMB, y + ROW_H),
            fill=(255, 255, 255),
            outline=(0xAA, 0xAA, 0xAA),
        )
        if r.aligned_input is not None:
            thumb = Image.fromarray(r.aligned_input, "RGB")
            thumb.thumbnail((THUMB - 6, ROW_H - 6))
            tx = col_x[0] + (THUMB - thumb.width) // 2
            ty = y + (ROW_H - thumb.height) // 2
            canvas.paste(thumb, (tx, ty))

        draw.rectangle(
            (col_x[1], y, col_x[1] + NAME_W, y + ROW_H), fill=(0xFA, 0xFA, 0xFA)
        )
        draw.text(
            (col_x[1] + 6, y + ROW_H // 2 - 8), r.path.name, font=f_small, fill=(0, 0, 0)
        )

        draw.rectangle(
            (col_x[2], y, col_x[2] + OUT_W, y + ROW_H),
            fill=(255, 255, 255),
            outline=(0xAA, 0xAA, 0xAA),
        )
        if r.output is not None and r.aligned_input is not None:
            overlay_arr = draw_overlay(r.aligned_input, r.output)
            overlay_img = Image.fromarray(overlay_arr, "RGB")
            overlay_img.thumbnail((OUT_W - 6, ROW_H - 6))
            ox = col_x[2] + (OUT_W - overlay_img.width) // 2
            oy = y + (ROW_H - overlay_img.height) // 2
            canvas.paste(overlay_img, (ox, oy))
        else:
            _draw_centered_text(draw, col_x[2], y, OUT_W, ROW_H, "FAIL", f_fail, MISMATCH)

        if r.routing is not None:
            draw.rectangle(
                (col_x[3], y, col_x[3] + BAND_W, y + ROW_H),
                fill=ROUTING_COLORS[r.routing],
            )
            _draw_centered_text(
                draw, col_x[3], y, BAND_W, ROW_H, r.routing, f_body, (255, 255, 255)
            )
        else:
            draw.rectangle(
                (col_x[3], y, col_x[3] + BAND_W, y + ROW_H), fill=(0x9E, 0x9E, 0x9E)
            )
            _draw_centered_text(
                draw, col_x[3], y, BAND_W, ROW_H, "n/a", f_body, (255, 255, 255)
            )

        ok = r.error is None
        draw.rectangle(
            (col_x[4], y, col_x[4] + CHECK_W, y + ROW_H),
            fill=(MATCH if ok else MISMATCH),
        )
        mark = "✓" if ok else "✗"
        bbox = draw.textbbox((0, 0), mark, font=f_big)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (col_x[4] + (CHECK_W - tw) // 2, y + (ROW_H - th) // 2 - 2),
            mark,
            font=f_big,
            fill=(255, 255, 255),
        )

    canvas.save(SPC_GRID_PATH, "PNG", optimize=True)
    print(f"Saved {SPC_GRID_PATH.relative_to(REPO_ROOT)}")


# ─── Stage 3 L2 feature_segment testbed ───────────────────────────────


@dataclass(frozen=True)
class L2FeatureSegmentResult:
    path: Path
    routing: Literal["standard", "inverted", "burst"] | None
    aligned_input: np.ndarray | None
    l1_output: SegmentClassification | None
    output: FeatureSegmentation | None
    error: str | None


def _l2_kind_counts(out: FeatureSegmentation) -> dict[str, int]:
    kc = {"local": 0, "global": 0, "defer": 0}
    for s in out.segments:
        if isinstance(s, Feature):
            kc[s.kind] += 1
    return kc


def _l2_summary(out: FeatureSegmentation) -> str:
    """`L3 G4 D0  •  half_circle, annular_sector×3` — kind counts + hint tally."""
    kc = _l2_kind_counts(out)
    hints: dict[str, int] = {}
    for s in out.segments:
        if isinstance(s, Feature):
            h = s.shape_hint or "—"
            hints[h] = hints.get(h, 0) + 1
    hint_str = ", ".join(f"{h}×{c}" if c > 1 else h for h, c in sorted(hints.items()))
    return f"L{kc['local']} G{kc['global']} D{kc['defer']}  •  {hint_str}"


def run_l2_feature_segment_testbed() -> int:
    fixtures = sorted(FIXTURES.glob("*.png"))
    if not fixtures:
        print(f"error: no PNG fixtures found under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2

    L2FS_DIR.mkdir(parents=True, exist_ok=True)
    _setup_run(fixtures)
    with _pool(len(fixtures)) as pool:
        results = list(pool.imap_unordered(_worker_l2_feature_segment, fixtures))
    results.sort(key=lambda r: r.path.name)

    n = len(results)
    hist = {lbl: sum(1 for r in results if r.routing == lbl) for lbl in ROUTING_LABELS}
    passed = sum(1 for r in results if r.error is None)
    failures = [r for r in results if r.error is not None]

    totals = {"local": 0, "global": 0, "defer": 0}
    for r in results:
        if r.output is None:
            continue
        for k, v in _l2_kind_counts(r.output).items():
            totals[k] += v

    print("Routing histogram: " + " ".join(f"{lbl}={hist[lbl]}" for lbl in ROUTING_LABELS))
    print(f"{passed}/{n} fixtures passed (no exception, contract OK)")
    print(f"Total features: local={totals['local']} global={totals['global']} defer={totals['defer']}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for r in failures:
            print(f"  {r.path.name}: {r.error}")

    _compose_l2_feature_segment_grid(results, hist, totals, passed, n)
    return 0 if not failures else 1


def _check_l2_feature_segment_contract(out: object) -> str | None:
    if not isinstance(out, FeatureSegmentation):
        return f"output type {type(out).__name__}, expected FeatureSegmentation"
    if not isinstance(out.segments, list):
        return f"segments type {type(out.segments).__name__}, expected list"
    for i, seg in enumerate(out.segments):
        if not isinstance(seg, (Classified, Feature)):
            return f"segments[{i}] type {type(seg).__name__}, expected Classified|Feature"
        if isinstance(seg, Feature):
            if not isinstance(seg.mask, np.ndarray) or seg.mask.dtype != np.uint8:
                return f"segments[{i}].mask dtype {getattr(seg.mask, 'dtype', None)}, expected uint8 ndarray"
            if seg.kind not in ("local", "global", "defer"):
                return f"segments[{i}].kind {seg.kind!r}, expected local|global|defer"
    return None


def _write_l2_feature_segment_strip(r: L2FeatureSegmentResult) -> None:
    """Per-fixture H7 strip: aligned (L0) | L1 primitives | L2 features + caption."""
    PANEL = 512
    CAPTION_H = 36
    total_w = PANEL * 3
    total_h = PANEL + CAPTION_H

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_big = _load_font(48)

    if r.aligned_input is not None:
        canvas.paste(_fit_to_panel(Image.fromarray(r.aligned_input, "RGB"), PANEL), (0, 0))
    else:
        _draw_centered_text(draw, 0, 0, PANEL, PANEL, "—", f_big, (0xAA, 0xAA, 0xAA))

    if r.l1_output is not None and r.aligned_input is not None:
        l1_arr = draw_overlay(r.aligned_input, r.l1_output)
        canvas.paste(_fit_to_panel(Image.fromarray(l1_arr, "RGB"), PANEL), (PANEL, 0))
    else:
        _draw_centered_text(draw, PANEL, 0, PANEL, PANEL, "—", f_big, (0xAA, 0xAA, 0xAA))

    if r.output is not None and r.aligned_input is not None:
        l2_arr = draw_feature_overlay(r.aligned_input, r.output)
        canvas.paste(_fit_to_panel(Image.fromarray(l2_arr, "RGB"), PANEL), (PANEL * 2, 0))
    else:
        _draw_centered_text(draw, PANEL * 2, 0, PANEL, PANEL, "FAIL", f_big, MISMATCH)

    for x in (PANEL, PANEL * 2):
        draw.line([(x, 0), (x, PANEL)], fill=(0xCC, 0xCC, 0xCC), width=1)

    draw.rectangle((0, PANEL, total_w, total_h), fill=(0xF4, 0xF4, 0xF4))
    f = _load_font(13)
    if r.routing is not None:
        chip_w, chip_h = 80, 20
        chip_y = PANEL + (CAPTION_H - chip_h) // 2
        draw.rectangle((8, chip_y, 8 + chip_w, chip_y + chip_h), fill=ROUTING_COLORS[r.routing])
        _draw_centered_text(draw, 8, chip_y, chip_w, chip_h, r.routing, f, (255, 255, 255))
        text_x = 8 + chip_w + 12
    else:
        text_x = 8
    if r.output is not None:
        summary = "  •  " + _l2_summary(r.output)
    elif r.error:
        summary = f"  •  ERROR: {r.error}"
    else:
        summary = ""
    caption = r.path.name + summary
    draw.text((text_x, PANEL + (CAPTION_H - 14) // 2), caption, font=f,
              fill=(0, 0, 0) if r.error is None else MISMATCH)

    canvas.save(L2FS_DIR / r.path.name, "PNG", optimize=True)


def _compose_l2_feature_segment_grid(
    results: list[L2FeatureSegmentResult],
    hist: dict[str, int],
    totals: dict[str, int],
    passed: int,
    n: int,
) -> None:
    THUMB = 96
    OUT_W = 128
    SUM_W = 150
    CHECK_W = 36
    NAME_W = 160
    PAD = 8
    ROW_H = THUMB

    # thumb | name | L1 overlay | L2 overlay | L/G/D summary | check
    cols = [THUMB, NAME_W, OUT_W, OUT_W, SUM_W, CHECK_W]
    col_x = [PAD]
    for w in cols[:-1]:
        col_x.append(col_x[-1] + w + PAD)
    total_w = col_x[-1] + cols[-1] + PAD

    header_h = 120
    body_h = (ROW_H + PAD) * len(results) + PAD
    total_h = header_h + body_h

    canvas = Image.new("RGB", (total_w, total_h), (0xF4, 0xF4, 0xF4))
    draw = ImageDraw.Draw(canvas)
    f_title = _load_font(18)
    f_body = _load_font(13)
    f_small = _load_font(11)
    f_big = _load_font(22)
    f_fail = _load_font(14)

    draw.text((PAD, PAD), "Stage 3 L2 feature_segment — testbed grid", font=f_title, fill=(0, 0, 0))
    hist_text = " ".join(f"{lbl}={hist[lbl]}" for lbl in ROUTING_LABELS)
    draw.text((PAD, PAD + 28), f"Routing histogram: {hist_text}", font=f_body, fill=(0, 0, 0))
    pass_color = MATCH if passed == n else MISMATCH
    draw.text((PAD, PAD + 50), f"{passed}/{n} fixtures passed (no exception, contract OK)",
              font=f_body, fill=pass_color)
    draw.text(
        (PAD, PAD + 72),
        f"Total features: local={totals['local']}  global={totals['global']}  defer={totals['defer']}"
        "   (columns: L1 primitives → L2 feature split)",
        font=f_body, fill=(0, 0, 0),
    )

    for r_idx, r in enumerate(results):
        y = header_h + r_idx * (ROW_H + PAD)

        draw.rectangle((col_x[0], y, col_x[0] + THUMB, y + ROW_H),
                       fill=(255, 255, 255), outline=(0xAA, 0xAA, 0xAA))
        if r.aligned_input is not None:
            thumb = Image.fromarray(r.aligned_input, "RGB")
            thumb.thumbnail((THUMB - 6, ROW_H - 6))
            canvas.paste(thumb, (col_x[0] + (THUMB - thumb.width) // 2,
                                 y + (ROW_H - thumb.height) // 2))

        draw.rectangle((col_x[1], y, col_x[1] + NAME_W, y + ROW_H), fill=(0xFA, 0xFA, 0xFA))
        draw.text((col_x[1] + 6, y + ROW_H // 2 - 8), r.path.name, font=f_small, fill=(0, 0, 0))

        # L1 primitive overlay
        draw.rectangle((col_x[2], y, col_x[2] + OUT_W, y + ROW_H),
                       fill=(255, 255, 255), outline=(0xAA, 0xAA, 0xAA))
        if r.l1_output is not None and r.aligned_input is not None:
            l1_img = Image.fromarray(draw_overlay(r.aligned_input, r.l1_output), "RGB")
            l1_img.thumbnail((OUT_W - 6, ROW_H - 6))
            canvas.paste(l1_img, (col_x[2] + (OUT_W - l1_img.width) // 2,
                                  y + (ROW_H - l1_img.height) // 2))
        else:
            _draw_centered_text(draw, col_x[2], y, OUT_W, ROW_H, "—", f_fail, (0xAA, 0xAA, 0xAA))

        # L2 feature overlay
        draw.rectangle((col_x[3], y, col_x[3] + OUT_W, y + ROW_H),
                       fill=(255, 255, 255), outline=(0xAA, 0xAA, 0xAA))
        if r.output is not None and r.aligned_input is not None:
            l2_img = Image.fromarray(draw_feature_overlay(r.aligned_input, r.output), "RGB")
            l2_img.thumbnail((OUT_W - 6, ROW_H - 6))
            canvas.paste(l2_img, (col_x[3] + (OUT_W - l2_img.width) // 2,
                                  y + (ROW_H - l2_img.height) // 2))
        else:
            _draw_centered_text(draw, col_x[3], y, OUT_W, ROW_H, "FAIL", f_fail, MISMATCH)

        # L/G/D summary band
        draw.rectangle((col_x[4], y, col_x[4] + SUM_W, y + ROW_H), fill=(0xFA, 0xFA, 0xFA))
        if r.output is not None:
            kc = _l2_kind_counts(r.output)
            _draw_centered_text(draw, col_x[4], y, SUM_W, ROW_H,
                                f"L{kc['local']} G{kc['global']} D{kc['defer']}", f_body, (0, 0, 0))
        else:
            _draw_centered_text(draw, col_x[4], y, SUM_W, ROW_H, "n/a", f_body, (0x9E, 0x9E, 0x9E))

        ok = r.error is None
        draw.rectangle((col_x[5], y, col_x[5] + CHECK_W, y + ROW_H), fill=(MATCH if ok else MISMATCH))
        mark = "✓" if ok else "✗"
        bbox = draw.textbbox((0, 0), mark, font=f_big)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((col_x[5] + (CHECK_W - tw) // 2, y + (ROW_H - th) // 2 - 2),
                  mark, font=f_big, fill=(255, 255, 255))

    canvas.save(L2FS_GRID_PATH, "PNG", optimize=True)
    print(f"Saved {L2FS_GRID_PATH.relative_to(REPO_ROOT)}")


# ─── bench harness ────────────────────────────────────────────────────


WORKER_BY_STAGE: dict = {
    "categorize": _worker_categorize,
    "preprocess": _worker_preprocess,
    "align":      _worker_align,
    "segment_prim_classify": _worker_segment_prim_classify,
    "L2_feature_segment": _worker_l2_feature_segment,
}


def run_bench(
    stage: str | None = None,
    serial: bool = False,
    label: str | None = None,
) -> int:
    """Benchmark the implemented testbed stages end-to-end and emit a JSON
    report under .cache/bench/. Use as the H1 baseline before / after any
    perf change.

    stage:  one of WORKER_BY_STAGE keys, or None to run all of them.
    serial: run in the master process, letting BLAS use multiple threads.
            Default forks the same pool testbed uses.
    label:  free-form tag appended to the JSON filename.
    """
    fixtures = sorted(FIXTURES.glob("*.png"))
    if not fixtures:
        print(f"error: no PNG fixtures found under {FIXTURES.relative_to(REPO_ROOT)}")
        return 2

    if stage is None:
        stages_to_run = list(WORKER_BY_STAGE.keys())
    else:
        if stage not in WORKER_BY_STAGE:
            print(
                f"error: bench stage {stage!r} not in {sorted(WORKER_BY_STAGE)}",
                file=sys.stderr,
            )
            return 2
        stages_to_run = [stage]

    timings: dict[str, dict[str, float]] = {}

    t0 = time.perf_counter()
    _prewarm_fixtures(fixtures)
    decode_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    prewarm(fixtures, _PREWARM_STAGES)
    prewarm_ms = (time.perf_counter() - t0) * 1000

    timings["_master"] = {
        "decode_ms": decode_ms,
        "cache_prewarm_ms": prewarm_ms,
    }

    for s in stages_to_run:
        t0 = time.perf_counter()
        if serial:
            for p in fixtures:
                WORKER_BY_STAGE[s](p)
        else:
            with _pool(len(fixtures)) as pool:
                list(pool.imap_unordered(WORKER_BY_STAGE[s], fixtures))
        wall_ms = (time.perf_counter() - t0) * 1000
        timings[s] = {
            "wall_ms": wall_ms,
            "per_fixture_ms": wall_ms / len(fixtures),
        }

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = f"{sha or 'nosha'}-{ts}"
    if label:
        slug += f"-{label}"
    out_path = REPO_ROOT / ".cache" / "bench" / f"{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "sha": sha,
        "ts": ts,
        "serial": serial,
        "label": label,
        "n_fixtures": len(fixtures),
        "stages": stages_to_run,
        "timings_ms": timings,
    }, indent=2))

    print(f"bench  sha={sha or '?'}  serial={serial}  n={len(fixtures)}")
    m = timings["_master"]
    print(f"  {'_master.decode':>20s}  {m['decode_ms']:8.1f} ms")
    print(f"  {'_master.prewarm':>20s}  {m['cache_prewarm_ms']:8.1f} ms")
    for s in stages_to_run:
        t = timings[s]
        print(
            f"  {s:>20s}  wall={t['wall_ms']:8.1f} ms  "
            f"per_fixture={t['per_fixture_ms']:7.2f} ms"
        )
    print(f"saved {out_path.relative_to(REPO_ROOT)}")
    return 0
