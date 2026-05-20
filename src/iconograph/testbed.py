"""Testbed driver. First implementation: `testbed categorize`.

Loads ground truth from fixtures/README.md, runs categorize() on every
PNG fixture, prints the 3×3 routing confusion matrix + per-field
accuracies to stdout, and writes testbed/categorize/grid.png. Per H7
this is the single artifact under review; per H8 the grid is
byte-identical on rerun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iconograph.application.categorize import Categorization, categorize, load_rgba
from iconograph.application.preprocess import preprocess

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
    results: list[tuple[Path, GroundTruth, Categorization]] = []
    skipped: list[str] = []
    for p in fixtures:
        gt = truth.get(p.name)
        if gt is None:
            skipped.append(p.name)
            continue
        pred = categorize(p)
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

    results: list[PreprocessResult] = []
    for p in fixtures:
        try:
            cat = categorize(p)
        except Exception as e:
            results.append(
                PreprocessResult(p, None, None, f"categorize: {type(e).__name__}: {e}")
            )
            continue
        try:
            rgba = load_rgba(p)
            out = preprocess(cat, rgba)
        except Exception as e:
            results.append(
                PreprocessResult(
                    p, cat.routing, None, f"preprocess: {type(e).__name__}: {e}"
                )
            )
            continue
        contract_err = _check_preprocess_contract(out)
        if contract_err is not None:
            results.append(PreprocessResult(p, cat.routing, None, contract_err))
            continue
        results.append(PreprocessResult(p, cat.routing, out, None))

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

    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        _write_preprocess_strip(r)
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
