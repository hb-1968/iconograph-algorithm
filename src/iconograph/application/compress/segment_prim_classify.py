"""Stage 3 L1 — binarize, 8-connectivity connected-components, classify each
CC as one of {Circle, Rectangle, Triangle, Ellipse} or pass it to L2 as an
L1Miss.

Pipeline position:
    align → segment_prim_classify (THIS) → L2 feature-segmentation (TBD)

Input is align()'s (S, S, 3) uint8 RGB on a white background, axis-aligned.
The output is a SegmentClassification holding the binarized mask, the CC
label image, and a per-CC list of Classified | L1Miss. CCs classify when
one of the four primitive fits clears IOU_ACCEPT; everything else is L1Miss
with its bbox-cropped mask retained for L2 to segment into near-fittable
sub-features (shape fitting deferred to a later layer).

Segmentation rule: strictly disjoint shapes — if two visually distinct
shapes share a single foreground pixel they are one CC and they become
one L1 miss for L2. No watershed, no morphological neck-splitting. The
Adobe Illustrator "Ai" fixture is the canonical case: Circle (i-dot),
Rectangle (i-body), L1Miss (A).

"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from iconograph.application.compress.common import (
    Circle,
    Ellipse,
    Primitive,
    Rectangle,
    Triangle,
    _fit_circle,
    _fit_ellipse,
    _fit_rectangle,
    _fit_triangle,
    _iou_bool,
    iou_analytical,
)


WHITE_THRESH = 250
IOU_ACCEPT = 0.92
MIN_AREA_FRAC = 5e-4
# Axis-aligned ellipses (and circles) are reflection-symmetric about both
# bbox axes. Shapes that match an inscribed ellipse on raw IoU but break
# this symmetry — Signal's chat-bubble-with-tail (sym ≈ 0.955), Slack's
# corner droplets (sym ≈ 0.88) — must reject. Threshold 0.96 sits above
# Signal's 0.955 and well below Ai's clean i-dot 0.99.
SYMMETRY_MIN_IOU = 0.96
# Triangle concavity gate (the vue class): a min-enclosing triangle papers
# over any concavity, and IoU under-weights a NARROW deep notch — vue's bold
# V fits at IoU 0.9436 ≥ IOU_ACCEPT while its nested-V identity dies at this
# stage. Census (the ONLY two L1-Classified Triangles in the fixture cache):
# vue defect 0.2015 / notch 0.0552; youtube-music#1 0.0025 / 0.0000. BOTH
# thresholds must hold to reject — a deep hairline crack alone is boundary
# noise, and a wide shallow leg-gap (adobe's A hull) alone is not identity.
TRI_MAX_DEFECT_FRAC = 0.04   # max convexityDefects depth / max(h, w)
TRI_MAX_NOTCH_FRAC = 0.02    # largest hull∖CC component area / CC area

_GREEN = (0x2E, 0x7D, 0x32)
_RED = (0xC6, 0x28, 0x28)


@dataclass(frozen=True)
class Classified:
    primitive: Primitive
    bbox: tuple[int, int, int, int]
    color_rgb: tuple[int, int, int]
    area_px: int
    iou: float


@dataclass
class L1Miss:
    bbox: tuple[int, int, int, int]
    color_rgb: tuple[int, int, int]
    area_px: int
    mask: np.ndarray
    best_attempt: tuple[str, float]


@dataclass
class SegmentClassification:
    image_shape: tuple[int, int, int]
    binarized: np.ndarray
    labels: np.ndarray
    segments: list[Classified | L1Miss]


def _axis_symmetry(roi: np.ndarray) -> float:
    """min(IoU(roi, h_flip), IoU(roi, v_flip)). Equals 1.0 for shapes
    symmetric about both bbox centerlines (true axis-aligned ellipse,
    circle, axis-aligned rectangle). Drops below 1 when the shape carries
    an asymmetric protrusion (Signal's chat-bubble tail) or a flat side
    (Slack corner droplets)."""
    rb = roi.astype(bool)
    hf = np.flip(rb, axis=1)
    vf = np.flip(rb, axis=0)
    return min(_iou_bool(rb, hf), _iou_bool(rb, vf))


def _tri_concavity(roi: np.ndarray, contour: np.ndarray) -> tuple[float, float]:
    """(max convexity-defect depth / max(h, w), largest hull∖roi component
    area / roi area) — how DEEP and how LARGE the biggest bite the
    min-enclosing triangle would paper over is. `contour` must be the SAME
    int32 CHAIN_APPROX_NONE contour the fits use (mixing a resampled contour
    desynchronizes convexityDefects' indices). (0.0, 0.0) on any degenerate
    case (short contour, collapsed hull, already-convex CC)."""
    if contour.shape[0] < 4:
        return 0.0, 0.0
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0.0, 0.0
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return 0.0, 0.0
    if defects is None:
        return 0.0, 0.0
    # the 4th defects column is ×256 fixed-point pixels
    dep = float(defects[:, 0, 3].max()) / 256.0 / float(max(roi.shape))
    hull_mask = np.zeros_like(roi)
    cv2.fillPoly(hull_mask, [contour[hull_idx.flatten()]], 255)
    notch = ((hull_mask > 0) & (roi == 0)).astype(np.uint8)
    n, _lbl, stats, _c = cv2.connectedComponentsWithStats(notch, connectivity=8)
    if n <= 1:
        return dep, 0.0
    nfr = float(stats[1:, cv2.CC_STAT_AREA].max()) / float(np.count_nonzero(roi))
    return dep, nfr


def segment_prim_classify(img: np.ndarray) -> SegmentClassification:
    s = int(img.shape[0])
    image_shape = (int(img.shape[0]), int(img.shape[1]), int(img.shape[2]))

    fg = (np.any(img < WHITE_THRESH, axis=2).astype(np.uint8)) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8, ltype=cv2.CV_32S,
    )
    labels = labels.astype(np.int32, copy=False)

    if num_labels <= 1:
        return SegmentClassification(
            image_shape=image_shape,
            binarized=fg,
            labels=labels,
            segments=[],
        )

    min_area = MIN_AREA_FRAC * s * s
    segments: list[Classified | L1Miss] = []

    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if w < 3 or h < 3:
            continue

        roi = ((labels[y:y + h, x:x + w] == lbl).astype(np.uint8)) * 255

        contours, _ = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if contour.shape[0] < 3:
            continue

        pix = img[labels == lbl]
        color_rgb = tuple(int(c) for c in np.median(pix, axis=0))

        circle, fit_c, area_c = _fit_circle(contour, h, w, x, y)
        rect, fit_r, area_r = _fit_rectangle(contour, h, w, x, y)
        tri, fit_t, area_t = _fit_triangle(contour, h, w, x, y)
        ell, fit_e, area_e = _fit_ellipse(contour, h, w, x, y)

        # Axis-symmetry gate: Circle and Ellipse demand reflection-symmetry
        # about both bbox centerlines. Rectangles are symmetric by
        # construction (the fit IS the bbox), so any shape that hits the
        # rect IoU floor is already symmetric enough. Triangles get the
        # CONCAVITY gate below instead — min-enclosing triangles are rarely
        # axis-symmetric, but a deep+large bite means compound glyph (vue).
        sym_ok = _axis_symmetry(roi) >= SYMMETRY_MIN_IOU
        iou_circle = iou_analytical(roi, fit_c, area_c) if sym_ok else 0.0
        iou_ellipse = iou_analytical(roi, fit_e, area_e) if sym_ok else 0.0
        # Triangle concavity gate: rejecting the candidate makes the CC an
        # L1Miss — L2 re-fits hint="triangle" and decompose carves the notch
        # as subtracts (vue: 4 parts, a0.981/b0.885, the V ~85% carved).
        dep, nfr = _tri_concavity(roi, contour)
        tri_ok = not (dep >= TRI_MAX_DEFECT_FRAC and nfr >= TRI_MAX_NOTCH_FRAC)
        iou_tri = iou_analytical(roi, fit_t, area_t) if tri_ok else 0.0

        candidates: list[tuple[str, float, Primitive]] = [
            ("circle", iou_circle, circle),
            ("rectangle", iou_analytical(roi, fit_r, area_r), rect),
            ("triangle", iou_tri, tri),
            ("ellipse", iou_ellipse, ell),
        ]
        best_name, best_iou, best_prim = max(candidates, key=lambda c: c[1])

        bbox = (x, y, w, h)
        if best_iou >= IOU_ACCEPT:
            segments.append(Classified(
                primitive=best_prim,
                bbox=bbox,
                color_rgb=color_rgb,
                area_px=area,
                iou=float(best_iou),
            ))
        else:
            segments.append(L1Miss(
                bbox=bbox,
                color_rgb=color_rgb,
                area_px=area,
                mask=roi.copy(),
                best_attempt=(best_name, float(best_iou)),
            ))

    segments.sort(key=lambda s: (-s.area_px, s.bbox[1], s.bbox[0]))

    return SegmentClassification(
        image_shape=image_shape,
        binarized=fg,
        labels=labels,
        segments=segments,
    )


def draw_overlay(img: np.ndarray, sc: SegmentClassification) -> np.ndarray:
    canvas = img.copy()
    for seg in sc.segments:
        if isinstance(seg, Classified):
            p = seg.primitive
            if isinstance(p, Circle):
                cv2.circle(
                    canvas,
                    (int(round(p.cx)), int(round(p.cy))),
                    int(round(p.r)),
                    _GREEN, 2,
                )
            elif isinstance(p, Rectangle):
                cv2.rectangle(
                    canvas,
                    (p.x, p.y), (p.x + p.w, p.y + p.h),
                    _GREEN, 2,
                )
            elif isinstance(p, Triangle):
                pts = np.array([p.v1, p.v2, p.v3], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, _GREEN, 2)
            elif isinstance(p, Ellipse):
                cv2.ellipse(
                    canvas,
                    (int(round(p.cx)), int(round(p.cy))),
                    (max(1, int(round(p.rx))), max(1, int(round(p.ry)))),
                    0, 0, 360, _GREEN, 2,
                )
        else:
            x, y, w, h = seg.bbox
            contours, _ = cv2.findContours(
                seg.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            for c in contours:
                shifted = c + np.array([[x, y]], dtype=c.dtype).reshape(1, 1, 2)
                cv2.drawContours(canvas, [shifted], -1, _RED, 2)
    return canvas
