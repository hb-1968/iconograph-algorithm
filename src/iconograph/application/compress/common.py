"""Stage 3 compress — shared primitive vocabulary, rasterizing fitters, IoU.

Lifted out of segment_prim_classify.py (L1) so both L1 and classify.py (L2)
share one definition of the {Circle, Rectangle, Triangle, Ellipse} primitives,
the per-primitive rasterizing fitters, and the analytical-area IoU (H5 — reuse
across the stage; one source of truth, not N copies).

L1 fits ONE min-enclosing / bounding primitive per connected component. L2
reuses these dataclasses + iou_analytical and adds its own *inscribed* fitters
(see classify.py) to build union-only compounds. _axis_symmetry stays in L1 —
it gates single-primitive classification and has no role in L2 decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Triangle:
    v1: tuple[float, float]
    v2: tuple[float, float]
    v3: tuple[float, float]


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    rx: float
    ry: float


Primitive = Circle | Rectangle | Triangle | Ellipse


def iou_analytical(roi: np.ndarray, fit: np.ndarray, true_fit_area: float) -> float:
    """IoU(CC, primitive) using the primitive's ANALYTICAL area as the union
    denominator instead of the rasterized-and-clipped count. Without this,
    a primitive whose true geometry extends beyond the CC bbox (most often
    the min-enclosing triangle's apex) gets its excess area silently
    clipped by cv2.fillPoly and the IoU is wildly inflated.
    """
    roi_bool = roi.astype(bool)
    fit_bool = fit.astype(bool)
    inter = int(np.logical_and(roi_bool, fit_bool).sum())
    roi_area = int(roi_bool.sum())
    union = roi_area + true_fit_area - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _fit_circle(
    contour: np.ndarray, h: int, w: int, ox: int, oy: int
) -> tuple[Circle, np.ndarray, float]:
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fit, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    true_area = float(np.pi * r * r)
    return Circle(cx=float(cx + ox), cy=float(cy + oy), r=float(r)), fit, true_area


def _fit_rectangle(
    contour: np.ndarray, h: int, w: int, ox: int, oy: int
) -> tuple[Rectangle, np.ndarray, float]:
    xr, yr, wr, hr = cv2.boundingRect(contour)
    fit = np.zeros((h, w), dtype=np.uint8)
    # numpy slicing — avoids cv2.rectangle's inclusive-endpoint off-by-one.
    fit[yr:yr + hr, xr:xr + wr] = 255
    true_area = float(wr * hr)
    return Rectangle(x=int(xr + ox), y=int(yr + oy), w=int(wr), h=int(hr)), fit, true_area


def _fit_triangle(
    contour: np.ndarray, h: int, w: int, ox: int, oy: int
) -> tuple[Triangle, np.ndarray, float]:
    area, tri = cv2.minEnclosingTriangle(contour)
    if tri is None:
        return Triangle(v1=(0.0, 0.0), v2=(0.0, 0.0), v3=(0.0, 0.0)), np.zeros((h, w), dtype=np.uint8), 0.0
    pts = tri.astype(np.int32)
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fit, [pts], 255)
    flat = tri.reshape(3, 2)
    return Triangle(
        v1=(float(flat[0, 0] + ox), float(flat[0, 1] + oy)),
        v2=(float(flat[1, 0] + ox), float(flat[1, 1] + oy)),
        v3=(float(flat[2, 0] + ox), float(flat[2, 1] + oy)),
    ), fit, float(area)


def _fit_ellipse(
    contour: np.ndarray, h: int, w: int, ox: int, oy: int
) -> tuple[Ellipse, np.ndarray, float]:
    xr, yr, wr, hr = cv2.boundingRect(contour)
    rx = wr / 2.0
    ry = hr / 2.0
    cx = xr + rx
    cy = yr + ry
    fit = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        fit,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(rx))), max(1, int(round(ry)))),
        0, 0, 360, 255, -1,
    )
    true_area = float(np.pi * rx * ry)
    return Ellipse(
        cx=float(cx + ox), cy=float(cy + oy),
        rx=float(rx), ry=float(ry),
    ), fit, true_area
