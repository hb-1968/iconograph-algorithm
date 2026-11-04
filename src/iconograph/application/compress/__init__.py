"""Alignment, primitive classification and feature segmentation.

The production renderer uses contour-based decomposition after L2. L3_fit
retains the older beam-search fitter for comparison.
"""

from iconograph.application.compress.align import align
from iconograph.application.compress.common import (
    Part,
    Polygon,
    RotatedEllipse,
    RotatedRectangle,
    RoundedRectangle,
    Wedge,
)
from iconograph.application.compress.render import render, render_debug
from iconograph.application.compress.L2_feature_segment import (
    Feature,
    FeatureSegmentation,
    draw_feature_overlay,
    segment_features,
)
from iconograph.application.compress.L3_fit import (
    Deferral,
    FittedDecomposition,
    FittedShape,
    draw_fit_overlay,
    fit_features,
)
from iconograph.application.compress.segment_prim_classify import (
    Circle,
    Classified,
    Ellipse,
    L1Miss,
    Rectangle,
    SegmentClassification,
    Triangle,
    draw_overlay,
    segment_prim_classify,
)

__all__ = [
    "align",
    "segment_prim_classify",
    "draw_overlay",
    "SegmentClassification",
    "Classified",
    "L1Miss",
    "Circle",
    "Rectangle",
    "Triangle",
    "Ellipse",
    "RotatedRectangle",
    "RotatedEllipse",
    "RoundedRectangle",
    "Wedge",
    "Polygon",
    "Part",
    "render",
    "render_debug",
    "segment_features",
    "draw_feature_overlay",
    "Feature",
    "FeatureSegmentation",
    "fit_features",
    "draw_fit_overlay",
    "FittedShape",
    "Deferral",
    "FittedDecomposition",
]
