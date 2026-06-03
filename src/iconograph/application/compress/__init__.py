"""Stage 3 — compress sub-pipeline.

Sub-stages in order:
    align (L0)  →  segment_prim_classify (L1)  →  L2_feature_segment (L2)  →  L3_fit (L3)

L0 axis-aligns dominant straight features. L1 binarizes, labels 8-connected
components, and per-region fits one of {Circle, Rectangle, Triangle, Ellipse};
regions that don't fit any single primitive are tagged L1Miss and carry their
cropped mask forward to L2. L2 splits each miss (it does NOT fit it): global
structures (ring / holed / sunburst) are kept whole — a detected ring is
polar-unrolled and resolved into annular sub-features — and a body+appendages
shape is split, each part tagged with a `shape_hint`. L3 fits every miss-derived
feature into an overlapping-CSG construction (union / negation of rotated
primitives, two-pass greedy + cleanup); a feature it cannot fit to an adequate IoU
becomes a `Deferral` instead of a forced low-quality shape.
"""

from iconograph.application.compress.align import align
from iconograph.application.compress.common import (
    Part,
    RotatedEllipse,
    RotatedRectangle,
)
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
    "Part",
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
