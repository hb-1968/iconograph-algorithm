"""Stage 3 — compress sub-pipeline.

Sub-stages in order:
    align (L0)  →  segment_prim_classify (L1)  →  classify (L2, TBD)

L0 axis-aligns dominant straight features. L1 binarizes, watershed-splits
touching shapes, and per-region fits one of {Circle, Rectangle, Triangle,
Ellipse}; regions that don't fit any primitive trivially are tagged L1Miss
and carry their cropped mask forward for L2 to decompose via unions and
negates of primitives.
"""

from iconograph.application.compress.align import align
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
]
