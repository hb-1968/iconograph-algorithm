"""Dispatch categorization to the preprocessing branches.

Color-essential marks use palette grouping and separated features. Inverted
marks use the inner mask; color-carried marks retain contrast as cutouts.
Other marks use the standard mask. Return framed RGB uint8 on white.
"""
import numpy as np
from iconograph.application.categorize import Categorization
from . import standard, inverted, color_essential, color_carried
from .common import frame_with_margin


def preprocess(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
    if cat.color_essential:
        rgba_universal = color_essential.universalize(cat, rgba)
        framed = frame_with_margin(rgba_universal)
        _, features = color_essential.separate_features(framed)
        return color_essential.compose_features(features)
    if cat.routing == "inverted":
        bw = inverted.universalize(cat, rgba)
    elif cat.color_carried:
        bw = color_carried.universalize(cat, rgba)
    else:
        bw = standard.universalize(cat, rgba)
    return frame_with_margin(bw)
