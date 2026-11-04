"""Render the selected foreground as black-on-white RGB.

Outer-precedence compounds subtract the inner mask; inner-precedence compounds
use it directly. Other marks use the alpha threshold shared with categorize.
"""

import numpy as np
from iconograph.application.categorize import (
    Categorization,
    ALPHA_OPAQUE_THRESHOLD,
)
from .common import compose_bw

def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
    if cat.is_compound and cat.precedence == "outer":
        # Keep inner glyphs visible as cutouts in the outer silhouette.
        mask = cat.envelope_mask & ~cat.inner_mask
    elif cat.is_compound and cat.precedence == "inner":
        mask = cat.inner_mask
    else:
        mask = rgba[..., 3] > ALPHA_OPAQUE_THRESHOLD

    return compose_bw(mask)