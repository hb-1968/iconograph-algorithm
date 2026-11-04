"""Render the compound inner mask as black-on-white RGB uint8."""

import numpy as np
from iconograph.application.categorize import Categorization
from .common import compose_bw

def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
    mask = cat.inner_mask
    return compose_bw(mask)