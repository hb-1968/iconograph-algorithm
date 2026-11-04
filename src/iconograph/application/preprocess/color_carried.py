"""Preserve interior color contrast as holes in an envelope silhouette.

Used when the outer shape alone would discard the identifying marks.
"""

import numpy as np
from skimage.color import rgb2lab

from iconograph.application.categorize import ALPHA_OPAQUE_THRESHOLD, Categorization
from iconograph.application.common import interior_mark_mask
from .common import compose_bw


def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
    opaque = rgba[..., 3] > ALPHA_OPAQUE_THRESHOLD
    lab = rgb2lab(rgba[..., :3].astype(np.float32) / 255.0)
    marks = interior_mark_mask(lab, opaque)
    badge = opaque & ~marks          # enclosed marks become white holes in the badge
    return compose_bw(badge)
