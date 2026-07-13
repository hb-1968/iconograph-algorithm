"""Stage 2 — color-carried cutout branch.

For a solid single-primitive badge whose identity lives in interior colour-contrast
marks (categorize.color_carried: EasyEffects' EQ knobs/tracks on a blue disc, Ptyxis'
prompt on a dark square), the alpha-only `standard` branch collapses the whole thing
to one filled blob and the marks vanish. This branch keeps the badge and CARVES the
enclosed marks out as negative-space HOLES, so stage 3's recursive hole decomposition
renders them (a disc/square outline with mark-shaped counters) instead of a block.

The marks are the enclosed pixels far (Lab ΔE) from the area-dominant badge colour —
`application.common.interior_mark_mask`, the SAME definition categorize._color_carried
and the pre-compression gate use, so "the marks" is defined once. Like standard.py /
inverted.py, `universalize` returns a SOURCE-resolution (H,W,3) BW image; the
dispatcher in preprocess/__init__.py applies frame_with_margin.
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
