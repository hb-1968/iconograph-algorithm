"""Flag color-carried marks whose interior detail disappears in preprocessing.

Compare envelope simplicity and enclosed color contrast. This diagnostic is
report-only and does not change categorization or stop application builds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab

from iconograph.application.categorize import ALPHA_OPAQUE_THRESHOLD
from iconograph.application.common import (
    COLLAPSE_IOU, MARK_FRAC_FLOOR, fit_simple_primitive, interior_mark_mask)


@dataclass(frozen=True)
class GateResult:
    """One logo's pre-compression verdict (report-only)."""
    suspect: bool               # colour-carried — interior detail lost by the alpha-only path
    recommended: str | None     # "color-difference" when suspect, else None
    collapse_kind: str          # best simple-primitive fit: circle|rect|ellipse|none
    collapse_iou: float         # its IoU (high = silhouette collapses to a simple shape)
    mark_frac: float            # enclosed interior colour-contrast marks / opaque area

    def as_dict(self) -> dict:
        """JSON-friendly form for the build manifest (report-only field)."""
        return {"suspect": self.suspect, "recommended": self.recommended,
                "collapse_kind": self.collapse_kind,
                "collapse_iou": round(self.collapse_iou, 4),
                "mark_frac": round(self.mark_frac, 4)}


def evaluate_precompress(cat, preprocessed: np.ndarray,
                         rgba: np.ndarray) -> GateResult:
    """cat: the stage-1 Categorization (uses .color_essential, .is_compound).
    preprocessed: the stage-2 BW output (unused here — kept for signature stability /
    future structural checks). rgba: (H,W,4) uint8 SOURCE. Returns a GateResult."""
    opaque = rgba[..., 3] > ALPHA_OPAQUE_THRESHOLD
    if not opaque.any():
        return GateResult(False, None, "none", 0.0, 0.0)
    lab = rgb2lab(rgba[..., :3].astype(np.float32) / 255.0)
    kind, iou = fit_simple_primitive(opaque.astype(np.uint8))
    mark_frac = float(interior_mark_mask(lab, opaque).sum()) / float(opaque.sum())
    color_aware = cat.color_essential or cat.is_compound
    suspect = (iou >= COLLAPSE_IOU and mark_frac >= MARK_FRAC_FLOOR
               and not color_aware)
    return GateResult(suspect, "color-difference" if suspect else None,
                      kind, float(iou), mark_frac)
