"""Shared panel helpers for the single-process debug harnesses (H5 lift).

d1_debug / l3_debug / diff_debug / render_debug each emit H7 comparison strips
from the same three moves: caption a panel, stack panels with separators, and
normalize panel height. One definition here instead of one copy per harness.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_PANEL_H = 420


def labeled(panel: np.ndarray, text: str) -> np.ndarray:
    """A white caption bar with text across the top of one panel."""
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 20), (255, 255, 255), -1)
    cv2.putText(out, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def hstack(panels: list[np.ndarray], sep: int = 3) -> np.ndarray:
    """Horizontally stack equal-height panels with thin separators."""
    bar = np.full((panels[0].shape[0], sep, 3), 0xCC, np.uint8)
    out: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        out.append(panel)
        if i < len(panels) - 1:
            out.append(bar)
    return np.hstack(out)


def shrink(img: np.ndarray, panel_h: int = DEFAULT_PANEL_H) -> np.ndarray:
    """Scale to panel_h height, preserving aspect (INTER_AREA — display quality)."""
    s = panel_h / img.shape[0]
    return cv2.resize(img, (max(1, int(round(img.shape[1] * s))), panel_h),
                      interpolation=cv2.INTER_AREA)
