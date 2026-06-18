"""Application-pipeline-wide shared helpers.

Helpers used by more than one stage live here so per-stage common.py
files do not need to import sideways across stage boundaries (e.g.
stage 3 reaching into preprocess/common.py).
"""

import cv2
import numpy as np

# Fedora hicolor target sizes — the per-target grids stage 3's render tier (R1),
# stage 4's min-feature enforcement, and stage 6's fill-maximization all work over.
HICOLOR_SIZES: tuple[int, ...] = (16, 22, 24, 32, 48, 64, 128, 256)


def frame_with_margin(img: np.ndarray, margin_frac: float = 0.10) -> np.ndarray:
    """Crop img to content bbox; center on a square white canvas with
    margin_frac padding per side (relative to the larger content axis).

    img: (H, W, 3) uint8 RGB OR (H, W, 4) uint8 RGBA. RGBA inputs are
         composited onto white before measuring content.
    Returns: (S, S, 3) uint8 RGB. S = round(max(content_w, content_h) * (1 + 2*margin_frac)).
    """
    if img.ndim == 3 and img.shape[2] == 4:
        rgb_f = img[..., :3].astype(np.float32)
        a = img[..., 3:4].astype(np.float32) / 255.0
        rgb = np.clip(rgb_f * a + 255.0 * (1.0 - a), 0, 255).astype(np.uint8)
    else:
        rgb = img

    not_white = np.any(rgb < 255, axis=2).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(not_white)
    if w == 0 or h == 0:
        return np.full((64, 64, 3), 255, dtype=np.uint8)

    content = rgb[y:y + h, x:x + w]
    m = max(w, h)
    s = int(round(m * (1.0 + 2.0 * margin_frac)))
    out = np.full((s, s, 3), 255, dtype=np.uint8)
    ox = (s - w) // 2
    oy = (s - h) // 2
    out[oy:oy + h, ox:ox + w] = content
    return out
