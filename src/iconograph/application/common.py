"""Application-pipeline-wide shared helpers.

Helpers used by more than one stage live here so per-stage common.py
files do not need to import sideways across stage boundaries (e.g.
stage 3 reaching into preprocess/common.py).
"""

import cv2
import numpy as np

# Fedora hicolor feed sizes — the per-target grids stage 3's render tier (R1) and
# stage 4's min-feature enforcement work over. Locked to the 16/24 feed resolution
# (user ruling): the pixel-cubify render owns T ≤ 24, and these are the sizes fed
# onward to stage-4 refinement. Stage 5's compose then places each onto its N=T+8
# output grid (16→24, 24→32). The geometry render path is retained for T > 24
# (synthetic regression only) but production never requests those sizes.
HICOLOR_SIZES: tuple[int, ...] = (16, 24)

# 3x3 rect structuring element = the full 8-neighbourhood (incl. diagonals).
_SE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def masks_8adjacent(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff foreground of `a` touches foreground of `b` under 8-connectivity
    (sharing an edge OR a diagonal corner). `a`, `b`: same-shape masks (bool or
    uint8, nonzero = foreground).

    Dilating `a` by the 3x3 SE grows it into its 8-neighbourhood; intersecting
    with `b` is non-empty exactly when the two are ≤1px apart. So `not
    masks_8adjacent(x, logo)` is the "x is ≥1px 8-separated from the logo"
    guarantee used by the stage-5 monogram fitter. Consolidates the 8-adjacency
    idiom that recurs inline across stages 3-4 (e.g. the `_adjacent` closure in
    compress/render.py) into one importable helper; stage 5 uses it directly.
    Converging the pre-existing stage-3/4 inline copies onto this is deferred —
    it touches determinism-critical upstream code and needs its own byte-identity
    re-verification (H6), so it is a separate cleanup, not part of this stage.
    """
    return bool(np.count_nonzero(cv2.dilate(a.astype(np.uint8), _SE3) & b.astype(np.uint8)))


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
