"""Separate color-essential marks before monochrome composition.

Group opaque colors in Lab, retain regions above CE_FEATURE_FLOOR of canvas
area, and erode each by CE_EROSION_PX to separate adjacent features. Return
black-on-white RGB; empty feature sets produce a white 64x64 canvas.
"""

import cv2
import numpy as np

from iconograph.application.categorize import (
    Categorization,
    ALPHA_OPAQUE_THRESHOLD,
)

from .common import compose_bw, shade_groups

# Tunable thresholds — single source so testbed iteration is fast (H1).
CE_PALETTE_K = 6         # KMeans centroid count — matched to expected dominant colours
CE_SHADE_DELTA = 70.0    # Lab ΔE76 distance threshold for "same shade family"
CE_EROSION_PX = 20       # per-feature mask erosion radius in pixels (Phase 3)
CE_FEATURE_FLOOR = 0.10  # min (colour area / total canvas area) to keep a colour


def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
    # Shade-group the opaque palette (lifted to common.shade_groups, H5) and remap
    # every opaque pixel to its group's area-dominant colour.
    alpha = rgba[..., 3]
    opaque = alpha > ALPHA_OPAQUE_THRESHOLD
    opaque_rgb = rgba[opaque][:, :3]
    universal_rgb, _area, pixel_group = shade_groups(
        opaque_rgb, CE_PALETTE_K, CE_SHADE_DELTA)
    out = rgba.copy()
    out[opaque, :3] = universal_rgb[pixel_group]
    return out


def separate_features(
    rgb: np.ndarray,
) -> tuple[np.ndarray, list[tuple[tuple[int, int, int], np.ndarray]]]:
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    
    WHITE = np.array([255, 255, 255], dtype=np.uint8)
    non_white = ~(colors == WHITE).all(axis=1)
    colors = colors[non_white]
    counts = counts[non_white]

    total_canvas = rgb.shape[0] * rgb.shape[1]
    fractions = counts / total_canvas
    keep = fractions > CE_FEATURE_FLOOR

    cleaned = rgb.copy()
    for color in colors[~keep]:
        cleaned[(rgb == color).all(axis=-1)] = WHITE
      
    features: list[tuple[tuple[int,int, int], np.ndarray]] = []
    for color in colors[keep]:
        mask = (cleaned == color).all(axis=-1)
        features.append((tuple(int(c) for c in color), mask))

    return cleaned, features


def compose_features(
    features: list[tuple[tuple[int, int, int], np.ndarray]],
) -> np.ndarray:
    """Erode each feature mask by CE_EROSION_PX pixels, union the eroded
    masks, and render as (H, W, 3) uint8 black-on-white RGB. The erosion
    introduces a thin negative-whitespace gap between adjacent features.

    features: output of separate_features (post-Phase 2). Empty list
              returns a (64, 64, 3) white canvas (matches frame_with_margin
              empty-input fallback).
    """
    if not features:
        return np.full((64, 64, 3), 255, dtype=np.uint8)

    k = 2 * CE_EROSION_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    H, W = features[0][1].shape
    union = np.zeros((H, W), dtype=bool)
    for _, mask in features:
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        union |= eroded

    return compose_bw(union)
