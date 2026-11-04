"""Shared mask composition and deterministic Lab palette grouping.

compose_bw maps a boolean mask to black-on-white RGB uint8. lab_kmeans returns
Lab centroids and pixel labels with seed 0 and n_init=10. shade_groups merges
nearby colors and selects each group's area-dominant representative.
"""

import warnings

import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning

from iconograph.application.common import frame_with_margin  # re-exported for stage-2 callers; lifted to application/common.py per H5


def compose_bw(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape + (3,), 255, dtype=np.uint8)
    out[mask] = 0
    return out


def lab_kmeans(
    pixels_rgb: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    k_eff = min(k, pixels_rgb.shape[0])
    lab = rgb2lab(pixels_rgb.astype(np.float32) / 255.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        km = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit(lab)
    return km.cluster_centers_, km.labels_


def shade_groups(
    opaque_rgb: np.ndarray, k: int, delta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group the opaque palette into perceptual ΔE shade-groups with per-group area.

    Over-quantize with lab_kmeans, then agglomerate the fine centroids by Lab
    distance (< delta merges "blue" with "darker-blue" but not with "white"), so the
    group COUNT is the perceptual distinct-colour count. Returns:
      universal_rgb  (G,3) uint8  - area-dominant colour of each shade group,
      per_group_area (G,)  int64  - pixel count per shade group,
      pixel_group    (N,)  int    - shade-group id for each input pixel.
    Deterministic (lab_kmeans seeds KMeans; Agglomerative is exact). Shared by
    color_essential.universalize and the pre-compression gate / color_carried cutout.
    """
    centroids_lab, labels = lab_kmeans(opaque_rgb, k)
    ac = AgglomerativeClustering(
        n_clusters=None, distance_threshold=delta, linkage="average",
        metric="euclidean", compute_full_tree=True,
    ).fit(centroids_lab)
    shade = ac.labels_
    n_groups = int(shade.max()) + 1
    per_group_area = np.bincount(shade[labels], minlength=n_groups)
    # area-dominant centroid within each group = its "universal" colour
    counts = np.bincount(labels, minlength=centroids_lab.shape[0])
    universal_idx = np.array([
        members[int(np.argmax(counts[members]))]
        for g in range(n_groups)
        for members in (np.where(shade == g)[0],)
    ])
    universal_lab = centroids_lab[universal_idx]
    universal_rgb = (lab2rgb(universal_lab[np.newaxis, ...]) * 255.0).round() \
        .clip(0, 255).astype(np.uint8)[0]
    return universal_rgb, per_group_area.astype(np.int64), shade[labels]
