"""
TASK: Reduce an RGB image to K dominant colors (palette quantization).

WHAT IT'S DOING (concept):
Every pixel is a 3D point in color space. Summarize H*W points by K
"representatives" such that each pixel's nearest rep is close. KMeans
is the standard answer: iteratively (assign → update) finds K centroids
minimizing total squared distance from points to their nearest centroid.
Converges to a *local* minimum (not global) — hence random_state matters
even on the same input.

WHY LAB (not RGB):
RGB Euclidean distance ≠ perceived color difference. Lab is designed so
equal distance ≈ equal perceived change. Cluster in Lab → perceptually
better palettes; in RGB the centroids drift toward bright primaries
because primaries are far apart in RGB but not perceptually farther
than mid-tones.

LIBRARIES (preferred → fallback):
  sklearn.cluster.KMeans    — full control; deterministic via
                              random_state; ~ms for 1024×1024
  PIL.Image.quantize        — built-in median-cut (recursively splits
                              color box at median); faster, less control
  scipy.cluster.vq.kmeans2  — alternative; rarely needed here

API (relevant subset):
  KMeans(n_clusters, random_state, n_init=10)
    .fit(X)              X shape (N, D)
    .cluster_centers_    shape (n_clusters, D) — the palette
    .labels_             shape (N,) — cluster index per point

COLOR SPACE:
  skimage.color.rgb2lab(rgb)   # uint8 [0,255] or float [0,1] → Lab
  skimage.color.lab2rgb(lab)   # Lab → float [0,1]

STEPS:
  1. RGB → Lab
     skimage.color.rgb2lab(img)
  2. Flatten (H, W, 3) → (H*W, 3) — KMeans wants 2D point arrays
     numpy: img.reshape(-1, 3)
  3. Cluster (seed mandatory per H8)
     sklearn.cluster.KMeans(n_clusters=K, random_state=SEED).fit(pixels)
  4. Map each pixel to its centroid via fancy indexing
     km.cluster_centers_[km.labels_]
  5. Reshape → Lab → RGB → cast to input dtype

GOTCHAS:
  - random_state mandatory (H8)
  - lab2rgb returns float [0,1]; multiply by 255 + cast for uint8 output
  - K too small (≤3) → blotchy; K too large (≥32) → defeats the purpose
  - KMeans converges to local minima; n_init=10 (default) mitigates
"""
