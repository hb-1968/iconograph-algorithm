'''
TASK
  Reduce an RGB image to K dominant colors (palette quantization).

WHAT YOU ARE WRITING
  File:        src/iconograph/application/compress/quantize.py
  Position:    top-level module function, directly below this docstring
  Imports:
      import numpy as np
      from sklearn.cluster import KMeans
      from skimage import color
  Signature:
      def quantize(img: np.ndarray, k: int) -> np.ndarray:
          """img: (H, W, 3) uint8 RGB. k: number of palette colors.
          Returns: (H, W, 3) uint8 RGB with at most k unique colors.
          """
  Called by:   src/iconograph/application/pipeline.py — stage 3 driver
               (TBD — stage 3 orchestrator not yet wired; per CLAUDE.md §3
                this is the single shared compression stage)

CONCEPT
  Goal:    pick k colors that best represent every pixel of the image.
  Method:  KMeans clustering on pixels in Lab color space.

  - Each pixel is a 3D point in color space.
  - KMeans finds k centroids that minimize the total squared
    distance from every pixel to its nearest centroid.
  - The k centroids ARE the palette; replacing each pixel with
    its nearest centroid is what quantizes the image.
  - KMeans converges to a *local* minimum, not a global one —
    the seed matters even on identical input (per H8).

WHY LAB (not RGB)
  Equal RGB distance does NOT mean equal perceived color change —
  RGB is uniform per channel, not per perception. Lab is designed
  so equal Lab distance ≈ equal perceived difference. Clustering
  in Lab gives visually balanced palettes; in RGB the centroids
  drift toward bright primaries because primaries are far apart
  in RGB but not perceptually farther than mid-tones.

LIBRARIES
  Use:    sklearn.cluster.KMeans     — clustering; deterministic via
                                       random_state; ~ms for 1024×1024
          skimage.color.rgb2lab      — RGB → Lab (perceptual space)
          skimage.color.lab2rgb      — Lab → RGB (returns float [0,1])
  Alt:    PIL.Image.quantize         — built-in median-cut (recursively
                                       splits color box at median);
                                       faster, less control
          scipy.cluster.vq.kmeans2   — alternative; rarely needed here

STEP-BY-STEP

  1. Convert RGB → Lab.
       lab = skimage.color.rgb2lab(img)
     - rgb2lab accepts uint8 [0,255] or float [0,1] input.
     - Returns float Lab. Shape unchanged: (H, W, 3).

  2. Flatten the image to a list of points.
       pixels = lab.reshape(-1, 3)
     - Shape (H, W, 3) → (H*W, 3). KMeans expects a 2D array of
       points, not an image.

  3. Run KMeans on the pixel cloud.
       km = sklearn.cluster.KMeans(
                n_clusters=k,
                random_state=SEED,
                n_init=10,
            ).fit(pixels)
     - random_state is REQUIRED for determinism (H8).
     - n_init=10 runs the algorithm 10 times from different
       starting centroids and keeps the best result —
       mitigates the local-minimum problem.
     - After .fit(), the model exposes:
         km.cluster_centers_   shape (k, 3)  — the palette
         km.labels_            shape (H*W,)  — cluster index per pixel

  4. Replace each pixel with its centroid.
       quantized_lab = km.cluster_centers_[km.labels_]
     - "Fancy indexing": for each label in km.labels_, look up the
       corresponding centroid in km.cluster_centers_.
     - Output shape (H*W, 3) in Lab.

  5. Reshape and convert back to uint8 RGB.
       quantized_lab = quantized_lab.reshape(lab.shape)
       rgb = skimage.color.lab2rgb(quantized_lab)
       return (rgb * 255).astype(np.uint8)
     - lab2rgb returns float in [0, 1] — multiply by 255 and cast
       to uint8 to match the input dtype.

GOTCHAS
  - random_state mandatory (H8 determinism).
  - lab2rgb returns float [0, 1]; ×255 + cast for uint8 output.
  - k too small (≤ 3) → blotchy; k too large (≥ 32) → defeats
    the point of quantization.
  - KMeans converges to local minima; n_init=10 (the default)
    mitigates but does not eliminate this.
'''

import numpy as np
from sklearn.cluster import KMeans
from skimage import color

def quantize(img: np.ndarray, k: int) -> np.ndarray:
    SEED = 0

    lab = color.rgb2lab(img)
    pixels = lab.reshape(-1, 3)

    km = KMeans(
                n_clusters=k,
                random_state=SEED,
                n_init=10,
            ).fit(pixels)

    quantized_lab = km.cluster_centers_[km.labels_]
    quantized_lab = quantized_lab.reshape(lab.shape)
    rgb = color.lab2rgb(quantized_lab)

    return (rgb * 255).astype(np.uint8)