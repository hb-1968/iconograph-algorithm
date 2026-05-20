'''
TASK
  Render a boolean foreground mask as a black-on-white RGB image.

WHAT YOU ARE WRITING
  File:        src/iconograph/application/preprocess/common.py
  Position:    top-level module function, directly below this docstring
  Imports:
      import numpy as np
  Signature:
      def compose_bw(mask: np.ndarray) -> np.ndarray:
          """mask: (H, W) bool — True where foreground.
          Returns: (H, W, 3) uint8 RGB — (0, 0, 0) where mask is True,
          (255, 255, 255) elsewhere.
          """
  Called by:   src/iconograph/application/preprocess/standard.py  — universalize
               src/iconograph/application/preprocess/inverted.py  — universalize

CONCEPT
  Goal:    paint the foreground mask as black ink on a white canvas.
  Method:  numpy broadcast — where mask is True write (0,0,0), else (255,255,255).

  - The mask is a 2-D bool array; the output is a 3-channel uint8 RGB.
  - Output has exactly two unique pixel values: pure black and pure white.
  - Deterministic by construction — no randomness, no library state.
  - Pure function; safe to call repeatedly in the testbed / cache writer.

WHY (H, W, 3) RGB INSTEAD OF (H, W) BINARY
  Downstream stages (compress, refine, fit) all accept (H, W, 3) uint8
  RGB as their image dtype — stage 3's quantize() is already typed this
  way in src/iconograph/application/compress/quantize.py. Emitting a
  binary (H, W) here would force every consumer to widen to RGB,
  duplicating that boilerplate at three+ call sites. Widening at the
  source keeps the pipeline contract uniform: every stage hands the
  next stage an (H, W, 3) uint8 image (except color-essential pass-
  through, which remains RGBA — see preprocess/__init__.py).

LIBRARIES
  Use:    numpy.full              — allocate an (H, W, 3) uint8 canvas
                                     pre-filled with 255 (white) in one
                                     pass; dtype is locked at allocation
          ndarray boolean idx     — out[mask] = 0 paints the foreground
                                     in-place; broadcasts the scalar 0
                                     across all 3 channels of the rows
                                     selected by the (H, W) bool mask
  Alt:    numpy.where(cond, 0, 255)
                                  — does NOT work here as a one-liner.
                                     np.where takes its shape from cond:
                                     mask[..., np.newaxis] gives (H, W, 1)
                                     and the result stays (H, W, 1) —
                                     scalar x/y do not promote it to
                                     (H, W, 3). Forcing the third axis
                                     needs x and y to be 3-element arrays
                                     (np.array([0,0,0]), np.array([255,255,255])),
                                     at which point np.full is simpler.
          PIL.Image.fromarray     — not relevant here; PIL handles
                                     serialization (ndarray ↔ image),
                                     not per-pixel value selection
          skimage                 — no equivalent primitive

STEP-BY-STEP

  1. Allocate the white canvas in the final (H, W, 3) uint8 shape.
       out = np.full(mask.shape + (3,), 255, dtype=np.uint8)
     - mask.shape is (H, W); mask.shape + (3,) is (H, W, 3).
     - np.full(shape, fill, dtype) makes a fresh array — no broadcast
       surprises, no int64 widening. The dtype is correct from the
       first byte, which matters because the cache PNG and stage-3
       input both read this same array.

  2. Paint the foreground in-place with boolean indexing.
       out[mask] = 0
     - out[mask] selects the rows of `out` where `mask` is True; each
       selected row has shape (3,). Assigning scalar 0 broadcasts to
       (3,) so all three channels become 0 — that's pure black.
     - In-place write; no second allocation. Returns nothing; mutates
       `out` directly.

  3. Return the painted canvas.
       return out
     - dtype already uint8, shape already (H, W, 3) — no cast needed.

GOTCHAS
  - mask MUST be 2-D bool. (H, W, 1) or float masks silently produce
    wrong-shape or wrong-value output via numpy broadcasting rules.
  - The implicit color convention here is "ink = black, canvas = white."
    Inverted-branch callers do NOT pre-invert the mask — they pass
    cat.inner_mask directly, since stage 1 already extracted the
    originally-light pixels as True in that mask.
  - Determinism (H8) holds by construction: no random state, no library
    state, no thread-local. Two calls with the same input produce
    byte-identical output.
  - ALPHA_OPAQUE_THRESHOLD is NOT redefined here. The constant lives
    in src/iconograph/application/categorize.py:62 and is imported
    by preprocess/standard.py from there. common.py is the rendering
    boundary; the constants boundary stays in stage 1.

─────────────────────────────────────────────────────────────────────

TASK
  Over-quantize a flat list of RGB pixels into K Lab centroids,
  returning the centroids and per-pixel labels.

WHAT YOU ARE WRITING
  File:        src/iconograph/application/preprocess/common.py
  Position:    top-level module function, after frame_with_margin
  Imports:
      import warnings
      import numpy as np
      from skimage.color import rgb2lab
      from sklearn.cluster import KMeans
      from sklearn.exceptions import ConvergenceWarning
  Signature:
      def lab_kmeans(
          pixels_rgb: np.ndarray, k: int
      ) -> tuple[np.ndarray, np.ndarray]:
          """pixels_rgb: (N, 3) uint8 RGB — flat list of pixels.
          k: target cluster count (clamped internally to min(k, N)).
          Returns:
            centroids: (k', 3) float64 Lab — k' = min(k, N).
            labels:    (N,) int — cluster index per input pixel.
          """
  Called by:   src/iconograph/application/preprocess/color_essential.py
                — universalize (over-quantize the opaque palette before
                  perceptual-grouping by ΔE76)

CONCEPT
  Goal:    decompose a pixel cloud into K colour centroids in a
           perceptually-uniform space, exposing the model outputs
           (centroids + labels) instead of a re-mapped image.
  Method:  rgb2lab → KMeans(n_clusters=k, random_state=0, n_init=10).

  - The input is already flat (N, 3). skimage.color.rgb2lab accepts
    this shape directly (it treats it as (N, 1, 3) under the hood).
  - The output is the *raw* KMeans model state: cluster_centers_ and
    labels_. No back-conversion to RGB — that is the caller's job
    because callers may post-process the centroids first (group them,
    swap them for area-dominant peers, etc.).
  - Determinism (H8): seed = 0 across the project; matches
    categorize._KM_SEED (categorize.py:65) and compress.quantize's
    random_state.

WHY EXPOSE (centroids, labels) — NOT A RE-MAPPED IMAGE
  compress/quantize.py already does the full round-trip and returns
  uint8 RGB; that interface is right for Stage 3 where the palette is
  finalized in one shot. color_essential.universalize, by contrast,
  post-processes the centroids (clusters them by Lab ΔE76 and picks
  one per group by area) before re-mapping. It needs the model
  outputs, not a finished image. Lifting this intermediate-output
  helper to common.py keeps the primitive reusable for any future
  preprocess branch that does palette analysis (per H5 — reuse across
  branches; do not re-solve).

LIBRARIES
  Use:    skimage.color.rgb2lab            — RGB → Lab; perceptually uniform
          sklearn.cluster.KMeans           — clustering; seeded for H8
          sklearn.exceptions.ConvergenceWarning
                                            — silenced via warnings.catch_warnings
                                              (same intent as
                                              categorize._silence_kmeans_convergence,
                                              lines 25-29)
  Alt:    PIL.Image.quantize               — median-cut, no Lab option;
                                              wrong granularity for shade analysis
          sklearn.cluster.MiniBatchKMeans  — faster on huge inputs; not needed
                                              at fixture resolution

STEP-BY-STEP

  1. Clamp k to the number of input pixels.
       k_eff = min(k, pixels_rgb.shape[0])
     - KMeans requires k ≤ N. Tiny opaque regions are unlikely
       (categorize._color_essential floors at 1000 px) but the clamp
       is cheap insurance and keeps the function safe to call with
       small inputs in future contexts.

  2. Convert flat RGB → flat Lab.
       lab = rgb2lab(pixels_rgb.astype(np.float32) / 255.0)
     - rgb2lab accepts (N, 3) and returns (N, 3) float64. The input
       dtype matters: uint8 [0, 255] also works, but explicit float
       avoids skimage's per-call dtype check.

  3. Fit KMeans inside a local convergence-warning silencer.
       with warnings.catch_warnings():
           warnings.simplefilter("ignore", category=ConvergenceWarning)
           km = KMeans(
               n_clusters=k_eff,
               n_init=10,
               random_state=0,
           ).fit(lab)
     - n_init=10 is the project convention; mitigates the local-
       minimum problem.
     - random_state=0 is the project convention for H8 determinism.
     - The warning silencer mirrors categorize.py's
       _silence_kmeans_convergence intent. It's inlined here rather
       than imported to avoid pulling categorize internals across
       the stage boundary; the duplication is three lines.

  4. Return centroids + labels.
       return km.cluster_centers_, km.labels_
     - cluster_centers_: (k_eff, 3) float64 Lab.
     - labels_:          (N,)       int — one index per input pixel.

GOTCHAS
  - ConvergenceWarning is silenced locally. KMeans emits it on small
    or near-singular pixel clouds; that warning is noise in this
    pipeline because we always have far more pixels than centroids.
  - rgb2lab on a (N, 3) input is supported in skimage ≥ 0.19. The
    explicit float32 / 255.0 normalization avoids skimage emitting
    "dtype check" warnings on uint8 input.
  - Determinism: seed=0, n_init=10 are NOT exposed as parameters —
    they're project-wide constants. If a future caller needs a
    different seed, lift them to module-level constants here first.
  - This function does NOT convert Lab back to RGB. Callers that
    need a recoloured image must call lab2rgb on the centroids
    themselves.
'''

import warnings

import cv2
import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning


def compose_bw(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape + (3,), 255, dtype=np.uint8)
    out[mask] = 0
    return out


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


def lab_kmeans(
    pixels_rgb: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    k_eff = min(k, pixels_rgb.shape[0])
    lab = rgb2lab(pixels_rgb.astype(np.float32) / 255.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        km = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit(lab)
    return km.cluster_centers_, km.labels_