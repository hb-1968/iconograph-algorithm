'''
TASK
  Rotate an icon to maximize axis-alignment of its dominant straight features.

WHAT YOU ARE WRITING
  File:        src/iconograph/application/compress/align.py
  Position:    top-level module function, directly below this docstring
  Imports:
      import numpy as np
      import cv2
      from skimage.transform import hough_line, hough_line_peaks
  Signature:
      def align(img: np.ndarray) -> np.ndarray:
          """img: (S, S, 3) uint8 RGB from preprocess() — white background,
          content centered. Returns: (S, S, 3) uint8 RGB rotated so the
          dominant straight features land on the horizontal/vertical axes,
          with exposed corners filled white.
          """
  Called by:   src/iconograph/application/compress/__init__.py — re-export
                 (added in a follow-up turn, after this function body lands)
               src/iconograph/testbed.py — run_align_testbed
                 (added in a follow-up turn; mirrors run_preprocess_testbed)
               src/iconograph/application/pipeline.py — stage 3 orchestrator
                 (TBD; not yet wired)

CONCEPT
  Goal:    rotate so the dominant straight features land on 0° or 90°.
  Method:  Hough line transform ONCE on the edge map, then sweep candidate
           rotations α and score each by axis-aligned vote weight; argmax
           gives the rotation to apply.

  - A line at angle θ in the original image is at angle θ + α after
    rotating the image by α. So you do NOT need to re-run Hough at every
    candidate α — detect lines once, then add α to each detected θ
    arithmetically and score.
  - Axis-alignment is periodic with period 90° (horizontal and vertical
    are both "on the grid"), so the irreducible sweep range is [-45°, 45°).
    Anything outside is a re-labeling of an angle inside.
  - The score is a soft, magnitude-weighted measure: each detected line
    contributes its Hough vote weight, multiplied by how close it would
    land to a grid axis after rotation. It is not a hard threshold —
    smooth scoring yields a unimodal argmax with no ties on the sweep.

WHY HOUGH-ONCE-THEN-SWEEP-θ (not detect-per-rotation)
  Re-running edge detection and Hough at each of 181 candidate angles is
  O(N) Hough calls — hundreds of milliseconds. Detecting lines once and
  sweeping their angles arithmetically is O(1) Hough plus a vectorized
  numpy product over (α, θ_i) — microseconds. Both produce the same answer
  because rotation preserves which pixels lie on which line; only the
  line's θ parameter shifts by α.

LIBRARIES
  Use:    cv2.cvtColor                          — RGB → grayscale (luminance)
          cv2.Canny                             — edge map (single-pixel ridges)
          skimage.transform.hough_line          — full accumulator + θ axis
          skimage.transform.hough_line_peaks    — peak (vote_weight, θ_i) pairs
          cv2.getRotationMatrix2D               — build 2×3 affine matrix
          cv2.warpAffine                        — apply rotation + border fill
  Alt:    cv2.HoughLines                        — discards per-peak vote weight
                                                  in the Python binding;
                                                  unusable for weighted scoring
          cv2.HoughLinesP                       — returns segments (x1,y1,x2,y2),
                                                  not (θ, weight); wrong shape
                                                  for the score function
          skimage.transform.rotate              — cval is a scalar (no RGB tuple);
                                                  float-roundtrip overhead
          cv2.minAreaRect                       — bounding-box orientation, not
                                                  line-feature orientation;
                                                  unreliable for asymmetric or
                                                  curved logos

STEP-BY-STEP

  1. Grayscale.
       gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     - COLOR_RGB2GRAY (not BGR2GRAY) — preprocess emits RGB channel order;
       BGR2GRAY would swap R and B luminance weights and give a different
       gray on saturated-color logos.
     - Result: (S, S) uint8. Channel dim is dropped, not set to 1.

  2. Edge detection.
       edges = cv2.Canny(gray, threshold1=50, threshold2=150)
     - Returns (S, S) uint8 with values strictly 0 or 255 — already binary.
     - Defaults are non-critical here because preprocess output is flat-color
       on white; edges are sharp regardless of threshold within the 1:2 to
       1:3 ratio range.

  3. Hough line transform — full accumulator.
       hspace, theta_axis, rho_axis = hough_line(edges)
     - hspace is the (ρ × θ) accumulator, dtype uint64.
     - theta_axis is radians in [-π/2, π/2). 180 bins by default
       (1° resolution) — sufficient for the score function.
     - hough_line accepts any nonzero pixel as an edge; the uint8 {0, 255}
       output of Canny goes in directly with no cast.

  4. Extract peaks with vote weights.
       accum, peak_angles, _ = hough_line_peaks(
           hspace, theta_axis, rho_axis,
           threshold=0.2 * hspace.max(),
           num_peaks=200,
       )
     - accum shape (K,) uint64 — vote weight per detected line.
     - peak_angles shape (K,) — radians in [-π/2, π/2).
     - 0.2 * max is empirically better than the skimage default of 0.5 * max
       for icons with short or thin features (the default returns too few
       peaks on sparse logos).
     - If K == 0 (no peaks): this is a curve-only logo (Apple- / Edge-style).
       Return img.copy() immediately — degeneracy gate 1.

  5. Sweep α and score.
       alphas = np.deg2rad(np.arange(-45.0, 45.0, 0.5))           # (181,)
       rotated = peak_angles[None, :] + alphas[:, None]            # (181, K)
       residual = ((rotated + np.pi / 4) % (np.pi / 2)) - np.pi / 4
       weights = np.cos(2 * residual) ** 2                         # (181, K)
       scores = (weights * accum[None, :]).sum(axis=1)             # (181,)
       alpha_best = alphas[int(np.argmax(scores))]
     - residual ∈ [-π/4, π/4] is the signed distance to the nearest 0°/90°
       axis after rotating that line by α.
     - cos²(2r) peaks at r=0 (perfectly axis-aligned) and falls smoothly
       to 0 at r=±π/4 (perfect diagonal).
     - np.argmax is deterministic — returns the first occurrence on ties.
     - Score curve is unimodal in practice; no coarse-then-fine pass needed.

  6. Degeneracy gates (return identity).
       if scores.max() < 0.05 * float(accum.sum()):
           return img.copy()                  # gate 2: no clear winner
       if abs(np.rad2deg(alpha_best)) < 0.25:
           return img.copy()                  # gate 3: already aligned
     - Gate 1 (K == 0) is handled in step 4.
     - Gate 2 catches logos where every α scores roughly the same —
       45°-diagonal-dominated logos (vscode.png), symmetric stars,
       circle-only fixtures whose few weak peaks cancel.
     - Gate 3 avoids burning a resampling pass on sub-step rotation.

  7. Rotate.
       alpha_deg = float(np.rad2deg(alpha_best))
       S = img.shape[0]
       M = cv2.getRotationMatrix2D(
               center=(S / 2.0, S / 2.0),
               angle=alpha_deg,
               scale=1.0,
           )
       return cv2.warpAffine(
           img, M, (S, S),
           flags=cv2.INTER_LINEAR,
           borderMode=cv2.BORDER_CONSTANT,
           borderValue=(255, 255, 255),
       )
     - cv2's angle is counter-clockwise positive.
     - center is (x, y) i.e. (col, row), not (row, col).
     - dsize is (width, height), not (height, width). For a square image
       both are S.
     - INTER_LINEAR — NOT INTER_CUBIC. The image goes straight into quantize
       which re-discretizes colors; cubic ringing creates intermediate
       palette entries that don't survive quantization cleanly.
     - borderValue MUST be the 3-tuple (255, 255, 255). Passing just `255`
       silently broadcasts wrong for a 3-channel image.

GOTCHAS
  - hough_line's θ range is [-π/2, π/2); the residual math must wrap modulo
    π/2. The `+ π/4` offset before the mod centers the residual on zero so
    cos²(2r) is symmetric.
  - cv2 vs Hough angle conventions differ: Hough θ is the angle of the
    NORMAL to each line; cv2.warpAffine's angle rotates the image plane.
    The sign relating the two MUST be verified empirically on a deliberately
    tilted fixture before the algorithm is trusted on production fixtures —
    rotate gmail.png by +10° in a paint tool, run align(), expect it to
    return alpha_deg ≈ +10 (or -10, depending on convention) and restore
    the original orientation.
  - Curve-only fixtures (microsoft-edge.png swoosh, Apple-rainbow-style)
    MUST hit gate 1 — zero detected lines, return identity. This is the
    expected behavior, not a bug.
  - 45°-diagonal-dominated fixtures (vscode.png) score cos²(2·±π/4) = 0
    across all α — gate 2 (low-score) returns identity. Also expected.
  - All ops are deterministic for fixed input — no random_state needed
    (unlike quantize.py). Reruns are byte-identical without seeding.
'''

import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks

from iconograph.application.common import frame_with_margin


def align(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    hspace, theta_axis, rho_axis = hough_line(edges)
    accum, peak_angles, _ = hough_line_peaks(
        hspace, theta_axis, rho_axis,
        threshold=0.25 * hspace.max(),
        num_peaks=200,
    )

    alphas = np.deg2rad(np.arange(-45.0, 45.0, 0.5))
    rotated = peak_angles[None, :] + alphas[:, None]
    residual = ((rotated + np.pi / 4) % (np.pi / 2)) - np.pi / 4
    weights = np.cos(2*residual) ** 2
    scores = (weights * accum[None, :]).sum(axis=1)
    alpha_best = alphas[int(np.argmax(scores))]

    if scores.max() < 0.05 * float(accum.sum()) or abs(np.rad2deg(alpha_best)) < 0.25:
        return img.copy()
    
    alpha_deg = float(np.rad2deg(alpha_best))
    S = img.shape[0]
    M = cv2.getRotationMatrix2D(
        center=(S / 2.0, S / 2.0),
        angle=alpha_deg,
        scale=1.0,
    )

    rotated = cv2.warpAffine(
        img, M, (S, S),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return frame_with_margin(rotated)