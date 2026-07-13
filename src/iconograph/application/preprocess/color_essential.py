'''
TASK
  Collapse perceptual shade-groups in a color_essential logo to a
  single universal colour per group (chosen by area dominance).

WHAT YOU ARE WRITING
  File:        src/iconograph/application/preprocess/color_essential.py
  Position:    top-level module function, directly below this docstring
  Imports:
      import numpy as np
      from skimage.color import lab2rgb
      from sklearn.cluster import AgglomerativeClustering
      from iconograph.application.categorize import (
          Categorization,
          ALPHA_OPAQUE_THRESHOLD,
      )
      from .common import lab_kmeans
  Signature:
      def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
          """cat: stage-1 Categorization with cat.color_essential=True.
          rgba: (H, W, 4) uint8 — original RGBA from categorize._load_rgba.
          Returns: (H, W, 4) uint8 RGBA at source resolution — same shape,
          same alpha; opaque pixels are recoloured to the area-dominant
          centroid of their perceptual shade-group. The dispatcher applies
          frame_with_margin afterwards (which handles the RGBA→RGB
          composite onto white).
          """
  Called by:   src/iconograph/application/preprocess/__init__.py — preprocess()
               (when cat.color_essential is True)

CONCEPT
  Goal:    reduce a multi-hue palette to one representative per
           perceptually-similar shade family, by area dominance.
  Method:  KMeans over Lab → AgglomerativeClustering on the centroid
           set → argmax(pixel count) inside each group picks the universal.

  - K (CE_PALETTE_K=6) is matched to the expected count of dominant
    colours per fixture. KMeans is forced to consolidate AA-fringe
    pixels into the dominant clusters they border instead of giving
    them their own centroids; that prevents the fringes from chaining
    dominant centroids together at high ΔE.
  - The agglomerative metric is Lab Euclidean (ΔE76). CE_SHADE_DELTA
    controls how broad a "shade family" is — at 70.0 it merges
    Firefox's red/yellow/orange shades into one warm universal while
    keeping Chrome's red/yellow/green/blue wedges distinct (no
    bridge centroids exist to chain them).
  - "Universal" = the actual centroid colour from KMeans with the
    highest pixel count in its group — not a synthesized average.
    Preserves a colour that physically appears in the source.
  - Firefox's warm shades (yellow/orange/red/magenta) collapse to one
    warm universal; the purple-blue globe stays separate. Chrome's
    red/green/yellow/blue wedges stay distinct because the inter-hue
    Lab distance exceeds CE_SHADE_DELTA and no bridge centroids exist
    at K=6 to chain them.

WHY AGGLOMERATIVE WITH 'average' LINKAGE
  Single-link (and DBSCAN with the same eps) chain through bridge
  points: A-B-C with dist(A,B) < T and dist(B,C) < T collapses ALL
  three into one cluster even when dist(A,C) > T. For shade collapse
  that is a correctness bug — salmon and burgundy would merge through
  a coral bridge, eating the very distinction we want to preserve.
  Average linkage compares the average pairwise distance between
  members of two candidate clusters before merging, so the bridge
  never carries the merger across. AgglomerativeClustering is
  deterministic by construction (no random_state parameter) — H8
  holds without extra seeding.

WHY (centroids, labels) FROM lab_kmeans — NOT A RE-MAPPED IMAGE
  The colour we keep at the end is the area-dominant centroid in each
  shade group, not the per-pixel KMeans output. We need access to the
  raw centroid set and the per-pixel labels so we can post-process
  before re-mapping; the higher-level "give me back a quantized image"
  interface in compress/quantize.py is the wrong shape for this stage.
  lab_kmeans (in preprocess.common) exposes the model outputs directly.

WHY RGBA OUTPUT (not pre-composited RGB)
  The dispatcher already composites RGBA → RGB on white via
  frame_with_margin. Returning RGBA here matches the *current*
  pass-through behaviour and keeps the compositing concentrated at
  one location — the dispatcher's framing tail. It also preserves
  the alpha channel for any future debug inspection of this stage's
  raw output.

LIBRARIES
  Use:    common.lab_kmeans                       — over-quantize to K Lab centroids
                                                     (centroids, labels) interface
          sklearn.cluster.AgglomerativeClustering — group centroids by ΔE76
              (n_clusters=None, distance_threshold=CE_SHADE_DELTA,
               linkage='average', metric='euclidean',
               compute_full_tree=True)
          numpy.bincount                          — pixel counts per centroid
          skimage.color.lab2rgb                   — convert universals back to RGB
  Alt:    sklearn.cluster.DBSCAN                  — single-link chaining bug
                                                     (rejected; see WHY above)
          scipy.cluster.hierarchy.fcluster        — equivalent but scipy is
                                                     not declared in pyproject.toml
          KMeans with very low k                  — can't separate "shade-of"
                                                     vs "distinct-hue" without
                                                     the agglomerative threshold

STEP-BY-STEP

  1. Extract opaque pixels.
       alpha = rgba[..., 3]
       opaque = alpha > ALPHA_OPAQUE_THRESHOLD
       opaque_rgb = rgba[opaque][:, :3]   # (N, 3) uint8
     - Only opaque pixels participate so the transparent background
       doesn't pollute the palette.
     - ALPHA_OPAQUE_THRESHOLD=16 is imported from categorize; same
       constant standard.py uses (single source of truth).

  2. Over-quantize the opaque palette.
       centroids_lab, labels = lab_kmeans(opaque_rgb, CE_PALETTE_K)
     - centroids_lab: (K', 3) float64 Lab. labels: (N,) int.
       lab_kmeans clamps K' = min(CE_PALETTE_K, N) internally.

  3. Cluster centroids into shade-groups by Lab Euclidean distance.
       ac = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=CE_SHADE_DELTA,
                linkage='average',
                metric='euclidean',
                compute_full_tree=True,
            ).fit(centroids_lab)
       shade_labels = ac.labels_   # (K',)
     - distance_threshold is exclusive (sklearn issue #24176):
       merges happen only when pairwise distance < T. If you tune T
       at a boundary value, expect "off by an epsilon" behavior.
     - compute_full_tree=True is REQUIRED whenever n_clusters=None +
       distance_threshold is set.

  4. Pick the area-dominant centroid per shade-group.
       counts = np.bincount(labels, minlength=centroids_lab.shape[0])
       n_groups = int(shade_labels.max()) + 1
       universal_idx = np.zeros(n_groups, dtype=np.int32)
       for g in range(n_groups):
           members = np.where(shade_labels == g)[0]
           universal_idx[g] = members[int(np.argmax(counts[members]))]
       universal_lab = centroids_lab[universal_idx]   # (n_groups, 3)
     - The loop is over n_groups (≤ K' ≤ 24), not pixel-sized. A pure
       vectorized version exists (segmented argmax) but is harder to
       read; the loop is fine at this scale.

  5. Convert universals back to uint8 RGB.
       universal_rgb = (
           (lab2rgb(universal_lab[np.newaxis, ...]) * 255.0)
           .round()
           .clip(0, 255)
           .astype(np.uint8)[0]
       )   # (n_groups, 3) uint8
     - lab2rgb expects (..., 3) with at least one spatial dim, so we
       add a length-1 row axis and strip it back out.
     - Multiply, round, clip, cast — same idiom as quantize.py.

  6. Two-hop fancy index to re-map opaque pixels.
       shade_per_pixel     = shade_labels[labels]              # (N,)
       universal_per_pixel = universal_rgb[shade_per_pixel]    # (N, 3) uint8
     - Two-stage lookup: pixel → centroid → group representative.
       This is the same idiom quantize.py uses at line 115, applied
       once for each hop.

  7. Write back into the output RGBA.
       out = rgba.copy()
       out[opaque, :3] = universal_per_pixel
       return out
     - Alpha untouched. Transparent pixels' RGB stays as-is — they
       become white when frame_with_margin composites RGBA→RGB
       downstream. We never write to those pixels here.

GOTCHAS
  - distance_threshold semantics: ΔE76 (Lab Euclidean), not ΔE2000.
    At K=6 the centroids are stable averages of large regions so
    ΔE76 is sufficient; CIEDE2000 is unnecessary complexity here.
  - lab2rgb returns float in [0, 1] — multiply, round, clip, cast.
    Without the clip, occasional sub-zero or > 1.0 values from
    out-of-gamut Lab centroids will overflow uint8.
  - shade_labels are NOT guaranteed contiguous-from-zero in
    pathological cases (single-element groups landing at high
    indices); `int(max()) + 1` upper bound + members-via-`np.where`
    is robust to gaps.
  - Determinism (H8): KMeans is seeded inside lab_kmeans;
    AgglomerativeClustering has no random state. Two calls with the
    same RGBA input produce byte-identical output.
  - K' clamping happens inside lab_kmeans, so this branch does NOT
    need to defend against tiny opaque regions. categorize already
    requires ≥ 1000 opaque pixels to set color_essential=True
    (categorize.py:130), so K=6 is always safe in the H9 set.
  - CE_PALETTE_K and CE_SHADE_DELTA are JOINTLY calibrated. Raising
    K above ~8 reintroduces AA-fringe centroids that bridge
    dominant centroids together via average linkage — at ΔE=70
    Chrome's white ring then absorbs the green and yellow wedges.
    Conversely, lowering ΔE below ~60 stops Firefox's warms from
    merging into one universal. The current pair (K=6, ΔE=70) has
    ~10 units of slack on either side of ΔE before either fixture
    breaks.
  - This function MUST be called only for cat.color_essential=True.
    The dispatcher in preprocess/__init__.py routes accordingly;
    no defensive check is added inside.

─────────────────────────────────────────────────────────────────────

TASK
  Filter the framed RGB to identity-bearing colours by area dominance
  against the total canvas, and emit per-colour binary masks for
  downstream stages.

WHAT YOU ARE WRITING
  File:        src/iconograph/application/preprocess/color_essential.py
  Position:    top-level module function, directly below `universalize`
  Imports:
      import numpy as np      # already imported at file top
  Signature:
      def separate_features(
          rgb: np.ndarray,
      ) -> tuple[np.ndarray, list[tuple[tuple[int, int, int], np.ndarray]]]:
          """rgb: (H, W, 3) uint8 — framed RGB (post-frame_with_margin)
          with white as negative space.
          Returns:
            cleaned:  (H, W, 3) uint8 — input with sub-threshold colours
                      replaced by white.
            features: list of (colour, mask) pairs, one per surviving
                      distinct colour. colour: (R, G, B) int tuple.
                      mask: (H, W) bool — True where the colour appears
                      in `cleaned`.
          """
  Called by:   TBD — Phase 2 wiring is deferred. Exposed as a standalone
               primitive in color_essential.py; future callers in
               refine/, monogram/, or fit/ will consume the features.

CONCEPT
  Goal:    isolate the colour-essential logo's identity-bearing
           features by area dominance against the framed canvas.
  Method:  unique-row count → fraction-of-canvas filter →
           per-colour boolean mask.

  - "Distinct colour" = a unique RGB triple in the input.
  - "Negative space" = pure white (255, 255, 255). Excluded from the
    distinct-colour list and the feature output. NOT subtracted from
    the denominator — the denominator is the whole canvas.
  - "Total canvas area" = H * W (rgb.shape[0] * rgb.shape[1]). Includes
    white pixels in the count.
  - "Sub-threshold" = area / total_canvas ≤ CE_FEATURE_FLOOR (0.15).
    These colours are not identity-bearing; their pixels are repainted
    white.
  - Order of `features` is deterministic: lexicographic on (R, G, B) as
    returned by `np.unique(axis=0)`. Re-runs are byte-identical (H8).

WHY % OF TOTAL CANVAS (not % of positive area)
  Measured against positive (non-white) area, Chrome's inner blue
  disc lands at 17.2 % — above a 15 % floor, so the disc would survive
  even though it is the structural "inner element" we want dropped.
  Measured against the total canvas (mostly white margin + ring), the
  same disc is 12.3 %, comfortably under the floor.

  The canvas denominator is also intrinsic: it does NOT depend on the
  logo's fill ratio inside the frame. A logo that fills 90 % of the
  canvas and one that fills 50 % can be compared on the same scale.
  Positive area shifts with fill — a tightly framed logo has higher
  positive-fractions for every colour than a sparsely framed one even
  when the viewer sees the same proportion of the canvas painted.

WHY POST-FRAME_WITH_MARGIN INPUT (not pre-framing RGBA)
  The "white = negative space" semantics only hold *after* framing:
  frame_with_margin is what composites RGBA onto a white canvas, and
  what guarantees the negative region is pure (255, 255, 255). On the
  RGBA upstream of framing, transparent pixels carry arbitrary RGB
  (often zeros from PIL), so "white" wouldn't be the right marker.
  The user's stated rule — "white is excluded" — is what fixes this
  function's input contract at the framed RGB.

WHY ONE FUNCTION (not three)
  Counting, demoting, and mask-emission share state (the unique-colour
  array, the keep mask). Splitting them would force re-traversing
  the image multiple times or threading the colour table through
  three call sites. The user asked for one function with these four
  effects; the brief honours that.

LIBRARIES
  Use:    numpy.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
              — unique rows + per-row counts in one pass
          numpy.all(rgb == colour, axis=-1)
              — per-pixel match for a specific colour → (H, W) bool
          ndarray boolean indexing assignment
              — `cleaned[mask] = WHITE` paints sub-threshold pixels
                white in place
  Alt:    PIL.Image.getcolors      — caps at 256 colours and returns
                                      [(count, colour), ...]; the numpy
                                      output is directly indexable and
                                      matches repo convention (H2)
          cv2.inRange              — returns uint8 0/255; needs extra
                                      cast vs. numpy's native bool
          scipy.ndimage.label      — labels spatial CCs; wrong primitive
                                      (we want colour-membership, not
                                      connected components)

STEP-BY-STEP

  1. Find unique RGB triples + their pixel counts.
       flat = rgb.reshape(-1, 3)
       colours, counts = np.unique(flat, axis=0, return_counts=True)
     - colours: (M, 3) uint8 — every unique colour (including white).
     - counts:  (M,)   int   — pixel count per unique colour.
     - axis=0 is mandatory — without it, np.unique flattens to scalars
       and counts individual byte values, not triples.

  2. Drop the white row from the candidate list (NOT from the denominator).
       WHITE = np.array([255, 255, 255], dtype=np.uint8)
       non_white = ~(colours == WHITE).all(axis=1)
       colours = colours[non_white]
       counts  = counts[non_white]
     - Element-wise compare colours against WHITE; reduce on the
       colour axis (axis=1, the (R, G, B) row); invert to keep
       non-white rows. White's pixel count is discarded from the
       candidate list but remains implicit in the H * W denominator
       computed in step 3.

  3. Compute per-colour fractions against total canvas; identify survivors.
       total_canvas = rgb.shape[0] * rgb.shape[1]
       fractions = counts / total_canvas
       keep = fractions > CE_FEATURE_FLOOR
     - No guard needed — total_canvas is always > 0 for a valid image.
     - Strict `>` matches the user's "≤ 15 % is removed" rule: a colour
       at exactly 15.0 % is demoted.

  4. Build `cleaned` by painting demoted colours white.
       cleaned = rgb.copy()
       for colour in colours[~keep]:
           cleaned[(rgb == colour).all(axis=-1)] = WHITE
     - Loop is over demoted colours (typically a handful); each
       iteration is one vectorized comparison + one boolean assignment.
       Not a per-pixel hot loop.
     - Read the membership mask from `rgb` (the input) rather than
       `cleaned`; reading from `cleaned` mid-loop would race with the
       in-place repaint above.

  5. Emit (colour, mask) pairs for survivors.
       features: list[tuple[tuple[int, int, int], np.ndarray]] = []
       for colour in colours[keep]:
           mask = (cleaned == colour).all(axis=-1)
           features.append((tuple(int(c) for c in colour), mask))
     - Read membership from `cleaned`, not `rgb` — after step 4,
       demoted pixels are white in `cleaned`, so the survivor mask
       cannot accidentally include them.
     - colour is a tuple of Python ints — hashable, dict-keyable,
       logging-friendly. Raw (3,) ndarrays are not hashable.

  6. Return the pair.
       return cleaned, features

GOTCHAS
  - Strict-`>` threshold: colour at exactly 15 % is demoted, not kept.
    If a fixture lands precisely on the boundary and that reads wrong,
    consider lifting to `>=` rather than changing CE_FEATURE_FLOOR.
  - White is hardcoded to (255, 255, 255). This matches
    frame_with_margin's canvas convention; if the canvas colour ever
    changes (e.g. transparent-PNG output), lift WHITE to a module
    constant shared with common.py.
  - Internal white features are conflated with the canvas (e.g.
    Chrome's thin inner ring). The "white is excluded" rule accepts
    this trade.
  - The 15 %-of-canvas rule drops both Chrome's inner blue disc
    (~12.3 % of canvas) and Firefox's inner purple globe (~12.1 %
    of canvas). The rule treats inner-element-style features
    symmetrically across both color_essential fixtures. If a future
    fixture has an identity-bearing inner element under the floor,
    the fix is structural (intersect with Categorization's inner_mask)
    rather than raising CE_FEATURE_FLOOR (which would defeat Chrome's
    drop).
  - `features` order is lexicographic (R, G, B), stable across runs.
    If a caller needs area-descending order, sort on counts at the
    call site:
        features.sort(key=lambda cm: -cm[1].sum())
  - mask dtype is bool, NOT uint8. cv2 callers must cast
    (`mask.astype(np.uint8)`) if needed.
  - Boolean assignment `cleaned[mask] = WHITE` requires `cleaned` to
    be C-contiguous. `rgb.copy()` returns a C-contiguous array; do
    not skip the copy or operate on a view / slice.
  - The function does NOT validate cat.color_essential — it is a pure
    image-in / image-out helper. The dispatcher guarantees only
    color_essential paths reach it (once Phase 2 wiring lands).
'''

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
