'''
TASK
  Dispatch a Categorization + RGBA pair to the right stage-2 branch and
  finalize the canvas via frame_with_margin so every output has the
  same square, white-background, 10%-padded contract.

WHAT YOU ARE WRITING
  File:        src/iconograph/application/preprocess/__init__.py
  Position:    top-level module function, directly below this docstring
  Imports:
      import numpy as np
      from iconograph.application.categorize import Categorization
      from . import standard, inverted, color_essential
      from .common import frame_with_margin
  Signature:
      def preprocess(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
          """cat: stage-1 output. rgba: (H, W, 4) uint8 from
          categorize._load_rgba.
          Returns: (S, S, 3) uint8 RGB — square canvas, white background,
          content centered with 10% margin per side relative to the
          larger content axis.
          """
  Called by:   src/iconograph/application/pipeline.py — stage 2 driver
               (TBD — application orchestrator not yet wired; per
                CLAUDE.md §3 this is the universalization stage feeding
                stage 3 compression)
               src/iconograph/testbed.py — run_preprocess_testbed (TBD)

CONCEPT
  Goal:    one entrypoint per fixture; route on Categorization fields;
           every output is a uniform (S, S, 3) uint8 square RGB.
  Method:  three-way switch on (color_essential, routing) →
           per-branch mask/RGBA selection → frame_with_margin tail.

  - color_essential=True (Chrome, Firefox) →
    color_essential.universalize(cat, rgba) → frame_with_margin.
    The branch collapses perceptual shade-groups in the opaque
    palette to one universal colour per group (picked by area
    dominance); the RGBA output is then composited onto white inside
    frame_with_margin (no transparent regions survive).
  - routing="inverted" → inverted.universalize(cat, rgba) → frame.
  - routing in {"standard", "burst"} → standard.universalize(cat, rgba)
    → frame. Burst (Claude only, non-color-essential) folds into the
    standard branch — at stage 2 the mask-selection logic is identical
    to any inner-precedence compound logo.

WHY COLOR-ESSENTIAL IS NOW SHADE-COLLAPSED (no longer pass-through)
  The earlier dispatcher framed the source RGBA unchanged, letting the
  multi-hue palette flow into Stage 3 verbatim. That is too lossy on
  Firefox (4-5 warm shades on the flame should collapse to one warm
  universal so Stage 3 doesn't fragment them into noise) and the
  pass-through gave Stage 3's KMeans no help distinguishing "shades
  of one hue" from "distinct identity hues" (Chrome's red/green/
  yellow wedges). Phase 1 of color_essential processing inserts a
  shade-collapse step before framing: opaque palette is over-quantized
  in Lab, centroids are grouped by perceptual ΔE76, and the
  area-dominant centroid wins each group as the universal. See
  preprocess/color_essential.py for the algorithm. The canvas contract
  is still uniform — every Stage 2 output is (S, S, 3) uint8 RGB on
  white with 10% margin — and Stage 3 still branches on
  cat.color_essential (higher k for Chrome/Firefox, k=2 for the rest).

WHY BURST FOLDS INTO STANDARD (no burst.py)
  Burst was a stage-1 distinction about spatial structure (Claude's
  sunburst vs Chrome's gradient). Once categorize has extracted
  Claude's inner_mask, painting it black-on-white is the same primitive
  as for any inner-precedence compound logo. The burst identity flows
  downstream via cat.routing — stages 4 (refine) and 5 (monogram)
  re-diverge on it; stage 2 does not.

LIBRARIES
  Use:    none directly — pure Python control flow. frame_with_margin
          (the canvas-hygiene tail) lives in preprocess.common and is
          the only piece of stage 2 that calls a library primitive
          beyond numpy.

STEP-BY-STEP

  1. Route color-essential through the three-phase color_essential
     pipeline (universalize → frame → separate_features → compose_features),
     emitting BW RGB that matches the contract of the other branches.
       if cat.color_essential:
           rgba_universal = color_essential.universalize(cat, rgba)
           framed         = frame_with_margin(rgba_universal)
           _, features    = color_essential.separate_features(framed)
           return color_essential.compose_features(features)
     - Phase 1 (universalize) shade-collapses the opaque palette.
     - frame_with_margin composites RGBA → RGB on white at framed scale.
     - Phase 2 (separate_features) drops sub-threshold colours and
       emits per-colour boolean masks.
     - Phase 3 (compose_features) erodes each mask by CE_EROSION_PX,
       unions them, and renders BW — already at framed resolution, so
       no second frame_with_margin call is needed.

  2. Branch on routing for the remaining fixtures.
       if cat.routing == "inverted":
           bw = inverted.universalize(cat, rgba)
       else:
           bw = standard.universalize(cat, rgba)
       return frame_with_margin(bw)
     - The per-branch universalize functions still return (H, W, 3)
       black-on-white at source resolution. The framing tail is what
       enforces the (S, S, 3) square contract.

GOTCHAS
  - Every output is (S, S, 3) uint8 square RGB. Downstream consumers
    no longer need to discriminate on cat.color_essential to know the
    output shape — only to choose Stage 3's k.
  - cat.envelope_mask / cat.inner_mask may be None for simple
    (non-compound) logos. The dispatcher does NOT validate them —
    that is the branches' responsibility.
  - Import topology: standard.py and inverted.py both `from .common
    import compose_bw`; this dispatcher also imports frame_with_margin
    from common. common.py does NOT import from the sibling branches.
'''
import numpy as np
from iconograph.application.categorize import Categorization
from . import standard, inverted, color_essential, color_carried
from .common import frame_with_margin


def preprocess(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
    if cat.color_essential:
        rgba_universal = color_essential.universalize(cat, rgba)
        framed = frame_with_margin(rgba_universal)
        _, features = color_essential.separate_features(framed)
        return color_essential.compose_features(features)
    if cat.routing == "inverted":
        bw = inverted.universalize(cat, rgba)
    elif cat.color_carried:
        bw = color_carried.universalize(cat, rgba)
    else:
        bw = standard.universalize(cat, rgba)
    return frame_with_margin(bw)
