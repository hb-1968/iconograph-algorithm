'''
TASK
  Universalize a non-inverted logo (standard or non-color-essential burst)
  to black-on-white RGB.

WHAT YOU ARE WRITING
  File:        src/iconograph/application/preprocess/standard.py
  Position:    top-level module function, directly below this docstring
  Imports:
      import numpy as np
      from iconograph.application.categorize import (
          Categorization,
          ALPHA_OPAQUE_THRESHOLD,
      )
      from .common import compose_bw
  Signature:
      def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
          """cat: stage-1 Categorization with routing in {"standard","burst"}
          and color_essential=False.
          rgba: (H, W, 4) uint8 — original RGBA from categorize._load_rgba.
          Returns: (H, W, 3) uint8 RGB at source resolution — black
          foreground on white. The dispatcher applies frame_with_margin
          afterwards; this function does NOT square or pad the canvas.
          """
  Called by:   src/iconograph/application/preprocess/__init__.py — preprocess()
               (when cat.color_essential is False and cat.routing != "inverted")

CONCEPT
  Goal:    pick the right mask from Categorization, paint it black on white.
  Method:  three-way mask selection on (is_compound, precedence), then compose_bw.

  - Simple (non-compound) logo → foreground = opaque alpha silhouette.
    The brand IS the visible region (Discord, Slack, Edge, Whatsapp,
    Figma, VSCode, Obsidian, Apple, React, Docker, Homebrew, Android).
  - Compound + precedence="outer" → foreground = envelope_mask.
    The envelope IS the brand mark; the inner is a generic primitive
    (YouTube's play triangle) that gets discarded by NOT being included.
    The "white triangle" effect for YouTube falls out naturally — pixels
    in the inner region are not in envelope_mask, so they render white.
  - Compound + precedence="inner" → foreground = inner_mask.
    The inner glyph IS the brand mark (Claude's cream sunburst inside
    its dark square; Gmail's M-shaped paper inside the envelope).

WHY MASK-FIRST (no per-pixel threshold at stage 2)
  Stage 1 already ran KMeans + connected-component analysis to extract
  these masks at 100% accuracy on the fixture set. Re-thresholding
  luminance here would (a) duplicate work the upstream stage already
  did, (b) potentially diverge from stage 1's decision when boundaries
  are subtle, (c) lose precedence-aware selection — no per-pixel test
  matches "the envelope of a compound logo with outer precedence."
  The H5 reuse boundary sits exactly here: stage 1 produces masks;
  stage 2 renders them.

WHY BURST IS ABSORBED HERE (no burst.py file)
  At stage 2 the only non-color-essential burst is Claude, which
  categorize routes as is_compound=True + precedence="inner". The
  mask-selection rule for Claude is therefore identical to any other
  inner-precedence compound logo: foreground = inner_mask. There is
  no burst-specific code path at this stage — the cream sunburst is
  shape-equivalent to a glyph here. The burst.py scaffold was removed
  in the same commit that lands this brief.

LIBRARIES
  Use:    numpy comparison        — alpha > ALPHA_OPAQUE_THRESHOLD for
                                     the simple (non-compound) branch;
                                     returns a (H, W) bool array directly
          (compose_bw)            — rendering primitive in preprocess.common
  Alt:    cv2.threshold           — would work for the alpha test but
                                     returns (retval, uint8 array); the
                                     numpy comparison is one expression
                                     with the right dtype

STEP-BY-STEP

  1. Branch on compound + precedence to pick the mask.
       if cat.is_compound and cat.precedence == "outer":
           mask = cat.envelope_mask
       elif cat.is_compound and cat.precedence == "inner":
           mask = cat.inner_mask
       else:
           mask = rgba[..., 3] > ALPHA_OPAQUE_THRESHOLD
     - Compound checks come FIRST; the simple fallback is the else
       branch.
     - cat.envelope_mask / cat.inner_mask are (H, W) bool arrays.
       They are not None on the compound branches (is_compound=True
       implies categorize populated them — see categorize.py:88-103).
     - The alpha-threshold path constructs a fresh (H, W) bool array
       from the source RGBA; this is the only path that reads rgba.

  2. Render.
       return compose_bw(mask)
     - compose_bw enforces the (H, W, 3) uint8 output contract. This
       branch file does not handle RGB construction directly — keep
       the rendering primitive in one place (common.py).

GOTCHAS
  - Branch ordering matters: compound checks come before the alpha
    fallback. The else branch must remain reachable for simple logos —
    it is the only path that constructs a mask from rgba.
  - For compound + outer (YouTube), do NOT fill the inner cutout.
    envelope_mask already excludes the inner pixels and that is the
    intended behavior — the play triangle renders as a white shape
    inside the black rounded rect, exactly per the user's "disregard
    the white" direction.
  - ALPHA_OPAQUE_THRESHOLD is imported from categorize (single source
    of truth — categorize.py:62). Do NOT redefine 16 here.
  - This function MUST NOT be called for color_essential=True logos;
    the dispatcher in preprocess/__init__.py short-circuits before
    reaching here. No defensive check is added.
'''

import numpy as np
from iconograph.application.categorize import (
    Categorization,
    ALPHA_OPAQUE_THRESHOLD,
)
from .common import compose_bw

def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
    if cat.is_compound and cat.precedence == "outer":
        # Subtract inner from env so opaque-white inner glyphs
        # (YouTube's play triangle, Mastodon's m) remain visible as
        # white cutouts inside the black envelope silhouette.
        mask = cat.envelope_mask & ~cat.inner_mask
    elif cat.is_compound and cat.precedence == "inner":
        mask = cat.inner_mask
    else:
        mask = rgba[..., 3] > ALPHA_OPAQUE_THRESHOLD

    return compose_bw(mask)