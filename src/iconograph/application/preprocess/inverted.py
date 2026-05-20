'''
TASK
  Universalize an inverted logo (light foreground on dark background)
  to black-on-white RGB.

WHAT YOU ARE WRITING
  File:        src/iconograph/application/preprocess/inverted.py
  Position:    top-level module function, directly below this docstring
  Imports:
      import numpy as np
      from iconograph.application.categorize import Categorization
      from .common import compose_bw
  Signature:
      def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
          """cat: stage-1 Categorization with routing="inverted"
          (and therefore is_compound=True, precedence="inner",
          inner_mask is not None).
          rgba: (H, W, 4) uint8 — present for signature parity with
          standard.universalize; not read by this branch.
          Returns: (H, W, 3) uint8 RGB at source resolution — black
          foreground on white. The dispatcher applies frame_with_margin
          afterwards; this function does NOT square or pad the canvas.
          """
  Called by:   src/iconograph/application/preprocess/__init__.py — preprocess()
               (when cat.color_essential is False and cat.routing == "inverted")

CONCEPT
  Goal:    paint the originally-light glyph as black on white.
  Method:  pull cat.inner_mask, render via compose_bw.

  - cat.inner_mask covers the originally-light region. Stage 1's
    _compound_via_light_inner (Lab L > 88, see categorize.py:252-265)
    or _compound_via_alpha_hole (categorize.py:268-291) extracted it.
    The mask therefore ALREADY represents "what should be black" after
    universalization — no further inversion or thresholding is needed.
  - Fixtures landing here: Telegram, Signal, Mastodon, LinkedIn,
    YouTube-Music, Spotify, Instagram, Adobe Illustrator (per the
    fixtures/README.md sub-case labels).

WHY INVERTED LIVES IN ITS OWN FILE (not collapsed into common.py)
  The "inverted" label describes the SOURCE polarity (light foreground
  on dark background), not the output transformation. The stage-2
  output is black-on-white in both branches — what differs is the
  mask-selection rule. standard.py has a three-way switch on
  (is_compound, precedence); inverted.py has a one-line rule
  (`mask = cat.inner_mask`) because categorize guarantees
  routing="inverted" implies is_compound=True + precedence="inner" +
  inner_mask is not None. The branch boundary encodes that contract
  at the directory level — collapsing the two into a common helper
  with a routing arg would obscure it and violate H4 ("branching is
  in the directory layout, not in a runtime framework").

WHY "INVERTING THE INVERTED CLASSES TO NORMALIZE THEM"
  Per the user's stage-2 direction. For an inverted source like
  Mastodon (white "m" on purple), painting cat.inner_mask as black IS
  the inversion: the originally-bright region becomes ink, the
  originally-dark region becomes canvas. After this step, Mastodon is
  shape-equivalent to a standard-routing logo (black glyph on white),
  and stages 3-6 can treat all non-color-essential fixtures uniformly.

LIBRARIES
  Use:    (compose_bw)            — rendering primitive in preprocess.common
  Alt:    none — this branch has no library calls of its own.

STEP-BY-STEP

  1. Pull the inner mask.
       mask = cat.inner_mask
     - (H, W) bool array. Categorize guarantees it is not None for
       routing="inverted"; an `assert cat.inner_mask is not None` is
       acceptable as a tripwire if you want one. The contract is
       enforced upstream in categorize.py.

  2. Render.
       return compose_bw(mask)
     - Same rendering helper as standard.universalize.

GOTCHAS
  - The rgba argument is unused. Keep it in the signature anyway —
    the dispatcher in preprocess/__init__.py calls both branches with
    the same (cat, rgba) shape. Removing it would break the dispatcher
    or force a special case there.
  - Do NOT invert / negate the mask before passing it to compose_bw.
    cat.inner_mask is already the foreground (the light glyph in the
    source); compose_bw paints True pixels as black, which is exactly
    the intended polarity flip relative to the original image.
  - If cat.inner_mask is somehow None at call time, the bug is
    upstream in categorize.py — do not paper over it here. A defensive
    assert is fine; a fallback path is not.
'''

import numpy as np
from iconograph.application.categorize import Categorization
from .common import compose_bw

def universalize(cat: Categorization, rgba: np.ndarray) -> np.ndarray:
    mask = cat.inner_mask
    return compose_bw(mask)