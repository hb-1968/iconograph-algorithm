"""Stage 5 phase 3 — the monogram STROKE font (scalable 1px geometric letter forms).

A vector stroke font: each letter is a list of strokes, each stroke `(points, closed)`
with `points` in normalized [0,1]^2 (x left->right, y top->bottom). `render_letter`
scales the points to an arbitrary (w, h) box and rasterizes each stroke as a crisp 1px
`cv2.LINE_8` polyline. Non-uniform stretch is free (x and y scale independently), so the
same "D" fills a tall-narrow box or a squat-wide one.

These are MINIMAL GEOMETRIC forms (per the user's ideal): a closed loop where the letter
has a bowl (D, O), an open polyline otherwise (C, M, V, ...). They read by silhouette, not
by typographic fidelity, and each is a continuous 1px stroke (snake-connectable in the
future phase 5). NOT a bitmap (the old FONT_5X7/etc. upscale made thick FILLED strokes)
and NOT TTF (counters -> multi-loop, and mush at small sizes).

Author fixture letters first (D,C,G,M,V,S,L,F + '?'); the rest fall back to '?' until A-Z
is filled in.
"""
from __future__ import annotations

import cv2
import numpy as np

# char -> list of (points, closed). points: list[(x, y)] in [0,1]^2.
STROKES: dict[str, list[tuple[list[tuple[float, float]], bool]]] = {
    # bowl letters: closed loops, flat/round sides distinguish D vs O
    "D": ([([(0.0, 0.0), (0.7, 0.0), (1.0, 0.28), (1.0, 0.72), (0.7, 1.0), (0.0, 1.0)], True)]),
    "O": ([([(0.3, 0.0), (0.7, 0.0), (1.0, 0.3), (1.0, 0.7), (0.7, 1.0),
             (0.3, 1.0), (0.0, 0.7), (0.0, 0.3)], True)]),
    # open-bowl letters
    "C": ([([(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)], False)]),
    "G": ([([(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 0.5)], False)]),
    # open polylines
    "M": ([([(0.0, 1.0), (0.0, 0.0), (0.5, 0.55), (1.0, 0.0), (1.0, 1.0)], False)]),
    "V": ([([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)], False)]),
    "S": ([([(1.0, 0.1), (0.2, 0.0), (0.0, 0.25), (0.25, 0.5), (0.75, 0.5),
             (1.0, 0.75), (0.8, 1.0), (0.0, 0.9)], False)]),
    "L": ([([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)], False)]),
    # F as a FRAME: spine full height + top bar + a HIGH full-width 2nd bar, opening a
    # big lower body for the logo + other letter to nest in (matches the user's ideal).
    "F": ([([(1.0, 0.0), (0.0, 0.0), (0.0, 1.0)], False),
           ([(0.0, 0.25), (1.0, 0.25)], False)]),
    "?": ([([(0.1, 0.2), (0.5, 0.0), (0.9, 0.2), (0.5, 0.5), (0.5, 0.62)], False),
           ([(0.5, 0.82), (0.5, 0.9)], False)]),
}


def render_letter(ch: str, w: int, h: int) -> np.ndarray:
    """(h, w) bool 1px-stroke raster of `ch` stretched to fill the (w, h) box.

    x and y scale independently (non-uniform), so the letter fills the box to reach the
    canvas boundary. Unknown chars fall back to '?'. Deterministic — static data + a
    fixed rasterizer, no RNG.
    """
    strokes = STROKES.get(ch.upper(), STROKES["?"])
    canvas = np.zeros((h, w), np.uint8)
    sx, sy = max(1, w - 1), max(1, h - 1)
    for points, closed in strokes:
        pts = np.array([[round(x * sx), round(y * sy)] for x, y in points], np.int32).reshape(-1, 1, 2)
        # LINE_4 (not LINE_8): a 4-connected staircase, so diagonal strokes (V/M/S, the
        # D/C bevels) stay FACE-connected with no diagonal-only "cuts" at the corners
        # (the same defect refine's connect_disjoint fixes on the logos).
        cv2.polylines(canvas, [pts], closed, 255, 1, cv2.LINE_4)
    m = canvas > 0
    # Two thin strokes meeting at a sharp vertex (V point, M notch) can leave a 2px
    # blob (a fully 8-surrounded pixel). Guarantee <=1px by skeletonizing ONLY when
    # that occurs (topology-preserving, deterministic; clean glyphs are untouched).
    nb = cv2.filter2D(canvas // 255, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    if np.any((nb >= 9) & m):
        from skimage.morphology import skeletonize
        m = skeletonize(m)
    return m


def _ascii(mask: np.ndarray) -> str:
    """Dev helper: render a bool mask as '#'/'·' rows for eyeballing in a terminal."""
    return "\n".join("".join("#" if v else "·" for v in row) for row in mask)
