"""Render scalable, one-pixel geometric frame letters.

Each letter bounds open space for the logo or partner letter. Composition
chooses dimensions and layout; this module rasterizes the glyph strokes.
"""
from __future__ import annotations

import cv2
import numpy as np

from iconograph.application.common import face_connect

# char -> list of (points, closed). points: list[(x, y)] in [0,1]^2. Letters are RECTILINEAR
# frame forms: predominantly HORIZONTAL / VERTICAL strokes (a diagonal only where the letter's
# identity requires it — N, V, W, X, Z, K, R's leg). Straight strokes render crisp (no
# staircase) and leave clean RECTANGULAR interior bays for the logo to weave into, per the user
# ("letters should be almost entirely straight line renders … opens up better weaving").
STROKES: dict[str, list[tuple[list[tuple[float, float]], bool]]] = {
    # bowls → near-rectangles with MINIMAL 1-step corner bevels (D flat-left, O symmetric)
    "D": ([([(0.0, 0.0), (0.85, 0.0), (1.0, 0.2), (1.0, 0.8), (0.85, 1.0), (0.0, 1.0)], True)]),
    "O": ([([(0.25, 0.0), (0.75, 0.0), (1.0, 0.25), (1.0, 0.75), (0.75, 1.0),
             (0.25, 1.0), (0.0, 0.75), (0.0, 0.25)], True)]),
    "C": ([([(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)], False)]),
    "G": ([([(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 0.5)], False)]),
    "Q": ([([(0.25, 0.0), (0.75, 0.0), (1.0, 0.25), (1.0, 0.75), (0.75, 1.0),
             (0.25, 1.0), (0.0, 0.75), (0.0, 0.25)], True),
           ([(0.6, 0.7), (1.0, 1.0)], False)]),                        # O + a tail
    # square bowls: spine + top bar + right vertical + mid bar → a clean rectangular lower bay
    "P": ([([(0.0, 1.0), (0.0, 0.0), (0.9, 0.0), (0.9, 0.45), (0.0, 0.45)], False)]),
    "R": ([([(0.0, 1.0), (0.0, 0.0), (0.9, 0.0), (0.9, 0.45), (0.0, 0.45)], False),
           ([(0.4, 0.45), (1.0, 1.0)], False)]),                       # leg (the one R diagonal)
    "B": ([([(0.0, 1.0), (0.0, 0.0), (0.9, 0.0), (0.9, 1.0)], True),   # 日 — two stacked bays
           ([(0.0, 0.5), (0.9, 0.5)], False)]),
    "U": ([([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], False)]),  # square cup
    # spine-and-bar SKELETON frames (all H/V; the strongest hosts, like F)
    "E": ([([(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)], False),   # C + a mid-bar
           ([(0.0, 0.5), (0.8, 0.5)], False)]),
    "F": ([([(1.0, 0.0), (0.0, 0.0), (0.0, 1.0)], False),
           ([(0.0, 0.25), (1.0, 0.25)], False)]),
    "H": ([([(0.0, 0.0), (0.0, 1.0)], False),                          # two spines + crossbar
           ([(1.0, 0.0), (1.0, 1.0)], False),
           ([(0.0, 0.5), (1.0, 0.5)], False)]),
    "T": ([([(0.0, 0.0), (1.0, 0.0)], False),                          # top bar + centre spine
           ([(0.5, 0.0), (0.5, 1.0)], False)]),
    "I": ([([(0.15, 0.0), (0.85, 0.0)], False),                        # serifed spine (thin)
           ([(0.5, 0.0), (0.5, 1.0)], False),
           ([(0.15, 1.0), (0.85, 1.0)], False)]),
    "L": ([([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)], False)]),
    "S": ([([(1.0, 0.0), (0.0, 0.0), (0.0, 0.25), (1.0, 0.25), (1.0, 1.0), (0.0, 1.0)], False)]),
    # A / M / Y → RECTILINEAR (Π + crossbar / Π + centre stub / bracket + stem) — big rect bays
    "A": ([([(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], False),
           ([(0.0, 0.55), (1.0, 0.55)], False)]),
    "M": ([([(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], False),
           ([(0.5, 0.0), (0.5, 0.45)], False)]),
    "Y": ([([(0.0, 0.0), (0.0, 0.4), (1.0, 0.4), (1.0, 0.0)], False),
           ([(0.5, 0.4), (0.5, 1.0)], False)]),
    # inherently-diagonal letters — kept minimal (mostly vertical, a short converging point)
    "N": ([([(0.0, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, 0.0)], False)]),
    "V": ([([(0.0, 0.0), (0.0, 0.6), (0.5, 1.0), (1.0, 0.6), (1.0, 0.0)], False)]),
    "W": ([([(0.0, 0.0), (0.0, 0.55), (0.25, 1.0), (0.5, 0.5), (0.75, 1.0),
             (1.0, 0.55), (1.0, 0.0)], False)]),
    "K": ([([(0.0, 0.0), (0.0, 1.0)], False),                          # spine + two arms
           ([(1.0, 0.0), (0.0, 0.5), (1.0, 1.0)], False)]),
    "X": ([([(0.0, 0.0), (1.0, 1.0)], False),
           ([(1.0, 0.0), (0.0, 1.0)], False)]),
    "Z": ([([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)], False)]),
    "J": ([([(1.0, 0.0), (1.0, 0.9), (0.5, 1.0), (0.1, 0.9), (0.1, 0.75)], False)]),
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
    # Guarantee FACE-connectivity (connectivity-4): skeletonize's 8-connected output and LINE_4
    # segments meeting at a sharp vertex can leave diagonal-only steps that split a narrow
    # M/W/Q under connectivity-4 (compose's `_singly_connected` gate + the phase-5 snake need
    # 4-conn). A no-op on the already-4-connected raster → existing letters stay byte-identical.
    return face_connect(m)


def _ascii(mask: np.ndarray) -> str:
    """Dev helper: render a bool mask as '#'/'·' rows for eyeballing in a terminal."""
    return "\n".join("".join("#" if v else "·" for v in row) for row in mask)
