"""Stage 5 phase 3 — compose the native logo + two 1px-outline letters on the canvas.

Per the user's ideal (`~/Pictures/ideal_discord_preconnect.png`): the letters are big
1px-stroke GEOMETRIC OUTLINE forms (from `glyphs.render_letter`) stretched NON-uniformly
to fill the canvas to its boundary (maximize coverage); the logo stays NATIVE (1:1, never
rescaled) and nests inside a letter's interior negative space (or a gap). Everything is
1px; all three components are kept 1px-separated (an 8-connected gap) — a future phase 5
connects them into ONE continuous snake-line.

Layout: both letters span (near) full height, flush to the left/right edges with a gap
between; the letter that hosts the logo gets a box wide enough for its stretched hollow to
contain the native logo + a 1px gap, the other takes the rest. Fallback (neither letter has
a usable interior — e.g. V, S): letters flanking with a reserved central gap for the logo.

Separation is a mask op: `masks_8adjacent(a, b)` is true iff a and b are <=1px apart.
Deterministic (H8): fixed host order, integer geometry, `cv2.polylines`/`dilate` only.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from iconograph.application.common import masks_8adjacent
from iconograph.application.monogram.common import split_monogram
from iconograph.application.monogram.glyphs import render_letter

_SE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_MARGIN = 1          # 1px canvas margin so letters don't sit flush against the very edge
_MIN_W = 4           # a letter box narrower than this is illegible
# Letters that naturally FRAME a big open body (spine + bar[s]) — used as a full-canvas
# frame hosting the logo + other letter. Bowl/round letters (C/D/O/G) read better in
# side-by-side, so they are NOT frame letters. (S/others can join once reshaped.)
_FRAME_LETTERS = frozenset("FETH")


@dataclass(frozen=True)
class Letter:
    ch: str
    y: int
    x: int
    h: int
    w: int
    mask: np.ndarray                 # (N,N) uint8 {0,255} — the placed 1px outline


@dataclass(frozen=True)
class ComposeResult:
    canvas: np.ndarray               # (N,N,3) uint8 — ink on white
    union: np.ndarray                # (N,N) uint8 {0,255} — logo | letters
    letters: np.ndarray              # (N,N) uint8 {0,255} — letters only
    logo: np.ndarray                 # (N,N) uint8 {0,255} — placed logo only
    l1: "Letter | None"
    l2: "Letter | None"


def _tight_crop(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _place_bool(g, y, x, N):
    m = np.zeros((N, N), np.uint8)
    h, w = g.shape
    m[y:y + h, x:x + w][g] = 255
    return m


def _place_logo(logo_c, y, x, N):
    m = np.zeros((N, N), np.uint8)
    h, w = logo_c.shape
    m[y:y + h, x:x + w][logo_c > 0] = 255
    return m


def _nest_logo(logo_c, letters, box, N):
    """Search `box`=(y,x,h,w) for a top-left position where the native logo is >=1px
    (8-connected) clear of `letters`. Returns (y,x) closest to the box centre, or None."""
    by, bx, bh, bw = box
    lh, lw = logo_c.shape
    if bh < lh or bw < lw:
        return None
    cy, cx = by + bh / 2.0, bx + bw / 2.0
    best = None
    for y in range(by, by + bh - lh + 1):
        for x in range(bx, bx + bw - lw + 1):
            if masks_8adjacent(_place_logo(logo_c, y, x, N), letters):
                continue
            d = (y + lh / 2.0 - cy) ** 2 + (x + lw / 2.0 - cx) ** 2
            if best is None or d < best[0]:
                best = (d, y, x)
    return (best[1], best[2]) if best else None


def _letter(ch, y, x, h, w, N):
    g = render_letter(ch, w, h)
    return Letter(ch, y, x, h, w, _place_bool(g, y, x, N))


def _largest_rect(free):
    """(y, x, h, w) of the largest all-True axis-aligned rectangle in bool `free`, or
    None. Histogram method, O(H*W)."""
    H, W = free.shape
    heights = np.zeros(W, np.int32)
    best = (0, 0, 0, 0, 0)                                # area, y, x, h, w
    for r in range(H):
        heights = np.where(free[r], heights + 1, 0)
        stack = []                                        # (start_col, height)
        for c in range(W + 1):
            cur = int(heights[c]) if c < W else 0
            start = c
            while stack and stack[-1][1] > cur:
                sx, sh = stack.pop()
                area = sh * (c - sx)
                if area > best[0]:
                    best = (area, r - sh + 1, sx, sh, c - sx)
                start = sx
            stack.append((start, cur))
    return None if best[0] == 0 else (best[1], best[2], best[3], best[4])


def _side_by_side(logo_c, ch1, ch2, N, ink_rgb):
    """Two letters flush left/right (full height); the logo nests in the host letter's
    negative space. Smallest host box that nests -> the other letter gets the remainder
    (matches the discord ideal: narrow D, wide C, logo in C)."""
    lh, lw = logo_c.shape
    h = N - 2 * _MARGIN
    usable = N - 2 * _MARGIN
    for host in (2, 1):
        for w_host in range(lw + 2, min(usable - _MIN_W - 1, lw + 8) + 1):
            w_other = usable - w_host - 1
            if w_other < _MIN_W:
                break
            (w1, w2) = (w_other, w_host) if host == 2 else (w_host, w_other)
            x1, x2 = _MARGIN, N - _MARGIN - w2
            l1 = _letter(ch1, _MARGIN, x1, h, w1, N)
            l2 = _letter(ch2, _MARGIN, x2, h, w2, N)
            if masks_8adjacent(l1.mask, l2.mask):
                continue
            hx, hw = (x2, w2) if host == 2 else (x1, w1)
            pos = _nest_logo(logo_c, l1.mask | l2.mask, (0, hx, N, hw), N)
            if pos is None:
                continue
            logo = _place_logo(logo_c, pos[0], pos[1], N)
            if masks_8adjacent(logo, l1.mask) or masks_8adjacent(logo, l2.mask):
                continue
            return _result(logo, l1, l2, N, ink_rgb)
    return None


def _frame_weave(logo_c, ch1, ch2, N, ink_rgb):
    """Stretch ONE letter to a full-canvas FRAME and nest the native logo + the other
    letter (stretched) in its largest open pocket/body — the aggressive-coverage look
    (figma ideal: F fills the canvas, logo + M woven in its body). None if it can't fit."""
    lh, lw = logo_c.shape
    m = _MARGIN
    best = None                                          # (coverage, logo, l1, l2)
    for frame_first, fch, och in ((True, ch1, ch2), (False, ch2, ch1)):
        if fch.upper() not in _FRAME_LETTERS:            # only genuine frame letters host
            continue
        frame = Letter(fch, m, m, N - 2 * m, N - 2 * m, _place_bool(render_letter(fch, N - 2 * m, N - 2 * m), m, m, N))
        free = cv2.dilate(frame.mask, _SE3) == 0         # 1px-clear of the frame strokes
        free[:m] = False
        free[-m:] = False
        free[:, :m] = False
        free[:, -m:] = False
        rect = _largest_rect(free)
        if rect is None:
            continue
        ry, rx, rh, rw = rect
        if rh < lh or rw < lw + 1 + _MIN_W:              # body must hold logo + other letter
            continue
        pos = _nest_logo(logo_c, frame.mask, (ry, rx, rh, lw + 2), N)   # logo flush-left in body
        if pos is None:
            continue
        logo = _place_logo(logo_c, pos[0], pos[1], N)
        ox = rx + lw + 2
        ow = rx + rw - ox
        if ow < _MIN_W:
            continue
        other = _letter(och, ry, ox, rh, ow, N)          # other letter fills the rest of the body
        if masks_8adjacent(other.mask, frame.mask) or masks_8adjacent(other.mask, logo):
            continue
        l1, l2 = (frame, other) if frame_first else (other, frame)
        cov = int((frame.mask > 0).sum() + (other.mask > 0).sum())
        if best is None or cov > best[0]:
            best = (cov, logo, l1, l2)
    return None if best is None else _result(best[1], best[2], best[3], N, ink_rgb)


def compose_icon(logo_ink: np.ndarray, monogram: str, N: int, *,
                 ink_rgb: tuple[int, int, int] = (0, 0, 0)) -> ComposeResult:
    """Compose the final icon: native `(T,T)` logo `logo_ink` + the two `monogram`
    letters as big 1px outlines, maximizing coverage. Tries side-by-side nesting and a
    full-canvas frame-weave, keeps whichever covers the most; falls back otherwise.
    Returns a ComposeResult; `.canvas` is (N,N,3) ink-on-white."""
    if N < logo_ink.shape[0]:
        raise ValueError(f"canvas N={N} must be >= logo T={logo_ink.shape[0]}")
    ch1, ch2 = split_monogram(monogram)
    logo_c = _tight_crop(logo_ink)
    # Frame-weave (a frame letter fills the canvas, logo + other letter woven in its body)
    # when a genuine frame letter is present and it fits; else side-by-side nesting (bowl
    # letters read better this way); else the flush-side fallback.
    return (_frame_weave(logo_c, ch1, ch2, N, ink_rgb)
            or _side_by_side(logo_c, ch1, ch2, N, ink_rgb)
            or _fallback(logo_c, ch1, ch2, N, ink_rgb))


def _fallback(logo_c, ch1, ch2, N, ink_rgb):
    """Neither letter can host the logo (no interior — V, S, F, M...): flush the native
    logo to each side in turn, fill the OPPOSITE region with the two biggest letters
    (side-by-side if the region is wide, stacked if tall), and keep the layout that
    covers the most area."""
    lh, lw = logo_c.shape
    m = _MARGIN
    best = None                                          # (area, logo, l1, l2)
    for mode in ("left", "right", "top", "bottom"):
        if mode == "left":
            ly, lx, ry, rx, rh, rw = (N - lh) // 2, m, m, m + lw + 1, N - 2 * m, N - 2 * m - lw - 1
        elif mode == "right":
            ly, lx, ry, rx, rh, rw = (N - lh) // 2, N - m - lw, m, m, N - 2 * m, N - 2 * m - lw - 1
        elif mode == "top":
            ly, lx, ry, rx, rh, rw = m, (N - lw) // 2, m + lh + 1, m, N - 2 * m - lh - 1, N - 2 * m
        else:
            ly, lx, ry, rx, rh, rw = N - m - lh, (N - lw) // 2, m, m, N - 2 * m - lh - 1, N - 2 * m
        if rh < _MIN_W or rw < _MIN_W:
            continue
        logo = _place_logo(logo_c, ly, lx, N)
        if rw >= rh:                                     # wide region -> letters side by side
            we = (rw - 1) // 2
            if we < _MIN_W:
                continue
            l1 = _letter(ch1, ry, rx, rh, we, N)
            l2 = _letter(ch2, ry, rx + we + 1, rh, rw - we - 1, N)
        else:                                            # tall region -> letters stacked
            he = (rh - 1) // 2
            if he < _MIN_W:
                continue
            l1 = _letter(ch1, ry, rx, he, rw, N)
            l2 = _letter(ch2, ry + he + 1, rx, rh - he - 1, rw, N)
        letters = l1.mask | l2.mask
        if masks_8adjacent(l1.mask, l2.mask) or masks_8adjacent(logo, letters):
            continue
        area = l1.h * l1.w + l2.h * l2.w
        if best is None or area > best[0]:
            best = (area, logo, l1, l2)
    if best is not None:
        return _result(best[1], best[2], best[3], N, ink_rgb)
    return _logo_only(logo_c, N, ink_rgb)


def _result(logo, l1, l2, N, ink_rgb):
    letters = l1.mask | l2.mask
    canvas = np.full((N, N, 3), 255, np.uint8)
    canvas[(logo > 0) | (letters > 0)] = np.asarray(ink_rgb, np.uint8)
    union = np.zeros((N, N), np.uint8)
    union[(logo > 0) | (letters > 0)] = 255
    return ComposeResult(canvas, union, letters, logo, l1, l2)


def _logo_only(logo_c, N, ink_rgb):
    lh, lw = logo_c.shape
    logo = _place_logo(logo_c, (N - lh) // 2, (N - lw) // 2, N)
    canvas = np.full((N, N, 3), 255, np.uint8)
    canvas[logo > 0] = np.asarray(ink_rgb, np.uint8)
    return ComposeResult(canvas, logo.copy(), np.zeros((N, N), np.uint8), logo, None, None)
