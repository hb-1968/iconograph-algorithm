"""Application-pipeline-wide shared helpers.

Helpers used by more than one stage live here so per-stage common.py
files do not need to import sideways across stage boundaries (e.g.
stage 3 reaching into preprocess/common.py).
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Fedora hicolor feed sizes — the per-target grids stage 3's render tier (R1) and
# stage 4's min-feature enforcement work over. Locked to the 16/24 feed resolution
# (user ruling): the pixel-cubify render owns T ≤ 24, and these are the sizes fed
# onward to stage-4 refinement. Stage 5's compose then places each onto its N=T+8
# output grid (16→24, 24→32). The geometry render path is retained for T > 24
# (synthetic regression only) but production never requests those sizes.
HICOLOR_SIZES: tuple[int, ...] = (16, 24)

# 3x3 rect structuring element = the full 8-neighbourhood (incl. diagonals).
_SE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Colour-carried detection (categorize._color_carried / color_carried cutout /
# the pre-compression gate). "If discarding colour collapses the silhouette to a
# single simple primitive AND the interior has enclosed colour-contrast marks, the
# identity lives in colour" (user ruling). Tunables shared by all three consumers.
COLLAPSE_IOU = 0.92      # fit_simple_primitive IoU at/above which the silhouette "collapses"
MARK_DELTA = 24.0        # Lab ΔE beyond the badge colour that counts as a contrast mark.
                         # 24 (was 40) so pale interior LINES survive, not just strong dials
                         # (EasyEffects' slider tracks): regression-verified — no new
                         # color_carried firings, existing cutouts unchanged (ptyxis/electron/
                         # zoom identical), and it drops a virt-viewer false positive.
MARK_MIN_AREA_FRAC = 3e-4  # despeckle: drop mark components below this fraction of opaque area
MARK_FRAC_FLOOR = 0.015  # interior marks must cover >= this fraction of opaque to fire
_DOM_BIN = 8.0           # coarse Lab bin width for the area-dominant (badge) colour


def masks_8adjacent(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff foreground of `a` touches foreground of `b` under 8-connectivity
    (sharing an edge OR a diagonal corner). `a`, `b`: same-shape masks (bool or
    uint8, nonzero = foreground).

    Dilating `a` by the 3x3 SE grows it into its 8-neighbourhood; intersecting
    with `b` is non-empty exactly when the two are ≤1px apart. So `not
    masks_8adjacent(x, logo)` is the "x is ≥1px 8-separated from the logo"
    guarantee used by the stage-5 monogram fitter. Consolidates the 8-adjacency
    idiom that recurs inline across stages 3-4 (e.g. the `_adjacent` closure in
    compress/render.py) into one importable helper; stage 5 uses it directly.
    Converging the pre-existing stage-3/4 inline copies onto this is deferred —
    it touches determinism-critical upstream code and needs its own byte-identity
    re-verification (H6), so it is a separate cleanup, not part of this stage.
    """
    return bool(np.count_nonzero(cv2.dilate(a.astype(np.uint8), _SE3) & b.astype(np.uint8)))


def fit_simple_primitive(mask: np.ndarray) -> tuple[str, float]:
    """Best single-primitive fit to a binary silhouette: try min-enclosing circle,
    bounding rectangle, and fitted ellipse; return (kind, IoU) of the best fit. A high
    IoU means the silhouette "collapses to a simple shape" — the stage-1 signal that a
    logo's identity may live in colour rather than outline (categorize._color_carried /
    the pre-compression gate). Self-contained here (application/common) so stage-1
    categorize needn't import the stage-3 compress package for it — a decoupled sibling
    of segment_prim_classify's per-CC fitters, not a reuse of them. Deterministic: exact
    cv2 fits + solid (non-antialiased) fills, no interpolation (H8). kind in
    {"circle","rect","ellipse","none"}."""
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return "none", 0.0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return "none", 0.0
    c = max(cnts, key=cv2.contourArea)

    def _iou(fit: np.ndarray) -> float:
        inter = int(np.count_nonzero(fit & m))
        union = area + int(fit.sum()) - inter
        return inter / union if union else 0.0

    candidates: list[tuple[str, np.ndarray]] = []
    (cx, cy), r = cv2.minEnclosingCircle(c)
    fc = np.zeros_like(m)
    cv2.circle(fc, (round(cx), round(cy)), round(r), 1, -1)
    candidates.append(("circle", fc))
    x, y, bw, bh = cv2.boundingRect(c)
    fr = np.zeros_like(m)
    fr[y:y + bh, x:x + bw] = 1
    candidates.append(("rect", fr))
    if len(c) >= 5:                                    # fitEllipse needs >= 5 points
        fe = np.zeros_like(m)
        cv2.ellipse(fe, cv2.fitEllipse(c), 1, -1)
        candidates.append(("ellipse", fe))

    best_kind, best = "none", 0.0
    for kind, fit in candidates:
        v = _iou(fit)
        if v > best:
            best_kind, best = kind, v
    return best_kind, best


def interior_mark_mask(
    lab: np.ndarray, opaque: np.ndarray,
    delta: float = MARK_DELTA, min_area_frac: float = MARK_MIN_AREA_FRAC,
) -> np.ndarray:
    """Enclosed colour/luminance-contrast marks inside a solid badge — the interior
    EQ knobs/tracks (EasyEffects) or terminal prompt (Ptyxis) that share the alpha
    silhouette with the badge and so vanish when colour is discarded. `lab`: (H,W,3)
    Lab of the source; `opaque`: (H,W) bool badge silhouette. Returns an (H,W) bool
    mask of the marks (to carve as holes, or measure for detection).

    A KMeans palette would miss these — they're too few pixels to earn a centroid
    (the shade-collapse failure) — so we work directly from the area-DOMINANT badge
    colour (mode of a coarse Lab binning) and keep pixels beyond `delta` ΔE from it.
    Only ENCLOSED marks qualify (a light region touching the badge boundary is
    background, not a hole — `masks_8adjacent` vs the badge complement), and tiny AA
    speckle is dropped by area. Deterministic (H8): mode + ΔE + connected components,
    no RNG, no interpolation. Shared by categorize._color_carried, the color_carried
    cutout, and the pre-compression gate — one definition of "the marks"."""
    n_op = int(opaque.sum())
    if n_op == 0:
        return np.zeros(opaque.shape, bool)
    olab = lab[opaque]                                  # (N, 3)
    binned = np.round(olab / _DOM_BIN).astype(np.int32)
    _, inv, counts = np.unique(binned, axis=0, return_inverse=True, return_counts=True)
    dom_lab = olab[inv == int(np.argmax(counts))].mean(0)
    far = ((np.sqrt(((lab - dom_lab) ** 2).sum(-1)) > delta) & opaque).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(far, connectivity=8)
    min_area = max(1, int(min_area_frac * n_op))
    not_badge = (~opaque).astype(np.uint8)
    keep = np.zeros(opaque.shape, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cc = lbl == i
        if not masks_8adjacent(cc.astype(np.uint8), not_badge):   # enclosed by badge
            keep |= cc
    return keep


def nb4_degree(mask: np.ndarray) -> np.ndarray:
    """Per-pixel 4-neighbour DEGREE of a face-connected 1px curve (int16, background 0):
    endpoint 1, through-pixel 2, T-junction 3, X/4-way 4. The plus-kernel count masked to the
    foreground. (compress/common.py::_nb4count is the unmasked base of the same kernel;
    converging them is a deferred cross-stage cleanup, per the face_connect precedent.)"""
    u = (mask > 0).astype(np.uint8)
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
    return cv2.filter2D(u, cv2.CV_16S, k, borderType=cv2.BORDER_CONSTANT) * u


def face_connect(mask: np.ndarray, max_passes: int = 8) -> np.ndarray:
    """Make a 1-px `mask` FACE-connected (4-connectivity) by ADDITIVELY filling one pixel at
    each diagonal-only 2x2 touch, iterated to a fixpoint. A raster that is already 4-connected
    (a clean `cv2.LINE_4` stroke) has no diagonal-only 2x2, so this is a byte-identical no-op —
    it only closes the 8-connected-but-not-4-connected steps that `skimage.skeletonize` and
    LINE_4 segments meeting at a sharp vertex can leave (a narrow M/W/Q then splits under
    connectivity-4). Additive only, so it never opens a stroke or changes 8-connected topology.
    `mask`: (H,W) bool or {0,255}. Returns (H,W) bool. Deterministic (H8) — vectorised shifted
    slices, a fixed TR/TL fill choice, no RNG. (Same diagonal-touch idiom as refine's
    `_diag_fill`, lifted here as the stage-agnostic 4-connect primitive.)"""
    b = mask > 0
    for _ in range(max_passes):
        tl, tr = b[:-1, :-1], b[:-1, 1:]
        bl, br = b[1:, :-1], b[1:, 1:]
        pat_a = tl & br & ~tr & ~bl              # fg on the ╲ diagonal → fill TR
        pat_b = tr & bl & ~tl & ~br              # fg on the ╱ diagonal → fill TL
        fill = np.zeros_like(b)
        fill[:-1, 1:] |= pat_a
        fill[:-1, :-1] |= pat_b
        if not fill.any():
            break
        b = b | fill
    return b


def nearest_gap(ca: np.ndarray, cb: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Closest pixel pair between two (n,2) coord arrays: `(pa, pb, sqdist)`. Exact squared-int
    distance; deterministic lexsort tiebreak (lowest a-index, then b-index). (Lifted from
    monogram/connect.py — shared by the stage-5 snake connector and the stage-3 render tier's
    cross-construction source-gap bridge.)"""
    D = cdist(ca, cb, metric="sqeuclidean")
    ties = np.argwhere(D == D.min())
    ties = ties[np.lexsort((ties[:, 1], ties[:, 0]))]
    i, j = ties[0]
    return tuple(int(v) for v in ca[i]), tuple(int(v) for v in cb[j]), int(round(float(D[i, j])))


def mst_bridges(comps: list[np.ndarray]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """MST over the components by nearest-pixel gap → n-1 SHORT bridge pixel pairs
    `[(pa, pb), ...]` in the order added. Deterministic Prim's, `(sqdist,i,j)` tiebreak.
    (Lifted from monogram/connect.py alongside `nearest_gap`.)"""
    coords = [np.argwhere(c) for c in comps]
    n = len(comps)
    gap: dict[tuple[int, int], tuple[int, tuple[int, int], tuple[int, int]]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pa, pb, d = nearest_gap(coords[i], coords[j])
            gap[(i, j)] = (d, pa, pb)
    in_tree = [False] * n
    in_tree[0] = True
    bridges = []
    for _ in range(n - 1):
        best = None                                          # (d, i, j, pa, pb)
        for i in range(n):
            if not in_tree[i]:
                continue
            for j in range(n):
                if in_tree[j] or i == j:
                    continue
                d, pa, pb = gap[(min(i, j), max(i, j))]
                cand = (d, i, j, pa, pb)
                if best is None or cand[:3] < best[:3]:
                    best = cand
        _, i, j, pa, pb = best
        in_tree[j] = True
        bridges.append((pa, pb))
    return bridges


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
