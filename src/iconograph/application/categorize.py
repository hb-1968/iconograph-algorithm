"""Stage 1 sub-case detector.

Emits a Categorization (routing + structural metadata + masks) for a
logo PNG. Deterministic, library-only, side-effect-free. Algorithm
sketched in /home/splicelord/.claude/plans/given-we-have-our-elegant-bumblebee.md
and the README's "Stage-1 routing-label columns" section.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from iconograph.application.common import (
    COLLAPSE_IOU, MARK_FRAC_FLOOR, fit_simple_primitive, interior_mark_mask)


@contextmanager
def _silence_kmeans_convergence():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        yield


def _kmeans_fit(X: np.ndarray, n_clusters: int) -> KMeans:
    with _silence_kmeans_convergence():
        return KMeans(
            n_clusters=n_clusters, n_init=_KM_N_INIT, random_state=_KM_SEED
        ).fit(X)

Routing = Literal["standard", "inverted", "burst"]
Precedence = Literal["outer", "inner"]

# Tunable thresholds — single source so testbed iteration is fast (H1).
ENVELOPE_DOM = 0.72            # KMeans(2)-on-opaque dominance to call uniform envelope
INNER_AREA_FLOOR = 0.025       # inner_largest_cc / bbox_area floor
INNER_LIGHT_L = 88.0           # L threshold for light-inner fallback (white only, not gradient corner)
LIGHT_GLYPH_FRAC = 0.15        # a light glyph this large ON a silhouette that COLLAPSES to a simple
                               # primitive (generic badge) is the identity even if it reaches the badge
                               # edge (Steam's pipe) — admit despite non-containment. homebrew (37% light
                               # but non-collapsing mug) is excluded by the collapse gate.
INNER_CONTAINMENT_MARGIN = 0.04  # inner-bbox must sit inside overall bbox by this fraction
FILLED_ENVELOPE = 0.55         # envelope_area / envelope_bbox floor for alpha-hole fallback
MINOR_AB_STD_MAX = 22.0        # max within-cluster ab-spread of the minor cluster (bimodality)
MAJOR_AB_STD_MAX = 28.0        # max ab-spread of the *dominant* cluster (rejects chrome: warm wedges merged)
ENV_NEAR_FILL = 0.70           # env_area / env_bbox; above this, env is a "container" and inner wins precedence
CE_K = 5                       # KMeans-K for the central-element search in color_essential
CE_CENTRAL_FRAC_RANGE = (0.02, 0.22)  # frac range for a "central element" candidate cluster
CE_CENTRAL_OFFSET = 0.18       # max centroid offset / opaque-radius for "central"
CE_SUB_HUES_MIN = 3            # min major-sub-clusters that read as distinct hues
CE_SUB_AB_STD_MIN = 18.0       # min ab-spread of the surrounding (non-central) region
SIGNIFICANT_CC_PX = 200        # absolute pixel floor for "significant" inner connected component
TRIANGLE_OUTER_FIT = 0.85      # if inner's best primitive is a CLEAN triangle (≥ this fit), keep outer — YouTube's play triangle is 0.99; Telegram's plane is 0.74 (triangular silhouette but not a clean primitive)
CONCAVITY_INNER = 0.10         # if inner has ≥ this concavity, keep inner (signal-bubble case)
RADIAL_BURST_POLY_TOL = 0.01   # approxPolyDP epsilon as fraction of perimeter
RADIAL_BURST_MIN_VTX = 20      # polygon vertex count floor for "starlike with ≥10 spikes" — Claude=24, all other inner glyphs ≤ 17 so this alone discriminates
RADIAL_BURST_R_CV = 0.20       # radial r(θ) coefficient-of-variation floor — Claude bare sunburst is 0.24 (was inflated to 0.45+ by the alpha-hole corner caps that used to leak into inner)
RADIAL_BURST_N_BINS = 36       # angular bins for r(θ) sampling
ENV_DISTINCTIVE_OUTER_CONCAVITY = 0.03  # filled-outer-contour concavity above which a near-fill envelope is distinctive (Mastodon's chatbubble tail) → outer-precedence
ALPHA_OPAQUE_THRESHOLD = 16

_KM_SEED = 0
_KM_N_INIT = 10


@dataclass(frozen=True)
class Categorization:
    routing: Routing
    color_essential: bool
    is_compound: bool
    precedence: Precedence | None
    envelope_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    inner_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    color_carried: bool = False   # solid badge whose identity is interior colour marks
                                  # (routes stage 2 to the color_carried cutout branch)


def _color_carried(lab: np.ndarray, opaque: np.ndarray) -> bool:
    """Colour-carried: discarding colour collapses the silhouette to ONE simple
    primitive AND the interior holds enclosed colour-contrast marks (EasyEffects' EQ
    knobs on a disc, Ptyxis' prompt on a square) — so the identity lives in colour,
    not outline (user ruling). Only meaningful when NOT already color_essential /
    is_compound; the caller gates that. Routes stage 2 to the color_carried cutout."""
    _, iou = fit_simple_primitive(opaque.astype(np.uint8))
    if iou < COLLAPSE_IOU:
        return False
    marks = interior_mark_mask(lab, opaque)
    return int(marks.sum()) >= MARK_FRAC_FLOOR * int(opaque.sum())


def categorize(image_path: Path) -> Categorization:
    rgba = _load_rgba(Path(image_path))
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    opaque = alpha > ALPHA_OPAQUE_THRESHOLD
    if not opaque.any():
        return Categorization("standard", False, False, None)

    lab = rgb2lab(rgb.astype(np.float32) / 255.0)
    color_essential = _color_essential(lab, opaque)
    is_compound, env_mask, inner_mask = _compound(lab, opaque)

    if color_essential:
        return Categorization("burst", True, is_compound, None, env_mask, inner_mask)
    if not is_compound:
        # A solid single-primitive silhouette whose interior colour-contrast marks
        # would be lost to the alpha-only standard branch → route the cutout instead.
        return Categorization("standard", False, False, None,
                              color_carried=_color_carried(lab, opaque))
    prec = _precedence(env_mask, inner_mask)
    initial_routing: Routing = "inverted" if prec == "inner" else "standard"
    # Final routing precedence — applied conceptually AFTER Stage 2 strips
    # the envelope. The revealed inner may itself be a radial burst pattern
    # (Claude's cream sunburst): if so, the final routing is `burst`, not
    # `inverted`. When Stage 2 lands, it will call this same function on
    # its actual post-strip output; we call it here so Stage 1 reports the
    # final-pipeline routing today.
    final_routing = _post_strip_recategorize(inner_mask) or initial_routing
    return Categorization(final_routing, False, True, prec, env_mask, inner_mask)


# load


def _load_rgba(path: Path) -> np.ndarray:
    # PIL .convert('RGBA') handles mode P + tRNS, RGB, RGBA uniformly.
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    return np.asarray(img, dtype=np.uint8)


# Public alias for outside callers (testbed, future pipeline orchestrator)
# that need the canonical RGBA loader without reaching into a private name.
load_rgba = _load_rgba


# (a) color_essential: radial multihue


def _color_essential(lab: np.ndarray, opaque: np.ndarray) -> bool:
    """True iff the logo has a small *central element* surrounded by a
    *multi-hue region* — the chrome/firefox signature (blue center disc
    inside red/green/yellow wedges; dark globe inside warm flame orbit).
    K=2 merges the center with one of the surrounding hues, so we search
    K=CE_K clusters and accept the first that fits the central-element
    profile.
    """
    if int(opaque.sum()) < 1000:
        return False

    M = cv2.moments(opaque.astype(np.uint8))
    if M["m00"] == 0:
        return False
    ocx = M["m10"] / M["m00"]
    ocy = M["m01"] / M["m00"]
    _, _, w, h = cv2.boundingRect(opaque.astype(np.uint8))
    radius = max(1.0, 0.5 * min(w, h))

    lab_op = lab[opaque]
    n_samples = lab_op.shape[0]
    if n_samples < CE_K * 20:
        return False
    km = _kmeans_fit(lab_op, CE_K)
    counts = np.bincount(km.labels_, minlength=CE_K)
    labels_grid = np.full(opaque.shape, -1, dtype=np.int8)
    labels_grid[opaque] = km.labels_

    lo, hi = CE_CENTRAL_FRAC_RANGE
    for c in range(CE_K):
        frac = counts[c] / n_samples
        if not (lo <= frac <= hi):
            continue
        m = labels_grid == c
        Mc = cv2.moments(m.astype(np.uint8))
        if Mc["m00"] == 0:
            continue
        ccx = Mc["m10"] / Mc["m00"]
        ccy = Mc["m01"] / Mc["m00"]
        if np.hypot(ccx - ocx, ccy - ocy) > CE_CENTRAL_OFFSET * radius:
            continue
        # Surrounding region = everything except this central cluster.
        other_pts = lab_op[km.labels_ != c]
        if other_pts.shape[0] < 200:
            continue
        ab_std = float(np.linalg.norm(other_pts[:, 1:3].std(axis=0)))
        if ab_std < CE_SUB_AB_STD_MIN:
            continue
        km_sub = _kmeans_fit(other_pts, 4)
        sub_counts = np.bincount(km_sub.labels_, minlength=4)
        n_sub = int((sub_counts >= 0.08 * other_pts.shape[0]).sum())
        if n_sub >= CE_SUB_HUES_MIN:
            return True
    return False


# (b) compound detection


def _compound(
    lab: np.ndarray, opaque: np.ndarray
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    om8 = opaque.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(om8)
    bbox_area = w * h
    if bbox_area == 0:
        return False, None, None

    for path in (
        _compound_via_cluster_envelope,
        _compound_via_light_inner,
        _compound_via_alpha_hole,
    ):
        env, inner = path(lab, opaque, x, y, w, h)
        if env is not None and inner is not None and _inner_is_real(inner, bbox_area):
            return True, env, inner
    return False, None, None


def _compound_via_cluster_envelope(
    lab: np.ndarray, opaque: np.ndarray, x: int, y: int, w: int, h: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """KMeans(2) on all opaque pixels. If one cluster dominates ≥ENVELOPE_DOM
    AND the minor cluster is *clean* (low ab-spread → it's one color, not
    a mix of 3+ colors squished into one cluster), that's the envelope-
    inner split. Also require the largest inner CC to be contained inside
    the overall bbox so AA fringe rings on single-color silhouettes don't
    fire (android/docker/obsidian/homebrew).
    """
    lab_op = lab[opaque]
    if lab_op.shape[0] < 200:
        return None, None
    km = _kmeans_fit(lab_op, 2)
    counts = np.bincount(km.labels_, minlength=2)
    dom = int(counts.argmax())
    if counts[dom] / counts.sum() < ENVELOPE_DOM:
        return None, None

    # Bimodality: both clusters must be tight in ab (not multi-color
    # palettes forcibly merged). Gmail's "non-white" minor spans
    # red+blue+yellow; chrome's "warm" dominant spans red+green+yellow.
    # Both should fail.
    minor_idx = 1 - dom
    minor_pts = lab_op[km.labels_ == minor_idx]
    major_pts = lab_op[km.labels_ == dom]
    if minor_pts.shape[0] < 30:
        return None, None
    minor_ab_std = float(np.linalg.norm(minor_pts[:, 1:3].std(axis=0)))
    if minor_ab_std > MINOR_AB_STD_MAX:
        return None, None
    major_ab_std = float(np.linalg.norm(major_pts[:, 1:3].std(axis=0)))
    if major_ab_std > MAJOR_AB_STD_MAX:
        return None, None

    flat_labels = np.full(opaque.shape, -1, dtype=np.int8)
    flat_labels[opaque] = km.labels_
    env = flat_labels == dom
    inner = opaque & (flat_labels >= 0) & (flat_labels != dom)

    env_bbox = cv2.boundingRect(env.astype(np.uint8))
    ex, ey, ew, eh = env_bbox
    if ew * eh > 0 and (env.sum() / (ew * eh)) >= FILLED_ENVELOPE:
        # Promote env to the full visible silhouette: the dominant cluster
        # was only part of the envelope (e.g. Mastodon's gradient half).
        # The actual brand envelope is the entire opaque region — that is
        # what _precedence's outer-shape tests need to see. Stage-2
        # rendering for outer-precedence paints (env & ~inner) so opaque
        # inner glyphs (YouTube's play triangle, Mastodon's m) remain
        # visible as white cutouts.
        #
        # NOTE: do NOT union alpha-hole pixels into inner here. The real
        # inner glyph's pixels are already captured by (opaque & non-dom)
        # — every fixture in the H9 dataset has an opaque inner. Adding
        # alpha-holes pulled in the rounded-rect corner caps which then
        # rendered as 4 black L-wedges on Claude / Adobe Illustrator /
        # Spotify / Telegram via inner.universalize.
        env = opaque.copy()

    if not _contained_largest_cc(inner, x, y, w, h):
        return None, None
    return env, inner


def _compound_via_light_inner(
    lab: np.ndarray, opaque: np.ndarray, x: int, y: int, w: int, h: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Envelope is non-uniform (e.g. instagram's rainbow gradient) but
    contains a very-light connected component (white camera glyph)
    bounded inside the overall bbox.
    """
    L = lab[..., 0]
    high = (L > INNER_LIGHT_L) & opaque
    if _contained_largest_cc(high, x, y, w, h):
        inner = _significant_components(high, SIGNIFICANT_CC_PX, contain_bbox=(x, y, w, h))
    else:
        # Steam case: a SUBSTANTIAL light glyph on a silhouette that COLLAPSES to a
        # simple primitive is the identity even though it reaches the badge edge (the
        # Steam pipe). The gradient badge is discardable background → inverted, not a
        # cutout. Distinguished from AA fringe rings (no light area) and from real
        # shapes like homebrew (badge does NOT collapse) by the collapse + area gates.
        if high.sum() < LIGHT_GLYPH_FRAC * opaque.sum():
            return None, None
        if fit_simple_primitive(opaque.astype(np.uint8))[1] < COLLAPSE_IOU:
            return None, None
        inner = _significant_components(high, SIGNIFICANT_CC_PX)   # allow edge-touch
    if not inner.any():
        return None, None
    env = opaque & ~inner
    return env, inner


def _compound_via_alpha_hole(
    lab: np.ndarray, opaque: np.ndarray, x: int, y: int, w: int, h: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Envelope is a filled opaque region with a meaningful alpha-zero
    hole inside (mastodon's white 'm' cutout). Requires the envelope
    bbox to be densely opaque (filled square / rect), so we don't
    accidentally fire on hollow outlines like the whatsapp speech-bubble.
    """
    # Envelope = whole opaque region for this path.
    om8 = opaque.astype(np.uint8)
    ex, ey, ew, eh = cv2.boundingRect(om8)
    if ew * eh == 0:
        return None, None
    fill_ratio = opaque.sum() / (ew * eh)
    if fill_ratio < FILLED_ENVELOPE:
        return None, None
    env_box = np.zeros_like(opaque)
    env_box[ey : ey + eh, ex : ex + ew] = True
    holes = (~opaque) & env_box
    if not _contained_largest_cc(holes, x, y, w, h):
        return None, None
    inner = _significant_components(holes, SIGNIFICANT_CC_PX, contain_bbox=(x, y, w, h))
    env = opaque.copy()
    return env, inner


def _contained_largest_cc(mask: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    bbox_area = w * h
    floor = max(50, INNER_AREA_FLOOR * bbox_area)
    if mask.sum() < floor:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return False
    sizes = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(sizes)) + 1
    if stats[idx, cv2.CC_STAT_AREA] < floor:
        return False
    cx = stats[idx, cv2.CC_STAT_LEFT]
    cy = stats[idx, cv2.CC_STAT_TOP]
    cw = stats[idx, cv2.CC_STAT_WIDTH]
    ch_ = stats[idx, cv2.CC_STAT_HEIGHT]
    mx = INNER_CONTAINMENT_MARGIN * w
    my = INNER_CONTAINMENT_MARGIN * h
    return (
        cx > x + mx
        and cy > y + my
        and cx + cw < x + w - mx
        and cy + ch_ < y + h - my
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(mask)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(sizes)) + 1
    return labels == idx


def _significant_components(
    mask: np.ndarray,
    min_size: int,
    contain_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Union of all CCs with area ≥ min_size. If contain_bbox=(x,y,w,h) is
    given, only include CCs whose bbox sits strictly inside (x,y,w,h) with
    INNER_CONTAINMENT_MARGIN slack on every side — used to drop alpha-hole
    rounded-corner caps that would otherwise render as black wedges at the
    canvas corners (Spotify wave-bars, Instagram-style envelopes).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    size_ok = stats[1:, cv2.CC_STAT_AREA] >= min_size
    if contain_bbox is None:
        keep_idx = np.where(size_ok)[0] + 1
    else:
        x, y, w, h = contain_bbox
        mx = INNER_CONTAINMENT_MARGIN * w
        my = INNER_CONTAINMENT_MARGIN * h
        cx = stats[1:, cv2.CC_STAT_LEFT]
        cy = stats[1:, cv2.CC_STAT_TOP]
        cw = stats[1:, cv2.CC_STAT_WIDTH]
        ch_ = stats[1:, cv2.CC_STAT_HEIGHT]
        contained = (cx > x + mx) & (cy > y + my) & (cx + cw < x + w - mx) & (cy + ch_ < y + h - my)
        keep_idx = np.where(size_ok & contained)[0] + 1
    if keep_idx.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, keep_idx)


def _inner_is_real(inner: np.ndarray, bbox_area: int) -> bool:
    """Inner is meaningful only if its largest connected component
    clears INNER_AREA_FLOOR (otherwise it's AA fringe / scattered noise).
    """
    floor = max(50, INNER_AREA_FLOOR * bbox_area)
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        inner.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return False
    sizes = stats[1:, cv2.CC_STAT_AREA]
    return int(sizes.max()) >= floor


# (c) precedence


def _precedence(env: np.ndarray, inner: np.ndarray) -> Precedence:
    """Ordered tests, first-match wins:
      1. If the inner has ≥2 *contained* CCs (excluding rounded-corner
         alpha-hole artifacts at env's bbox edges), it's a structured
         multi-piece glyph — keep inner. (Adobe Ai, Signal outline +
         dashes, Spotify wave bars, Instagram lens+ring+accent,
         YouTube-Music ring + triangle.)
      2. If the inner's best primitive is a clean triangle, it's a
         play-button-style universal UI primitive — keep outer (YouTube).
      3. If the env is a near-filled container (mastodon's purple
         square, telegram's blue disc), the inner is the contained brand
         glyph — keep inner.
      4. Concavity fallback: any remaining non-convex inner is treated
         as structured (signal's bubble-with-tail).
      5. Otherwise primitive-fit comparison: the better-fitting primitive
         is the generic shape and gets dropped.
    """
    env_bbox = cv2.boundingRect(env.astype(np.uint8))
    fit_env, _ = _primitive_fit(env)
    fit_inner, kind_inner = _primitive_fit(inner)

    if _count_contained_ccs(inner, env_bbox) >= 2:
        return "inner"
    if kind_inner == "triangle" and fit_inner >= TRIANGLE_OUTER_FIT:
        return "outer"
    _, _, ew, eh = env_bbox
    if ew * eh > 0 and int(env.sum()) / (ew * eh) >= ENV_NEAR_FILL:
        # Env is a filled container — but is the container itself distinctive?
        # Mastodon's chatbubble (rounded square + tail) → outer-precedence.
        # Telegram disc / LinkedIn square / Adobe Illustrator rrect → inner.
        if _outer_concavity(env) >= ENV_DISTINCTIVE_OUTER_CONCAVITY:
            return "outer"
        return "inner"
    if _concavity(inner) >= CONCAVITY_INNER:
        return "inner"
    return "inner" if fit_env > fit_inner else "outer"


def _count_contained_ccs(mask: np.ndarray, env_bbox: tuple[int, int, int, int]) -> int:
    """Count CCs of `mask` that are significant AND whose bounding box
    lies inside `env_bbox` with INNER_CONTAINMENT_MARGIN slack. This
    drops rounded-rect corner artifacts (alpha-holes at the env's bbox
    edges) which would otherwise inflate the CC count.
    """
    ex, ey, ew, eh = env_bbox
    if ew == 0 or eh == 0:
        return 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return 0
    mx = INNER_CONTAINMENT_MARGIN * ew
    my = INNER_CONTAINMENT_MARGIN * eh
    count = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < SIGNIFICANT_CC_PX:
            continue
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch_ = stats[i, cv2.CC_STAT_HEIGHT]
        if (
            cx > ex + mx
            and cy > ey + my
            and cx + cw < ex + ew - mx
            and cy + ch_ < ey + eh - my
        ):
            count += 1
    return count


def _post_strip_recategorize(inner_mask: np.ndarray | None) -> Routing | None:
    """Conceptual post-Stage-2 step. After Stage 2 strips a compound
    logo's envelope, the revealed inner mark may itself be a burst-style
    radial pattern (Claude's cream sunburst). Detect via two combined
    signals:

      (1) Polygon-vertex count of the inner contour after approxPolyDP
          tolerance RADIAL_BURST_POLY_TOL × perimeter. A k-spike sunburst
          has ~2k vertices (one at each spike tip + one at each valley
          between spikes). ≥ RADIAL_BURST_MIN_VTX captures 10+-spike
          sunbursts; smaller monograms (linkedin's "in", adobe's "Ai")
          stay below.

      (2) Radial profile r(θ): for each of RADIAL_BURST_N_BINS angular
          bins around the inner's centroid, take the max distance from
          centroid to any in-mask pixel. The coefficient of variation
          (std/mean) of this profile is high for starlike shapes
          (alternating spike-tip / valley) and near-zero for rings,
          discs, and outlined bubbles. Floor at RADIAL_BURST_R_CV.

    Vertex count alone catches mastodon's "m" (17 vertices) — too close.
    r_cv alone catches linkedin's "in" (0.43, same as claude). Together
    they isolate the sunburst.

    Returns "burst" if both signals clear their floors; None otherwise
    (caller keeps the initial Stage-1 routing).
    """
    if inner_mask is None or not inner_mask.any():
        return None
    contours, _ = cv2.findContours(
        inner_mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None
    approx = cv2.approxPolyDP(cnt, RADIAL_BURST_POLY_TOL * peri, True)
    if len(approx) < RADIAL_BURST_MIN_VTX:
        return None

    moments = cv2.moments(inner_mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    yy, xx = np.where(inner_mask)
    theta = np.arctan2(yy - cy, xx - cx)
    r = np.hypot(xx - cx, yy - cy)
    edges = np.linspace(-np.pi, np.pi, RADIAL_BURST_N_BINS + 1)
    r_max = np.zeros(RADIAL_BURST_N_BINS)
    for i in range(RADIAL_BURST_N_BINS):
        in_bin = (theta >= edges[i]) & (theta < edges[i + 1])
        if in_bin.any():
            r_max[i] = float(r[in_bin].max())
    mean_r = float(r_max.mean())
    if mean_r < 1.0:
        return None
    if r_max.std() / mean_r < RADIAL_BURST_R_CV:
        return None
    return "burst"


def _concavity(mask: np.ndarray) -> float:
    """1 - IoU(mask, convex_hull). 0 for convex shapes; rises toward 1
    as the mask develops non-convex features (a tail, notches, spikes).
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return 0.0
    hull = cv2.convexHull(cnt)
    hull_canvas = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(hull_canvas, [hull], 1)
    hull_area = int(hull_canvas.sum())
    if hull_area == 0:
        return 0.0
    return 1.0 - int(mask.sum()) / hull_area


def _outer_concavity(mask: np.ndarray) -> float:
    """Concavity of the largest external contour, holes ignored.

    Returns 1 - contour_area / convex_hull_area in [0, 1). Discriminates
    envelopes whose *outer boundary* is non-convex (Mastodon's tail/curl)
    from envelopes whose mask is non-convex only because of inner cutouts
    (YT Music's ring+triangle cutout makes the raw mask concave even though
    the outer disc IS convex).
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return 0.0
    cnt_area = float(cv2.contourArea(cnt))
    if cnt_area <= 0:
        return 0.0
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return 0.0
    return 1.0 - cnt_area / hull_area


def _primitive_fit(mask: np.ndarray) -> tuple[float, str]:
    m8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, "none"
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return 0.0, "none"
    scores = {
        "circle": _iou_with_circle(mask, cnt),
        "rrect": _iou_with_rrect(mask, cnt),
        "triangle": _iou_with_triangle(mask, cnt),
    }
    kind = max(scores, key=lambda k: scores[k])
    return scores[kind], kind


def _iou_with_circle(mask: np.ndarray, cnt: np.ndarray) -> float:
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    if r < 1:
        return 0.0
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(canvas, (int(round(cx)), int(round(cy))), int(round(r)), 1, thickness=-1)
    return _iou(mask, canvas.astype(bool))


def _iou_with_rrect(mask: np.ndarray, cnt: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 2 or h < 2:
        return 0.0
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    radius = max(1, int(round(min(w, h) * 0.15)))
    cv2.rectangle(canvas, (x + radius, y), (x + w - radius, y + h), 1, thickness=-1)
    cv2.rectangle(canvas, (x, y + radius), (x + w, y + h - radius), 1, thickness=-1)
    for (cx, cy) in (
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius),
    ):
        cv2.circle(canvas, (cx, cy), radius, 1, thickness=-1)
    return _iou(mask, canvas.astype(bool))


def _iou_with_triangle(mask: np.ndarray, cnt: np.ndarray) -> float:
    try:
        area, triangle = cv2.minEnclosingTriangle(cnt)
    except cv2.error:
        return 0.0
    if triangle is None:
        return 0.0
    pts = triangle.reshape(-1, 2).astype(np.int32)
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return _iou(mask, canvas.astype(bool))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return inter / union
