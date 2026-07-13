"""Stage 5 phase 5 — connect the composed pieces into ONE continuous 1px stroke.

`compose_icon` (phase 3) leaves the native logo + two frame-letters 1px-SEPARATED, so
`ComposeResult.union` holds several 8-connected 1px components (the logo is often itself a
few pieces — discord = face ring + 2 eyes + mouth). `connect_snake` joins them into a single
pen-stroke with the user's **stricter-snake rule: the connecting line must never form a
degree-3 T-JUNCTION; a degree-4 X / 4-way crossing is fine.**

Mechanism (CONNECT + the connection-side DISCONNECT, per the user rulings):
  - MST over the components by nearest-pixel gap → n-1 single-point 1px `LINE_4` bridges (each
    pair joined once).
  - Each bridge lands mid-stroke, which would be a degree-3 T. **OPEN the crossed stroke**:
    remove one of the join pixel's stroke-neighbours (topology-guarded to stay connected) so the
    join is a clean degree-2 pass — a closed loop (O/D bowl, the discord face ring) opens into
    an arc. Removal-based (no added pixel) → no whisker stubs, unlike the add-a-4-way-crossing
    alternative (the user picked 'open — clean arcs' over 'cross — keep loops closed').
  - Scope = the CONNECTION joins only; each letter's and each logo's OWN internal junctions
    (the react atom's branches, F's bar) are left intact.

Deterministic (H8): `cv2.connectedComponents` row-major labels, `cdist` sqeuclidean + lexsort
tiebreak, hand-Prim's MST with a `(dist,i,j)` tiebreak, `cv2.LINE_4`, fixed 4-neighbour fill
order. No RNG, no interpolating cv2 op.
"""
from __future__ import annotations

import cv2
import numpy as np

from iconograph.application.common import (
    face_connect,
    mst_bridges as _mst_bridges,
    nb4_degree,
)


def _component_masks(union: np.ndarray) -> list[np.ndarray]:
    """Per-label 8-connected component boolean masks of `union`, row-major label order."""
    b = (union > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(b, connectivity=8)
    return [labels == i for i in range(1, n)]


def _open_junction(out: np.ndarray, P: tuple[int, int]) -> None:
    """Make bridge-attach `P` a clean degree-2 pass by REMOVING one of its stroke-neighbours —
    opening the crossed stroke (a closed loop becomes an arc) — chosen so the snake stays ONE
    connected component. No pixel added (no whisker, no collateral T). No-op if `P` isn't a
    degree-3 T or if no removal keeps it connected + degree-2."""
    y, x = P
    if nb4_degree(out)[y, x] != 3:
        return
    base = cv2.connectedComponents((out > 0).astype(np.uint8), connectivity=4)[0]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = y + dy, x + dx
        if not (0 <= ny < out.shape[0] and 0 <= nx < out.shape[1]) or out[ny, nx] == 0:
            continue
        out[ny, nx] = 0
        if (cv2.connectedComponents((out > 0).astype(np.uint8), connectivity=4)[0] == base
                and nb4_degree(out)[y, x] == 2):
            return
        out[ny, nx] = 255                                    # revert — that cut disconnected it


def connect_snake(union: np.ndarray) -> np.ndarray:
    """Join the 8-connected pieces of `union` (uint8 {0,255} or bool 1px strokes) into ONE
    4-connected snake via an MST of single-point 1px `LINE_4` bridges, then try to make each
    bridge junction a clean degree-2 pass instead of a degree-3 T (the user's stricter-snake
    rule) by OPENING the crossed stroke — removing one of the join's stroke-neighbours so a
    closed LOOP becomes an arc — but only when that keeps the snake one connected component.

    **This only fixes joins that land on a closed loop (measured ~57% of bridge joins).** A
    join onto an OPEN-stroke MIDPOINT can't be opened without severing the snake (the topology
    guard reverts the cut), so it STAYS a degree-3 T; `face_connect` can also re-promote a few
    near-join pixels. Removal-based → no whisker stubs, which the ADD-a-crossing alternative
    left in dense icons (the user picked 'open — clean arcs' over 'cross — keep loops closed').
    Use `join_report` / the harness `joins` column to see the true degree-2 rate per icon.
    Returns (H,W) uint8 {0,255}. Single-component/empty input is a byte-identical no-op.
    Deterministic (H8)."""
    out = ((union > 0).astype(np.uint8)) * 255
    comps = _component_masks(out)
    if len(comps) <= 1:
        return out
    bridges = _mst_bridges(comps)
    for pa, pb in bridges:
        cv2.line(out, (pa[1], pa[0]), (pb[1], pb[0]), 255, 1, cv2.LINE_4)   # (x,y); LINE_4
    for pa, pb in bridges:
        _open_junction(out, pa)
        _open_junction(out, pb)
    return (face_connect(out).astype(np.uint8)) * 255


def join_report(union: np.ndarray) -> tuple[int, int]:
    """(#bridge joins, #joins that are NOT a degree-3 T in the final snake). The snake's
    raison-d'être is T-free connection joins, which the harness's `_singly_connected`
    connectivity gate does NOT measure — this exposes the true degree-2 rate (H1)."""
    comps = _component_masks(union)
    if len(comps) <= 1:
        return (0, 0)
    d = nb4_degree(connect_snake(union))
    joins = [int(d[p[0], p[1]]) for pa, pb in _mst_bridges(comps) for p in (pa, pb)]
    return (len(joins), sum(1 for g in joins if g != 3))
