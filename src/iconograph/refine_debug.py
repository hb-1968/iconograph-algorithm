"""Lightweight stage-4 refine harness — pixel-level perimeter outline + disjoint repair.

    python -m iconograph.refine_debug                  # default: discord, gmail, vscode, claude, figma
    python -m iconograph.refine_debug <stem> [stem …]  # explicit set
    python -m iconograph.refine_debug --twice          # H8 determinism assert

Per fixture × feed size {16, 24}: composite the stage-3 render → silhouette → 1-px boundary
perimeter (refine.common.silhouette_outline) → diagonal "disjoint" repair
(refine.common.connect_disjoint), plus the full refine chain un-blot → widen → deburr edge nibs
(refine.common.deburr_nibs, ref-gated). Emits a side-by-side strip
`render | outline raw | connected (fills red) | refined outline (refine red / deburr blue)` to
testbed/refine/<stem>.png. Reuses render_debug's fixture→masks seam (no decomposition
re-plumbing). BLAS pinned to 1 thread.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import contextlib  # noqa: E402
import io  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# Pin cv2 to one thread BEFORE the iconograph imports below: their module-level cv2 ops
# (e.g. refine.common's getStructuringElement) otherwise warm the thread pool multi-threaded,
# which makes warpPolar in _is_radial nondeterministic (H8) — and a later pin can't shrink a warm
# pool. Must run before those imports, not just in main().
cv2.setNumThreads(1)

from iconograph.application.compress.render import reference_mask  # noqa: E402
from iconograph.application.refine.common import (  # noqa: E402
    connect_disjoint,
    deburr_nibs,
    normalize_stroke_width,
    silhouette_outline,
    symmetrize_render,
    unblot_features,
    widen_underrepresented,
)
from iconograph.cache import CACHE_ROOT  # noqa: E402
from iconograph.debug_common import hstack, labeled  # noqa: E402
from iconograph.render_debug import (  # noqa: E402
    DEFAULT_FIXTURES,
    GATE_SIZES,
    _all_cached,
    _fixture_constructions,
    _half_cores,
    assemble_render,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMP_DIR = REPO_ROOT / "testbed" / "refine"
PANEL_H = 240

# The inner-feature continuity widen (widen_underrepresented) now runs on EVERY feed-res fixture;
# it self-gates two ways: a no-op at @24 (structure_ref=None, the authoritative reference) and a
# no-op on the radial-blade family (claude/firefox sunbursts, _is_radial) which the project
# excludes from structure mechanisms. Everything else is widened toward its @24 footprint.


def _panel(mask: np.ndarray, added: np.ndarray | None = None,
           removed: np.ndarray | None = None) -> np.ndarray:
    """Black-on-white render of a {0,255} mask, integer-upscaled NEAREST to ~PANEL_H so the
    pixels stay crisp; `added` (bool) pixels are tinted red (disjoint fills / widen adds),
    `removed` (bool) pixels are tinted blue (deburr edge-nib deletions)."""
    t = mask.shape[0]
    rgb = np.full((t, t, 3), 255, np.uint8)
    rgb[mask > 0] = (30, 30, 30)
    if added is not None:
        rgb[added] = (220, 40, 40)
    if removed is not None:
        rgb[removed] = (40, 80, 220)
    s = max(1, PANEL_H // t)
    return cv2.resize(rgb, (t * s, t * s), interpolation=cv2.INTER_NEAREST)


def _vstack(rows: list[np.ndarray]) -> np.ndarray:
    """Stack equal-width rows with a thin grey separator."""
    bar = np.full((4, rows[0].shape[1], 3), 0xCC, np.uint8)
    out: list[np.ndarray] = []
    for i, r in enumerate(rows):
        out.append(r)
        if i < len(rows) - 1:
            out.append(bar)
    return np.vstack(out)


def _components(mask: np.ndarray, conn: int) -> int:
    """Foreground component count under the given connectivity (4 or 8)."""
    return cv2.connectedComponents(mask, connectivity=conn)[0] - 1


def _process(stem: str, sizes: tuple[int, ...], twice: bool) -> bool:
    if not (CACHE_ROOT / stem / "L2_feature_segment.pkl").exists():
        print(f"{stem}: no cached L2 under .cache/stages/ — run the render/testbed first")
        return False
    S, _aligned, cons = _fixture_constructions(stem)
    if not cons:
        print(f"{stem}: nothing to render")
        return False
    sils: dict[int, np.ndarray] = {}
    for t in sizes:
        masks, _icon = assemble_render(cons, S, t)
        s = np.zeros((t, t), np.uint8)
        for m in masks:
            s |= m
        sils[t] = s
    ref_t = max(sizes)
    rows: list[np.ndarray] = []
    for t in sizes:
        sil = sils[t]
        # the larger feed sibling (@24) guides @16's widen; @24 itself gets no reference
        structure_ref = sils[ref_t] if t < ref_t else None
        # ideal-geometry reference (union of per-construction rasters) — the deburr's outside-ref oracle
        reference = np.zeros((t, t), np.uint8)
        for parts, _c, _l in cons:
            reference |= reference_mask(parts, (S, S), t)
        ring = silhouette_outline(sil)
        conn = connect_disjoint(ring, sil, t)
        if twice:
            again = connect_disjoint(silhouette_outline(sil), sil, t)
            print(f"  determinism({stem}@{t}): "
                  f"{'IDENTICAL' if np.array_equal(conn, again) else 'MISMATCH'}")
        added = (conn > 0) & ~(ring > 0)
        c8, c4 = _components(conn, 8), _components(conn, 4)
        # full refine order: un-blot -> widen (@24-gated) -> deburr (ref-gated edge nibs)
        ub = unblot_features(sil, t)
        wid = widen_underrepresented(ub, t, structure_ref)   # self-gates: no-op @24 / radial family
        deb = deburr_nibs(wid, t, reference)                 # self-gates: no-op if radial / nothing outside-ref
        nrm = normalize_stroke_width(deb, t)                 # 2px lines -> 1px (thin) or 3px (widen)
        sym = symmetrize_render(nrm, cons, (S, S), t)        # fold to the label's mirror axes
        w_conn = connect_disjoint(silhouette_outline(sym), sym, t)   # FINAL refine_perimeter product (outline)
        removed = (wid > 0) & (sym == 0)                     # net silhouette deletions (blue markers)
        sil_added = (sym > 0) & (wid == 0)                   # net silhouette additions (red markers)
        wc8, wc4 = _components(w_conn, 8), _components(w_conn, 4)
        moved = int(np.count_nonzero(ub != sil))
        widened = int(np.count_nonzero(wid != ub))
        deburred = int(np.count_nonzero(deb != wid))
        stroked = int(np.count_nonzero(nrm != deb))
        symd = int(np.count_nonzero(sym != nrm))
        if twice:
            ub2 = unblot_features(sil, t)
            wid2 = widen_underrepresented(ub2, t, structure_ref)
            deb2 = deburr_nibs(wid2, t, reference)
            nrm2 = normalize_stroke_width(deb2, t)
            sym2 = symmetrize_render(nrm2, cons, (S, S), t)
            print(f"  refine-determinism({stem}@{t}): "
                  f"{'IDENTICAL' if np.array_equal(sym, sym2) else 'MISMATCH'}")
        print(f"  {stem:<12} @{t:<3} connected: comp8={c8} comp4={c4}  "
              f"unblot {'moved ' + str(moved) + 'px' if moved else 'no-op'}  "
              f"widen {'+' + str(widened) + 'px' if widened else 'no-op'}  "
              f"deburr {'-' + str(deburred) + 'px' if deburred else 'no-op'}  "
              f"stroke {stroked}px  sym {symd}px")
        rows.append(hstack([
            labeled(_panel(sil), f"{stem}@{t} render"),
            labeled(_panel(ring), f"outline raw  c4={_components(ring, 4)} c8={_components(ring, 8)}"),
            labeled(_panel(conn, added), f"connected  c4={c4} c8={c8}  +{int(added.sum())}px"),
            labeled(_panel(w_conn, sil_added, removed),
                    f"refined outline  c4={wc4} c8={wc8}  +{int(sil_added.sum())}/-{int(removed.sum())}px"),
        ]))
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_vstack(rows), "RGB").save(DUMP_DIR / f"{stem}.png", "PNG", optimize=True)
    return True


def _fixture_worker(args: tuple[str, tuple[int, ...], bool]) -> tuple[bool, str]:
    """Pool worker: process ONE fixture single-threaded (cv2 serial — inherited at fork, re-asserted
    here so warpPolar in _is_radial stays deterministic, H8), capturing its stdout so the parent
    emits each fixture's block whole (not interleaved). Returns (ok, captured-log); a per-fixture
    failure is reported in the log and yields ok=False (the batch never dies on one bad fixture)."""
    stem, sizes, twice = args
    cv2.setNumThreads(1)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ok = _process(stem, sizes, twice)
    except Exception as e:  # noqa: BLE001 — debug harness: report and continue
        return False, f"{stem}: ERROR {type(e).__name__}: {e}\n"
    return ok, buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cv2.setNumThreads(1)
    twice = "--twice" in argv
    serial = "--serial" in argv
    want_all = "--all" in argv
    argv = [a for a in argv if a not in ("--twice", "--serial", "--all")]
    wanted = _all_cached() if want_all else (argv if argv else list(DEFAULT_FIXTURES))
    n = 0
    if serial or len(wanted) <= 1:
        for stem in wanted:
            try:
                if _process(stem, GATE_SIZES, twice):
                    n += 1
            except Exception as e:  # noqa: BLE001 — debug harness: report and continue
                print(f"{stem}: ERROR {type(e).__name__}: {e}")
    else:
        # Process-level parallelism: N single-threaded workers (one fixture each) use all N cores
        # while every worker stays cv2-serial for H8 (fork inherits the parent's 1-thread pool + the
        # OMP/BLAS=1 env). ICONOGRAPH_JOBS overrides the default half-cores budget (e.g. =16 for all).
        env = os.environ.get("ICONOGRAPH_JOBS")
        cap = max(1, int(env)) if env else _half_cores()
        njobs = max(1, min(cap, len(wanted)))
        print(f"  [parallel: {njobs} workers of {os.cpu_count()} cores, "
              f"each single-threaded for H8 determinism]", flush=True)
        work = [(stem, GATE_SIZES, twice) for stem in wanted]
        with mp.get_context("fork").Pool(njobs) as pool:
            for i, (ok, log) in enumerate(pool.imap_unordered(_fixture_worker, work), 1):
                print(f"  [{i}/{len(work)} done]\n{log}", end="", flush=True)  # stream as each finishes
                n += int(ok)
    print(f"{n} fixture strip(s) → {DUMP_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
