"""Lightweight stage-4 refine harness — pixel-level perimeter outline + disjoint repair.

    python -m iconograph.refine_debug                  # default: discord, gmail, vscode, claude, figma
    python -m iconograph.refine_debug <stem> [stem ...]  # explicit set
    python -m iconograph.refine_debug --twice          # H8 determinism assert (pixels + trace order)
    python -m iconograph.refine_debug --trace          # per-component normalize decision table

Per fixture x feed size {16, 24}: composite the stage-3 render -> full refine chain un-blot ->
widen (@24-gated) -> deburr edge nibs (ref-gated) -> normalize_stroke_width (per-stroke
thin/hollow-widen/keep arbitration on the blot metric; trace threaded here) -> symmetrize
(retention-guarded accents + fold-then-repair) -> perimeter_ring (the arbitrated outline tail: plain
AND pocket-averse connect_disjoint -> reopen_channels chain-blot carve ->
relieve_tight_pockets outward wall relief, competing on count_tight_pockets), wrapped by
the production accessory passes: carve_accent before / restore_accent after (protected
assembly accents), fill_glyph_channels before the chain and stamp_glyph_ink after
perimeter_ring (mastodon's m ink-lines). The chain is
IDENTICAL to production refine_perimeter. Emits a side-by-side strip `render | outline raw |
connected (fills red) | stroke (normalize diff, red adds / blue thins) | refined outline
(net red/blue)` to testbed/refine/<stem>.png. Report columns per cell: pass px deltas,
fired=<acted>/<comps>, blot=<2x2 all-ink blocks> and tight=<count_tight_pockets> on the
FINAL ring (the two gate metrics), the tail px delta, and the per-stage thin-residue chain
resid sil/ub/wid/deb/nrm. Reuses render_debug's fixture->masks seam (no decomposition
re-plumbing). BLAS pinned to 1 thread.

PROTECTED ACCENT (assemble_render's third return): a deliberately-placed assembly accent
(instagram's separated camera dot) is CARVED OUT of the silhouette before the chain and
OR-ed back onto the final ring — mirroring production refine_perimeter's `protected=` —
because refine's oracles are blind to it (its reference is empty at the placed position).
Column semantics with an accent present: blot + tight* + the c4/c8 caption are measured on
the RESTORED ring (the shipped product; tight* excludes the accent zone — instagram@24's
3x3-OUTLINE dot carries a user-ratified 1x1 interior hole that must not read as wireframe
tightness); raw tight/asym/ret/stray stay on the chain's OWN pre-restore product (the
arbitration never saw the dot — its metrics aren't charged for it, and the dot is an
INTENTIONAL asymmetry/non-ref accent that would misread as asym/stray). With a glyph
channel present, `stray` additionally excludes the glyph-channel/valley zone (the fill
and the K ink are deliberate ref-empty ink, not render noise); blot/tight/tight* stay
charged for the glyph zone (K is trimmed 8-separated off the ring, probed blot-clean).

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

from iconograph.application.compress.render import _flip_mask  # noqa: E402
from iconograph.application.refine.common import (  # noqa: E402
    _SE3,
    _best_flip_axis,
    _blot2x2,
    _enclosed_bg,
    _label_axes,
    _min_feat_at,
    _retention_loss,
    _significant_holes,
    build_reference,
    carve_accent,
    connect_disjoint,
    count_tight_pockets,
    count_tight_ref,
    deburr_nibs,
    despeck_artifacts,
    fill_glyph_channels,
    glyph_ink_lines,
    normalize_stroke_width,
    perimeter_ring,
    restore_accent,
    silhouette_outline,
    stamp_glyph_ink,
    symmetrize_render,
    unblot_features,
    widen_underrepresented,
)
from iconograph.cache import CACHE_ROOT, TESTBED_ROOT  # noqa: E402
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
DUMP_DIR = TESTBED_ROOT / "refine"
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


def _thin_px(mask: np.ndarray) -> int:
    """Count of <=2px-stroke pixels: foreground a 3x3 MORPH_OPEN removes (the
    normalize-stroke detection idiom) — the per-stage residue column."""
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _SE3,
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return int(np.count_nonzero((mask > 0) & (opened == 0)))


def _col_asym(ring: np.ndarray, cons, canvas_hw) -> int | None:
    """Final-ring asymmetry px about the LABEL's mirror axes (max over axes); None when
    the label has no accepted axis — the detail-retention rework's symmetry column."""
    axes = _label_axes(cons, canvas_hw)
    if not axes:
        return None
    worst = 0
    for vert in axes:
        fa = _best_flip_axis(ring, vert)
        if fa is None:
            return None
        worst = max(worst, int(np.count_nonzero(
            (ring > 0) != (_flip_mask(ring, fa[0], vert) > 0))))
    return worst


def _col_ret(sil_in: np.ndarray, sil_out: np.ndarray, reference: np.ndarray) -> int:
    """RETENTION column: chain-INPUT ink that is reference-supported and whose 3x3-local
    ink the final silhouette lost (call-through to the shared retention formula)."""
    return _retention_loss(sil_in, sil_out, reference)


def _col_blot(ring: np.ndarray, protected: np.ndarray | None) -> int:
    """blot column: 2x2 all-ink blocks in the final ring, EXCLUDING blocks
    touching the 1px-dilated protected-accent zone (the tight* doctrine: a
    deliberately-placed attached accent — signal's tail nub meeting the ring
    wall — is a placement decision, not line-art smudge; the chain's
    arbitration never saw it)."""
    if protected is None or not protected.any():
        return _blot2x2(ring)
    pz = cv2.dilate((protected > 0).astype(np.uint8), _SE3,
                    borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    b = (ring > 0)
    blocks = b[:-1, :-1] & b[:-1, 1:] & b[1:, :-1] & b[1:, 1:]
    pzb = pz[:-1, :-1] | pz[:-1, 1:] | pz[1:, :-1] | pz[1:, 1:]
    return int(np.count_nonzero(blocks & ~pzb))


def _col_stray(sil_out: np.ndarray, reference: np.ndarray) -> int:
    """Final-silhouette ink with NO reference ink anywhere in its 3x3 (spill/artifacts)."""
    refn = cv2.dilate(reference, _SE3, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return int(np.count_nonzero((sil_out > 0) & ~refn))


def _col_lost_holes(ring: np.ndarray, reference: np.ndarray, target: int) -> int:
    """REFERENCE significant holes with no enclosed-bg overlap in the final ring — a
    lost counter (discord's eyes) scores 1 each."""
    encl = np.zeros(ring.shape, bool)
    for cells, *_r in _enclosed_bg(ring):
        for c in cells:
            encl[c] = True
    return sum(1 for h in _significant_holes(reference, _min_feat_at(target))
               if not bool(np.any((h > 0) & encl)))


def _process(stem: str, sizes: tuple[int, ...], twice: bool, trace_on: bool = False) -> bool:
    if not (CACHE_ROOT / stem / "L2_feature_segment.pkl").exists():
        print(f"{stem}: no cached L2 under .cache/stages/ — run the render/testbed first")
        return False
    S, _aligned, cons, src = _fixture_constructions(stem)
    if not cons:
        print(f"{stem}: nothing to render")
        return False
    sils: dict[int, np.ndarray] = {}
    prots: dict[int, np.ndarray | None] = {}
    for t in sizes:
        masks, _icon, prot = assemble_render(cons, S, t, src)
        s = np.zeros((t, t), np.uint8)
        for m in masks:
            s |= m
        sils[t] = s
        prots[t] = prot
    ref_t = max(sizes)
    rows: list[np.ndarray] = []
    for t in sizes:
        sil_full = sils[t]
        # protected-accent carve-out, mirroring production refine_perimeter: the chain runs
        # on the accent-free silhouette (refine's oracles are blind to a deliberately-placed
        # assembly accent — its ref is empty at the placed position); the accent is OR-ed
        # back onto the final ring for the product columns/panels. tight* excludes its zone
        # (the @24 dot's 1x1 interior is a user-ratified form, not wireframe tightness);
        # raw tight/asym/stray stay measured on the chain's OWN product — the arbitration
        # never saw the dot, so its metrics aren't charged for it.
        prot = prots[t]
        sil, acc, acc_ccs, acc_att = carve_accent(sil_full, prot)
        # W3 mirror — mastodon's m: channel FILL before the chain, zone-guarded
        # widen/normalize, K ink-lines stamped after perimeter_ring (production
        # refine_perimeter order exactly). stray excludes the zone (the fill is
        # deliberate ref-empty ink, not render noise).
        gz = glyph_ink_lines(cons, (S, S), t)
        gzone = gz[0] if gz is not None else None
        if gz is not None:
            sil = fill_glyph_channels(sil, gzone)
        # the larger feed sibling (@24) guides @16's widen; @24 itself gets no
        # reference. The sibling's PROTECTED ACCENTS (comb stamps, dots, tails)
        # are carved from it — widen must never chase a deliberately-placed
        # accent's footprint (production callers pass the same carved sibling).
        if t < ref_t:
            structure_ref = sils[ref_t]
            if prots[ref_t] is not None:
                structure_ref = structure_ref.copy()
                structure_ref[prots[ref_t] > 0] = 0
        else:
            structure_ref = None
        # ideal-geometry reference — the shared build (W4 relayout substitution +
        # valley zone), production refine_perimeter's exact oracle
        reference, valley_zone = build_reference(cons, (S, S), t)
        if valley_zone is not None:
            gzone = valley_zone if gzone is None else (gzone | valley_zone)
        ring = silhouette_outline(sil)
        conn = connect_disjoint(ring, sil, t)
        if twice:
            again = connect_disjoint(silhouette_outline(sil), sil, t)
            print(f"  determinism({stem}@{t}): "
                  f"{'IDENTICAL' if np.array_equal(conn, again) else 'MISMATCH'}")
        added = (conn > 0) & ~(ring > 0)
        c8, c4 = _components(conn, 8), _components(conn, 4)
        # full refine order: un-blot -> despeck -> widen (@24-gated) -> deburr (ref-gated edge nibs)
        ub = unblot_features(sil, t)
        dsp = despeck_artifacts(ub, t, reference)            # artifact removal (pre-widen)
        wid = widen_underrepresented(dsp, t, structure_ref,
                                     protected_bg=gzone)     # self-gates: no-op @24 / radial family
        deb = deburr_nibs(wid, t, reference)                 # dual-radial gate / outside-ref nibs
        tr: list[dict] = []
        nrm = normalize_stroke_width(deb, t, reference=reference, trace=tr,
                                     protected_bg=gzone)     # 2px -> 1/3px, arbitrated
        n_comps = tr[0]["n_comps"] if tr else 0
        fired = sum(1 for r in tr[1:]
                    if str(r["verdict"]).startswith(("thin", "widen", "radial-thin")))
        sym = symmetrize_render(nrm, cons, (S, S), t, reference)   # fold to the label's mirror axes
        conn0 = connect_disjoint(silhouette_outline(sym), sym, t)
        w_conn = perimeter_ring(sym, t)                      # FINAL refine_perimeter product (outline)
        if gz is not None:
            w_conn = stamp_glyph_ink(w_conn, gz[1], t)       # the m as 1px interior ink lines
        w_conn_full = restore_accent(w_conn, acc, acc_ccs, acc_att)   # + accent, attach-repaired
        removed = (wid > 0) & (sym == 0)                     # net silhouette deletions (blue markers)
        sil_added = (sym > 0) & (wid == 0)                   # net silhouette additions (red markers)
        wc8, wc4 = _components(w_conn_full, 8), _components(w_conn_full, 4)
        moved = int(np.count_nonzero(ub != sil))
        despd = int(np.count_nonzero(dsp != ub))
        widened = int(np.count_nonzero(wid != dsp))
        deburred = int(np.count_nonzero(deb != wid))
        stroked = int(np.count_nonzero(nrm != deb))
        symd = int(np.count_nonzero(sym != nrm))
        taild = int(np.count_nonzero(w_conn != conn0))   # reopen + relief + averse, vs plain connect
        if twice:
            ub2 = unblot_features(sil, t)
            dsp2 = despeck_artifacts(ub2, t, reference)
            wid2 = widen_underrepresented(dsp2, t, structure_ref, protected_bg=gzone)
            deb2 = deburr_nibs(wid2, t, reference)
            tr2: list[dict] = []
            nrm2 = normalize_stroke_width(deb2, t, reference=reference, trace=tr2,
                                          protected_bg=gzone)
            sym2 = symmetrize_render(nrm2, cons, (S, S), t, reference)
            w_conn2 = perimeter_ring(sym2, t)
            if gz is not None:
                w_conn2 = stamp_glyph_ink(w_conn2, gz[1], t)
            same = (np.array_equal(sym, sym2) and np.array_equal(w_conn, w_conn2)
                    and tr == tr2)                       # pixel AND decision-order drift
            print(f"  refine-determinism({stem}@{t}): {'IDENTICAL' if same else 'MISMATCH'}")
        print(f"  {stem:<12} @{t:<3} connected: comp8={c8} comp4={c4}  "
              f"unblot {'moved ' + str(moved) + 'px' if moved else 'no-op'}  "
              f"desp {'-' + str(despd) + 'px' if despd else 'no-op'}  "
              f"widen {'+' + str(widened) + 'px' if widened else 'no-op'}  "
              f"deburr {'-' + str(deburred) + 'px' if deburred else 'no-op'}  "
              f"stroke {stroked}px  sym {symd}px  tail {taild}px  "
              f"fired={fired}/{n_comps}  "
              f"blot={_col_blot(w_conn_full, prot)}  tight={count_tight_pockets(w_conn)}  "
              f"tight*={count_tight_ref(w_conn_full, reference, protected=prot)}  "
              f"asym={'-' if (av := _col_asym(w_conn, cons, (S, S))) is None else av}  "
              f"ret={_col_ret(sil, sym, reference)}  "
              f"stray={_col_stray(sym if gzone is None else np.where(gzone, 0, sym).astype(np.uint8), reference)}  "
              f"lost={_col_lost_holes(w_conn, reference, t)}  "
              f"resid sil={_thin_px(sil)} ub={_thin_px(ub)} "
              f"wid={_thin_px(wid)} deb={_thin_px(deb)} nrm={_thin_px(nrm)}")
        if trace_on and tr:
            h = tr[0]
            print(f"    trace: radial={h['radial']} thin_px={h['thin_px']} n_comps={h['n_comps']}")
            for r in tr[1:]:
                print(f"    comp{r['comp']}: area={r['area']} len={r['len']} "
                      f"frag={r['frag']:.2f} -> {r['verdict']}")
        stroke_add = (nrm > 0) & (deb == 0)
        stroke_del = (deb > 0) & (nrm == 0)
        rows.append(hstack([
            labeled(_panel(sil_full), f"{stem}@{t} render"),
            labeled(_panel(ring), f"outline raw  c4={_components(ring, 4)} c8={_components(ring, 8)}"),
            labeled(_panel(conn, added), f"connected  c4={c4} c8={c8}  +{int(added.sum())}px"),
            labeled(_panel(nrm, stroke_add, stroke_del),
                    f"stroke  +{int(stroke_add.sum())}/-{int(stroke_del.sum())}px  fired={fired}/{n_comps}"),
            labeled(_panel(w_conn_full, sil_added, removed),
                    f"refined outline  c4={wc4} c8={wc8}  +{int(sil_added.sum())}/-{int(removed.sum())}px"),
        ]))
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_vstack(rows), "RGB").save(DUMP_DIR / f"{stem}.png", "PNG", optimize=True)
    return True


def _fixture_worker(args: tuple[str, tuple[int, ...], bool, bool]) -> tuple[bool, str]:
    """Pool worker: process ONE fixture single-threaded (cv2 serial — inherited at fork, re-asserted
    here so warpPolar in _is_radial stays deterministic, H8), capturing its stdout so the parent
    emits each fixture's block whole (not interleaved). Returns (ok, captured-log); a per-fixture
    failure is reported in the log and yields ok=False (the batch never dies on one bad fixture)."""
    stem, sizes, twice, trace_on = args
    cv2.setNumThreads(1)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ok = _process(stem, sizes, twice, trace_on)
    except Exception as e:  # noqa: BLE001 — debug harness: report and continue
        return False, f"{stem}: ERROR {type(e).__name__}: {e}\n"
    return ok, buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cv2.setNumThreads(1)
    twice = "--twice" in argv
    serial = "--serial" in argv
    want_all = "--all" in argv
    trace_on = "--trace" in argv
    argv = [a for a in argv if a not in ("--twice", "--serial", "--all", "--trace")]
    wanted = _all_cached() if want_all else (argv if argv else list(DEFAULT_FIXTURES))
    n = 0
    if serial or len(wanted) <= 1:
        for stem in wanted:
            try:
                if _process(stem, GATE_SIZES, twice, trace_on):
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
        work = [(stem, GATE_SIZES, twice, trace_on) for stem in wanted]
        with mp.get_context("fork").Pool(njobs) as pool:
            for i, (ok, log) in enumerate(pool.imap_unordered(_fixture_worker, work), 1):
                print(f"  [{i}/{len(work)} done]\n{log}", end="", flush=True)  # stream as each finishes
                n += int(ok)
    print(f"{n} fixture strip(s) → {DUMP_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
