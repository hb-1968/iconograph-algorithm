# iconograph-algorithm

A pipeline that decomposes application logos into a pixelated "abstract" +
monogram and ships them as a Fedora hicolor icon set. The architectural
inspiration is MIT Media Lab's algorithmic identity system: per-target
rescaling and recomposition instead of one canonical bitmap squeezed onto
every grid. Built on Python + OpenCV + scikit-image + scikit-learn + Pillow.

## Pipeline at a glance

```mermaid
flowchart TB
    A[inputs/logo-markings/*.png] --> TS[classify_top<br/>application vs file]
    TS -->|file| FT[file pipeline<br/>DEFERRED]
    TS -->|application| S1

    subgraph APP[application pipeline]
      direction TB
      S1[1. categorize<br/>sub-case detection] --> S2[2. preprocess<br/>universalization]
      S2 --> S3[3. compress<br/>SHARED geometric decomposition<br/>L0 align → L1 classify → L2 segment]
      S3 --> S4[4. refine]
      S4 --> S5[5. monogram]
      S5 --> S6[6. fit<br/>aggressive per-target sizing]
    end

    S6 --> OUT[outputs/<br/>+ manifest.json]
    OUT --> RV[review.html<br/>+ testbed strips]

    classDef shared fill:#e0f0ff,stroke:#06c
    classDef deferred fill:#eee,stroke:#999,stroke-dasharray:5 5
    class S3 shared
    class FT deferred
```

## Pipeline stages

This README documents each stage's **interface** — input contract, output
contract, decision logic, invocation. Algorithm internals live in
`src/iconograph/**`; downstream stages should plan against this README, not
against upstream source (per CLAUDE.md H6 — "applies to LLM context too").

### Stage 1 — categorize (sub-case detection)

**Status:** complete. 25/25 fixtures match ground truth on
`python -m iconograph testbed categorize`.

**Input contract**

- `path: pathlib.Path` pointing at a PNG (any mode — `_load_rgba` handles
  RGB, RGBA, indexed-with-tRNS uniformly via `Pillow.Image.convert("RGBA")`).

**Output contract**

Returns a frozen `Categorization` dataclass with these fields:

| Field             | Type                              | Meaning                                                                                                |
|-------------------|-----------------------------------|--------------------------------------------------------------------------------------------------------|
| `routing`         | `"standard" \| "inverted" \| "burst"` | Which Stage 2 branch the dispatcher will call.                                                          |
| `color_essential` | `bool`                            | Chrome/Firefox signature — small central element inside multi-hue surround. When `True`, routing is always `"burst"`. |
| `is_compound`     | `bool`                            | Envelope + inner-glyph composition (Telegram disc+plane, Claude square+sunburst).                       |
| `precedence`      | `"outer" \| "inner" \| None`        | For compound logos: which sub-element wins. `None` for non-compound.                                    |
| `envelope_mask`   | `(H, W) bool \| None`              | Outer boundary mask when compound; `None` otherwise.                                                    |
| `inner_mask`      | `(H, W) bool \| None`              | Inner glyph/hole mask when compound; `None` otherwise.                                                  |

**Decision tree (routing)**

```mermaid
flowchart TB
    start([categorize PNG]) --> ce{color_essential?}
    ce -->|yes| burst1[routing = burst]
    ce -->|no| comp{is_compound?}
    comp -->|no| std1[routing = standard]
    comp -->|yes| prec{precedence}
    prec -->|outer| std2[routing = standard]
    prec -->|inner| psr{inner is radial<br/>burst pattern?}
    psr -->|yes| burst2[routing = burst]
    psr -->|no| inv[routing = inverted]
```

The post-strip-recategorize check (`psr` node) re-runs after Stage 2 would
strip the envelope: if the revealed inner is a sunburst (Claude), the final
routing is `burst`, not `inverted`. The check uses polygon-vertex count
≥ 20 *and* radial-profile coefficient-of-variation ≥ 0.20 together — either
signal alone misfires.

**Routing distribution across the 25 fixtures**

```
standard  = 15   discord, gmail, whatsapp, slack, figma, android-head_flat,
                 vscode, mastodon, obsidian, apple-rainbow, docker-mark-black,
                 homebrew, react, microsoft-edge, youtube
inverted  =  7   adobe-illustrator, instagram, linkedin, signal, spotify,
                 telegram, youtube-music
burst     =  3   claude, chrome, firefox
```

**Invocation (Python)**

```python
from pathlib import Path
from iconograph.application.categorize import categorize, load_rgba

cat = categorize(Path("fixtures/claude.png"))
# cat.routing == "burst"
# cat.is_compound == True
# cat.precedence == "inner"

rgba = load_rgba(Path("fixtures/claude.png"))   # canonical RGBA loader
# returns (H, W, 4) uint8 cropped to non-transparent bbox
```

**Invocation (testbed)**

```
python -m iconograph testbed categorize
```

Emits `testbed/categorize/grid.png` (3×3 routing confusion matrix +
per-fixture chips + check/cross). Exit code 0 iff every fixture matches
the ground truth in `fixtures/README.md`.

### Stage 2 — preprocess (universalization)

**Status:** complete. 25/25 fixtures pass the contract check on
`python -m iconograph testbed preprocess`.

Stage 2 takes `(Categorization, RGBA)` and dispatches to one of three
branches: `standard.universalize`, `inverted.universalize`, or the
three-phase `color_essential` pipeline (shade-collapse → frame →
feature-separate → compose). All branches converge at `frame_with_margin`,
producing a `(S, S, 3) uint8` square RGB image on white with a 10% margin
around the content's tight bounding box. Burst logos (Claude) fold into the
standard branch at Stage 2 because the mask-selection primitive is
identical to any inner-precedence compound; the burst identity flows
downstream via `cat.routing` for stages 4 and 5 to re-diverge on.

### Stage 3 — compress (geometric decomposition)

**Status:** L0 align + L1 primitive-classify + L2 feature-segment complete. The **contour-driven
decomposer** — D1 `contour_parse` (DONE) + D2/D3 `decompose` — is the **mature primary path**: most
fixtures decompose to **≥0.85 IoU as clean, contiguous constructions** (arc-trace + spill-carve +
notch-subtract + holes + annular ring/sector + holed fill-fit + iterative contour-refinement, all
chosen through one **boundary-aware** `min(area, boundary)` arbitration with a structured-annulus
preference). The **L3 beam** (`L3_fit`) is kept *only* as the `testbed decompose --panels` panel-4
baseline — whole-shape IoU **blob-fitting**, **never reused by D2** (it breaks contiguity). See
*"D1–D3 — contour-driven decomposer"* below for the full design, per-fixture results, and the
**Next (future sessions)** roadmap.

Stage 3 "compresses" the universalized image into a minimal **geometric**
description — fitted primitives where one suffices, split sub-features where
none does — that the deferred fitting layer turns into the final pixel
abstract. It is a sub-pipeline of three implemented sub-stages, each cached
(H6) with its own testbed grid, plus a deferred fourth:

```
align (L0) → segment_prim_classify (L1) → L2_feature_segment (L2) → ┬ L3_fit (beam — IoU baseline)
                                                                    └ contour_parse (D1) → decompose (D2+D3)  ◀ ACTIVE
```

#### D1–D3 — contour-driven decomposer (the active decomposition paradigm)

The decomposer identifies **the clean primitives whose boundaries trace a shape's contour** —
triangles for straight/angular runs, ellipses for curved runs, combined by union (add) + negation
(subtract) — leaving **curve-junction labels** between primitives and a small **complex residual**.
*Why:* clean primitive identity enables idealized geometric pixel-art (a later render stage can
de-rotate / "linearize" e.g. Android's angled pills to straight rects, and overrepresent the
labelled curves), recovering structure that nearest-neighbour pixelation destroys. It is **not**
area-IoU blob-fitting (L3's muddy whole-shape fits — Docker's 7-part 0.92 whale — are the
anti-pattern). Built layer-by-layer; each layer is its own module + `*_debug.py` testbed.

- **D1 — `contour_parse.py` (`parse_shape`).** A feature mask → `ParsedShape{outer, holes}`. Each
  closed contour (via `approxPolyDP`) is split into maximal **straight** and **arc** runs (by the
  **sagitta ratio**, `STRAIGHT_DEV_FRAC=0.12` — semicircle≈0.50, 45°-arc≈0.10), each run + corner
  tagged **convex/concave** (curvature cross-product on a CW-normalized contour), and **interior
  holes** parsed (`RETR_CCOMP`). Types: `ContourSegment{kind, points, start, end, convex}`,
  `Corner{point, kind, convex}`, `ParsedShape` (in `common.py`). Run: `python -m iconograph.d1_debug`
  → strip: aligned | typed contour (straight=blue, convex arc=red, concave arc=purple, hole=amber,
  peak=green, valley=cyan). **DONE, validated, deterministic.**

- **D2+D3 — `decompose.py` (`decompose_shape(mask, min_feat, hint, ring, bbox_xy) → Decomposition`).** Edge-traces
  the boundary: a **convex arc → an ellipse that ADDs**; a **concave arc → an ellipse that
  SUBTRACTs** (so Apple's bite / a ring's inner edge carve generically — *no hint needed*, the
  convexity sign is the linchpin); an added ellipse that **spills** outside the shape is **carved by
  a half-plane** from a straight run (the whale hull's flat top → `ellipse − rect` = half-disc); the
  straight-edged remainder is covered by per-component best-fit primitives; a **concave straight
  notch** (an over-covered spill component) is carved by an `approxPolyDP` **triangle SUBTRACT** —
  inscribed, eats-guarded (the Gmail M's V); **holes subtract** a fitted primitive; a
  **single-primitive `shape_hint`** is fit directly; a **`holed` / annular `shape_hint`** builds a
  **fill-fit ring** (`_fit_holed`: best primitive over the hole-FILLED body ADD − each detected hole
  SUBTRACT) and, when `RingGeom` is threaded, an **annular** construction (`Circle(outer) ADD −
  Circle(inner) SUBTRACT [− two radial half-plane SUBTRACTs]`, `_fit_annular`) — **both compete** with
  the edge-trace (so a ragged ring geometry can't beat a clean fill-fit; youtube-music's ring → 0.95). A
  **structured-construction preference** then selects the clean few-part annulus over a higher-fidelity
  many-part trace when it covers ≥ `STRUCT_MIN_AREA` and is within `STRUCT_TOL` of the best score
  (chrome's global ring `outer − inner − cuts` beats its 19-part local-ellipse trace). Each arc-ellipse fit is
  **bbox-constrained** (axis ≤ `AXIS_BBOX_FRAC` × the feature box, + a centre-escape reject) so a
  partial, ill-conditioned arc can't balloon. Acceptance is **global containment**: the convex-arc
  ellipses are carved (deterministic order), then kept as a group iff the carved construction's spill
  (`|outside S| / |construction|`) ≤ `ACCEPT_SPILL`; an un-traceable / un-carvable run → a **junction
  label**; coverage < `ADEQUATE_COV=0.90` leaves a **residual** mask. **The construction is then
  REFINED** while sub-adequate: the trace loops its own contour-fit passes over the shrinking residual
  (under-coverage → a best-fit ADD; over-coverage → a generalized eats-guarded SUBTRACT,
  `_best_overcover_prim`; both net-positive-guarded, bounded by `REFINE_MAX_ITERS`/`MIN_REFINE_GAIN`) —
  filling under-carved counters and missing negates while staying **one contiguous, contour-aligned
  piece** (deliberately *not* the L3 IoU-beam, which scatters non-contiguous blobs). Sub-adequate
  features also **arbitrate** a whole-shape single primitive and a straight-only trace. **Arbitration is
  contour-aware**: candidates rank by **`min(area IoU, boundary IoU)`** — Boundary IoU (Cheng 2021) is
  IoU restricted to a `d`-px boundary rim (`d = 0.02 × bbox diagonal`), so a high-area but boundary-wrong
  construction (a curved ellipse shaving a flat edge — Adobe's 'A' top; a local ellipse over a global
  ring) loses to the boundary-aligned one. The same `min` keys the spill-subtractor pick
  (`_best_overcover_prim`), flipping Adobe's flat top from a negated ellipse to a straight subtract.
  `coverage` (area IoU) still drives the residual/adequacy gates; `boundary_cov` rides along for display.
  Runs on a padded canvas (spill room). `Decomposition{parts: list[Part], junctions, residual,
  coverage (area IoU), boundary_cov, …}`. Run: **`python -m iconograph testbed decompose`** (cache-only;
  per-fixture `target | D2 overlay` + **split chips** top=area / bottom=boundary IoU, so a high-area /
  low-boundary false positive reads green-over-red) or **`python -m iconograph testbed decompose --panels`**
  (4-panel: target | D1 contour | D2 | old L3 beam). **Mature, deterministic.**

**Output contract:** `{ primitives (signed `Part`s — adds OR'd, then subtracts carved), junction
labels (curves the renderer overrepresents), residual mask (low-res render tier) }`.

**Render tiers — stage 6, NOT built:** R1 idealize/linearize primitives → crisp pixel-art; R2
overrepresent junctions; R3 low-res nearest-neighbour the residual, layered last.

**Current results:** Apple → `body ellipse − bite + leaf` (0.93; leaf 0.95); Docker
whale → `hull half-disc (ellipse − waterline carve) + remainder`, the old balloon eliminated (0.79;
tail is honest residual); android dome → half-disc via flat carve (0.95). The **iterative contour-refinement** then carves the
remaining negates/constructions — Adobe's "A" → `triangle − counter − leg-gap` (0.80→**0.97**, zero
residual) — and lifts nearly the whole sub-adequate set with **zero regressions**: telegram 0.78→0.95,
linkedin 0.69→0.91, obsidian 0.86→0.92, discord 0.80→0.92, chrome sectors 0.76→0.90, docker 0.79→0.89,
spotify ~0.66→0.88, vscode 0.75→0.87, gmail 0.74→0.86, figma 0.72→0.84, whatsapp-sector 0.56→0.82 — all
staying one contiguous piece. The
**balloon overshoot is fixed** (bbox-constrained fit + global-spill acceptance),
`ALIGN_TAU`/`ALIGN_FRAC`/`SPILL_FRAC` are retired into one spill gate. **Tuning roadmap (pick-up points):** **polygon recognition** (try one
dominant primitive, else **ear-triangles**) is still unimplemented — the concave straight-notch
subtract now carves gmail's M-valley (0.74) and any notched logo (vscode 0.75, signal), but crisping
it wants the exact valley triangle (D1's corner + the two inner-run endpoints) and rounded-corner
handling; obsidian (a faceted gem) still needs full straight-sided decomposition; the **ring/annular fitter**
(`Circle(outer) − Circle(inner) [− radial cuts]` from L2's threaded `RingGeom`) plus the **`_fit_holed`
fill-fit**, both competing under the boundary-aware arbitration with a structured-annulus preference,
now render **chrome as a clean 3-sector global annulus** (selected over its 19-part trace, area 0.91→0.76
— a deliberate parsimony tradeoff), fix **youtube-music** (0.00→0.95) and the **whatsapp / instagram**
bands (0.00→0.86 / 0.67), and hold microsoft-edge (0.87); react's thin overlapping orbits stay an honest
residual (0.53). Dataset mean area-IoU **0.817→0.862**, clean fixtures zero-regression; **sunbursts** (claude, firefox) are the next clear gap — one connected blob the residual
loop can't segment, so they fall to a bounding-circle fit (~0.50); they need a contour-driven
**spoke-segmentation** step (separate the radial spikes, fit a thin rotated-rect each); the **capsule**
isn't reused for `rotated_rect` hints (bare rect).

**Next (future sessions):**
- **Sunburst spoke-segmentation** (claude 0.50, firefox 0.69) — a single connected blob the residual
  loop can't split, so it falls to a bounding-circle fit. Add a contour-driven spoke pass: separate the
  radial spikes (distance-erode / convexity defects), fit a thin rotated-rect per spoke. Stay inside the
  contour methodology — **do NOT borrow the L3 beam** (it scatters non-contiguous blobs).
- **Thin overlapping rings** (react = 0.53 honest partial; youtube-music now fixed via `_fit_holed`) —
  react's concentric thin orbit bands still need an annulus-of-annuli / orbit-segmentation path, bundled
  with the sunburst spoke-segmentation above.
- **Decompose polish** (partially DONE — de-mud pass): the 4× component-fit loop is now one
  `_cover_components`; the arbitration's post-hoc STRUCT override is folded into one `_arbitrate`
  (a `structured` flag + a keyed `+STRUCT_TOL` bonus, provably the same selection); `_half_plane_rect`
  is lifted to `common.py`; `NOTCH_POLY_TOL` now derives from D1's `CONTOUR_EPS_FRAC` (×4). All
  byte-identical (zero per-feature delta across the dataset). **Still open:** compute `recon` once per
  call (incremental compound mask — matters for stage-6's 8 hicolor sizes); the optional annular →
  `compress/annular.py` split. **Reverted:** deriving `AXIS_BBOX_FRAC` from the arc sagitta — measured
  mean-neutral, it trades docker's area (0.89→0.86) for boundary (+0.08) / parsimony (17→6 parts) and
  adds an `AXIS_SAFETY` knob, so it's a deliberate quality tradeoff, not a free cleanup (re-land separately).
- **Deferred pipeline stages**: 4 refine · 5 monogram · **6 aggressive-fit — the rendering phase**
  (per-target rasterization onto each hicolor grid + the two-tier render: idealized primitives + low-res
  residual). Then the **file-icon pipeline** (deferred until the application path ships clean outputs).

The **L3 beam** (below) is kept only as the `testbed decompose --panels` panel-4 baseline — it does whole-shape IoU
**blob-fitting** (non-contiguous, non-contour-aligned), so it is **not** the path forward and is never
reused by D2; all decompose refinement stays inside the contour-driven methodology.

#### L0 — align

Axis-aligns dominant straight features so downstream primitive fits aren't
fighting a few degrees of rotation. Input and output are both the
`(S, S, 3) uint8` RGB square (contract-identical to Stage 2's output).
Testbed: `python -m iconograph testbed align`.

#### L1 — segment_prim_classify (primitive classification)

Binarizes, labels 8-connected components, and fits each region to the best
single primitive — **Circle / Rectangle / Triangle / Ellipse** — when one
clears the IoU gate; a region no primitive fits becomes an `L1Miss` carrying
its cropped mask forward.

**Output:** `SegmentClassification{image_shape, binarized, labels,
segments: list[Classified | L1Miss]}`, where
`Classified{primitive, bbox, color_rgb, area_px, iou}` and
`L1Miss{bbox, color_rgb, area_px, mask, best_attempt}`. The shared primitive
fitters and `iou_analytical` live in `compress/common.py` (reused by L2).
Testbed: `python -m iconograph testbed segment_prim_classify`.

#### L2 — L2_feature_segment (feature segmentation)

Splits each `L1Miss` — it does **not** fit it (fitting is L3's job). Per miss:

- **global structure kept whole** — `holed` (interior counterform),
  `sunburst` (radial spikes), or `ring` (a thin band around an empty centre);
- **a detected ring is polar-resolved** — `cv2.warpPolar` unrolls it about
  its centre, connected components in polar space split it into its
  sub-features (Chrome's 3 sectors, Edge's band, Signal's open arc), each
  inverse-warped back to cartesian and tagged `annular_band` /
  `annular_sector`. A seam stitch across θ=0/2π keeps a sector that crosses
  the +x axis as one feature;
- **otherwise body + appendages** — morphological opening peels thin
  protrusions (the body is kept exact as `mask − appendages`); each part is
  identified by its best-fitting primitive, with a `half_circle` fallback,
  else `defer`.

A cross-miss pre-pass groups shared-centre misses that tile one ring
(Chrome's sectors arrive as three separate `L1Miss`es) before resolving.

**Output:** `FeatureSegmentation{image_shape, segments: list[Classified |
Feature]}`, where `Feature{mask, bbox, kind, color_rgb, best_fit_iou,
shape_hint}`, `kind ∈ {local, global, defer}`, and `shape_hint` (e.g.
`circle`, `rotated_rect`, `half_circle`, `annular_sector`, `holed`) routes
the deferred fitter. Testbed: `python -m iconograph testbed L2_feature_segment`
— a 3-panel strip per fixture (aligned | L1 primitives | L2 features) plus
`grid.png`.

#### L3 — L3_fit (geometric shape fitting)

L3 turns every miss-derived `Feature` into a *construction* — signed primitives
(rotated rectangles / ellipses / triangles, circles) combined by union (add) and
negation (subtract), overlap allowed — that reconstructs the shape. Two passes per
feature: a bounded **beam search** (greedy with back-stepping) adds a primitive over the
uncovered residual or subtracts one over the spilled residual, scoring the *whole*
construction's IoU so overshoot is safe (a later subtract carves it); then a **cleanup**
pass drops redundant parts (backward elimination — the "several rectangles on one spot"
case) and gap-fills any remaining hole (the no-disjoint / arc-coverage guarantee). The
fit runs on a padded canvas so an enclosing primitive can overshoot into background and
be carved rather than silently clipped. The L2 `shape_hint` seeds the search
(single-primitive → 1-part fit; `half_circle` → enclosing circle); composite hints
(`holed` / `ring` / `sunburst` / `annular_*`) cold-start in v1 — dedicated templates are
a later pass.

**Deferral is a fit-adequacy decision:** a feature whose cleaned construction stays below
`ADEQUATE_IOU` becomes a `Deferral` (carrying its mask forward) rather than a forced
low-quality shape. This supersedes L2's `kind="defer"` tag, which now only signals "no
hint → cold start."

**Output:** `FittedDecomposition{image_shape, shapes: list[Classified | FittedShape |
Deferral]}`, where `FittedShape{bbox, color_rgb, parts, iou, shape_hint}` and
`Part{primitive, op ∈ {add, subtract}}` (rotated primitives + the construction algebra
live in `compress/common.py`). Determinism (H8): fixed-order candidates + total
`_part_key` tie-breaks, no RNG. Validated by the direct-run harness `python -m
iconograph.l3_debug` (default fixtures: android-head_flat, gmail, obsidian), which writes
a 4-panel strip per fixture (aligned | L2 features | L3 pass-1 | L3 pass-2) to
`testbed/L3_fit/`; full-testbed wiring is deferred.

### Stage 4 — refine

**Status:** not yet implemented.

Post-quantization cleanup: stray-pixel snapping, isolated-pixel artifact
removal, minimum-feature-size enforcement per hicolor target, edge-aware
corrections. Mostly shared logic in `refine/common.py`; class-specific
passes (e.g. radial-symmetry preservation for burst) only where shared
logic visibly fails on a sub-case.

### Stage 5 — monogram (selection + generation)

**Status:** not yet implemented.

Pick the monogram glyph (extracted letter, embedded mark, or
filename-derived fallback) and render it at canonical working resolution.
Per-class: `inverted` uses a different luminance threshold for letter
extraction; `monogram-only` logos forward their own glyph unchanged;
`standard` logos go through connected-component / OCR letter extraction.

### Stage 6 — fit (aggressive per-target sizing)

**Status:** not yet implemented.

For each Fedora hicolor size (16, 22, 24, 32, 48, 64, 128, 256), search
over `(scale, x-offset, y-offset)` to maximize the proportion of the N×N
canvas occupied by ink, subject to: tight bounding-box preservation,
minimum legible monogram glyph height per target, and a per-size padding
floor. Per-target rasterization happens here. Writes the Fedora hicolor
tree + updates `manifest.json`.

## File-icon pipeline

**Deferred** until the application pipeline ships. `src/iconograph/file/`
is a stub directory; `src/iconograph/classify_top.py` routes every
current fixture to `"application"` and raises `NotImplementedError` for
`"file"` so the unimplemented boundary surfaces in stack traces rather
than being silently absorbed.

## CLI

```
python -m iconograph build                  # batch over inputs/logo-markings/, skip unchanged
python -m iconograph build --force          # reprocess everything
python -m iconograph build --only "<glob>"  # reprocess matching inputs
python -m iconograph review                 # regenerate review.html
python -m iconograph testbed <stage>        # in-development stage runner
python -m iconograph bench [<stage>] [--serial]   # time the implemented stages (H1 baseline)
```

Currently wired testbed stages: `categorize`, `preprocess`, `align`,
`segment_prim_classify`, `L2_feature_segment`. Others return
`not implemented` until their stage lands. The single-process
`l2seg_debug` module (`python -m iconograph.l2seg_debug [stem ...]`) is a
fast numeric L2 report + `--dump <stem>` overlay for tight iteration
without forking a worker pool.

## Repository layout

```
src/iconograph/
    __main__.py            CLI entrypoint
    pipeline.py            top-level dispatch (application vs file)
    classify_top.py        icon-class detector
    manifest.py            hash + skip-don't-rebuild (per H8)
    review.py              review.html generator
    testbed.py             cached-stage runner + comparison-strip emitter
    application/
        pipeline.py        stage 1 → 6 orchestrator
        categorize.py      Stage 1
        preprocess/        Stage 2 — standard.py, inverted.py, color_essential.py, common.py
        compress/          Stage 3 — align.py (L0), segment_prim_classify.py (L1),
                                     L2_feature_segment.py (L2), common.py (shared fitters)
        refine/            Stage 4 — common.py, burst.py
        monogram/          Stage 5 — standard.py, inverted.py, common.py
        fit/               Stage 6 — search.py, compose.py, common.py
    file/                  DEFERRED
fixtures/                  ≥20 reference logos spanning sub-cases (per H9)
inputs/logo-markings/      user-provided, gitignored
outputs/                   generated, gitignored
.cache/stages/             per-fixture intermediate outputs (per H6), gitignored
testbed/                   in-development comparison strips (per H7), gitignored
```
