# Implemented pipeline

The build command calls `application.pipeline.build_icon`. The older top-level
`pipeline.dispatch` is not the production orchestrator. Stages operate on arrays
and geometric constructions; no model is trained on the fixture set.

| Stage | Input → output | Main implementation |
| :-- | :-- | :-- |
| Categorize | PNG → routing, compound masks and color flags | `application/categorize.py` |
| Preprocess | Categorization + RGBA → framed RGB on white | `application/preprocess/` |
| Align/classify | RGB → aligned image and fitted connected components | `application/compress/align.py`, `segment_prim_classify.py` |
| Segment/decompose | Unfitted components → features and signed primitive groups | `L2_feature_segment.py`, `contour_parse.py`, `decompose.py` |
| Render | Constructions → 16×16 and 24×24 masks, with protected accents | `compress/render.py`, `cubify.py`, `label_clean.py` |
| Refine | Rendered masks → one-pixel outlines | `application/refine/common.py` |
| Compose/connect | Outline + app name → 24×24 and 32×32 masks | `application/monogram/` |

## Shape extraction

Categorization distinguishes standard, inverted and radial structures, including
compound envelopes and interior marks. Color-essential logos use palette grouping;
color-carried logos preserve interior contrast as cutouts. Other branches retain
the selected foreground mask. `frame_with_margin` supplies a square white canvas.

Alignment scores dominant Hough-line orientations over rotations from −45° to
44.5° in 0.5° steps. L1 fits single primitives to connected regions. L2 separates
unfitted regions while preserving global rings, holes and radial structures.
The contour decomposer constructs signed primitives and measures area and boundary
agreement. The older L3 beam fitter remains a comparison baseline, not the main
build path. Optional differentiable refiners optimize an individual construction
at test time; this is separate from training a learned logo model.

## Small-grid rendering

At the production feed sizes (16 and 24), simple round labels become square forms
or square rings. Compound constructions use silhouette-based pixel edits subject
to topology and shape constraints. This deliberately changes geometry to suit the
grid. The larger-target geometry renderer remains available to debug harnesses.

Refinement removes blocky or isolated artifacts, adjusts strokes, preserves
protected accents and forms the perimeter. The 16-pixel pass uses the 24-pixel
silhouette with protected accents removed as its structural reference. A placed
accent must not cause the smaller outline to widen into its footprint.

## Monograms and output

Letter selection uses the app name: its initial followed by a ranked word initial,
syllable onset, interior consonant or vowel. Pyphen supplies syllable boundaries;
this is a naming heuristic, not OCR. Composition keeps the logo at its native
resolution and compares frame-weave, side-by-side and fallback layouts. Letter
legibility precedes coverage in layout ranking.

The connection pass joins components and opens connection-created T-junctions
where topology permits. Existing junctions inside a letter or logo are retained.
A 16-pixel outline is composed on a 24-pixel canvas; a 24-pixel outline on a
32-pixel canvas. Both themes use the same mask, filled black or white on transparency.
Larger SVGs encode the 32-pixel mask as vector squares.

## State and limits

The stage cache keys source-image content and application source text. The build
manifest records source hashes, output paths, status and code version after each
app. Its incremental skip rule checks the source and outputs, not code version;
`build --update` requests a full rebuild.

Fixture checks cover geometry, routing and deterministic output. They are not a
human recognition study or a held-out evaluation of arbitrary app logos. Earlier
25- and 30-fixture measurements in [the development record](DEVELOPMENT.md) belong
to those historical runs. The current fixture catalogs retain their own provenance
and labels. No new image-quality claim is made by this documentation revision.
