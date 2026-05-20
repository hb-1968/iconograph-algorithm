# fixtures/

Per CLAUDE.md §3 (Bootstrap) and **H9**: maintain **n ≥ 20** application logos
covering the sub-case space. Single-fixture wins are over-fits — every accepted
heuristic must hold across the full dataset.

Bootstrap is **not complete** until the dataset is at n ≥ 20 with each entry
labeled below.

## Sub-case labels (H9 coverage tags)

Reference labels — extend as new sub-cases are encountered (per H3 recurrence):

- `standard` — opaque foreground on transparent/solid background
- `inverted` — light foreground on dark background
- `burst` — radial / burst-pattern abstract
- `monogram-only` — already a single letter or glyph
- `wordmark-with-icon` — text + icon mark
- `gradient-heavy` — large continuous gradients dominate
- `single-color-flat` — one foreground color, no shading

## Stage-1 routing-label columns

These columns are the **Stage-1 categorizer ground truth** that
`src/iconograph/application/categorize.py` must reproduce on every fixture.
They are a separate ontology from the H9 `sub-case` column above — `sub-case`
is for dataset diversity auditing; the routing fields decide pipeline
branching:

- `color_essential` — `T` if shape alone would not carry brand identity (the
  hue/gradient must survive palette quantization). Sharp-edged multi-region
  color identity. Currently true for Firefox + Chrome only.
- `is_compound` — `T` if the logo is an envelope + inner-glyph composition
  (Telegram's blue disc + white plane, Claude's coral square + cream sunburst).
  `F` for single-mark-on-canvas logos (Gmail, Slack, Figma) and for
  `color_essential` radial-multihue logos with no flat envelope (Firefox,
  Chrome).
- `precedence` — for compound logos, which sub-element wins. `inner` when
  the inner glyph is the distinctive shape and the envelope is a generic
  primitive (Telegram → keep plane; Claude → keep sunburst). `outer` when
  the envelope is the brand carrier and the inner is a generic primitive
  (YouTube → keep red rounded-rect; the play triangle is universal UI).
  `n/a` when `is_compound` is `F`.
- `routing` — Stage-2 preprocess branch (final, after the post-strip
  re-categorize step):
  - `burst` when `color_essential` (radial multihue: Chrome, Firefox)
    OR when the stripped inner is itself a fine-fold radial pattern
    (Claude's cream sunburst inside the coral envelope).
  - `inverted` when compound + inner-precedent AND the inner is not a
    radial burst pattern.
  - `standard` otherwise (simple logos, and compound + outer-precedent
    after Stage 2 strips the inner).

## Dataset

The five non-negotiable seeds (per H9) are listed first. Add additional rows
until n ≥ 20.

| filename | sub-case | color_essential | is_compound | precedence | routing | rationale |
| --- | --- | --- | --- | --- | --- | --- |
| discord.png | single-color-flat | F | F | n/a | standard | Official brand asset (rasterized from Discord-Symbol-Black.svg, viewBox 126.644×96 → 1280×968 RGBA); isolated controller-face glyph in solid black with no wordmark — single flat color, only landscape-canvas fixture in the set. |
| gmail.png | standard | F | F | n/a | standard | Wikimedia Commons (public domain SVG render); multi-color envelope M-shape on transparent bg, classic opaque foreground. |
| youtube-music.png | single-color-flat | F | T | inner | inverted | Wikimedia Commons; solid pure-red circle (≈#FF0000) with white play-triangle inside a white ring — flat fields, no shading or gradient. |
| claude.png | single-color-flat | F | T | inner | burst | uxwing.com (free commercial license); two flat fields — coral rounded-square background and cream sunburst/spark mark — no gradient. Once Stage 2 strips the coral envelope, the cream sunburst's 12-spike fine rotational symmetry reads as a burst pattern → final routing is `burst`, not `inverted`. |
| microsoft-edge.png | gradient-heavy | F | F | n/a | standard | Wikimedia Commons MIT-licensed original PNG; teal-to-blue-to-green wave with 7780 unique RGB values — multi-band gradient. Smooth transitions, so color_essential=F despite many hues. |
| signal.png | inverted | F | T | inner | inverted | Wikimedia Commons; white dashed chat-bubble outline on filled blue square — light foreground on a colored background, parallels the telegram pattern. |
| whatsapp.png | single-color-flat | F | F | n/a | standard | Wikimedia Commons (WhatsApp Logo green SVG); solid green phone-in-bubble mark on transparent bg, only anti-aliasing variation, no shading. Single-color shape on transparent canvas — no light foreground to extract; routes standard. |
| telegram.png | inverted | F | T | inner | inverted | Wikimedia Commons; white paper-plane mark on solid filled blue circle — canonical light foreground on dark background. |
| instagram.png | gradient-heavy | F | T | inner | inverted | Wikimedia Commons (2022 SVG); continuous rainbow-to-magenta gradient fills the entire square icon, 12507 unique RGB. White camera outline inside the rainbow envelope wins precedence; gradient is decorative, not identity-bearing. |
| firefox.png | burst | T | F | n/a | burst | Wikimedia Commons MPL-2.0 original PNG; radial flame orbit wrapping a globe is the canonical burst-pattern structure. Warm gradient — color carries identity (flat-quantized firefox is unrecognizable). |
| chrome.png | burst | T | T | n/a | burst | Wikimedia Commons (2022 SVG render); radial pie-wedge color wheel (red/green/yellow thirds + blue center disc with white ring) is textbook burst geometry. Four distinct hues with sharp boundaries — color is identity. `is_compound=T` because the blue center disc + white ring is structurally an inner element; `precedence` is `n/a` because routing is captured by `color_essential` ahead of compound logic. |
| spotify.png | single-color-flat | F | T | inner | inverted | Wikimedia Commons (logo without text SVG); solid green disc with three white wave bars — only two flat fields, no shading. Inner white wave bars win precedence over the green disc envelope. |
| slack.png | standard | F | F | n/a | standard | Wikimedia Commons (Slack icon 2019 SVG); eight interlocking pill-shaped segments in four distinct hues (cyan/green/pink/yellow) on transparent bg — multi-color opaque foreground, no gradients. |
| figma.png | standard | F | F | n/a | standard | Wikimedia Commons; five abstract pill/disc shapes in five distinct hues (red, salmon, purple, cyan, green) composing the F-mark — multi-color opaque foreground, no letter glyph. |
| android-head_flat.png | single-color-flat | F | F | n/a | standard | Official brand asset (flat variant); solid green Android robot head with two black eye dots, no wordmark, no shading — minimal single-color glyph with two-tone accent dots. |
| adobe-illustrator.png | monogram-only | F | T | inner | inverted | Wikimedia Commons (public domain); flat orange "Ai" lettermark on flat dark-maroon rounded-square — literal two-letter monogram, no gradient. Inner Ai wins precedence over the maroon envelope. |
| vscode.png | standard | F | F | n/a | standard | Wikimedia Commons (VS Code 1.35 SVG); blue angle-bracket / slab logo on transparent bg, multi-tone blue with no smooth gradient — opaque foreground. |
| linkedin.png | inverted | F | T | inner | inverted | Wikimedia Commons; white "in" lettering on solid blue square — light foreground on colored background, parallels the telegram/signal pattern. |
| mastodon.png | standard | F | T | outer | standard | Wikimedia Commons (2022 purple hue SVG); white lowercase "m" glyph cut out of a purple rounded-square with a distinctive bottom-right tail/curl forming a chatbubble silhouette — the envelope (chatbubble) is the brand carrier, the m cutout is the generic glyph. Routes outer-precedence via the env-outer-concavity test (Mastodon's tail clears `ENV_DISTINCTIVE_OUTER_CONCAVITY`); standard.universalize then paints the chatbubble envelope as ink, leaving the m-shape as white cutout. |
| obsidian.png | gradient-heavy | F | F | n/a | standard | Wikimedia Commons CC-BY-4.0 (2023 SVG); purple-violet shard with 3441 unique RGB, strong multi-tone gradient character. Single mark on transparent canvas — not compound. |
| apple-rainbow.png | monogram-only | F | F | n/a | standard | Wikimedia Commons; six-stripe rainbow apple glyph — pure symbol mark, no text, maximum hue diversity. Stripes have distinct luminances so the lum-component check correctly classifies as color-incidental. |
| youtube.png | inverted | F | T | outer | standard | Wikimedia Commons; white play triangle on filled red rounded-rectangle — light mark on dark filled background. **Outer-precedent**: the red rounded-rect carries identity, the play triangle is a universal UI primitive; Stage 2 strips the inner and routes to standard. |
| docker-mark-black.png | single-color-flat | F | F | n/a | standard | Official brand asset (mark-only black variant); whale silhouette with cargo-container blocks stacked on top, solid near-black on transparent bg, no wordmark — single flat color. |
| homebrew.png | standard | F | F | n/a | standard | Wikimedia Commons BSD-2 (Homebrew SVG); outlined beer-mug-with-apple mascot on transparent bg — flat orange beer, white foam, grey apple, black outlines. |
| react.png | single-color-flat | F | F | n/a | standard | Wikimedia Commons CC-BY-SA (React-icon SVG); cyan atom orbital mark rendered as a single hue, only AA variation. |

Totals: `color_essential` = 2 (firefox, chrome); `is_compound` = 11; `precedence`=outer = 2 (youtube, mastodon); `routing` = {standard: 15, inverted: 7, burst: 3} — Claude's sunburst is the third burst, picked up by the post-strip re-categorization step described above.
