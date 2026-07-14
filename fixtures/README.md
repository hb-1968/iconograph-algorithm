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
- `letterform-counter` — bold letterform whose interior counter geometry is non-trivial (multiple enclosed voids from the glyph's stroke valleys); exercises recursive hole decomposition
- `stroke-glyph` — mark built entirely from thick straight/angled polyline strokes or chevrons that joint at vertices, ideally mirror-symmetric; exercises stroke-coordination and mirror-symmetry refiner
- `compound-mascot` — organic multi-part mascot silhouette whose semantic parts (head, ears, appendages) join at concavity necks as one merged blob; exercises neck-split fallback and grouped CSG

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

| mathematica.png | burst | F | T | inner | inverted | Wikimedia Commons (public domain); Wolfram Mathematica "Spikey" — a flat red 10-point radial star with concentric nested-star negative-space rings, on transparent bg. A `burst` sub-case for dataset diversity. Does NOT exercise `_fit_rot_spurs`: the categorizer reads the white nested-ring negative space as light foreground (compound + inner-precedent → routing `inverted`), and L2 segments the concentric red bands as two `annular_band` features, not a `sunburst`. A clean solid-hub blade sunburst is still needed to give the spur detector a second positive (H9, open). |
| electron.png | inverted | F | F | n/a | standard | Wikimedia Commons (public domain); Electron framework — three woven orbital ellipses + a nucleus dot + three terminal circles, light cyan on a filled dark navy disc. An `inverted`-looking atom mark for diversity. Does NOT exercise `_fit_rot_orbits`: the categorizer does not strip the dark disc (routes `standard`, is_compound=F), so the thin broken cyan strokes never form a clean feature (L2 defers them) — no Feature reaches the orbit detector. A solid, thick-ring atom is still needed for the orbit detector's second positive (H9, open). **Now a third `color_carried` case** (cyan atom on navy disc): the silhouette collapses to a disc and the cyan orbits are enclosed interior marks → `color_carried=True` → the cutout carves the full atom (orbits + nucleus + terminal circles) as holes, so the atom is captured as negative space instead of lost. |
| medium-m.png | letterform-counter | F | F | n/a | standard | Wikimedia Commons (public domain, File:Medium_icon.svg); Medium's bold display-weight M icon — a single near-black glyph on transparent bg. The M's two inner-valley legs create two distinct enclosed counter triangles (fill-rule:nonzero path), making this a multi-void letterform. est.: is_compound=F (the counters are interior negative space of the shape geometry, not a separate inner glyph); routes standard. Exercises recursive hole decomposition — each counter is a multi-primitive negation inside the glyph boundary. |
| vue.png | stroke-glyph | F | F | n/a | standard | Wikimedia Commons CC-BY 4.0 (File:Vue.js_Logo_2.svg); Vue.js V-chevron mark — two nested V-shaped strokes, outer in green (#41b883) and inner in dark slate (#34495e), both flat solid color on transparent bg. The outer and inner V meet at a shared apex joint; the whole mark is mirror-symmetric about the vertical axis. est.: is_compound=F (both V-layers are co-equal parts of the brand mark, not an envelope wrapping a generic primitive). Intended to exercise stroke-coordination and the mirror-symmetry refiner variant — but currently yields 0 L2 Features (the V-strokes are deferred upstream), a stage-1/2 routing gap of the same class as electron's; the stroke-glyph exercise is pending that fix (H9, open). |
| reddit-snoo.png | compound-mascot | F | F | n/a | standard | Wikimedia Commons (public domain, File:Snoo.svg); Reddit's Snoo alien head — a single merged flat orange silhouette comprising the main head dome, two ear-humps flanking it, an antenna rod, and a terminal circle ball. All parts are fused into one contiguous blob at pixel level; ears and antenna connect to the head at concavity necks. est.: is_compound=F (the mascot is a single merged silhouette, not an envelope+inner-glyph pair). Exercises semantic part-grouping via neck-split fallback and grouped CSG at Stage 3. |
| easyeffects.png | colored-badge-interior-mark | F | F | n/a | standard | Installed pack source (Flatpak com.github.wwmm.easyeffects, rasterized at 512 — byte-identical to the pack input). A solid opaque disc (blue gradient + gray/blue rings) with three WHITE equalizer slider tracks + white knobs + dark shadow circles painted ON TOP; all meaning is interior color/luminance contrast, NOT alpha. Currently mis-handled: categorize routes `standard` (color_essential=F, is_compound=F); preprocess builds the silhouette from alpha alone → one solid disc (fill 0.46, 1 CC) → L1 fits a single Ellipse (IoU 0.997) → a meaningless block. The **pinned motivating case** for the pre-compression color-difference gate (`precompress_gate_debug`): the source has ≥2 area-significant ΔE color groups but preprocess collapses them to one structureless blob — the same stage-1/2 detail-loss class flagged for electron/vue, here in its color-contrast form. |

| ptyxis.png | colored-badge-interior-mark | F | F | n/a | standard | Installed pack source (Flatpak org.gnome.Ptyxis, rasterized at 512). A solid dark rounded-square with a cyan `>_` terminal prompt — identity is the interior luminance-contrast prompt, not alpha. Second `color_carried` case (the "square" analog of easyeffects' "circle"): the silhouette collapses to a simple primitive (circle-fit IoU 0.999) and holds enclosed light marks. Routes standard but `color_carried=True` → the **cutout** stage-2 branch carves the `>_` as holes → renders a squircle outline with a prompt, not a solid block. Second fixture for the collapse-to-primitive + interior-mark detector. |
| steam.png | glyph-on-gradient-badge | F | T | inner | inverted | Installed pack source (Flatpak com.valvesoftware.Steam). A large white Steam logo (pipe + gear + terminal circle) on a navy-gradient disc. The COUNTERPOINT to color_carried: the gradient disc is discardable BACKGROUND and the white logo is the identity, so this must **invert** (extract the glyph, discard the disc), NOT cutout. Fails the normal `_compound_via_light_inner` containment check (the pipe reaches the badge edge) → admitted by the LIGHT_GLYPH_FRAC≥0.15 + collapse-to-primitive relaxation (homebrew's 37%-light non-collapsing mug is excluded by the collapse gate). is_compound=T, precedence=inner → routing inverted → extracts the Steam mark. The size discriminator vs cutout: a DOMINANT light glyph (≥15%, Steam) inverts; small interior accents (easyeffects knobs ~5%) cut out. |



Totals: `color_essential` = 2 (firefox, chrome); `is_compound` = 12; `precedence`=outer = 2 (youtube, mastodon); `color_carried` = 3 (easyeffects, ptyxis, electron — the cutout branch); `routing` = {standard: 21, inverted: 9, burst: 3} — Claude's sunburst remains the only flat-spike burst that L2 hints `sunburst`. n=33 (steam = glyph-on-gradient-badge → inverted). New sub-cases added: `letterform-counter` (medium-m), `stroke-glyph` (vue), `compound-mascot` (reddit-snoo), `colored-badge-interior-mark` (easyeffects, ptyxis). Note: mathematica + electron are general-diversity additions that do NOT exercise the symmetry detectors (`_fit_rot_spurs` / `_fit_rot_orbits`) — see their rows — so the H9 second-fixture gap for both detectors remains open.
