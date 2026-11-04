# Iconograph

Iconograph turns application logos into small, monochrome icons paired with a
two-letter monogram. I wanted the logo to be reconstructed for the pixel grid,
rather than reduced to a blurry thumbnail. The pipeline decomposes its shape,
redraws it at a small size, then joins the outline and letters.

The application pipeline builds light and dark freedesktop icon themes from
installed Linux apps. It produces 24×24 and 32×32 PNGs, plus SVGs that preserve
the 32-pixel design at larger display sizes. This is a heuristic design system:
complex logos can lose identifying details, and passing a geometry check does
not establish that an icon is recognizable.

## Run

Requires Python 3.11+. From the repository root:

```bash
python -m pip install -e .
python -m pip install cairosvg
python -m iconograph build --only 'org.gnome.Calculator' --cores 1
```

CairoSVG is needed to rasterize SVG source icons; it is not declared in the core
package dependencies. Optional differentiable fitting uses PyTorch:
`python -m pip install -e '.[diff]'`.

`build` writes both themes under `~/.local/share/icons/`. It does not select a
theme for the desktop. On GNOME, select one explicitly:

```bash
gsettings set org.gnome.desktop.interface icon-theme 'Iconograph'
gsettings set org.gnome.desktop.interface icon-theme 'Iconograph-Dark'
```

Restore your previous theme by setting its name with the same command.
Without `--only`, the build scans all eligible installed apps. Completed icons
are skipped when their source hash matches and their outputs exist. Use
`--update` after algorithm changes to rebuild them; code changes alone do not
trigger a full rebuild. `--cores N` sets the worker count.

## How it works

1. Resolve installed apps' source icons and classify their foreground structure.
2. Extract the relevant silhouette or color regions and align dominant edges.
3. Split complex regions, fit geometric constructions, and render them on a
   16- or 24-pixel grid.
4. Refine the outline, compose it with two letters selected from the app name,
   and connect the pieces on a 24- or 32-pixel canvas.

The SVG output enlarges that pixel design; it does not add detail. File icons,
OCR-based letter extraction and the `review` command remain unimplemented.
Absolute-path `Icon=` entries are excluded because they bypass theme lookup.

## Inspect the algorithm

```bash
python -m iconograph testbed categorize
python -m iconograph testbed preprocess
python -m iconograph testbed decompose --panels
python -m iconograph.render_debug --final --all
```

The testbeds write comparison images under `testbed/` and cache intermediate
stages under `.cache/stages/`. Source-text changes in the application package
invalidate those stage caches, including comment-only changes. Existing installed
themes still follow the explicit `--update` rule.

- [Methods](METHODS.md): stage contracts, rendering choices and limits.
- [Curated fixtures](fixtures/README.md): source provenance and routing labels.
- [Installed-app sample](fixtures-sample/README.md): a separate, unlabeled sample.
- [Development record](DEVELOPMENT.md): earlier designs, measurements and revisions.
