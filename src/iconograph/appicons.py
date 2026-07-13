"""appicons.py — installed-app census + icon resolution (the production source layer).

ONE home for "what apps exist on this machine and where do their icons live" (H5),
shared by the `build` command (per-name resolution) and the dev sampler
(`sample_icons.py`, pool scan). The resolution chain is the user-ratified order:
the app's OWN libraries lead, **Adwaita** is the designated fallback —

    hicolor (system) → hicolor (user, ~/.local — the Edge-PWA location) →
    flatpak exports → pixmaps → Adwaita → AdwaitaLegacy (incl. */legacy/ dirs) →
    absolute-path Icon= values

arcticons/breeze theme RESTYLES are deliberately NOT sources (round-2 lesson: theme
packs alias thousands of apps that are not installed; and their line art restyles
brand identity). Within a tier: SVG preferred, else the largest raster (≥128 px
preferred, any size accepted — AdwaitaLegacy generics top out at 48). Census =
.desktop files (Type=Application, not NoDisplay/Hidden; configparser with
interpolation=None, strict=False, optionxform=str — the three load-bearing options).

Deterministic (H8): sorted directory walks, fixed tier order, no RNG.
"""
from __future__ import annotations

import configparser
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

HOME = Path.home()

# (root, tier) — resolution priority = list order. Brand tiers first, the
# Adwaita fallback last (before absolute-path Icon= values, handled separately).
ICON_ROOTS: tuple[tuple[Path, str], ...] = (
    (Path("/usr/share/icons/hicolor"), "hicolor"),
    (HOME / ".local/share/icons/hicolor", "hicolor-user"),
    (Path("/var/lib/flatpak/exports/share/icons/hicolor"), "flatpak"),
    (HOME / ".local/share/flatpak/exports/share/icons/hicolor", "flatpak-user"),
    (Path("/usr/share/pixmaps"), "pixmaps"),
    (Path("/usr/share/icons/Adwaita"), "adwaita"),
    (Path("/usr/share/icons/AdwaitaLegacy"), "adwaita-legacy"),
)
# Search order = freedesktop XDG precedence, HIGHEST first (XDG_DATA_HOME before
# XDG_DATA_DIRS; flatpak prepends its export dirs, user before system). First-wins
# in installed_apps() means a user-overridden .desktop shadows the system one.
DESKTOP_DIRS: tuple[Path, ...] = (
    HOME / ".local/share/applications",
    HOME / ".local/share/flatpak/exports/share/applications",
    Path("/var/lib/flatpak/exports/share/applications"),
    Path("/usr/share/applications"),
)
MIN_RASTER_PREF = 128    # preferred raster floor within a tier (fallback: any size)
SVG_RASTER_W = 512       # cairosvg output width (aspect preserved)

_SIZE_DIR = re.compile(r"^(\d+)x\1$")
_GENERIC_SEGS = {"tray", "desktop", "app", "bin", "gtk", "qt", "icon"}


@dataclass(frozen=True)
class AppEntry:
    """One installed, launchable application."""
    icon: str            # the .desktop Icon= value VERBATIM (the pack-output basename)
    lookup: str          # the icon LOOKUP name (basename stem when Icon= is a path)
    name: str            # human app name (Name=)
    desktop: Path        # the .desktop file


@dataclass(frozen=True)
class IconSource:
    """A resolved icon file."""
    path: Path
    fmt: str             # "png" | "svg" | "path" (absolute Icon= value)
    tier: str            # which ICON_ROOTS tier (or "absolute")
    size: int            # raster size (0 for svg)


def installed_apps() -> dict[str, AppEntry]:
    """{lookup_name: AppEntry} for every visible, THEMEABLE installed app (.desktop
    census: Type=Application, not NoDisplay/Hidden; first .desktop wins per lookup
    name, USER dirs scanned before system per XDG precedence). Absolute-path Icon=
    values are EXCLUDED: the freedesktop spec has DEs load such files directly,
    bypassing the icon theme, so a themed override could never be displayed (and
    their basename-stem aliases distinct apps onto one lookup — e.g. many game
    cover.png files → 'cover')."""
    apps: dict[str, AppEntry] = {}
    for d in DESKTOP_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.desktop")):
            cp = configparser.ConfigParser(interpolation=None, strict=False)
            cp.optionxform = str            # .desktop keys are case-sensitive
            try:
                cp.read(f, encoding="utf-8")
                sec = cp["Desktop Entry"]
            except (configparser.Error, KeyError, UnicodeDecodeError):
                continue
            if sec.get("Type", "Application").strip() != "Application":
                continue
            if sec.get("NoDisplay", "").strip().lower() == "true":
                continue
            if sec.get("Hidden", "").strip().lower() == "true":
                continue
            icon, name = sec.get("Icon"), sec.get("Name")
            if not icon or not name:
                continue
            icon = icon.strip()
            if icon.startswith("/"):     # absolute path — un-themeable (see docstring)
                continue
            apps.setdefault(icon, AppEntry(icon=icon, lookup=icon,
                                           name=name.strip(), desktop=f))
    return apps


def _dir_size(p: Path) -> int:
    """Raster size from a themed path's NxN component, else 0 (probe lazily)."""
    for part in p.parts:
        m = _SIZE_DIR.match(part)
        if m:
            return int(m.group(1))
    return 0


def _png_size(p: Path) -> int:
    from PIL import Image
    try:
        with Image.open(p) as im:
            return min(im.size)
    except Exception:
        return 0


def _tier_candidates(root: Path, name: str) -> list[Path]:
    """Icon files named `name` under one root: themed roots search */apps and
    */legacy (AdwaitaLegacy keeps its content there); pixmaps is flat."""
    if root.name == "pixmaps":
        return sorted(p for p in (root / f"{name}.svg", root / f"{name}.png")
                      if p.is_file())
    out: list[Path] = []
    for sub in ("apps", "legacy"):
        out += sorted(root.glob(f"*/{sub}/{name}.svg"))
        out += sorted(root.glob(f"*/{sub}/{name}.png"))
    return out


def resolve_icon(entry_icon: str) -> IconSource | None:
    """Resolve an Icon= value through the chain. Within a tier: SVG wins, else
    the largest raster (>= MIN_RASTER_PREF preferred, any size accepted).
    Absolute-path Icon= values resolve directly (tier 'absolute')."""
    if "/" in entry_icon:
        p = Path(entry_icon)
        if p.is_file():
            fmt = p.suffix.lstrip(".").lower() or "path"
            return IconSource(p, "svg" if fmt == "svg" else "png", "absolute",
                              0 if fmt == "svg" else _png_size(p))
        entry_icon = p.stem                 # fall through to a name lookup
    for root, tier in ICON_ROOTS:
        if not root.exists():
            continue
        cands = _tier_candidates(root, entry_icon)
        if not cands:
            continue
        svgs = [c for c in cands if c.suffix.lower() == ".svg"]
        if svgs:
            return IconSource(svgs[0], "svg", tier, 0)
        best, best_sz = None, -1
        for c in cands:
            sz = _dir_size(c) or _png_size(c)
            if sz > best_sz:
                best, best_sz = c, sz
        pref = [c for c in cands if (_dir_size(c) or _png_size(c)) >= MIN_RASTER_PREF]
        if pref:
            best = max(pref, key=lambda c: _dir_size(c) or _png_size(c))
            best_sz = _dir_size(best) or _png_size(best)
        if best is not None:
            return IconSource(best, "png", tier, best_sz)
    return None


def materialize(src: IconSource, dst: Path, svg_width: int = SVG_RASTER_W) -> None:
    """Write `src` to `dst` as a PNG the pipeline can consume: SVGs rasterize via
    cairosvg (transparent bg — categorize.load_rgba composites RGBA itself);
    rasters copy RAW (no pre-crop/framing — that is the pipeline's job)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.fmt == "svg":
        import cairosvg
        cairosvg.svg2png(url=str(src.path), write_to=str(dst),
                         output_width=svg_width)
    else:
        shutil.copyfile(src.path, dst)


def slug(name: str) -> str:
    """Cache-namespace slug: lowercase, non-alnum runs -> single hyphen."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


def norm(name: str) -> str:
    """Lowercase alnum-only normal form (fuzzy dedupe/exclusion)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pretty_name(icon_name: str) -> str:
    """Fallback human name: last NON-GENERIC reverse-DNS segment (tray/desktop/…
    carry no identity — stage-5 monogram letters come from this), title-cased."""
    segs = [s for s in icon_name.split(".") if s]
    base = next((s for s in reversed(segs) if s.lower() not in _GENERIC_SEGS),
                segs[-1] if segs else icon_name)
    return " ".join(w.capitalize() for w in re.split(r"[-_]+", base) if w)
