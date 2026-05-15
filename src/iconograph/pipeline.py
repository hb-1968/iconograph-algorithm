"""Top-level dispatch: application vs file."""

from pathlib import Path

from . import classify_top


def dispatch(image_path: Path) -> None:
    """Route an input to its icon-class pipeline."""
    icon_class = classify_top.classify(image_path)
    if icon_class == "file":
        raise NotImplementedError(
            "file-icon pipeline is deferred — see CLAUDE.md §3"
        )
    if icon_class != "application":
        raise ValueError(f"unknown icon class: {icon_class!r}")
