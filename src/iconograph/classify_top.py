"""Top-level icon-class detector (application | file)."""

from pathlib import Path


def classify(image_path: Path) -> str:
    """Return the icon class for the given input.

    Currently always returns "application" — file-icon classification is
    deferred per CLAUDE.md §3 until the application pipeline ships.
    """
    return "application"
