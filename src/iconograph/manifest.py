"""Hash + skip-don't-rebuild logic shared across pipelines (per H8).

The `build` command's progress ledger. The manifest lives WITH the icon pack
(`<pack>/manifest.json`) so pack + state travel together. One entry per app
lookup name:

    {lookup: {"icon": <Icon= verbatim>, "name": <app name>,
              "source": <resolved path>, "source_hash": sha256[:16],
              "code": <cache._code_version()>,        # recorded for audit ONLY
              "outputs": ["24x24/apps/<icon>.png", ...],
              "status": "built" | "unresolved" | "failed", "reason": <str, failed only>}}

SKIP RULE (default incremental run): status == "built" AND source_hash matches the
current source file AND every output exists on disk. A code change does NOT
auto-trigger rebuilds — `build --update` is the user's explicit code-change lever
(no surprise multi-hour rebuilds after small edits); the recorded `code` field
makes stale entries auditable. The manifest is FLUSHED after every app so an
interrupted run resumes exactly where it stopped.
"""
from __future__ import annotations

import json
from pathlib import Path

from iconograph.cache import _code_version, _source_hash


def load(pack_dir: Path) -> dict:
    p = pack_dir / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def flush(pack_dir: Path, man: dict) -> None:
    """Atomic-ish write (tmp + rename) after EVERY app — the resume contract."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    tmp = pack_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    tmp.replace(pack_dir / "manifest.json")


def should_skip(man: dict, pack_dir: Path, lookup: str, source: Path) -> bool:
    """True iff this app is already built from THIS exact source file and all
    its outputs still exist (the incremental no-pointless-rerun rule)."""
    e = man.get(lookup)
    if not e or e.get("status") != "built":
        return False
    try:
        if e.get("source_hash") != _source_hash(source):
            return False
    except OSError:
        return False
    outs = e.get("outputs") or []
    return bool(outs) and all((pack_dir / o).exists() for o in outs)


def record_built(man: dict, lookup: str, *, icon: str, name: str, source: Path,
                 outputs: list[str], gate: dict | None = None) -> None:
    entry = {"icon": icon, "name": name, "source": str(source),
             "source_hash": _source_hash(source), "code": _code_version(),
             "outputs": outputs, "status": "built"}
    if gate is not None:                    # pre-compression gate (report-only)
        entry["gate"] = gate
    man[lookup] = entry


def prune_outputs(man: dict, pack_dir: Path, lookup: str) -> None:
    """Unlink a prior-BUILT entry's PNGs before its status is downgraded to
    unresolved/failed, so the pack tree never keeps serving an icon the manifest
    no longer claims (an app whose source icon moved or that now fails to build)."""
    e = man.get(lookup)
    if not e or e.get("status") != "built":
        return
    for o in e.get("outputs") or []:
        (pack_dir / o).unlink(missing_ok=True)


def record_unresolved(man: dict, lookup: str, *, icon: str, name: str) -> None:
    man[lookup] = {"icon": icon, "name": name, "source": None,
                   "source_hash": None, "code": _code_version(),
                   "outputs": [], "status": "unresolved"}


def record_failed(man: dict, lookup: str, *, icon: str, name: str, source: Path,
                  reason: str) -> None:
    man[lookup] = {"icon": icon, "name": name, "source": str(source),
                   "source_hash": _source_hash(source), "code": _code_version(),
                   "outputs": [], "status": "failed", "reason": reason}
