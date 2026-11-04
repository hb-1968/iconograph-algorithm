"""Cache per-fixture stage outputs in memory and on disk.

Keys bind source PNG bytes and application/cache source bytes; comment edits
also invalidate disk entries. Values are pickled. prewarm loads valid entries
before fork so workers inherit them through copy-on-write memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parents[1]
CACHE_ROOT = _REPO_ROOT / ".cache" / "stages"
# Override the fixture basis without changing the shared stem-based cache namespace.
FIXTURES_ROOT = Path(os.environ.get("ICONOGRAPH_FIXTURES",
                                    str(_REPO_ROOT / "fixtures")))
# Separate output directories by fixture basis.
TESTBED_ROOT = (_REPO_ROOT / "testbed"
                if FIXTURES_ROOT == _REPO_ROOT / "fixtures"
                else _REPO_ROOT / "testbed" / FIXTURES_ROOT.name)
_APPLICATION_ROOT = _PKG_ROOT / "application"

_code_version_cache: str | None = None
_RAM_CACHE: dict[tuple[str, str], Any] = {}
_MISSING: Any = object()


def _code_version() -> str:
    """SHA-256 over the application/ subtree + this cache module.

    Any algorithm change inside application/* (or cache logic itself) busts
    every fixture's cache; downstream caches survive changes elsewhere.
    """
    global _code_version_cache
    if _code_version_cache is not None:
        return _code_version_cache
    h = hashlib.sha256()
    for p in sorted(_APPLICATION_ROOT.rglob("*.py")):
        h.update(p.relative_to(_APPLICATION_ROOT).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    h.update(Path(__file__).read_bytes())
    _code_version_cache = h.hexdigest()[:16]
    return _code_version_cache


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _evict_ram(stem: str) -> None:
    for key in [k for k in _RAM_CACHE if k[0] == stem]:
        del _RAM_CACHE[key]


def _ensure_sentinel(cdir: Path, source: str, code: str) -> None:
    keys_file = cdir / "keys.json"
    expected = {"source": source, "code": code}
    if keys_file.exists():
        try:
            if json.loads(keys_file.read_text()) == expected:
                return
        except (json.JSONDecodeError, OSError):
            pass
    for p in cdir.glob("*.pkl"):
        p.unlink()
    keys_file.write_text(json.dumps(expected))
    # Disk wipe must invalidate RAM so a prewarm()ed-then-stale entry can't
    # outlive its on-disk counterpart in the same process.
    _evict_ram(cdir.name)


def _ram_get(stem: str, stage: str) -> Any:
    return _RAM_CACHE.get((stem, stage), _MISSING)


def _ram_put(stem: str, stage: str, value: Any) -> None:
    _RAM_CACHE[(stem, stage)] = value


def prewarm(fixture_paths: Iterable[Path], stages: Iterable[str]) -> None:
    """Bulk-load existing on-disk pickles into the RAM cache.

    Best-effort: silently skips fixtures with no cache dir, missing sentinel,
    or sentinel mismatch (source/code hash differs from current). Call this
    from the master process BEFORE forking a worker pool — the populated
    dict is copy-on-write inherited by every child.
    """
    code = _code_version()
    stage_list = list(stages)
    for p in fixture_paths:
        cdir = CACHE_ROOT / p.stem
        keys_file = cdir / "keys.json"
        if not keys_file.exists():
            continue
        try:
            saved = json.loads(keys_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if saved.get("code") != code or saved.get("source") != _source_hash(p):
            continue
        for s in stage_list:
            if _ram_get(p.stem, s) is not _MISSING:
                continue
            pkl = cdir / f"{s}.pkl"
            if pkl.exists():
                _ram_put(p.stem, s, pickle.loads(pkl.read_bytes()))


def cached(
    fixture_path: Path,
    stage: str,
    compute_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Return cached stage output, or compute + store + return.

    fixture_path: source PNG (cache namespace = its stem).
    stage:        short stage name, used as filename (e.g. "categorize").
    compute_fn:   callable invoked on miss; its return value is pickled.

    Two-tier cache: RAM dict in front of pickle store. RAM hit short-circuits
    before any I/O. Both layers populate on fresh compute and on disk hit.
    """
    hit = _ram_get(fixture_path.stem, stage)
    if hit is not _MISSING:
        return hit
    cdir = CACHE_ROOT / fixture_path.stem
    cdir.mkdir(parents=True, exist_ok=True)
    _ensure_sentinel(cdir, _source_hash(fixture_path), _code_version())
    stage_file = cdir / f"{stage}.pkl"
    if stage_file.exists():
        result = pickle.loads(stage_file.read_bytes())
    else:
        result = compute_fn(*args, **kwargs)
        stage_file.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    _ram_put(fixture_path.stem, stage, result)
    return result
