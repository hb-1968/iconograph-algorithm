"""Inspect the name-to-two-letter monogram selector."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from iconograph.application.monogram.common import select_monogram_letters

REPO_ROOT = Path(__file__).resolve().parents[2]
from iconograph.cache import FIXTURES_ROOT  # noqa: E402

NAMES_JSON = FIXTURES_ROOT / "names.json"   # env-overridable (ICONOGRAPH_FIXTURES)

# The 4 user examples (hard ground truth — acceptance gate) then the remaining seeds (observed).
DEFAULT_FIXTURES = (
    "discord", "adobe-illustrator", "android-head_flat", "medium-m",   # examples
    "gmail", "youtube-music", "claude", "microsoft-edge",              # remaining seeds
)
EXPECTED = {
    "discord": "DC",
    "adobe-illustrator": "AI",
    "android-head_flat": "AD",
    "medium-m": "MD",
}


def _load_names() -> dict[str, str]:
    return json.loads(NAMES_JSON.read_text())


def main(argv: list[str]) -> int:
    names = _load_names()
    wanted = argv or list(DEFAULT_FIXTURES)

    rows: list[tuple[str, str, str, str, str]] = []
    n_fail = 0
    for stem in wanted:
        name = names.get(stem)
        if name is None:
            rows.append((stem, "<no name in names.json>", "--", "", "SKIP"))
            continue
        got = select_monogram_letters(name)
        exp = EXPECTED.get(stem, "")
        if exp:
            status = "OK" if got == exp else "FAIL"
            n_fail += status == "FAIL"
        else:
            status = "-"
        rows.append((stem, name, got, exp, status))

    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    for stem, name, got, exp, status in rows:
        print(f"{stem:<{w0}}  {name:<{w1}}  {got:<3}  exp={exp or '·':<3}  {status}")

    if n_fail:
        print(f"\n{n_fail} gate example(s) FAILED")
        return 1
    print("\nall gate examples OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
