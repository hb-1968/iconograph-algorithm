"""Choose two letters from an application name.

Keep the first letter. Rank the second by word initial, syllable onset,
interior consonant, then vowel; position breaks ties within a tier. Pyphen
supplies syllable boundaries. The selector does not use OCR.
"""
from __future__ import annotations

import re

import pyphen

_VOWELS = frozenset("aeiou")

# One shared, deterministic hyphenator (dictionary lookup, no RNG -> H8-safe).
# Default left/right (2, 2): permissive enough for real syllable breaks
# (dis-cord, an-droid) without the spurious 1-char-first-piece splits that
# (1, 1) injects on out-of-dictionary brands (slack -> "s-lack").
_DIC = pyphen.Pyphen(lang="en_US")

# Word separators.
_SEP_RE = re.compile(r"[\s\-_]+")
# camelCase splitter: an all-caps run before a Word ("VSCode" -> VS|Code), then
# a normal Word, then a trailing all-caps run ("You"|"Tube", "Whats"|"App").
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")


def _is_consonant(ch: str) -> bool:
    """True for a letter that is not a vowel. 'y' counts as a consonant so it
    can serve as a word/syllable onset (e.g. YouTube -> Y)."""
    return ch.isalpha() and ch.lower() not in _VOWELS


def _has_alpha(s: str) -> bool:
    return any(c.isalpha() for c in s)


def _word_parts(name: str) -> list[str]:
    """Split a display name into ordered word-parts, letters-bearing only.

    A space / hyphen / underscore is a stronger boundary than a case change, so
    camelCase is applied only WITHIN a single separator-token: "YouTube Music"
    -> [YouTube, Music] (the space wins, so 'Music' supplies letter 2 -> YM),
    while the lone token "YouTube" -> [You, Tube] (-> YT).
    """
    tokens = [t for t in _SEP_RE.split(name) if _has_alpha(t)]
    if len(tokens) >= 2:
        return tokens
    if not tokens:
        return []
    return [p for p in _CAMEL_RE.findall(tokens[0]) if _has_alpha(p)]


def select_monogram_letters(name: str) -> str:
    """name: human-readable app name, e.g. "Adobe Illustrator".
    Returns: a 2-character uppercase monogram, e.g. "AI".
    """
    words = _word_parts(name)
    if not words:
        return "??"  # no letters at all -> degenerate placeholder

    # Letter 1 = first alphabetic char of the first word-part.
    first = words[0]
    letter1 = next(c for c in first if c.isalpha()).upper()

    # Letter-2 candidates, each tagged (tier, order); the min (tier, then order)
    # wins. Tier dominates position; position only breaks ties within a tier.
    cands: list[tuple[int, int, str]] = []

    # Tier 0 -- initial of every word-part after the first.
    for order, word in enumerate(words[1:]):
        for c in word:
            if c.isalpha():
                cands.append((0, order, c))
                break

    # Tier 1 -- single word-part only: onset consonant of a later syllable.
    if len(words) == 1:
        pieces = _DIC.inserted(first).split("-")
        for order, piece in enumerate(pieces[1:]):
            if piece and _is_consonant(piece[0]):
                cands.append((1, order, piece[0]))

    # Tier 2 (consonant) / Tier 3 (vowel) -- interior letters after letter 1.
    # Scan from the post-split words (same source as letter 1) so letters[0] ==
    # letter 1 holds even for a char _CAMEL_RE skips (e.g. a non-ASCII leading
    # letter), which would otherwise double-select the first letter.
    letters = [c for w in words for c in w if c.isalpha()]
    for order, c in enumerate(letters[1:]):
        cands.append((2 if _is_consonant(c) else 3, order, c))

    if not cands:
        return letter1 + letter1  # one-letter name -> "AA"

    letter2 = min(cands, key=lambda t: (t[0], t[1]))[2].upper()
    return letter1 + letter2


def split_monogram(mono: str) -> tuple[str, str]:
    """Split a 2-char monogram into its (letter1, letter2) for glyph rendering.

    Robust to the degenerate cases select_monogram_letters can emit: "" -> the
    "??" placeholder, a single char -> duplicated, >2 chars -> first two. Each
    output char is what glyphs.render_glyph consumes (missing chars fall back to
    '?' there).
    """
    chars = mono[:2] if mono else ""
    if len(chars) == 0:
        return "?", "?"
    if len(chars) == 1:
        return chars, chars
    return chars[0], chars[1]
