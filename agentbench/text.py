"""Small text utilities shared by the heuristic judges, fuzzy replay, and trajectory metrics."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or of to in on at by for with from as is are was were be been being it its this that
    these those which who whom what when where why how than then so such into over under about after
    before during between through per via also has have had do does did not no can could would should
    will may might their there they them he she his her our we you your i""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokens(text: str) -> list[str]:
    """Lowercase content tokens with stopwords removed (keeps decimals like 6.5 intact)."""
    return [t for t in _TOKEN.findall(_strip_thousands(text.lower())) if t not in STOPWORDS]


def _strip_thousands(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


def normalize_numbers(text: str) -> str:
    """Remove thousands separators so 299,792,458 and 299792458 compare equal."""
    return _strip_thousands(text)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def containment(needle: list[str], haystack: set[str]) -> float:
    """Fraction of needle tokens present in haystack."""
    if not needle:
        return 0.0
    return sum(1 for t in needle if t in haystack) / len(needle)
