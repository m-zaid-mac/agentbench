"""Parse a numbered Sources section out of a Markdown report."""
from __future__ import annotations

import re

from ..judges import _SOURCES_HEADING
from ..schema import Citation

# URLs may contain balanced parentheses, e.g. https://en.wikipedia.org/wiki/Raft_(algorithm)
_URL_BODY = r"https?://(?:[^()\s]|\([^()\s]*\))+"
_MD_LINK = re.compile(
    r"^\s*(?:[-*]\s*)?\[?(\d+)\]?[.):]?\s*\[([^\]]+)\]\((" + _URL_BODY + r")\)", re.M)
_PLAIN = re.compile(r"^\s*(?:[-*]\s*)?\[?(\d+)\]?[.):]?\s+(.*?)\s*<?(https?://[^\s>]+)", re.M)


def trim_url(url: str) -> str:
    """Strip trailing punctuation, but keep a closing ')' that balances an '(' in the URL."""
    while url:
        if url[-1] in ".,;]":
            url = url[:-1]
        elif url[-1] == ")" and url.count(")") > url.count("("):
            url = url[:-1]
        else:
            break
    return url


def parse_markdown_citations(report: str) -> list[Citation]:
    """Handles '[1] Title - https://..', '1. https://..' and '1. [Title](https://..)' lines."""
    m = _SOURCES_HEADING.search(report)
    section = report[m.start():] if m else report
    found: dict[int, Citation] = {}
    for mm in _MD_LINK.finditer(section):
        found.setdefault(int(mm.group(1)), Citation(int(mm.group(1)), mm.group(3), mm.group(2).strip()))
    for mm in _PLAIN.finditer(section):
        n = int(mm.group(1))
        if n not in found:
            url = trim_url(mm.group(3))
            title = mm.group(2).strip(" -–—:|*[]()")
            found[n] = Citation(n, url, title)
    return [found[k] for k in sorted(found)]