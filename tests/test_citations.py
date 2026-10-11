from agentbench.adapters.citations import parse_markdown_citations


def test_parses_common_source_formats():
    report = """Body text [1] [2] [3].

## Sources
[1] NASA JWST page - https://nasa.gov/jwst
2. [ESA Webb](https://esa.int/webb)
- [3] <https://example.org/x>.
"""
    cites = parse_markdown_citations(report)
    assert [(c.marker, c.url) for c in cites] == [
        (1, "https://nasa.gov/jwst"), (2, "https://esa.int/webb"), (3, "https://example.org/x")]
    assert cites[0].title == "NASA JWST page" and cites[1].title == "ESA Webb"


def test_urls_with_parentheses_are_not_truncated():
    report = """Raft [1], Rust [2], Paxos [3], BERT [4].

## Sources
1. Raft - https://en.wikipedia.org/wiki/Raft_(algorithm)
2. [Rust](https://en.wikipedia.org/wiki/Rust_(programming_language))
3. Paxos - https://en.wikipedia.org/wiki/Paxos_(computer_science).
4. BERT (Wikipedia) - https://en.wikipedia.org/wiki/BERT_(language_model)),
"""
    urls = [c.url for c in parse_markdown_citations(report)]
    assert urls == [
        "https://en.wikipedia.org/wiki/Raft_(algorithm)",
        "https://en.wikipedia.org/wiki/Rust_(programming_language)",
        "https://en.wikipedia.org/wiki/Paxos_(computer_science)",
        "https://en.wikipedia.org/wiki/BERT_(language_model)",
    ]


def test_percent_encoded_parentheses_match():
    from agentbench.urls import normalize_url

    assert normalize_url("https://en.wikipedia.org/wiki/Raft_%28algorithm%29") == \
        normalize_url("https://en.wikipedia.org/wiki/Raft_(algorithm)/")