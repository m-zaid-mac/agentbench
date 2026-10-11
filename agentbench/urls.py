from urllib.parse import unquote, urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    """Normalize a URL for comparison: lowercase scheme/host, decode %-escapes in the path
    (so Raft_%28algorithm%29 matches Raft_(algorithm)), drop fragment and trailing slash."""
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    path = unquote(parts.path).rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))