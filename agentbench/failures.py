"""Rule-based failure taxonomy so reports say *how* trials fail, not just that scores dropped."""
from __future__ import annotations

from typing import Any

MISSED_FACTS_BELOW = 0.5
UNFAITHFUL_BELOW = 0.7
SEARCH_LOOP_ABOVE = 0.3

DESCRIPTIONS = {
    "missed_facts": f"fact recall below {MISSED_FACTS_BELOW}",
    "unfaithful": f"citation faithfulness below {UNFAITHFUL_BELOW}",
    "phantom_citation": "cites a URL this trial never retrieved (state leaking between runs?)",
    "dangling_marker": "uses a [n] marker with no matching source entry",
    "numeric_corruption": "a required number is missing or altered",
    "search_error": "a search failed (in replay mode, usually a query missing from the cache); the agent answered without those results",
    "search_loop": f"over {int(SEARCH_LOOP_ABOVE * 100)}% of searches near-duplicate earlier ones",
    "no_citations": "report has no cited claims",
}


def classify(m: dict[str, Any]) -> list[str]:
    if m.get("status") != "ok":
        return [f"error:{m.get('error_type') or 'unknown'}"]
    labels = []
    if m.get("fact_recall") is not None and m["fact_recall"] < MISSED_FACTS_BELOW:
        labels.append("missed_facts")
    if m.get("faithfulness") is not None and m["faithfulness"] < UNFAITHFUL_BELOW:
        labels.append("unfaithful")
    if m.get("phantom_citations", 0) > 0:
        labels.append("phantom_citation")
    if m.get("dangling_markers", 0) > 0:
        labels.append("dangling_marker")
    if m.get("numeric_fidelity") is not None and m["numeric_fidelity"] < 1.0:
        labels.append("numeric_corruption")
    if m.get("n_search_errors", 0) > 0:
        labels.append("search_error")
    if m.get("duplicate_search_rate", 0) > SEARCH_LOOP_ABOVE:
        labels.append("search_loop")
    if m.get("n_claims", 0) == 0:
        labels.append("no_citations")
    return labels