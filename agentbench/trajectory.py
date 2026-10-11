"""Metrics computed from the trace: how the agent got to its answer, and what it cost."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .text import jaccard, tokens

DUPLICATE_THRESHOLD = 0.8


def load_pricing(path: str | Path | None) -> dict[str, dict[str, float]]:
    """Pricing JSON: {"models": {model_id: {"input_per_mtok": x, "output_per_mtok": y}}}."""
    if not path or not Path(path).exists():
        return {}
    models = json.loads(Path(path).read_text()).get("models", {})
    return {m: p for m, p in models.items()
            if p.get("input_per_mtok") is not None and p.get("output_per_mtok") is not None}


def trajectory_metrics(events: list[dict[str, Any]], pricing: dict[str, dict[str, float]]) -> dict[str, Any]:
    searches = [e for e in events if e["type"] == "search"]
    ok = [e for e in searches if "error" not in e]
    seen: list[set[str]] = []
    duplicates = 0
    for e in ok:
        q = set(tokens(e["query"]))
        if any(jaccard(q, s) >= DUPLICATE_THRESHOLD for s in seen):
            duplicates += 1
        seen.append(q)

    llm = [e for e in events if e["type"] == "llm_call"]
    in_tok = sum(e.get("input_tokens", 0) for e in llm)
    out_tok = sum(e.get("output_tokens", 0) for e in llm)
    cost: float | None = 0.0
    for e in llm:
        p = pricing.get(e.get("model", ""))
        if p is None:
            cost = None  # unknown price: report n/a rather than a misleading number
            break
        cost += (e.get("input_tokens", 0) * p["input_per_mtok"] + e.get("output_tokens", 0) * p["output_per_mtok"]) / 1e6
    if not llm:
        cost = None

    return {
        "n_searches": len(ok),
        "n_search_errors": len(searches) - len(ok),
        "duplicate_search_rate": duplicates / len(ok) if ok else 0.0,
        "n_llm_calls": len(llm),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "agent_cost_usd": round(cost, 6) if cost is not None else None,
        "n_subquestions": sum(len(e.get("items", [])) for e in events if e["type"] == "subquestions"),
        "n_nodes": sum(1 for e in events if e["type"] == "node"),
    }
