"""Per-trial context handed to the agent: the only way it should search, and where it logs its trace."""
from __future__ import annotations

import copy
import time
from typing import Any

from .urls import normalize_url


class RunContext:
    def __init__(self, task_id: str, trial_idx: int, search_tool: Any):
        self.task_id = task_id
        self.trial_idx = trial_idx
        self._search_tool = search_tool
        self._t0 = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.search_log: list[dict[str, Any]] = []

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def emit(self, type_: str, **data: Any) -> None:
        """Record a trace event. Conventional types: node, subquestions, search, llm_call, error."""
        self.events.append({"type": type_, "t": round(self.elapsed(), 4), **data})

    async def search(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Search through the record/replay layer. Returns [{title, url, content, score}]."""
        start = time.perf_counter()
        try:
            results = await self._search_tool.search(query, **params)
        except Exception as e:
            self.emit("search", query=query, error=f"{type(e).__name__}: {e}",
                      latency_s=round(time.perf_counter() - start, 4))
            raise
        self.search_log.append({"query": query, "params": params, "results": copy.deepcopy(results)})
        self.emit("search", query=query, n_results=len(results),
                  latency_s=round(time.perf_counter() - start, 4))
        return results

    def record_llm(self, model: str, input_tokens: int, output_tokens: int,
                   latency_s: float | None = None) -> None:
        self.emit("llm_call", model=model, input_tokens=int(input_tokens or 0),
                  output_tokens=int(output_tokens or 0), latency_s=latency_s)

    def known_urls(self) -> set[str]:
        """Every URL this trial's own searches returned."""
        return {normalize_url(r["url"]) for e in self.search_log for r in e["results"] if r.get("url")}

    def source_text(self, url: str) -> str | None:
        """Content this trial actually retrieved for `url` (None if it never retrieved it)."""
        target = normalize_url(url)
        chunks: list[str] = []
        for entry in self.search_log:
            for r in entry["results"]:
                if normalize_url(r.get("url", "")) == target:
                    text = (r.get("content") or "").strip()
                    if text and text not in chunks:
                        chunks.append(text)
        return "\n\n".join(chunks) if chunks else None
