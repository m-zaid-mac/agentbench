"""Adapter for the LangGraph + AWS Bedrock + Tavily research agent.

Usage:
    agentbench run --agent agentbench.adapters.research_agent:ResearchAgentAdapter ...

Requires the agent's `backend/` folder on PYTHONPATH. The harness relies on two things:
1. Every web search goes through `ctx.search`. The agent's nodes read an optional search
   function from `config["configurable"]["search"]`, so record/replay, phantom-citation
   checks, and faithfulness all see exactly what the agent retrieved.
2. A final Markdown report with [n] markers and a numbered Sources section.
"""
from __future__ import annotations

import time
from typing import Any

from ..context import RunContext
from ..schema import AgentResult
from .citations import parse_markdown_citations

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def tavily_compatible_search(ctx: RunContext):
    """An async drop-in for `TavilyClient.search` that returns Tavily's response shape."""

    async def search(query: str, **params: Any) -> dict[str, Any]:
        return {"query": query, "results": await ctx.search(query, **params)}

    return search


def make_usage_callback(ctx: RunContext, model_id: str):
    """LangChain callback that records token usage for every LLM call into the trace."""
    from langchain_core.callbacks import AsyncCallbackHandler

    class UsageCallback(AsyncCallbackHandler):
        def __init__(self) -> None:
            self._starts: dict[Any, float] = {}

        async def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
            self._starts[run_id] = time.perf_counter()

        async def on_llm_end(self, response, *, run_id, **kw):
            in_tok = out_tok = 0
            for gens in response.generations:
                for g in gens:
                    usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                    in_tok += usage.get("input_tokens", 0)
                    out_tok += usage.get("output_tokens", 0)
            if not (in_tok or out_tok):  # older integrations put usage in llm_output
                u = (response.llm_output or {}).get("usage", {}) or {}
                in_tok = u.get("prompt_tokens", u.get("input_tokens", 0))
                out_tok = u.get("completion_tokens", u.get("output_tokens", 0))
            start = self._starts.pop(run_id, None)
            ctx.record_llm(model_id, in_tok, out_tok,
                           latency_s=round(time.perf_counter() - start, 3) if start else None)

    return UsageCallback()


class ResearchAgentAdapter:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        # model_id labels token usage for cost tracking; the agent's own bedrock.py
        # decides which model is actually called, so keep the two in sync.
        from agent import build_graph  # type: ignore  # from the agent's backend/ folder
        self._build_graph = build_graph
        self.model_id = model_id

    async def run(self, question: str, ctx: RunContext) -> AgentResult:
        graph = self._build_graph()
        config = {
            "callbacks": [make_usage_callback(ctx, self.model_id)],
            "configurable": {"search": tavily_compatible_search(ctx)},
        }

        state: dict[str, Any] = {}
        async for update in graph.astream({"topic": question}, config=config,
                                          stream_mode="updates"):
            for node, delta in update.items():
                ctx.emit("node", name=node)
                if isinstance(delta, dict):
                    state.update(delta)
                    # Nodes return the full state, so record sub-questions only once.
                    if node == "decompose" and delta.get("sub_questions"):
                        ctx.emit("subquestions", items=list(delta["sub_questions"]))

        report = state.get("report") or ""
        return AgentResult(report=report, citations=parse_markdown_citations(report))