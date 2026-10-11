"""A deterministic stand-in agent for demos and tests, with optional seeded bugs.

The seeded bugs mirror two classic agent failures, so you can confirm the harness catches them:
- buggy_digits: a cleanup regex meant to strip '1. ' list numbering also eats digits before a
  period, so '8,848.86' becomes '8,86' and 'in 2021.' becomes 'in '.
- shared_state: sources are collected on the agent instance instead of per run, so concurrent
  runs overwrite each other and cite each other's URLs.
"""
from __future__ import annotations

import asyncio
import re

from ..context import RunContext
from ..schema import AgentResult, Citation

BUGS = {None, "buggy_digits", "shared_state"}


def decompose(question: str) -> list[str]:
    q = question.rstrip("?")
    parts = [p.strip() for p in re.split(r",\s*|;\s*|\s+and\s+", q) if len(p.strip().split()) >= 2]
    return [question] + parts[:4]


class MockResearchAgent:
    def __init__(self, bug: str | None = None, max_sources: int = 8):
        if bug not in BUGS:
            raise ValueError(f"unknown bug {bug!r}; choose from {sorted(b for b in BUGS if b)}")
        self.bug = bug
        self.max_sources = max_sources
        self._shared_sources: list[dict] = []  # only used by the shared_state bug

    async def run(self, question: str, ctx: RunContext) -> AgentResult:
        ctx.emit("node", name="plan")
        subqs = decompose(question)
        ctx.emit("subquestions", items=subqs)
        ctx.record_llm("mock", input_tokens=len(question) // 4 + 200, output_tokens=60)

        ctx.emit("node", name="search")
        batches = await asyncio.gather(*(ctx.search(q) for q in subqs))
        mine: list[dict] = []
        seen: set[str] = set()
        for r in sorted((r for b in batches for r in b), key=lambda r: -(r.get("score") or 0)):
            if r["url"] not in seen:
                seen.add(r["url"])
                mine.append(r)
        mine = mine[: self.max_sources]

        if self.bug == "shared_state":
            self._shared_sources.clear()
            self._shared_sources.extend(mine)
            await asyncio.sleep(0.005)  # e.g. awaiting the LLM; other runs mutate the list meanwhile
            sources = list(self._shared_sources)
        else:
            sources = mine

        ctx.emit("node", name="synthesize")
        body = " ".join(f"{s['content']} [{i}]" for i, s in enumerate(sources, 1))
        ctx.record_llm("mock", input_tokens=sum(len(s["content"]) for s in sources) // 4 + 300,
                       output_tokens=len(body) // 4)
        if self.bug == "buggy_digits":
            body = re.sub(r"\d+\.(?=\s|\d|$)", "", body)
        refs = "\n".join(f"[{i}] {s['title']} - {s['url']}" for i, s in enumerate(sources, 1))
        report = f"# Research report\n\n{question}\n\n## Findings\n\n{body}\n\n## Sources\n\n{refs}\n"
        citations = [Citation(i, s["url"], s["title"]) for i, s in enumerate(sources, 1)]
        return AgentResult(report=report, citations=citations)
