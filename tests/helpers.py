"""Test doubles."""
from __future__ import annotations

import json

from agentbench.llm import LLMResponse


class ScriptedLLM:
    """Returns canned replies in order and records prompts."""

    model_id = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete(self, system, prompt, max_tokens=1024):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        return LLMResponse(r if isinstance(r, str) else json.dumps(r), 100, 20)


class CountingBackend:
    def __init__(self, results=None):
        self.calls = 0
        self.results = results or [{"title": "t", "url": "https://x.org/a", "content": "c"}]

    async def search(self, query, **params):
        import asyncio

        self.calls += 1
        await asyncio.sleep(0.01)
        return [dict(r, content=f"{r['content']} for {query}") for r in self.results]
