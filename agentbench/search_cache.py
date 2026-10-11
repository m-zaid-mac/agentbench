"""Record/replay layer for web search.

Without this, a score change could come from your code OR from the web changing under you.
- record: return cached results when present, otherwise call the backend and cache them.
- replay: never touch the network; a cache miss is an error (optionally with fuzzy fallback).
- live:   always call the backend, never cache.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

from .text import jaccard, tokens

MODES = ("record", "replay", "live")


class CacheMiss(RuntimeError):
    pass


class SearchBackend(Protocol):
    async def search(self, query: str, **params: Any) -> list[dict[str, Any]]: ...


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def cache_key(query: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"q": normalize_query(query), "p": params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class TavilyBackend:
    def __init__(self, api_key: str | None = None, max_results: int = 5, search_depth: str = "basic"):
        try:
            from tavily import TavilyClient
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install tavily-python: pip install 'agentbench[live]'") from e
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self._client = TavilyClient(api_key=key)
        self.max_results = max_results
        self.search_depth = search_depth

    async def search(self, query: str, **params: Any) -> list[dict[str, Any]]:
        p = {"max_results": self.max_results, "search_depth": self.search_depth, **params}
        resp = await asyncio.to_thread(self._client.search, query=query, **p)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "content": r.get("content", ""), "score": r.get("score")}
            for r in resp.get("results", [])
        ]


class FakeCorpusBackend:
    """Offline keyword search over a small corpus. Used for the demo and tests."""

    def __init__(self, docs: list[dict[str, Any]], k: int = 4, latency_s: float = 0.002):
        self.docs = docs
        self.k = k
        self.latency_s = latency_s
        self._doc_tokens = [set(tokens(d["title"] + " " + d["content"])) for d in docs]

    @classmethod
    def from_tasks(cls, tasks: list[Any], **kw: Any) -> "FakeCorpusBackend":
        docs = []
        for t in tasks:
            for f in t.key_facts:
                docs.append({
                    "title": t.question,
                    "url": f"https://demo.local/{t.id}/{f.id}",
                    "content": f.text if f.text.endswith(".") else f.text + ".",
                })
        return cls(docs, **kw)

    async def search(self, query: str, **params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(self.latency_s)
        q = set(tokens(query))
        scored = [(len(q & dt) / (len(q) or 1), i) for i, dt in enumerate(self._doc_tokens)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1]))
        k = int(params.get("max_results", self.k))
        return [{**self.docs[i], "score": round(score, 3)} for score, i in scored[:k]]


class RecordReplaySearch:
    def __init__(self, cache_dir: str | Path, mode: str = "replay",
                 backend: SearchBackend | None = None, fuzzy_threshold: float | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "replay" and backend is None:
            raise ValueError(f"mode {mode!r} needs a backend")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.backend = backend
        self.fuzzy_threshold = fuzzy_threshold
        self.stats = {"hits": 0, "fuzzy_hits": 0, "misses": 0, "backend_calls": 0}
        self._locks: dict[str, asyncio.Lock] = {}
        self._index: list[tuple[set[str], Path]] | None = None

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _fuzzy_lookup(self, query: str) -> Path | None:
        """LLM-generated sub-questions drift between runs; match the closest cached query."""
        if self._index is None:
            self._index = []
            for p in sorted(self.cache_dir.glob("*.json")):
                try:
                    self._index.append((set(tokens(json.loads(p.read_text())["query"])), p))
                except (json.JSONDecodeError, KeyError):
                    continue
        q = set(tokens(query))
        best = max(((jaccard(q, t), p) for t, p in self._index), default=(0.0, None), key=lambda x: x[0])
        if best[1] is not None and best[0] >= (self.fuzzy_threshold or 1.0):
            return best[1]
        return None

    async def search(self, query: str, **params: Any) -> list[dict[str, Any]]:
        if self.mode == "live":
            self.stats["backend_calls"] += 1
            return await self.backend.search(query, **params)  # type: ignore[union-attr]

        key = cache_key(query, params)
        path = self._path(key)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:  # concurrent identical queries -> one backend call
            if path.exists():
                self.stats["hits"] += 1
                return json.loads(path.read_text())["results"]
            if self.mode == "replay":
                if self.fuzzy_threshold:
                    match = self._fuzzy_lookup(query)
                    if match is not None:
                        self.stats["fuzzy_hits"] += 1
                        return json.loads(match.read_text())["results"]
                self.stats["misses"] += 1
                raise CacheMiss(f"No cached search for {query!r} (key {key}); "
                                f"run with --search-mode record first")
            self.stats["misses"] += 1
            self.stats["backend_calls"] += 1
            results = await self.backend.search(query, **params)  # type: ignore[union-attr]
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"query": query, "params": params, "results": results,
                                       "recorded_at": time.time()}, indent=2))
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache entry
            self._index = None
            return results
