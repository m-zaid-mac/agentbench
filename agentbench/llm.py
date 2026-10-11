"""LLM clients for the judges: Bedrock (Converse API) plus a record/replay cache."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class LLM(Protocol):
    model_id: str

    async def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse: ...


_RETRYABLE = {"ThrottlingException", "ServiceUnavailableException",
              "ModelNotReadyException", "InternalServerException"}


class BedrockLLM:
    def __init__(self, model_id: str, region: str | None = None, temperature: float = 0.0,
                 max_retries: int = 6):
        import boto3  # imported lazily so the offline demo needs no AWS deps

        self.model_id = model_id
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = boto3.client(
            "bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def _call(self, system: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        return self._client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": self.temperature},
        )

    async def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        from botocore.exceptions import ClientError

        for attempt in range(self.max_retries + 1):
            try:
                resp = await asyncio.to_thread(self._call, system, prompt, max_tokens)
                break
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code not in _RETRYABLE or attempt == self.max_retries:
                    raise
                await asyncio.sleep(min(30.0, 2 ** attempt) * (0.5 + random.random()))
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        usage = resp.get("usage", {})
        return LLMResponse(text, usage.get("inputTokens", 0), usage.get("outputTokens", 0))


class CachedLLM:
    """Caches judge calls by prompt hash. Re-scoring a stored run or re-running CI costs nothing."""

    def __init__(self, inner: LLM, cache_dir: str | Path, mode: str = "record"):
        if mode not in ("record", "replay", "live"):
            raise ValueError("mode must be record, replay or live")
        self.inner = inner
        self.model_id = inner.model_id
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        if self.mode == "live":
            return await self.inner.complete(system, prompt, max_tokens)
        key = hashlib.sha256(json.dumps([self.model_id, system, prompt, max_tokens]).encode()).hexdigest()[:32]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            d = json.loads(path.read_text())
            return LLMResponse(d["text"], d["input_tokens"], d["output_tokens"], cached=True)
        if self.mode == "replay":
            raise RuntimeError(f"Judge cache miss ({key}) in replay mode")
        resp = await self.inner.complete(system, prompt, max_tokens)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"text": resp.text, "input_tokens": resp.input_tokens,
                                   "output_tokens": resp.output_tokens}))
        os.replace(tmp, path)
        return resp


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences or stray prose around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in model output: {text[:200]!r}")
