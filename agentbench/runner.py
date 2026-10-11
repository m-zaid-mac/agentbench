"""Runs tasks x trials concurrently, scores each trial, and writes everything to the store."""
from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .context import RunContext
from .failures import classify
from .judges import (ClaimJudgment, FactJudgment, extract_claims, faithfulness_score,
                     numeric_fidelity)
from .schema import AgentResult, Citation, Task
from .search_cache import CacheMiss
from .store import Store
from .trajectory import trajectory_metrics
from .urls import normalize_url


class Agent(Protocol):
    async def run(self, question: str, ctx: RunContext) -> AgentResult: ...


@dataclass
class RunConfig:
    variant: str
    trials: int = 3
    concurrency: int = 4
    timeout_s: float = 300.0
    pricing: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # free-form: agent spec, judge model, git sha...


@dataclass
class TrialRecord:
    task_id: str
    trial_idx: int
    category: str
    status: str
    error: str | None
    report: str
    citations: list[Citation]
    events: list[dict[str, Any]]
    search_log: list[dict[str, Any]]
    metrics: dict[str, Any]
    fact_judgments: list[FactJudgment]
    claim_judgments: list[ClaimJudgment]


async def run_trial(task: Task, trial_idx: int, agent: Agent, search: Any, fact_judge: Any,
                    faith_judge: Any, config: RunConfig) -> TrialRecord:
    ctx = RunContext(task.id, trial_idx, search)
    status, error, error_type = "ok", None, None
    result: AgentResult | None = None
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(agent.run(task.question, ctx), timeout=config.timeout_s)
        if not isinstance(result, AgentResult):
            raise TypeError(f"agent returned {type(result).__name__}, expected AgentResult")
    except asyncio.TimeoutError:
        status, error_type, error = "timeout", "timeout", f"exceeded {config.timeout_s}s"
    except CacheMiss as e:
        status, error_type, error = "error", "cache_miss", str(e)
    except Exception as e:  # the harness must survive any agent failure
        status, error_type = "error", type(e).__name__
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"
    latency = time.perf_counter() - start

    metrics: dict[str, Any] = {"status": status, "error_type": error_type, "latency_s": round(latency, 3)}
    metrics.update(trajectory_metrics(ctx.events, config.pricing))
    facts: list[FactJudgment] = []
    claim_js: list[ClaimJudgment] = []

    if result is None:
        # An answer that never arrives has zero recall; faithfulness is undefined.
        metrics.update(fact_recall=0.0, faithfulness=None, numeric_fidelity=None, n_claims=0)
    else:
        report = result.report or ""
        known = ctx.known_urls()
        markers = {c.marker for c in result.citations}
        phantom = sorted({normalize_url(c.url) for c in result.citations} - known)
        claims = extract_claims(report, result.citations)
        dangling = sum(1 for c in claims for m in c.markers if m not in markers)
        num_score, num_missing = numeric_fidelity(task, report)
        try:
            facts = await fact_judge.judge(task, report)
        except Exception as e:
            metrics["fact_judge_error"] = f"{type(e).__name__}: {e}"
        try:
            claim_js = await faith_judge.judge(claims, ctx.source_text)
        except Exception as e:
            metrics["faith_judge_error"] = f"{type(e).__name__}: {e}"
        metrics.update(
            fact_recall=(sum(f.present for f in facts) / len(facts)) if facts else None,
            facts_unverified=sum(1 for f in facts if f.judge_present and not f.verified),
            faithfulness=faithfulness_score(claim_js),
            numeric_fidelity=num_score,
            numbers_missing=num_missing,
            n_claims=len(claims),
            n_claims_judged=len(claim_js),
            n_citations=len(result.citations),
            phantom_citations=len(phantom),
            phantom_urls=phantom,
            dangling_markers=dangling,
            report_chars=len(report),
        )
    metrics["failure_labels"] = classify(metrics)
    return TrialRecord(task.id, trial_idx, task.category, status, error,
                       result.report if result else "", result.citations if result else [],
                       ctx.events, ctx.search_log, metrics, facts, claim_js)


async def run_eval(*, tasks: list[Task], agent: Agent, search: Any, fact_judge: Any, faith_judge: Any,
                   store: Store, config: RunConfig,
                   progress: Callable[[TrialRecord], None] | None = None) -> str:
    run_id = store.create_run(config.variant, {
        "trials": config.trials, "concurrency": config.concurrency, "timeout_s": config.timeout_s,
        "fact_judge": type(fact_judge).__name__, "faith_judge": type(faith_judge).__name__,
        **config.extra}, [t.id for t in tasks])
    sem = asyncio.Semaphore(config.concurrency)

    async def one(task: Task, k: int) -> None:
        async with sem:
            rec = await run_trial(task, k, agent, search, fact_judge, faith_judge, config)
        store.save_trial(run_id, rec)
        if progress:
            progress(rec)

    # Trial index on the outer loop so concurrent slots hold *different* tasks,
    # which is what surfaces shared-state bugs between runs.
    await asyncio.gather(*(one(t, k) for k in range(config.trials) for t in tasks))
    store.finish_run(run_id, {
        "search_stats": getattr(search, "stats", {}),
        "fact_judge_usage": fact_judge.usage.as_dict(),
        "faith_judge_usage": faith_judge.usage.as_dict(),
    })
    return run_id
