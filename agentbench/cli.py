"""Command-line entry point: `agentbench <command> --help`."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from . import labeling
from .adapters import load_agent
from .judges import (HeuristicFactJudge, HeuristicFaithfulnessJudge, LLMFactJudge,
                     LLMFaithfulnessJudge)
from .report import HEADLINE, render_comparison, render_run, summarize
from .runner import RunConfig, run_eval
from .schema import load_tasks
from .search_cache import FakeCorpusBackend, RecordReplaySearch, TavilyBackend
from .store import Store
from .trajectory import load_pricing

DEFAULT_DB = "results/eval.db"


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _build_judges(spec: str, cache_dir: str, cache_mode: str):
    if spec == "heuristic":
        return HeuristicFactJudge(), HeuristicFaithfulnessJudge(), None
    if spec.startswith("bedrock:"):
        from .llm import BedrockLLM, CachedLLM

        model = spec.split(":", 1)[1]
        llm = CachedLLM(BedrockLLM(model), cache_dir, mode=cache_mode)
        return LLMFactJudge(llm), LLMFaithfulnessJudge(llm), model
    raise SystemExit("--judge must be 'heuristic' or 'bedrock:<model-id>'")


def cmd_run(a: argparse.Namespace) -> int:
    tasks = load_tasks(a.tasks, tags=a.tags, ids=a.ids, limit=a.limit)
    if not tasks:
        raise SystemExit("no tasks selected")
    if a.search == "fake":
        backend = FakeCorpusBackend.from_tasks(load_tasks(a.tasks))
        mode = a.search_mode or "live"
    else:
        mode = a.search_mode or "replay"
        backend = TavilyBackend() if mode != "replay" else None
    search = RecordReplaySearch(a.cache_dir, mode=mode, backend=backend, fuzzy_threshold=a.fuzzy_replay)
    fact_j, faith_j, judge_model = _build_judges(a.judge, a.judge_cache_dir, a.judge_cache_mode)
    agent = load_agent(a.agent, **json.loads(a.agent_kwargs or "{}"))
    variant = a.variant or a.agent.replace(":", "-").split(".")[-1]
    config = RunConfig(variant=variant, trials=a.trials, concurrency=a.concurrency,
                       timeout_s=a.timeout, pricing=load_pricing(a.pricing),
                       extra={"agent": a.agent, "search": a.search, "search_mode": mode,
                              "judge_model": judge_model, "git_sha": _git_sha(), "notes": a.notes})
    store = Store(a.db)
    total = len(tasks) * a.trials
    done = 0

    def progress(rec) -> None:
        nonlocal done
        done += 1
        m = rec.metrics
        recall = "n/a" if m.get("fact_recall") is None else f"{m['fact_recall']:.2f}"
        flags = ",".join(m.get("failure_labels", [])) or "-"
        print(f"[{done:>4}/{total}] {rec.task_id:<28} t{rec.trial_idx} {rec.status:<7} "
              f"recall={recall} flags={flags}", file=sys.stderr)

    run_id = asyncio.run(run_eval(tasks=tasks, agent=agent, search=search, fact_judge=fact_j,
                                  faith_judge=faith_j, store=store, config=config, progress=progress))
    md = render_run(store, run_id)
    if a.report_out:
        Path(a.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report_out).write_text(md)
    print(md)
    print(f"\nrun_id: {run_id}", file=sys.stderr)
    return 0


def _write_or_print(text: str, out: str | None) -> None:
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)


def cmd_report(a) -> int:
    _write_or_print(render_run(Store(a.db), a.run), a.out)
    return 0


def cmd_compare(a) -> int:
    _write_or_print(render_comparison(Store(a.db), a.a, a.b), a.out)
    return 0


def cmd_runs(a) -> int:
    for r in Store(a.db).list_runs():
        print(f"{r['run_id']}\t{r['variant']}\t{'done' if r['finished_at'] else 'incomplete'}")
    return 0


def cmd_summary(a) -> int:
    store = Store(a.db)
    s = summarize(store, store.resolve(a.run))
    print(json.dumps({k: v["mean"] for k, v in s["metrics"].items()} if a.means_only else s, indent=2))
    return 0


def cmd_label_export(a) -> int:
    csv_path, key_path = labeling.export(Store(a.db), a.run, a.out, a.n_facts, a.n_claims, a.seed)
    print(f"Label {csv_path} (fill the human_label column). Judge verdicts are hidden in {key_path}.")
    return 0


def cmd_label_score(a) -> int:
    print(json.dumps(labeling.score(a.csv, a.key), indent=2))
    return 0


def cmd_ci_check(a) -> int:
    """Exit 1 if the run breaks absolute thresholds or regresses past allowed drops vs a baseline."""
    store = Store(a.db)
    means = {k: v["mean"] for k, v in summarize(store, store.resolve(a.run))["metrics"].items()}
    th = json.loads(Path(a.thresholds).read_text())
    higher_better = {k: hb for k, _, hb, _ in HEADLINE}
    failures = []
    for k, lo in th.get("min", {}).items():
        if means.get(k) is None or means[k] < lo:
            failures.append(f"{k}={means.get(k)} below min {lo}")
    for k, hi in th.get("max", {}).items():
        if means.get(k) is not None and means[k] > hi:
            failures.append(f"{k}={means[k]} above max {hi}")
    if a.baseline and not Path(a.baseline).exists():
        print(f"note: baseline {a.baseline} not found; skipping regression check "
              f"(create it with `agentbench summary --means-only > {a.baseline}`)")
    elif a.baseline:
        base = json.loads(Path(a.baseline).read_text())
        for k, allowed in th.get("max_drop", {}).items():
            if base.get(k) is None or means.get(k) is None:
                continue
            delta = means[k] - base[k] if higher_better.get(k, True) else base[k] - means[k]
            if delta < -allowed:
                failures.append(f"{k} regressed {abs(delta):.3f} vs baseline (allowed {allowed})")
    for k in sorted(set(th.get("min", {})) | set(th.get("max", {})) | set(th.get("max_drop", {}))):
        print(f"{k:<24} {means.get(k)}")
    if failures:
        print("\nCI CHECK FAILED:\n- " + "\n- ".join(failures))
        return 1
    print("\nCI check passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentbench", description="Evaluation harness for research agents")
    p.add_argument("--db", default=DEFAULT_DB, help=f"SQLite results file (default {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an eval")
    r.add_argument("--tasks", default="tasks/research_tasks.jsonl")
    r.add_argument("--agent", default="mock", help="'mock', 'mock:<bug>', or 'module:Class'")
    r.add_argument("--agent-kwargs", help="JSON kwargs for the agent class")
    r.add_argument("--variant", help="label for this run (default derived from --agent)")
    r.add_argument("--search", choices=["fake", "tavily"], default="fake")
    r.add_argument("--search-mode", choices=["record", "replay", "live"],
                   help="default: live for fake, replay for tavily")
    r.add_argument("--cache-dir", default="eval_cache/search")
    r.add_argument("--fuzzy-replay", type=float, default=None,
                   help="in replay mode, reuse the closest cached query at this Jaccard similarity")
    r.add_argument("--judge", default="heuristic", help="'heuristic' or 'bedrock:<model-id>'")
    r.add_argument("--judge-cache-dir", default="eval_cache/judge")
    r.add_argument("--judge-cache-mode", choices=["record", "replay", "live"], default="record")
    r.add_argument("--trials", type=int, default=3)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--tags", nargs="*")
    r.add_argument("--ids", nargs="*")
    r.add_argument("--limit", type=int)
    r.add_argument("--pricing", default="config/pricing.json")
    r.add_argument("--notes", default="")
    r.add_argument("--report-out")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="render a run report")
    s.add_argument("--run", default="latest")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_report)

    c = sub.add_parser("compare", help="paired comparison of two runs (B minus A)")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)

    sub.add_parser("runs", help="list runs").set_defaults(fn=cmd_runs)

    sm = sub.add_parser("summary", help="print run summary as JSON")
    sm.add_argument("--run", default="latest")
    sm.add_argument("--means-only", action="store_true")
    sm.set_defaults(fn=cmd_summary)

    le = sub.add_parser("label-export", help="export a blind sample for hand labeling")
    le.add_argument("--run", default="latest")
    le.add_argument("--out", default="labels/labels.csv")
    le.add_argument("--n-facts", type=int, default=30)
    le.add_argument("--n-claims", type=int, default=30)
    le.add_argument("--seed", type=int, default=0)
    le.set_defaults(fn=cmd_label_export)

    ls = sub.add_parser("label-score", help="judge-human agreement (Cohen's kappa)")
    ls.add_argument("--csv", default="labels/labels.csv")
    ls.add_argument("--key")
    ls.set_defaults(fn=cmd_label_score)

    ci = sub.add_parser("ci-check", help="fail on threshold breaches or regressions")
    ci.add_argument("--run", default="latest")
    ci.add_argument("--thresholds", required=True)
    ci.add_argument("--baseline", help="JSON of baseline means (from `summary --means-only`)")
    ci.set_defaults(fn=cmd_ci_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
