"""Markdown reports: single-run summaries and paired comparisons between runs."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .failures import DESCRIPTIONS
from .stats import bootstrap_ci, mean, paired_bootstrap_diff, stdev
from .store import Store

# (key, label, higher_is_better, format)
HEADLINE = [
    ("fact_recall", "Fact recall", True, "pct"),
    ("faithfulness", "Citation faithfulness", True, "pct"),
    ("numeric_fidelity", "Numeric fidelity", True, "pct"),
    ("success_rate", "Trial success rate", True, "pct"),
    ("phantom_rate", "Trials with phantom citations", False, "pct"),
    ("search_error_rate", "Trials with search errors", False, "pct"),
    ("duplicate_search_rate", "Duplicate search rate", False, "pct"),
    ("n_searches", "Searches per trial", False, "num"),
    ("latency_s", "Latency per trial (s)", False, "num"),
    ("input_tokens", "Agent input tokens per trial", False, "int"),
    ("output_tokens", "Agent output tokens per trial", False, "int"),
    ("agent_cost_usd", "Agent cost per trial ($)", False, "usd"),
]


def _trial_value(t: dict[str, Any], key: str) -> float | None:
    m = t["metrics"]
    if key == "success_rate":
        return 1.0 if t["status"] == "ok" else 0.0
    if key == "search_error_rate":
        return (1.0 if m.get("n_search_errors", 0) > 0 else 0.0) if t["status"] == "ok" else None
    if key == "phantom_rate":
        return 1.0 if m.get("phantom_citations", 0) > 0 else 0.0 if t["status"] == "ok" else None
    v = m.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def task_means(trials: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Average trials within each task first, so tasks (not trials) are the unit of analysis."""
    per: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        v = _trial_value(t, key)
        if v is not None:
            per[t["task_id"]].append(v)
    return {k: sum(v) / len(v) for k, v in per.items() if v}


def summarize(store: Store, run_id: str) -> dict[str, Any]:
    trials = store.trials(run_id)
    out: dict[str, Any] = {"run_id": run_id, "n_trials": len(trials),
                           "n_tasks": len({t["task_id"] for t in trials}), "metrics": {}}
    for key, *_ in HEADLINE:
        m, lo, hi = bootstrap_ci(list(task_means(trials, key).values()))
        out["metrics"][key] = {"mean": m, "ci_low": lo, "ci_high": hi}
    # Flakiness: how much the same task's recall varies across repeated trials.
    per: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        v = _trial_value(t, "fact_recall")
        if v is not None:
            per[t["task_id"]].append(v)
    out["recall_within_task_stdev"] = mean([stdev(v) for v in per.values() if len(v) > 1])
    return out


def _fmt(v: float | None, kind: str) -> str:
    if v is None:
        return "n/a"
    return {"pct": f"{v * 100:.1f}%", "num": f"{v:.2f}", "int": f"{v:,.0f}", "usd": f"${v:.4f}"}[kind]


def _fmt_diff(v: float | None, kind: str) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else "-"
    a = abs(v)
    return sign + {"pct": f"{a * 100:.1f} pts", "num": f"{a:.2f}", "int": f"{a:,.0f}", "usd": f"${a:.4f}"}[kind]


def render_run(store: Store, run_ref: str) -> str:
    run_id = store.resolve(run_ref)
    run = store.get_run(run_id)
    trials = store.trials(run_id)
    s = summarize(store, run_id)
    cfg = run["config"]
    lines = [f"# Eval report: `{run['variant']}`", "",
             f"Run `{run_id}`: {s['n_tasks']} tasks x {cfg.get('trials')} trials = {s['n_trials']} trials. "
             f"Judges: {cfg.get('fact_judge')} / {cfg.get('faith_judge')}"
             + (f" ({cfg['judge_model']})" if cfg.get("judge_model") else "") + ".", "",
             "Confidence intervals are 95% percentile bootstrap over tasks (trials averaged within each task first).", "",
             "## Headline metrics", "", "| Metric | Mean | 95% CI |", "|---|---|---|"]
    for key, label, _, kind in HEADLINE:
        m = s["metrics"][key]
        ci = "n/a" if m["mean"] is None else f"{_fmt(m['ci_low'], kind)} to {_fmt(m['ci_high'], kind)}"
        lines.append(f"| {label} | {_fmt(m['mean'], kind)} | {ci} |")
    flak = s["recall_within_task_stdev"]
    if flak is not None:
        lines += ["", f"Run-to-run variation: same task, different trial, fact recall varies by "
                      f"{flak * 100:.1f} pts (mean within-task standard deviation)."]

    # per category
    cats: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        cats[t["category"]].append(t)
    lines += ["", "## Fact recall by category", "", "| Category | Tasks | Fact recall |", "|---|---|---|"]
    for cat in sorted(cats):
        tm = task_means(cats[cat], "fact_recall")
        lines.append(f"| {cat} | {len(tm)} | {_fmt(mean(list(tm.values())), 'pct')} |")

    # failure taxonomy
    counts = Counter(lbl for t in trials for lbl in t["metrics"].get("failure_labels", []))
    lines += ["", "## Failure taxonomy", ""]
    if counts:
        lines += ["| Failure | Trials | Share | Meaning |", "|---|---|---|---|"]
        for lbl, n in counts.most_common():
            desc = DESCRIPTIONS.get(lbl, "agent raised or timed out" if lbl.startswith("error:") else "")
            lines.append(f"| `{lbl}` | {n} | {n / len(trials) * 100:.0f}% | {desc} |")
    else:
        lines.append("No failures detected.")

    # worst tasks
    recall = task_means(trials, "fact_recall")
    missed = Counter((f["task_id"], f["fact_id"], f["fact_text"])
                     for f in store.fact_judgments(run_id) if not f["present"])
    lines += ["", "## Lowest-recall tasks", ""]
    for task_id, r in sorted(recall.items(), key=lambda kv: kv[1])[:5]:
        lines.append(f"**{task_id}**: {_fmt(r, 'pct')} recall")
        for (tid, fid, text), n in missed.most_common():
            if tid == task_id:
                lines.append(f"- missed {fid} in {n}/{cfg.get('trials')} trials: {text}")
        lines.append("")

    errors = [t for t in trials if t["status"] != "ok"]
    if errors:
        lines += ["## Errors (first 5)", ""]
        for t in errors[:5]:
            first = (t["error"] or "").splitlines()[0] if t["error"] else ""
            lines.append(f"- `{t['task_id']}` trial {t['trial_idx']}: {first}")
        lines.append("")

    jf, jc = run["summary"].get("fact_judge_usage", {}), run["summary"].get("faith_judge_usage", {})
    if jf.get("calls") or jc.get("calls"):
        lines += ["## Judge usage", "",
                  f"Fact judge: {jf.get('calls', 0)} calls ({jf.get('cached_calls', 0)} cached), "
                  f"{jf.get('input_tokens', 0):,} in / {jf.get('output_tokens', 0):,} out tokens. "
                  f"Faithfulness judge: {jc.get('calls', 0)} calls ({jc.get('cached_calls', 0)} cached), "
                  f"{jc.get('input_tokens', 0):,} in / {jc.get('output_tokens', 0):,} out tokens.", ""]
        unverified = sum(t["metrics"].get("facts_unverified", 0) for t in trials)
        if unverified:
            lines += [f"{unverified} fact verdicts were overturned because the judge's quoted evidence "
                      f"was not found in the report.", ""]
    ss = run["summary"].get("search_stats", {})
    if ss:
        lines += ["## Search cache", "", ", ".join(f"{k}: {v}" for k, v in ss.items()), ""]
    return "\n".join(lines)


def compare(store: Store, ref_a: str, ref_b: str) -> list[dict[str, Any]]:
    a_id, b_id = store.resolve(ref_a), store.resolve(ref_b)
    ta, tb = store.trials(a_id), store.trials(b_id)
    rows = []
    for key, label, higher_better, kind in HEADLINE:
        ma, mb = task_means(ta, key), task_means(tb, key)
        d, lo, hi, n = paired_bootstrap_diff(ma, mb)
        if d is None:
            verdict = "n/a"
        elif lo is not None and (lo > 0 or hi < 0):
            better = (d > 0) == higher_better
            verdict = "B better" if better else "B worse"
        else:
            verdict = "no significant difference"
        rows.append({"key": key, "label": label, "kind": kind, "a": mean(list(ma.values())),
                     "b": mean(list(mb.values())), "diff": d, "lo": lo, "hi": hi, "n": n, "verdict": verdict})
    return rows


def render_comparison(store: Store, ref_a: str, ref_b: str) -> str:
    a_id, b_id = store.resolve(ref_a), store.resolve(ref_b)
    va, vb = store.get_run(a_id)["variant"], store.get_run(b_id)["variant"]
    rows = compare(store, a_id, b_id)
    lines = [f"# Comparison: A = `{va}` vs B = `{vb}`", "",
             f"A: `{a_id}`  \nB: `{b_id}`", "",
             "Differences are B minus A, paired by task, with 95% bootstrap CIs. "
             "A difference is called only when the CI excludes zero.", "",
             "| Metric | A | B | B - A | 95% CI | Verdict |", "|---|---|---|---|---|---|"]
    for r in rows:
        k = r["kind"]
        ci = "n/a" if r["lo"] is None else f"{_fmt_diff(r['lo'], k)} to {_fmt_diff(r['hi'], k)}"
        lines.append(f"| {r['label']} | {_fmt(r['a'], k)} | {_fmt(r['b'], k)} | "
                     f"{_fmt_diff(r['diff'], k)} | {ci} | {r['verdict']} |")
    return "\n".join(lines) + "\n"