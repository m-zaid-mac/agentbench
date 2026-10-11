"""Full pipeline on the mock agent, including proof that seeded bugs are caught."""
import asyncio
import json
from pathlib import Path

from agentbench import labeling
from agentbench.adapters import load_agent
from agentbench.cli import main
from agentbench.judges import HeuristicFactJudge, HeuristicFaithfulnessJudge
from agentbench.report import compare, render_comparison, render_run, summarize
from agentbench.runner import RunConfig, run_eval
from agentbench.schema import load_tasks
from agentbench.search_cache import FakeCorpusBackend, RecordReplaySearch
from agentbench.store import Store

TASKS = Path(__file__).resolve().parents[1] / "tasks" / "research_tasks.jsonl"


def _run(tmp_path, agent_spec, variant, concurrency=4, tasks=None, mode="live", trials=2):
    tasks = tasks or load_tasks(TASKS, limit=12)
    store = Store(tmp_path / "eval.db")
    search = RecordReplaySearch(tmp_path / "cache", mode, FakeCorpusBackend.from_tasks(load_tasks(TASKS)))
    run_id = asyncio.run(run_eval(
        tasks=tasks, agent=load_agent(agent_spec), search=search, fact_judge=HeuristicFactJudge(),
        faith_judge=HeuristicFaithfulnessJudge(), store=store,
        config=RunConfig(variant=variant, trials=trials, concurrency=concurrency,
                         pricing={"mock": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}})))
    return store, run_id


def test_clean_agent_scores_well(tmp_path):
    store, run_id = _run(tmp_path, "mock", "clean")
    m = summarize(store, run_id)["metrics"]
    assert m["success_rate"]["mean"] == 1.0
    assert m["fact_recall"]["mean"] > 0.9 and m["faithfulness"]["mean"] > 0.9
    assert m["phantom_rate"]["mean"] == 0.0 and m["numeric_fidelity"]["mean"] == 1.0
    assert m["agent_cost_usd"]["mean"] > 0
    md = render_run(store, run_id)
    assert "Headline metrics" in md and "Fact recall" in md


def test_digit_bug_is_caught_by_numeric_check(tmp_path):
    digits = load_tasks(TASKS, tags=["regression:digits"])
    store, clean = _run(tmp_path, "mock", "clean", tasks=digits)
    _, buggy = _run(tmp_path, "mock:buggy_digits", "buggy", tasks=digits)
    rows = {r["key"]: r for r in compare(store, clean, buggy)}
    assert rows["numeric_fidelity"]["verdict"] == "B worse"


def test_shared_state_race_is_caught_by_phantom_citations(tmp_path):
    store, clean = _run(tmp_path, "mock", "clean", concurrency=8)
    _, racy = _run(tmp_path, "mock:shared_state", "racy", concurrency=8)
    assert summarize(store, clean)["metrics"]["phantom_rate"]["mean"] == 0.0
    assert summarize(store, racy)["metrics"]["phantom_rate"]["mean"] > 0.3
    assert "phantom_citation" in render_run(store, racy)
    assert "B worse" in render_comparison(store, clean, racy)


def test_replay_mode_reproduces_recorded_run(tmp_path):
    tasks = load_tasks(TASKS, limit=4)
    store, a = _run(tmp_path, "mock", "rec", tasks=tasks, mode="record", trials=1)
    _, b = _run(tmp_path, "mock", "rep", tasks=tasks, mode="replay", trials=1)
    ra = {t["task_id"]: t["report"] for t in store.trials(a)}
    rb = {t["task_id"]: t["report"] for t in store.trials(b)}
    assert ra == rb and all(t["status"] == "ok" for t in store.trials(b))


def test_agent_exceptions_become_failed_trials(tmp_path):
    class Boom:
        async def run(self, question, ctx):
            raise RuntimeError("kaboom")

    store = Store(tmp_path / "e.db")
    search = RecordReplaySearch(tmp_path / "c", "live", FakeCorpusBackend([]))
    run_id = asyncio.run(run_eval(tasks=load_tasks(TASKS, limit=2), agent=Boom(), search=search,
                                  fact_judge=HeuristicFactJudge(), faith_judge=HeuristicFaithfulnessJudge(),
                                  store=store, config=RunConfig(variant="boom", trials=1)))
    trials = store.trials(run_id)
    assert all(t["status"] == "error" and "kaboom" in t["error"] for t in trials)
    assert all(t["metrics"]["failure_labels"] == ["error:RuntimeError"] for t in trials)


def test_labeling_roundtrip(tmp_path):
    store, run_id = _run(tmp_path, "mock", "clean", tasks=load_tasks(TASKS, limit=4), trials=1)
    csv_path, key_path = labeling.export(store, run_id, tmp_path / "labels.csv", n_facts=5, n_claims=5)
    assert "judge" not in csv_path.read_text().splitlines()[0]  # blind: no judge column
    key = json.loads(key_path.read_text())["items"]
    import csv
    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:  # a "human" who agrees with the judge everywhere
        r["human_label"] = key[r["item_id"]]["judge"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=labeling.FIELDS)
        w.writeheader()
        w.writerows(rows)
    res = labeling.score(csv_path)
    assert res["fact"]["agreement"] == 1.0 and res["claim"]["agreement"] == 1.0


def test_cli_run_and_ci_check(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "run", "--tasks", str(TASKS), "--limit", "3", "--trials", "1",
                 "--cache-dir", str(tmp_path / "c")]) == 0
    th = tmp_path / "th.json"
    th.write_text(json.dumps({"min": {"success_rate": 1.0}, "max": {"phantom_rate": 0.0}}))
    assert main(["--db", db, "ci-check", "--thresholds", str(th)]) == 0
    th.write_text(json.dumps({"min": {"fact_recall": 1.01}}))
    assert main(["--db", db, "ci-check", "--thresholds", str(th)]) == 1
