"""SQLite results store. One file holds every run, so any two runs can be compared later."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, variant TEXT, created_at REAL, finished_at REAL,
    config_json TEXT, task_ids_json TEXT, summary_json TEXT);
CREATE TABLE IF NOT EXISTS trials (
    run_id TEXT, task_id TEXT, trial_idx INTEGER, category TEXT, status TEXT, error TEXT,
    report TEXT, citations_json TEXT, events_json TEXT, search_log_json TEXT, metrics_json TEXT,
    PRIMARY KEY (run_id, task_id, trial_idx));
CREATE TABLE IF NOT EXISTS fact_judgments (
    run_id TEXT, task_id TEXT, trial_idx INTEGER, fact_id TEXT, fact_text TEXT,
    present INTEGER, judge_present INTEGER, evidence TEXT, verified INTEGER, method TEXT);
CREATE TABLE IF NOT EXISTS claim_judgments (
    run_id TEXT, task_id TEXT, trial_idx INTEGER, claim_idx INTEGER, claim TEXT,
    urls_json TEXT, verdict TEXT, rationale TEXT, method TEXT);
"""


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---------- writes
    def create_run(self, variant: str, config: dict[str, Any], task_ids: list[str]) -> str:
        run_id = f"{variant}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                          (run_id, variant, time.time(), None, json.dumps(config, default=str),
                           json.dumps(task_ids), None))
        self.conn.commit()
        return run_id

    def save_trial(self, run_id: str, rec: Any) -> None:
        c = self.conn
        c.execute("INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            run_id, rec.task_id, rec.trial_idx, rec.category, rec.status, rec.error, rec.report,
            json.dumps([asdict(x) for x in rec.citations]), json.dumps(rec.events, default=str),
            json.dumps(rec.search_log, default=str), json.dumps(rec.metrics, default=str)))
        c.executemany("INSERT INTO fact_judgments VALUES (?,?,?,?,?,?,?,?,?,?)", [
            (run_id, rec.task_id, rec.trial_idx, f.fact_id, f.fact_text, int(f.present),
             int(f.judge_present), f.evidence, int(f.verified), f.method) for f in rec.fact_judgments])
        c.executemany("INSERT INTO claim_judgments VALUES (?,?,?,?,?,?,?,?,?)", [
            (run_id, rec.task_id, rec.trial_idx, j.claim_idx, j.claim, json.dumps(j.urls),
             j.verdict, j.rationale, j.method) for j in rec.claim_judgments])
        c.commit()

    def finish_run(self, run_id: str, summary: dict[str, Any]) -> None:
        self.conn.execute("UPDATE runs SET finished_at=?, summary_json=? WHERE run_id=?",
                          (time.time(), json.dumps(summary, default=str), run_id))
        self.conn.commit()

    # ---------- reads
    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT run_id, variant, created_at, finished_at FROM runs "
                                 "ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def resolve(self, ref: str) -> str:
        """Accepts a full run id, a unique prefix, 'latest', or 'latest:<variant>'."""
        if ref == "latest" or ref.startswith("latest:"):
            q, args = "SELECT run_id FROM runs", []
            if ":" in ref:
                q += " WHERE variant=?"
                args.append(ref.split(":", 1)[1])
            row = self.conn.execute(q + " ORDER BY created_at DESC LIMIT 1", args).fetchone()
        else:
            rows = self.conn.execute("SELECT run_id FROM runs WHERE run_id LIKE ?", (ref + "%",)).fetchall()
            if len(rows) > 1:
                raise ValueError(f"run reference {ref!r} is ambiguous")
            row = rows[0] if rows else None
        if row is None:
            raise ValueError(f"no run matches {ref!r}")
        return row["run_id"]

    def get_run(self, run_id: str) -> dict[str, Any]:
        r = dict(self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())
        r["config"] = json.loads(r.pop("config_json") or "{}")
        r["task_ids"] = json.loads(r.pop("task_ids_json") or "[]")
        r["summary"] = json.loads(r.pop("summary_json") or "{}")
        return r

    def trials(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM trials WHERE run_id=? ORDER BY task_id, trial_idx",
                                 (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("citations", "events", "search_log", "metrics"):
                d[k] = json.loads(d.pop(f"{k}_json") or "null")
            out.append(d)
        return out

    def fact_judgments(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM fact_judgments WHERE run_id=?", (run_id,)).fetchall()]

    def claim_judgments(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM claim_judgments WHERE run_id=?", (run_id,)).fetchall()
        return [{**dict(r), "urls": json.loads(r["urls_json"])} for r in rows]
