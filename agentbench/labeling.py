"""Judge validation: export a blind sample for hand labeling, then measure judge-human agreement.

The CSV you label does NOT contain the judge's verdicts (they go in a separate key file),
so your labels aren't anchored by what the judge said.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .stats import cohens_kappa
from .store import Store

FIELDS = ["item_id", "item_type", "question_or_fact", "claim", "context", "allowed_labels", "human_label"]
_TRUE = {"1", "y", "yes", "true", "present", "t"}
_FALSE = {"0", "n", "no", "false", "absent", "f"}


def export(store: Store, run_ref: str, out_csv: str | Path, n_facts: int = 30, n_claims: int = 30,
           seed: int = 0, max_context: int = 6000) -> tuple[Path, Path]:
    run_id = store.resolve(run_ref)
    rng = random.Random(seed)
    reports = {(t["task_id"], t["trial_idx"]): t for t in store.trials(run_id)}
    facts = store.fact_judgments(run_id)
    claims = [c for c in store.claim_judgments(run_id) if c["verdict"] != "no_source"]
    rows, key = [], {}
    for f in rng.sample(facts, min(n_facts, len(facts))):
        item_id = f"F-{f['task_id']}-{f['trial_idx']}-{f['fact_id']}"
        report = reports[(f["task_id"], f["trial_idx"])]["report"]
        rows.append({"item_id": item_id, "item_type": "fact", "question_or_fact": f["fact_text"],
                     "claim": "", "context": report[:max_context], "allowed_labels": "yes / no",
                     "human_label": ""})
        key[item_id] = {"judge": "yes" if f["present"] else "no"}
    for c in rng.sample(claims, min(n_claims, len(claims))):
        item_id = f"C-{c['task_id']}-{c['trial_idx']}-{c['claim_idx']}"
        t = reports[(c["task_id"], c["trial_idx"])]
        sources = []
        for u in c["urls"]:
            for e in t["search_log"]:
                for r in e["results"]:
                    if r.get("url") == u and r.get("content") and r["content"] not in sources:
                        sources.append(f"[{u}]\n{r['content']}")
        rows.append({"item_id": item_id, "item_type": "claim", "question_or_fact": "",
                     "claim": c["claim"], "context": "\n\n".join(sources)[:max_context],
                     "allowed_labels": "supported / partial / unsupported", "human_label": ""})
        key[item_id] = {"judge": c["verdict"]}
    rng.shuffle(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    key_path = out_csv.with_suffix(".key.json")
    key_path.write_text(json.dumps({"run_id": run_id, "items": key}, indent=2))
    return out_csv, key_path


def _norm(item_type: str, label: str) -> str | None:
    label = label.strip().lower()
    if not label:
        return None
    if item_type == "fact":
        return "yes" if label in _TRUE else "no" if label in _FALSE else None
    return label if label in {"supported", "partial", "unsupported"} else None


def score(csv_path: str | Path, key_path: str | Path | None = None) -> dict[str, Any]:
    csv_path = Path(csv_path)
    key = json.loads(Path(key_path or csv_path.with_suffix(".key.json")).read_text())["items"]
    pairs: dict[str, list[tuple[str, str]]] = {"fact": [], "claim": []}
    skipped = 0
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            h = _norm(row["item_type"], row.get("human_label", ""))
            if h is None or row["item_id"] not in key:
                skipped += 1
                continue
            pairs[row["item_type"]].append((key[row["item_id"]]["judge"], h))
    out: dict[str, Any] = {"skipped_unlabeled_or_invalid": skipped}
    for kind, ps in pairs.items():
        if not ps:
            continue
        j, h = [p[0] for p in ps], [p[1] for p in ps]
        res = {"n": len(ps), "agreement": sum(a == b for a, b in ps) / len(ps),
               "kappa": cohens_kappa(j, h),
               "confusion(judge->human)": {f"{a}->{b}": n for (a, b), n in sorted(Counter(ps).items())}}
        if kind == "claim":  # binary view: is the claim fully supported?
            jb = ["supported" if x == "supported" else "not" for x in j]
            hb = ["supported" if x == "supported" else "not" for x in h]
            res["kappa_binary_supported"] = cohens_kappa(jb, hb)
        out[kind] = res
    return out
