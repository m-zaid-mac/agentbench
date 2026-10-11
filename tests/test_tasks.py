import re
from pathlib import Path

from agentbench.schema import load_tasks
from agentbench.text import normalize_numbers

TASKS = Path(__file__).resolve().parents[1] / "tasks" / "research_tasks.jsonl"


def test_task_file_is_valid():
    tasks = load_tasks(TASKS)
    assert len(tasks) >= 30
    assert all(len(t.key_facts) >= 2 for t in tasks)
    assert any("smoke" in t.tags for t in tasks)


def test_every_required_number_appears_in_its_fact_text():
    # Catches typos where 'numbers' and the fact sentence disagree.
    for t in load_tasks(TASKS):
        for f in t.key_facts:
            for n in f.numbers:
                assert re.search(re.escape(normalize_numbers(n)), normalize_numbers(f.text)), (t.id, f.id, n)


def test_tag_filter():
    smoke = load_tasks(TASKS, tags=["smoke"])
    assert smoke and all("smoke" in t.tags for t in smoke)
