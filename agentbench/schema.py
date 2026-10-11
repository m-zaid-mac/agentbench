"""Core data types shared across the harness."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class KeyFact:
    """A fact a good answer must contain. `numbers` are checked verbatim (deterministically)."""

    id: str
    text: str
    numbers: list[str] = field(default_factory=list)


@dataclass
class Task:
    id: str
    question: str
    category: str
    key_facts: list[KeyFact]
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        return cls(
            id=d["id"],
            question=d["question"],
            category=d.get("category", "general"),
            key_facts=[KeyFact(**kf) for kf in d["key_facts"]],
            tags=list(d.get("tags", [])),
        )


def load_tasks(
    path: str | Path,
    tags: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[Task]:
    """Load a JSONL task file. Lines starting with '#' are comments."""
    tasks: list[Task] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            task = Task.from_dict(json.loads(line))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"{path}:{lineno}: invalid task: {e}") from e
        if task.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate task id {task.id!r}")
        if not task.key_facts:
            raise ValueError(f"{path}:{lineno}: task {task.id!r} has no key_facts")
        seen.add(task.id)
        tasks.append(task)
    if tags:
        wanted = set(tags)
        tasks = [t for t in tasks if wanted & set(t.tags)]
    if ids:
        wanted_ids = set(ids)
        tasks = [t for t in tasks if t.id in wanted_ids]
    if limit:
        tasks = tasks[:limit]
    return tasks


@dataclass
class Citation:
    marker: int
    url: str
    title: str = ""


@dataclass
class AgentResult:
    """What an agent adapter must return. Citation markers like [1] in `report` map to `citations`."""

    report: str
    citations: list[Citation] = field(default_factory=list)
