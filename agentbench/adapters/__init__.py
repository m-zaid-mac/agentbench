"""Agent adapters. `load_agent("module.path:ClassName")` loads your own."""
from __future__ import annotations

import importlib
from typing import Any


def load_agent(spec: str, **kwargs: Any) -> Any:
    if spec == "mock" or spec.startswith("mock:"):
        from .mock_agent import MockResearchAgent

        bug = spec.split(":", 1)[1] if ":" in spec else None
        return MockResearchAgent(bug=bug)
    if ":" not in spec:
        raise ValueError("agent spec must be 'mock', 'mock:<bug>', or 'package.module:ClassName'")
    module_name, cls_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)
