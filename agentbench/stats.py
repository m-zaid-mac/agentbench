"""Statistics: agents are nondeterministic, so every headline number gets a confidence interval."""
from __future__ import annotations

import random
from collections import Counter
from typing import Sequence


def mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float | None, float | None, float | None]:
    """Percentile bootstrap over tasks. Returns (mean, lo, hi)."""
    vals = list(values)
    if not vals:
        return None, None, None
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(vals) / n, lo, hi


def paired_bootstrap_diff(a: dict[str, float], b: dict[str, float], n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0):
    """CI on mean(b - a) over tasks both runs scored. Pairing removes per-task difficulty variance."""
    shared = sorted(set(a) & set(b))
    diffs = [b[t] - a[t] for t in shared]
    m, lo, hi = bootstrap_ci(diffs, n_boot, alpha, seed)
    return m, lo, hi, len(shared)


def cohens_kappa(a: Sequence, b: Sequence) -> float | None:
    if len(a) != len(b):
        raise ValueError("label lists differ in length")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
