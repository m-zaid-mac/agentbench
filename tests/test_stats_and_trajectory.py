from agentbench.stats import bootstrap_ci, cohens_kappa, paired_bootstrap_diff
from agentbench.trajectory import trajectory_metrics


def test_cohens_kappa_known_values():
    assert cohens_kappa(["y", "n", "y", "n"], ["y", "n", "y", "n"]) == 1.0
    # classic example: po = 0.7, pe = 0.5 -> kappa = 0.4
    a = ["y"] * 5 + ["n"] * 5
    b = ["y"] * 4 + ["n"] + ["y"] * 2 + ["n"] * 3
    assert abs(cohens_kappa(a, b) - 0.4) < 1e-9


def test_bootstrap_ci_brackets_mean():
    m, lo, hi = bootstrap_ci([0.2, 0.4, 0.6, 0.8, 1.0] * 4)
    assert lo <= m <= hi and abs(m - 0.6) < 1e-9


def test_paired_diff_detects_consistent_improvement():
    a = {f"t{i}": 0.5 for i in range(20)}
    b = {f"t{i}": 0.6 + (i % 3) * 0.01 for i in range(20)}
    d, lo, hi, n = paired_bootstrap_diff(a, b)
    assert n == 20 and lo > 0


def test_trajectory_duplicates_and_cost():
    events = [
        {"type": "search", "query": "JWST launch date"},
        {"type": "search", "query": "jwst launch date?"},
        {"type": "search", "query": "Hubble mirror size"},
        {"type": "llm_call", "model": "m", "input_tokens": 1_000_000, "output_tokens": 0},
        {"type": "subquestions", "items": ["a", "b"]},
    ]
    m = trajectory_metrics(events, {"m": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}})
    assert m["n_searches"] == 3 and abs(m["duplicate_search_rate"] - 1 / 3) < 1e-9
    assert m["agent_cost_usd"] == 3.0 and m["n_subquestions"] == 2
    assert trajectory_metrics(events, {})["agent_cost_usd"] is None  # unknown price -> n/a


def test_search_errors_are_labeled():
    from agentbench.failures import classify

    labels = classify({"status": "ok", "fact_recall": 1.0, "faithfulness": 1.0, "n_claims": 3,
                       "n_search_errors": 2})
    assert "search_error" in labels