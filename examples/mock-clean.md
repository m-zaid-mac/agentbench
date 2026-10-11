# Eval report: `mock-clean`

Run `mock-clean-20261011-013908-165236`: 35 tasks x 3 trials = 105 trials. Judges: HeuristicFactJudge / HeuristicFaithfulnessJudge.

Confidence intervals are 95% percentile bootstrap over tasks (trials averaged within each task first).

## Headline metrics

| Metric | Mean | 95% CI |
|---|---|---|
| Fact recall | 99.4% | 98.3% to 100.0% |
| Citation faithfulness | 100.0% | 100.0% to 100.0% |
| Numeric fidelity | 100.0% | 100.0% to 100.0% |
| Trial success rate | 100.0% | 100.0% to 100.0% |
| Trials with phantom citations | 0.0% | 0.0% to 0.0% |
| Duplicate search rate | 3.6% | 0.7% to 7.1% |
| Searches per trial | 3.74 | 3.54 to 3.94 |
| Latency per trial (s) | 0.01 | 0.01 to 0.01 |
| Agent input tokens per trial | 592 | 583 to 602 |
| Agent output tokens per trial | 138 | 128 to 148 |
| Agent cost per trial ($) | $0.0000 | $0.0000 to $0.0000 |

Run-to-run variation: same task, different trial, fact recall varies by 0.0 pts (mean within-task standard deviation).

## Fact recall by category

| Category | Tasks | Fact recall |
|---|---|---|
| comparison | 3 | 100.0% |
| cs-algorithms | 1 | 100.0% |
| cs-systems | 5 | 100.0% |
| history | 5 | 100.0% |
| ml | 5 | 100.0% |
| science | 7 | 100.0% |
| space | 5 | 96.0% |
| technology | 4 | 100.0% |

## Failure taxonomy

| Failure | Trials | Share | Meaning |
|---|---|---|---|
| `search_loop` | 9 | 9% | over 30% of searches near-duplicate earlier ones |

## Lowest-recall tasks

**jwst**: 80.0% recall
- missed F5 in 3/3 trials: JWST is a collaboration between NASA, ESA and the Canadian Space Agency

**alexnet**: 100.0% recall

**alphafold**: 100.0% recall

**antarctic-treaty**: 100.0% recall

**apollo-11**: 100.0% recall

## Search cache

hits: 0, fuzzy_hits: 0, misses: 0, backend_calls: 393
