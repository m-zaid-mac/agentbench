# Comparison: A = `mock-clean` vs B = `mock-buggy-digits`

A: `mock-clean-20261011-013908-165236`  
B: `mock-buggy-digits-20261011-013908-02c532`

Differences are B minus A, paired by task, with 95% bootstrap CIs. A difference is called only when the CI excludes zero.

| Metric | A | B | B - A | 95% CI | Verdict |
|---|---|---|---|---|---|
| Fact recall | 99.4% | 98.0% | -1.4 pts | -3.6 pts to +0.0 pts | no significant difference |
| Citation faithfulness | 100.0% | 99.5% | -0.5 pts | -1.4 pts to +0.0 pts | no significant difference |
| Numeric fidelity | 100.0% | 30.7% | -69.3 pts | -80.0 pts to -58.8 pts | B worse |
| Trial success rate | 100.0% | 100.0% | +0.0 pts | +0.0 pts to +0.0 pts | no significant difference |
| Trials with phantom citations | 0.0% | 0.0% | +0.0 pts | +0.0 pts to +0.0 pts | no significant difference |
| Duplicate search rate | 3.6% | 3.6% | +0.0 pts | +0.0 pts to +0.0 pts | no significant difference |
| Searches per trial | 3.74 | 3.74 | +0.00 | +0.00 to +0.00 | no significant difference |
| Latency per trial (s) | 0.01 | 0.01 | +0.00 | +0.00 to +0.00 | B worse |
| Agent input tokens per trial | 592 | 592 | +0 | +0 to +0 | no significant difference |
| Agent output tokens per trial | 138 | 138 | +0 | +0 to +0 | no significant difference |
| Agent cost per trial ($) | $0.0000 | $0.0000 | +$0.0000 | +$0.0000 to +$0.0000 | no significant difference |
