# Comparison: A = `mock-clean` vs B = `mock-shared-state`

A: `mock-clean-20261011-013908-165236`  
B: `mock-shared-state-20261011-013909-879e44`

Differences are B minus A, paired by task, with 95% bootstrap CIs. A difference is called only when the CI excludes zero.

| Metric | A | B | B - A | 95% CI | Verdict |
|---|---|---|---|---|---|
| Fact recall | 99.4% | 18.3% | -81.2 pts | -86.7 pts to -75.4 pts | B worse |
| Citation faithfulness | 100.0% | 15.8% | -84.2 pts | -89.6 pts to -78.6 pts | B worse |
| Numeric fidelity | 100.0% | 22.6% | -77.4 pts | -84.0 pts to -70.5 pts | B worse |
| Trial success rate | 100.0% | 100.0% | +0.0 pts | +0.0 pts to +0.0 pts | no significant difference |
| Trials with phantom citations | 0.0% | 86.7% | +86.7 pts | +81.0 pts to +92.4 pts | B worse |
| Duplicate search rate | 3.6% | 3.6% | +0.0 pts | +0.0 pts to +0.0 pts | no significant difference |
| Searches per trial | 3.74 | 3.74 | +0.00 | +0.00 to +0.00 | no significant difference |
| Latency per trial (s) | 0.01 | 0.02 | +0.01 | +0.01 to +0.01 | B worse |
| Agent input tokens per trial | 592 | 606 | +14 | +5 to +24 | B worse |
| Agent output tokens per trial | 138 | 153 | +16 | +6 to +26 | B worse |
| Agent cost per trial ($) | $0.0000 | $0.0000 | +$0.0000 | +$0.0000 to +$0.0000 | no significant difference |
