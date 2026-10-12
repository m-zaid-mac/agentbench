# agentbench

An evaluation harness for LLM research agents. It runs an agent against a fixed task set, scores the final reports and the path the agent took to produce them, and tells you whether a change to prompts, models, or graph structure actually made things better, with confidence intervals rather than vibes.

Built to evaluate my [Autonomous AI Research Agent](https://github.com/m-zaid-mac/research-agent) (LangGraph, AWS Bedrock, Tavily). Any agent that implements one async method can plug in.

```bash
make demo-bugs     # offline demo with a mock agent: no API keys, about 5 seconds
```

## Results on the research agent

35 multi-part research tasks, 3 trials each (105 trials per run). Agent and judges: Claude Sonnet 4.5 on Bedrock. Differences are paired by task with 95% bootstrap confidence intervals, and a difference is only claimed when the interval excludes zero.

**Baseline**

| Metric                | Value  |
| --------------------- | ------ |
| Fact recall           | 93.5%  |
| Citation faithfulness | 83.7%  |
| Numeric fidelity      | 95.9%  |
| Searches per report   | 4.0    |
| Latency per report    | 51.8 s |
| Cost per report       | $0.062 |

Run-to-run variation: fact recall on the same task moves by about 5 points between trials, so a single run of a single task proves little.

**Experiment: numbered sources in the synthesizer context**

The baseline synthesizer receives search results as loose text and invents its own citation numbers. The variant numbers every retrieved source in the context (`[1]`, `[2]`, ...) and instructs the model to cite only those numbers and state only what they support.

| Metric                   | Baseline | Numbered sources | Difference (95% CI)                  |
| ------------------------ | -------- | ---------------- | ------------------------------------ |
| Citation faithfulness    | 83.7%    | 88.4%            | **+4.7 pts** (+2.3 to +6.9)          |
| Fact recall              | 93.5%    | 92.0%            | no significant change (-4.0 to +0.9) |
| Numeric fidelity         | 95.9%    | 96.7%            | no significant change                |
| Latency per report       | 51.8 s   | 44.6 s           | **-14%**                             |
| Output tokens per report | 2,740    | 2,247            | **-18%**                             |
| Cost per report          | $0.062   | $0.055           | **-12%**                             |

Citations became more trustworthy at no measurable cost to coverage, and reports got shorter, cheaper, and faster. Phantom citations went from 2.9% of trials to 0%, but with only three baseline cases that difference is not statistically significant.

## What the eval caught along the way

Three times a number looked wrong, and each time the cause was the measurement rather than the agent. Finding these was the most useful part of the project.

**1. A URL parser bug inflated phantom citations from 1% to 19%.** The first baseline reported that 19% of trials cited URLs the agent never retrieved, which looked like state leaking between concurrent runs. Printing the offending URLs showed they were all truncated at a parenthesis: `wikipedia.org/wiki/Raft_(algorithm` instead of `Raft_(algorithm)`. The citation parser stopped at the first `)`. Re-parsing the stored reports with the fix gave a true phantom rate of 1 in 105 trials. The fix handles balanced parentheses and percent-encoded URLs, with regression tests built from the real failing URLs.

**2. Silently swallowed search failures masked a 16-point recall drop.** A replay-mode rerun showed fact recall falling from 94% to 77% while trial success stayed at 100%. The agent generates sub-questions with an LLM, so they are phrased differently on every run and about half missed the search cache. The agent's search node deliberately turns failures into empty results, so it kept going and wrote reports with no search data. The harness now reports a "trials with search errors" metric, labels those trials in the failure taxonomy, and CI fails on any search error.

**3. An API quota confounded an experiment.** The first numbered-sources run showed a 4-point recall drop, which would have been reported as a faithfulness-for-recall tradeoff. The new search-error metric showed 15% of trials had failed searches; the error logs showed the Tavily plan's usage limit had been hit mid-run. With search working, the rerun showed no significant recall change.

## What it measures

| Metric                | How                                                                                                                                                                                                              | Why it matters                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Fact recall           | Per-fact LLM judge against hand-written key facts. The judge must quote the report sentence it relied on, and the harness verifies that quote appears in the report; unverifiable evidence overturns the verdict | Did the report answer the question                          |
| Citation faithfulness | Each cited claim is checked against the text the agent actually retrieved for that URL (supported / partial / unsupported)                                                                                       | Hallucinated or misattributed citations                     |
| Numeric fidelity      | Deterministic: required numbers must appear verbatim (`8,848.86` does not match `8,848`)                                                                                                                         | LLM judges are lenient about numbers; string checks are not |
| Phantom citations     | Cited URLs this trial never retrieved                                                                                                                                                                            | Fabricated sources, state leaking between runs              |
| Search errors         | Trials where any search failed                                                                                                                                                                                   | Agents that degrade silently instead of failing loudly      |
| Dangling markers      | `[n]` markers with no matching source entry                                                                                                                                                                      | Broken report assembly                                      |
| Trajectory            | Searches, near-duplicate search rate, sub-questions, LLM calls, tokens, latency, cost                                                                                                                            | What the answer cost, and whether the agent wandered        |
| Failure taxonomy      | Rule-based labels per trial (`missed_facts`, `unfaithful`, `phantom_citation`, `numeric_corruption`, `search_error`, `search_loop`, ...)                                                                         | How trials fail, not only that scores dropped               |

## Design decisions

**Record/replay search.** All agent search goes through `ctx.search`. In `record` mode responses are cached to disk (atomic writes; concurrent identical queries hit the backend once) and cached queries are reused. In `replay` mode a cache miss raises instead of silently going live. Optional fuzzy replay reuses the closest cached query by token similarity. For an agent whose queries are LLM-generated, record mode is the practical choice; see Limitations.

**Judge calls are cached**, so re-scoring a stored run or rerunning CI costs nothing.

**Statistics over tasks, not trials.** Trials are averaged within each task first, then a percentile bootstrap runs over tasks. Comparisons are paired by task, and run-to-run variation (within-task standard deviation) is reported separately.

**Deterministic checks under LLM judges.** The seeded-bug demo below shows why: an LLM-style fact check barely notices corrupted numbers, while the verbatim numeric check catches them immediately.

**Judge validation is part of the workflow.** `label-export` produces a blind CSV of judge decisions with verdicts hidden in a separate key file. After a human labels them, `label-score` reports Cohen's kappa per judge.

## Seeded-bug demo (offline)

The mock agent can reproduce two bugs I hit while building the research agent, which proves each check catches what it claims to:

- `mock:buggy_digits`: a cleanup regex meant to strip `1. ` list numbering also eats digits, so `8,848.86` becomes `8,86`.
- `mock:shared_state`: sources are collected on the agent instance instead of per run, so concurrent runs overwrite each other.

|                               | Clean | Digit bug                     | Shared-state race |
| ----------------------------- | ----- | ----------------------------- | ----------------- |
| Fact recall                   | 99.4% | 98.0% (no significant change) | 18.3%             |
| Numeric fidelity              | 100%  | 30.7%                         | 22.6%             |
| Trials with phantom citations | 0%    | 0%                            | 86.7%             |

These numbers use offline heuristic judges and a mock agent; they validate the harness, not any real agent. Example reports are in `examples/`.

## Quickstart

Python 3.10+. The core package has no dependencies.

```bash
pip install -e ".[dev]"
python -m pytest -q        # 31 tests
make demo-bugs             # mock runs plus paired comparisons -> reports/
```

Main commands:

```bash
agentbench run --agent mock --trials 3 --report-out reports/run.md
agentbench runs
agentbench compare --a latest:baseline --b latest:variant --out reports/compare.md
agentbench report --run latest:baseline --out reports/baseline.md
agentbench label-export --run latest --out labels/sample.csv
agentbench label-score --csv labels/sample.csv
agentbench ci-check --run latest --thresholds ci/thresholds.agent.json --baseline ci/baseline.json
```

## Evaluating a LangGraph agent

`agentbench/adapters/research_agent.py` connects the research agent. To adapt it to another graph:

1. **Route search through the harness.** Have your search node read an optional search function from `config["configurable"]["search"]` and fall back to the real client when it is absent, so the agent behaves normally outside the harness.
2. **Emit citations.** Reports need `[n]` markers and a numbered `## Sources` section of `n. Title - URL` lines.
3. **Set pricing.** Add per-million-token prices for your model to `config/pricing.json`, keyed by the exact model ID.
4. **Run:**
   ```bash
   pip install -e ".[live]"
   export TAVILY_API_KEY=...  AWS_REGION=us-east-1
   export PYTHONPATH=/path/to/research-agent/backend
   agentbench run --agent agentbench.adapters.research_agent:ResearchAgentAdapter \
     --variant baseline --search tavily --search-mode record \
     --judge "bedrock:<judge-model-id>" --trials 3 --concurrency 4 \
     --report-out reports/baseline.md
   ```
5. **Change one thing, run a variant, compare.** Before trusting any comparison, check that both runs show 0% search errors and the expected searches per trial.

Budget note: a full 105-trial run uses roughly 2 to 3 million Bedrock tokens across agent and judges, plus a few hundred Tavily searches in record mode. Daily Bedrock token quotas and Tavily plan limits are both easy to hit; the search-error metric will tell you when you have.

## CI

`.github/workflows/eval.yml` runs two jobs:

- **harness**: unit tests and an offline mock run gated by `ci/thresholds.mock.json`. No secrets needed.
- **agent-smoke**: the real agent on the `smoke` task subset with cached search and a Bedrock judge, gated by thresholds and maximum drops against `ci/baseline.json`. It skips until an `AWS_ROLE_ARN` secret and `JUDGE_MODEL` variable are configured.

## Limitations

- **Judge validation is in progress.** Agent and judges are currently the same model family (Claude Sonnet 4.5). Human-labeled agreement (Cohen's kappa) has not been measured yet, so treat absolute faithfulness numbers as provisional. Relative comparisons between variants are more robust, since both sides are scored by the same judge.
- **Search is only partly frozen.** Sub-questions are LLM-generated, so most queries differ between runs and variants see overlapping but not identical search results. Caching the decompose step's output per task would make comparisons fully reproducible.
- **Faithfulness is judged against search snippets,** not full pages. A claim supported by a page but not by its snippet is marked unsupported, which biases faithfulness downward.
- **35 tasks gives wide intervals for small effects.** Effects under about 3 points need a larger task set.
- **Key facts were written by hand,** so recall measures coverage of what I chose to ask about, not overall report quality.

## Layout

```
agentbench/
  runner.py         async trial runner: concurrency, timeouts, per-trial isolation
  context.py        RunContext: the agent's only search path, plus its trace log
  search_cache.py   record/replay search, fuzzy replay, Tavily and offline corpus backends
  llm.py            Bedrock Converse client with retries, plus cached LLM wrapper
  judges.py         claim extraction, numeric fidelity, LLM and heuristic judges
  prompts.py        judge rubrics
  trajectory.py     search, LLM, and cost metrics from trace events
  failures.py       failure taxonomy
  stats.py          bootstrap CIs, paired differences, Cohen's kappa
  store.py          SQLite results store
  report.py         Markdown run reports and comparisons
  labeling.py       blind export and agreement scoring
  adapters/         research agent adapter, mock agent with seeded bugs, citation parsing
tasks/              task set (35 tasks; `smoke` and `regression:digits` tags)
ci/                 thresholds
config/             pricing
examples/           example reports from the offline demo
```

## License

MIT
