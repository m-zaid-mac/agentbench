# agentbench

An evaluation harness for LLM research agents. It runs an agent against a fixed task set, scores the final reports and the path the agent took, and tells you whether a change to prompts, models, or graph structure actually made things better.

Built to evaluate my [Autonomous AI Research Agent](https://github.com/m-zaid-mac) (LangGraph + AWS Bedrock + Tavily), but any agent that implements one async method can plug in.

```
make demo-bugs     # offline, no API keys, about 5 seconds
```

## Why this exists

"The report looks good" is not an eval. An agent can produce fluent reports while citing sources that never say what it claims, mangling numbers, looping on near-duplicate searches, or leaking state between concurrent runs. Each of those failures has a dedicated check here, and every check reports a confidence interval, because agents are nondeterministic and a single run proves little.

## What it measures

| Metric | How | Why it matters |
|---|---|---|
| Fact recall | Per-fact LLM judge against hand-written key facts. The judge must quote the report sentence it relied on, and the harness verifies that quote really appears in the report; unverifiable evidence overturns the verdict | Did the report answer the question |
| Citation faithfulness | Each cited claim is checked against the text the agent actually retrieved for that URL (supported / partial / unsupported) | Hallucinated or misattributed citations |
| Numeric fidelity | Deterministic: required numbers must appear verbatim (no partial matches, so `8,848.86` does not match `8,848`) | LLM judges are lenient about numbers; string checks are not |
| Phantom citations | Cited URLs this trial never retrieved | State leaking between runs, fabricated sources |
| Dangling markers | `[n]` markers with no matching source entry | Broken report assembly |
| Trajectory | Searches per trial, near-duplicate search rate, sub-questions, LLM calls, tokens, latency, cost | What the answer cost, and whether the agent wandered |
| Failure taxonomy | Rule-based labels per trial (`missed_facts`, `unfaithful`, `phantom_citation`, `numeric_corruption`, `search_loop`, ...) | Reports say how trials fail, not only that scores dropped |

## Design decisions

**Record/replay search.** All agent search goes through `ctx.search`. In `record` mode responses are cached to disk (atomic writes; concurrent identical queries hit the backend once). In `replay` mode a cache miss raises instead of silently going live, so a run is reproducible and comparisons between agent versions measure the code rather than the web. Optional fuzzy replay (`--fuzzy-replay 0.75`) reuses the closest cached query by token Jaccard similarity, since LLM-generated queries drift slightly between runs.

**Judge calls are cached too.** Re-scoring a stored run or rerunning CI costs nothing.

**Statistics over tasks, not trials.** Trials are averaged within each task first, then a percentile bootstrap runs over tasks. Comparisons are paired by task and a difference is only called when the 95% CI excludes zero. Run-to-run variation (within-task standard deviation) is reported separately.

**Judge validation is part of the workflow.** `label-export` produces a blind CSV of judge decisions with verdicts hidden in a separate key file. After you label them, `label-score` reports Cohen's kappa per judge, so you know how much to trust each score.

**Seeded bugs prove the checks work.** The mock agent can reproduce two bugs I actually hit in my research agent:

- `mock:buggy_digits`: a cleanup regex meant to strip `1. ` list numbering also eats digits before a period, so `8,848.86` becomes `8,86`.
- `mock:shared_state`: sources are collected on the agent instance instead of per run, so concurrent runs overwrite each other.

From the demo (35 tasks x 3 trials each):

| | Clean | Digit bug | Shared-state race |
|---|---|---|---|
| Fact recall | 99.4% | 98.0% (no significant change) | 18.3% |
| Numeric fidelity | 100% | 30.7% | 22.6% |
| Trials with phantom citations | 0% | 0% | 86.7% |

The digit bug is the interesting row: the fuzzy fact-recall check barely moves, and only the deterministic numeric check catches it. That is the argument for layering deterministic checks under LLM judges.

## Quickstart

Python 3.10+. The core package has no dependencies.

```bash
pip install -e ".[dev]"
python -m pytest -q                 # 28 tests
make demo                           # clean mock run -> reports/mock-clean.md
make demo-bugs                      # adds both seeded bugs and paired comparisons
```

Main commands:

```bash
agentbench run --agent mock --trials 3 --report-out reports/run.md
agentbench runs                                     # list stored runs
agentbench compare --a latest:mock-clean --b latest:mock-buggy-digits --out reports/cmp.md
agentbench label-export --run latest --out labels/sample.csv
agentbench label-score --csv labels/sample.csv
agentbench summary --run latest --means-only > ci/baseline.json
agentbench ci-check --run latest --thresholds ci/thresholds.agent.json --baseline ci/baseline.json
```

## Evaluating the real research agent

1. **Fill in the adapter.** `agentbench/adapters/research_agent.py` has three TODOs: import your graph builder, inject `tavily_compatible_search(ctx)` in place of the direct Tavily client, and read the final report from your graph state. Reports need `[n]` markers and a numbered Sources section.
2. **Set pricing.** Put per-million-token prices for your Bedrock model in `config/pricing.json`. Models left `null` report cost as n/a rather than a wrong number.
3. **Record search once.**
   ```bash
   pip install -e ".[live]"
   export TAVILY_API_KEY=...  AWS_REGION=us-east-1
   agentbench run --agent agentbench.adapters.research_agent:ResearchAgentAdapter \
     --agent-kwargs '{"model_id": "<bedrock-model-id>"}' --variant baseline \
     --search tavily --search-mode record --judge bedrock:<judge-model-id> --trials 3
   ```
4. **Replay for every comparison after that.** Use `--search-mode replay --fuzzy-replay 0.75`, change one thing (model, prompt, parallel vs sequential search), and run `compare`.
5. **Validate the judges.** Export about 60 items, label them yourself, run `label-score`, and put the kappa values in this README.

Use a different model for the judge than for the agent where you can, so the judge is not grading its own style.

## Task set

`tasks/research_tasks.jsonl` has 35 multi-part questions across science, space, history, ML, and systems, each with 2 to 5 key facts. Facts were chosen to be stable and well documented. Facts carry a `numbers` list that the numeric check matches verbatim; a test enforces that every required number appears in its fact text. Tags: `smoke` (fast CI subset) and `regression:digits` (tasks that exercise number handling).

## CI

`.github/workflows/eval.yml` runs two jobs:

- **harness**: unit tests and an offline mock run gated by `ci/thresholds.mock.json`. No secrets.
- **agent-smoke**: the real agent on the `smoke` subset with replayed search and a live Bedrock judge, gated by absolute thresholds and maximum drops against `ci/baseline.json`. It skips cleanly until you add an `AWS_ROLE_ARN` secret (OIDC) and a `JUDGE_MODEL` variable, commit your recorded search cache, and install your agent package in the job.

## Layout

```
agentbench/
  runner.py         async trial runner: concurrency, timeouts, per-trial isolation
  context.py        RunContext: the agent's only search path, plus its trace log
  search_cache.py   record/replay search, fuzzy replay, Tavily and offline corpus backends
  llm.py            Bedrock Converse client with retries, plus cached LLM wrapper
  judges.py         claim extraction, numeric fidelity, LLM and heuristic judges
  prompts.py        judge rubrics
  trajectory.py     search/LLM/cost metrics from trace events
  failures.py       failure taxonomy
  stats.py          bootstrap CIs, paired differences, Cohen's kappa
  store.py          SQLite results store
  report.py         Markdown run reports and comparisons
  labeling.py       blind export and agreement scoring
  adapters/         mock agent (with seeded bugs), research agent adapter, citation parsing
tasks/              task set
ci/                 thresholds
config/             pricing
```

## Limitations

- The offline demo uses heuristic token-overlap judges. They are good enough to test the pipeline, not to produce real scores; real numbers need the Bedrock judges.
- Faithfulness is judged against the search snippets the agent retrieved, not the full pages. A claim supported by a page but not its snippet will be marked unsupported.
- 35 tasks gives wide confidence intervals for small effects. Grow the set before claiming small improvements.
- Key facts were written by hand, so recall measures coverage of what I chose to ask about, not overall report quality.
