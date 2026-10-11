.PHONY: test demo demo-bugs clean

test:
	python -m pytest -q

# Offline end-to-end run: mock agent, fake search corpus, heuristic judges. No API keys needed.
demo:
	python -m agentbench --db results/demo.db run --agent mock --variant mock-clean --trials 3 --report-out reports/mock-clean.md > /dev/null
	@echo "wrote reports/mock-clean.md"

# Seeded-bug demo: shows the harness catching a digit-mangling parser and a shared-state race.
demo-bugs: demo
	python -m agentbench --db results/demo.db run --agent mock:buggy_digits --variant mock-buggy-digits --trials 3 > /dev/null
	python -m agentbench --db results/demo.db run --agent mock:shared_state --variant mock-shared-state --trials 3 --concurrency 8 > /dev/null
	python -m agentbench --db results/demo.db compare --a latest:mock-clean --b latest:mock-buggy-digits --out reports/compare-digits.md
	python -m agentbench --db results/demo.db compare --a latest:mock-clean --b latest:mock-shared-state --out reports/compare-shared-state.md

clean:
	rm -rf results reports labels eval_cache/judge
