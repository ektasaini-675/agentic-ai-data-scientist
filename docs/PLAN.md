# Solo 3-day plan

Rule for every day: **end the day with something that runs.** `main` is never broken.
Architecture rule: **a number only reaches a report if a tool computed it and the ledger recorded it.**

## Day 1 - foundation and tools
- [x] Contracts: `ads/state.py`
- [x] Evidence ledger: `ads/ledger.py`
- [x] Run context + per-run folders: `ads/context.py`
- [x] Claim verifier: `ads/verify.py`
- [x] Tool registry + runner: `ads/tools/registry.py`
- [x] Data tools: profile, clean, identify_target (`ads/tools/data.py`)
- [x] Tests for all of the above
- [x] Stats tools: `test_association`, `segment_rates`, `top_correlations` (`ads/tools/stats.py`)
- [x] ML tools: `train_cv`, `permutation_importance`, `detect_leakage` (`ads/tools/ml.py`)
- [x] Viz tools: `plot_target_by_feature`, `plot_importance`, `plot_correlation` (`ads/tools/viz.py`)
- [x] LLM client with live / offline / fake modes + response cache (`ads/llm/client.py`)
- [x] Smoke script: every tool on every real dataset (`scripts/smoke_tools.py`)

## Day 2 - agents, graph, critic, evaluation
- [x] Generic agent loop (one function reused by all agents: LLM picks tools, step limit) (`ads/llm/loop.py`)
- [x] Offline "scripted policy" (same tools, fixed order) so everything works without an API key (`ads/agents/policy.py`)
- [x] Agents: data, eda, ml, viz (thin wrappers: prompt + tool allow-list) (`ads/agents/{data,eda,ml,viz}.py`)
- [x] Insight agent: outputs `Claim` objects, verifier gate, rewrite loop (`ads/agents/insight.py`)
- [x] Critic: leakage / sample size / CV variance / imbalance / multiplicity -> `Feedback` (`ads/agents/critic.py`)
- [x] Graph: orchestrator, targeted retry routing, replan budget (`ads/agents/orchestrator.py`, `ads/graph.py`)
- [x] Benchmark: 10 cases (4 datasets x seeded flaws) and systems B1 vs B2 vs B3 (`evaluation/benchmark.py`)

Checkpoint achieved: one command produces a report where every number links to evidence.

## Day 3 - app, polish, docs
- [x] Streamlit: evidence view, live trace, verified-claims panel (`app.py`)
- [x] Executive report renderer: HTML & Markdown with Audit Appendix (`ads/report/render.py`)
- [x] Pipeline entry point and CLI: `ads/pipeline.py` & `ads/cli.py`
- [x] Documentation & Architecture specifications: `docs/architecture.md`, `docs/evaluation.md`, `README.md`
- [x] Edge cases handled: offline fallback, missing targets, target leakage dropping, verifier rewrite loop
- [x] Integration test suite: `tests/test_pipeline.py` (all tests passing)

## Run the tests
```bash
pip install -r requirements-dev.txt
pytest
python scripts/smoke_tools.py
python scripts/check_llm.py
python -m ads.cli datasets/raw/sample.csv "predict diagnosis"
```
