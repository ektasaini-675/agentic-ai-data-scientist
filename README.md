# Autonomous Multi-Agent AI System for Data Analysis

B.Tech Final Year Project — Ekta (B23CS1018), Kashish Joshi (B23CS1025), Kurra Hema (B23CS1031)

## What this is
A multi-agent system (built on LangGraph) that takes a dataset + a natural-language
analytical goal, autonomously plans which analysis steps are needed, executes them
through specialized agents, validates the results, and produces a data-supported report.

## Status (Week 0)
The orchestration skeleton is fully wired and runs end-to-end on a placeholder dataset.
Orchestrator and Insight Agent are currently rule-based / template-based (no API key
required to test) — Data, EDA, ML, Visualization, and Critic agents are already doing
real computation on real data. See `docs/weekly_log.md` for details.

## Quickstart
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python graph.py datasets/raw/sample.csv "predict whether the tumor is malignant based on the measurements"
```
This runs the full pipeline and prints the agent trace + final report.
Charts land in `reports/charts/`.

## Project structure
```
agents/       - one file per agent (state.py holds the shared state schema all agents read/write)
tools/        - reusable helper functions agents call (to be filled in as agents get more complex)
datasets/     - raw/ and processed/ data; keep raw data untouched, write cleaned copies to processed/
evaluation/   - scripts comparing system output vs. baseline (single-prompt LLM) and ablations
notebooks/    - exploratory work only, not production code
reports/      - generated final reports + charts (system output, not manually written)
docs/         - weekly_log.md, architecture notes, meeting notes
graph.py      - wires all agents into the LangGraph StateGraph; run this directly to test
```

## Next steps (see docs/weekly_log.md for the live version)
1. Swap Orchestrator's rule-based planner for an LLM call (needs `ANTHROPIC_API_KEY` in `.env`)
2. Swap Insight Agent's template for an LLM call that only references stored state values (no invented numbers)
3. Replace placeholder dataset with real Telco Churn / UCI Heart Disease / finance datasets
4. Build Streamlit UI (`app.py`) showing live agent trace during a run
5. Build `evaluation/baseline_comparison.py`: run the same goal through a single LLM prompt and score both outputs on a rubric
6. Ablation: run the pipeline with the critic agent disabled and compare report quality
