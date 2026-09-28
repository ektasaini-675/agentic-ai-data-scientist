# Weekly Progress Log

Keep this updated every week — it becomes ~70% of your final report if you do this consistently.

## Week 0 (Setup)
- Built repo skeleton: agents/, tools/, datasets/, evaluation/, reports/, docs/
- Implemented rule-based Orchestrator (task planner) — will replace with LLM call once API key is wired up
- Implemented real Data Agent (schema detection, missing value / duplicate detection)
- Implemented real EDA Agent (correlation analysis, distribution summary)
- Implemented baseline ML Agent (RandomForestClassifier, crude target-column guessing)
- Implemented Visualization Agent (correlation heatmap + distribution plot)
- Implemented Critic Agent with 2 real checks: sample-size adequacy, EDA/ML feature overlap contradiction check
- Implemented Insight Agent (template-based report assembly, no hallucination risk since it only pulls from state)
- Wired everything with LangGraph `StateGraph`, including a genuine retry loop (critic can send state back to eda_agent)
- Smoke-tested end-to-end on sklearn's breast cancer dataset (placeholder) — pipeline runs, produces a full report + 2 charts

**Decisions made:**
- Chose LangGraph over CrewAI because we need explicit conditional edges for the critic's retry loop
- Orchestrator and Insight Agent are rule-based/template-based for now (no API key needed) — will upgrade to LLM calls next

**Open questions for prof:**
- [fill in based on tomorrow's meeting]

**Blocked on:**
- Need real datasets downloaded (Telco Churn, UCI Heart Disease, a finance/loan dataset) — placeholder dataset in use for now
- Need Anthropic/OpenAI API key provisioned for orchestrator + insight agent upgrade

## Week 1 — Real datasets + Orchestrator LLM upgrade
- Added 3 real datasets: Telco Customer Churn (7,043 rows), UCI Heart Disease (303 rows),
  Loan Approval (614 rows) — covers our 3 target domains (e-commerce, healthcare, finance)
- Ran the full pipeline end-to-end on all 3 and found two real bugs in the ML Agent's
  target-column detection:
  1. Naive substring matching picked "ca" out of the word "clinical" (heart disease goal)
     — fixed with word-boundary regex matching instead of raw substring search
  2. Picked "Loan_ID" over "Loan_Status" because both share the word "loan" and it took
     the first match — fixed by scoring ALL columns by word-overlap count and taking the
     best match, plus explicitly deprioritizing ID-like columns
- Built `agents/orchestrator_llm.py`: real Claude-based task planning that reasons about
  the goal + dataset schema, replacing the keyword-matching rule-based planner. Falls back
  to the rule-based planner automatically if no API key is set or the LLM call fails
  (validated both paths work)
- Added a `USE_LLM_ORCHESTRATOR` env var toggle in `graph.py` so we can A/B the two planners
  — this doubles as the setup for our ablation study later
- Built `evaluation/baseline_comparison.py`: runs the same goal through our multi-agent
  pipeline AND a single-prompt LLM baseline, saves both reports for manual scoring

**Decisions made:**
- Target-column detection stays rule-based (word-overlap + ID exclusion) rather than
  jumping straight to LLM-based detection — it's fast, free, and now correct on all 3
  datasets; may revisit if a 4th dataset breaks it

**Open questions for prof:**
- [fill in based on this week's meeting]

**Blocked on:**
- Need actual ANTHROPIC_API_KEY provisioned to test the LLM orchestrator for real and
  run the baseline comparison (currently only fallback path is tested)

## Week 2 — Advanced EDA, Multi-Model ML Benchmarking & Interactive Dashboard
- Implemented automated statistical hypothesis testing in `agents/eda_agent.py`:
  - Two-Sample Welch's t-test for binary target differences across continuous features.
  - One-Way ANOVA for multi-class target groups.
  - Chi-Square (χ²) Test of Independence with contingency tables for categorical relationships.
- Upgraded `agents/ml_agent.py` to evaluate a multi-model candidate pool:
  - Trains RandomForest, GradientBoosting, and LogisticRegression (with `StandardScaler`).
  - Evaluates both train score and test score to quantify generalization/overfitting gaps.
  - Robust handling for categorical targets and pyarrow chunked arrays.
- Built interactive Streamlit dashboard (`app.py`):
  - Visualizes real-time LangGraph execution trace and state transitions.
  - Interactive EDA correlation heatmap and distributions with Plotly & Matplotlib.
  - Model leaderboard comparison table and feature importance bar charts.
  - File uploader supporting custom CSV files and preloaded benchmark datasets.

**Decisions made:**
- Kept model pool to standard scikit-learn estimators to maintain sub-5s interactive execution on CPU without heavy GPU overhead.
- Implemented fallback handling for all statistical tests to handle zero variance or uniform distributions gracefully.

## Week 3 — Critic Evidence Confidence Scoring, Ablation Evaluation & PDF/HTML Export
- Implemented quantitative validation and confidence scoring in `agents/critic_agent.py`:
  - Real overfitting gap detection (flags when train score exceeds test score by > 15%).
  - Statistical hypothesis verification check (validating that findings meet p < 0.05).
  - Computed Evidence Confidence Score (0–100%) reflecting sample size, consistency, and generalizability.
- Implemented executive report generation in `agents/insight_agent.py`:
  - Actionable business recommendations prioritized by statistical and model impact.
  - Automated PDF export (`reports/executive_report.pdf` via `fpdf2`) and HTML report (`reports/executive_report.html`).
- Built architectural ablation study script (`evaluation/ablation_study.py`):
  - Evaluates Full System (with Critic Agent & retry loop) vs. Ablated System (Critic bypassed).
  - Produces structured JSON output and markdown reports for the final thesis defense.
- Updated project documentation and setup scripts to ensure zero-friction local reproduction.
