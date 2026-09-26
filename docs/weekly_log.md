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

## Week 1
-

## Week 2
-
