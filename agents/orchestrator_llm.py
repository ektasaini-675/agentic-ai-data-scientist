"""
LLM-based Multi-Provider Task Planning for Orchestrator.
Supports:
1. Google Gemini API (via google-genai / google.generativeai)
2. Anthropic Claude (via anthropic)
3. OpenAI (via openai)
4. Automatic graceful fallback to Heuristic Planner if keys or network are unavailable.
"""
import os
import json
import pandas as pd
from typing import List
from agents.state import AgentState
from agents.orchestrator import plan_tasks as plan_tasks_rule_based

VALID_AGENTS = ["data_agent", "eda_agent", "ml_agent", "viz_agent", "critic_agent", "insight_agent"]

SYSTEM_PROMPT = """You are the Lead Orchestrator of an Autonomous Multi-Agent AI Data Scientist System.
Given a user's analytical goal and a dataset's columns and shape, determine the optimal execution sequence
of specialized agents.

Available agents:
- data_agent: Data hygiene, schema detection, type inference, missing value imputation (MUST BE FIRST)
- eda_agent: Statistical distributions, Pearson/Spearman correlations, Welch's t-test, Chi-square, VIF
- ml_agent: Multi-model training (Random Forest, Gradient Boosting, Logistic Regression), 5-fold CV, XAI
- viz_agent: Publication charts (heatmaps, feature importance, distributions, confusion matrix)
- critic_agent: Empirical 4-pillar validation (overfitting gap, sample size adequacy, confidence score)
- insight_agent: Executive report synthesis & PDF export (MUST BE LAST)

Return ONLY a valid JSON array of strings containing the ordered agent names.
Example:
["data_agent", "eda_agent", "ml_agent", "viz_agent", "critic_agent", "insight_agent"]
"""


def plan_tasks_llm(state: AgentState) -> AgentState:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    try:
        sample_df = pd.read_csv(state["dataset_path"], nrows=5)
        user_msg = f"User goal: {state['user_goal']}\nDataset shape: {sample_df.shape}\nColumns: {list(sample_df.columns)}"

        raw_json_str = None
        provider = "none"

        # 1. Try Gemini if configured
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(f"{SYSTEM_PROMPT}\n\n{user_msg}")
                raw_json_str = resp.text
                provider = "Gemini"
            except Exception:
                pass

        # 2. Try Anthropic if configured
        if not raw_json_str and anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                resp = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw_json_str = resp.content[0].text
                provider = "Anthropic Claude"
            except Exception:
                pass

        # 3. Try OpenAI if configured
        if not raw_json_str and openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=200
                )
                raw_json_str = resp.choices[0].message.content
                provider = "OpenAI"
            except Exception:
                pass

        if raw_json_str:
            clean_str = raw_json_str.replace("```json", "").replace("```", "").strip()
            plan = json.loads(clean_str)
            filtered_plan = [p for p in plan if p in VALID_AGENTS]
            if filtered_plan:
                state["task_plan"] = filtered_plan
                state["current_task_index"] = 0
                state["plan_rationale"] = f"Orchestrated via {provider} LLM based on user goal and schema attributes."
                state.setdefault("agent_trace", []).append(
                    f"[orchestrator-llm] Decomposed plan via {provider}: {filtered_plan}"
                )
                return state

    except Exception as e:
        state.setdefault("agent_trace", []).append(
            f"[orchestrator-llm] LLM planning exception ({e}), engaging deterministic heuristic planner."
        )

    return plan_tasks_rule_based(state)
