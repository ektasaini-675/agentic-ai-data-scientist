"""
LLM-based task planning for the Orchestrator.

This replaces the keyword-matching in orchestrator.plan_tasks() with an
actual Claude call that reasons about the user's goal and the dataset's
schema. Falls back to the rule-based planner automatically if no API key
is set, so the graph never breaks during development.

Usage: import plan_tasks_llm and use it in place of orchestrator.plan_tasks
in graph.py once you're ready.
"""
import os
import json
import pandas as pd
from agents.state import AgentState
from agents.orchestrator import plan_tasks as plan_tasks_rule_based

VALID_AGENTS = ["data_agent", "eda_agent", "ml_agent", "viz_agent", "critic_agent", "insight_agent"]

SYSTEM_PROMPT = """You are the Orchestrator of a multi-agent data analysis system.
Given a user's analytical goal and a dataset's column names, decide which of the
following agents are needed, IN ORDER. Always include data_agent first, eda_agent
early, critic_agent before insight_agent, and insight_agent last.

Available agents:
- data_agent: cleaning, schema inspection (always needed)
- eda_agent: correlations, distributions, statistical patterns (almost always needed)
- ml_agent: trains a predictive model -- ONLY include if the goal asks to predict,
  classify, forecast, or estimate something
- viz_agent: generates charts
- critic_agent: validates findings before the final report (always needed)
- insight_agent: writes the final report (always needed, always last)

Respond with ONLY a JSON array of agent names, nothing else. Example:
["data_agent", "eda_agent", "ml_agent", "viz_agent", "critic_agent", "insight_agent"]
"""


def plan_tasks_llm(state: AgentState) -> AgentState:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # graceful fallback -- lets the rest of the team keep working without a key
        state.setdefault("agent_trace", []).append(
            "[orchestrator] no ANTHROPIC_API_KEY set, falling back to rule-based planner"
        )
        return plan_tasks_rule_based(state)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        columns = list(pd.read_csv(state["dataset_path"], nrows=5).columns)
        user_msg = f"User goal: {state['user_goal']}\nDataset columns: {columns}"

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        plan = json.loads(raw)

        # validate every step is a real agent name, drop anything hallucinated
        plan = [step for step in plan if step in VALID_AGENTS]
        if not plan:
            raise ValueError("LLM returned no valid agent names")

        state["task_plan"] = plan
        state["current_task_index"] = 0
        state.setdefault("agent_trace", []).append(
            f"[orchestrator-llm] goal='{state['user_goal']}' -> plan={plan}"
        )
        return state

    except Exception as e:
        # any failure (bad JSON, network error, etc.) -- fall back rather than crash the graph
        state.setdefault("agent_trace", []).append(
            f"[orchestrator-llm] LLM planning failed ({e}), falling back to rule-based planner"
        )
        return plan_tasks_rule_based(state)
