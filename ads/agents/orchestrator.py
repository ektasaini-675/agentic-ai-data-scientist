"""Orchestrator Agent: dynamic goal decomposition and execution routing."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from ads.config import SETTINGS
from ads.state import AgentState

VALID_AGENTS = ["data_agent", "eda_agent", "ml_agent", "viz_agent", "insight_agent", "critic_agent"]

DEFAULT_PLAN = ["data_agent", "eda_agent", "ml_agent", "viz_agent", "insight_agent", "critic_agent"]

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Orchestrator of an autonomous multi-agent data scientist system.
Given a user's analytical goal and the dataset column headers, formulate the optimal ordered sequence of agents.
Available agents:
- data_agent: profiling, cleaning, schema analysis, leakage detection (always needed, always first)
- eda_agent: statistical association tests, correlations, segment rates (needed for patterns and drivers)
- ml_agent: predictive modeling, CV evaluation, feature importances (include if prediction, classification, or estimation is relevant)
- viz_agent: plots and visual charts
- insight_agent: compiles verified claims and recommendations
- critic_agent: audits findings, leakage, and overfitting

Return ONLY a valid JSON list of agent names, with NO markdown formatting:
["data_agent", "eda_agent", "ml_agent", "viz_agent", "insight_agent", "critic_agent"]
"""


def plan_tasks(state: AgentState) -> Dict[str, Any]:
    """Determine the task execution plan using LLM reasoning or fallback policy."""
    client = state.get("client")
    ctx = state["ctx"]
    goal = state.get("goal") or ctx.goal
    columns = list(ctx.df.columns[:30])
    new_trace = list(state.get("trace") or [])

    plan = list(DEFAULT_PLAN)

    if client is not None and client.available:
        try:
            user_msg = f"Analytical Goal: '{goal}'\nDataset Columns: {columns}"
            resp = client.complete(
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=200,
            )
            clean = resp.text.strip()
            clean = re.sub(r"^```(?:json)?", "", clean, flags=re.MULTILINE)
            clean = re.sub(r"```$", "", clean, flags=re.MULTILINE).strip()
            candidate_plan = json.loads(clean)
            if isinstance(candidate_plan, list):
                valid = [a for a in candidate_plan if a in VALID_AGENTS]
                if "data_agent" in valid and "insight_agent" in valid:
                    if "critic_agent" not in valid:
                        valid.append("critic_agent")
                    plan = valid
            new_trace.append(f"[orchestrator] dynamic plan formulated via LLM: {plan}")
        except Exception:
            new_trace.append(f"[orchestrator] fallback to default plan: {plan}")
    else:
        new_trace.append(f"[orchestrator] offline mode; default plan: {plan}")

    return {
        "plan": plan,
        "plan_index": 0,
        "replans": 0,
        "retry_counts": {},
        "trace": new_trace,
    }


def route_next(state: AgentState) -> str:
    """Determine next agent in sequence based on current plan index."""
    plan = state.get("plan") or DEFAULT_PLAN
    idx = state.get("plan_index", 0)

    if idx < len(plan):
        return plan[idx]
    return "finalize"


def advance(state: AgentState) -> Dict[str, Any]:
    """Advance the plan pointer by 1."""
    return {"plan_index": state.get("plan_index", 0) + 1}

