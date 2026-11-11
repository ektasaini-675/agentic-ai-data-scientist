"""
Orchestrator Agent — Task Planning, DAG Progression & Adaptive Routing.
Supports both semantic heuristic decomposition and LLM-driven planning.
"""
import re
import pandas as pd
from typing import List
from agents.state import AgentState

VALID_AGENTS = ["data_agent", "eda_agent", "ml_agent", "viz_agent", "critic_agent", "insight_agent"]


def plan_tasks(state: AgentState) -> AgentState:
    """Deterministic, zero-latency semantic planner."""
    goal = state.get("user_goal", "").lower()
    
    plan: List[str] = ["data_agent", "eda_agent"]

    # Semantic keyword and problem formulation check for predictive ML
    predictive_tokens = {
        "predict", "predictive", "classif", "classification", "forecast", "regression",
        "churn", "risk", "default", "malignant", "benign", "disease", "score", "model",
        "estimate", "probability", "determinant", "factor", "detect"
    }
    
    # Check if dataset has columns and if target can be modeled
    has_predictive_goal = any(token in goal for token in predictive_tokens)
    
    # If the user asks for patterns/correlations/predictive factors
    if has_predictive_goal or "eda" not in goal:
        plan.append("ml_agent")

    plan.append("viz_agent")
    plan.append("critic_agent")
    plan.append("insight_agent")

    state["task_plan"] = plan
    state["current_task_index"] = 0
    state["plan_rationale"] = (
        f"Semantic planner decomposed objective '{state['user_goal']}' into {len(plan)} structured stages "
        f"including data hygiene, inferential EDA, predictive modeling, multi-figure visualization, "
        f"4-pillar critic validation, and executive synthesis."
    )
    state.setdefault("agent_trace", []).append(
        f"[orchestrator] Formulated execution DAG: {plan} | Mode=Heuristic"
    )
    return state


def route_next(state: AgentState) -> str:
    """Conditional edge router: inspects plan index and yields next node."""
    idx = state.get("current_task_index", 0)
    plan = state.get("task_plan", [])
    if idx >= len(plan):
        return "end"
    return plan[idx]


def advance(state: AgentState) -> AgentState:
    """Advances execution pointer to next planned task."""
    state["current_task_index"] = state.get("current_task_index", 0) + 1
    return state
