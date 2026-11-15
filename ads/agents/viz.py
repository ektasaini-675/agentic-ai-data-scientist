"""Visualization Agent: generates charts and plots for statistical and ML findings."""
from __future__ import annotations

from typing import Any, Dict

from ads.agents.base import run_specialist
from ads.agents.policy import get_target_info, viz_policy
from ads.state import AgentState

VIZ_SYSTEM_PROMPT = """You are the Visualization Agent in an autonomous data science system.
Your job is to generate charts illustrating permutation importances, feature relationships,
and correlation structures.
Rules:
1. Always call the plotting tools provided.
2. Stop when all key visual plots are saved.
"""


def run(state: AgentState) -> Dict[str, Any]:
    ctx = state["ctx"]
    target, problem_type, _ = get_target_info(ctx)
    digest = ctx.ledger.digest() or "(empty ledger)"

    task = (
        f"Generate visual plots for target '{target}'.\n"
        f"Current ledger:\n{digest}\n"
        "Plot model feature importance, segment outcome rates, and correlation heatmap."
    )

    tools = ["plot_target_by_feature", "plot_importance", "plot_correlation"]
    return run_specialist(state, "viz_agent", VIZ_SYSTEM_PROMPT, task, tools, viz_policy)

