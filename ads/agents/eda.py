"""EDA Agent: statistical association tests, correlations, and segment rates."""
from __future__ import annotations

from typing import Any, Dict

from ads.agents.base import run_specialist
from ads.agents.policy import eda_policy, get_target_info
from ads.state import AgentState

EDA_SYSTEM_PROMPT = """You are the Exploratory Data Analysis (EDA) & Statistics Agent.
Your job is to test feature associations with the target, find strong rank correlations,
and compute segment outcome rates.
Rules:
1. Always call the tools provided; never guess numbers or p-values.
2. Examine the strongest features and verify relationships.
3. Stop when statistical relationships are thoroughly captured in the ledger.
"""


def run(state: AgentState) -> Dict[str, Any]:
    ctx = state["ctx"]
    goal = state.get("goal") or ctx.goal or "Analyze patterns"
    target, problem_type, _ = get_target_info(ctx)
    digest = ctx.ledger.digest() or "(empty ledger)"

    task = (
        f"Perform exploratory and statistical analysis for target '{target}' with goal: '{goal}'.\n"
        f"Current ledger:\n{digest}\n"
        "Run statistical tests against the target, find top correlations, and inspect segment rates."
    )

    tools = ["test_association", "top_correlations", "segment_rates"]
    return run_specialist(state, "eda_agent", EDA_SYSTEM_PROMPT, task, tools, eda_policy)

