"""Data Agent: schema inspection, target identification, cleaning, and leakage mitigation."""
from __future__ import annotations

from typing import Any, Dict

from ads.agents.base import run_specialist
from ads.agents.policy import data_policy, get_target_info
from ads.state import AgentState

DATA_SYSTEM_PROMPT = """You are the Data Engineering Agent in an autonomous data science system.
Your job is to inspect the dataset structure, identify the prediction target based on the user's goal,
clean the data, check for target leakage, and drop suspect columns if needed.
Rules:
1. Always call the tools provided; never guess numbers or invent statistics.
2. If feedback requests dropping a column, call drop_columns.
3. Stop as soon as data preparation is complete.
"""


def run(state: AgentState) -> Dict[str, Any]:
    ctx = state["ctx"]
    goal = state.get("goal") or ctx.goal or "Analyze the dataset"
    digest = ctx.ledger.digest() or "(empty ledger)"

    task = (
        f"Inspect and prepare the dataset for analytical goal: '{goal}'.\n"
        f"Current ledger:\n{digest}\n"
        "Profile the data, infer the target, clean the dataset, and verify leakage."
    )

    tools = ["profile_dataset", "identify_target", "clean_data", "detect_leakage", "drop_columns"]
    update = run_specialist(state, "data_agent", DATA_SYSTEM_PROMPT, task, tools, data_policy)

    target, problem_type, _ = get_target_info(ctx)
    if target:
        update["target"] = target
    if problem_type:
        update["problem_type"] = problem_type

    return update

