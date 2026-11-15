"""Machine Learning Agent: cross-validation, baseline comparison, and feature importance."""
from __future__ import annotations

from typing import Any, Dict

from ads.agents.base import run_specialist
from ads.agents.policy import get_target_info, ml_policy
from ads.state import AgentState

ML_SYSTEM_PROMPT = """You are the Machine Learning Agent in an autonomous data science system.
Your job is to train predictive models with stratified cross-validation on the train split,
evaluate once on the held-out test set, and compute permutation feature importances.
Rules:
1. Always call the tools provided; never invent scores or model metrics.
2. Stop when model evaluation and feature ranking are complete.
"""


def run(state: AgentState) -> Dict[str, Any]:
    ctx = state["ctx"]
    target, problem_type, _ = get_target_info(ctx)
    digest = ctx.ledger.digest() or "(empty ledger)"

    task = (
        f"Train and evaluate predictive models for target '{target}' ({problem_type}).\n"
        f"Current ledger:\n{digest}\n"
        "Train models with cross-validation and compute permutation feature importances."
    )

    tools = ["train_cv", "permutation_importance", "detect_leakage"]
    return run_specialist(state, "ml_agent", ML_SYSTEM_PROMPT, task, tools, ml_policy)

