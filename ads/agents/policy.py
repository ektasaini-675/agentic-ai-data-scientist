"""Deterministic scripted policies for specialist agents.

Ensures that every agent can run and produce an identical ledger structure
even when no LLM API key is present, guaranteeing reliable offline execution.
"""
from __future__ import annotations

from typing import Optional, Tuple

from ads.context import RunContext
from ads.state import AgentState
from ads.tools import execute_tool


def get_target_info(ctx: RunContext) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Retrieve target name, problem type, and n_unique classes from the ledger."""
    for ev in reversed(ctx.ledger.all()):
        if ev.tool == "identify_target" and isinstance(ev.result, dict):
            return (
                ev.result.get("target"),
                ev.result.get("problem_type"),
                ev.result.get("n_unique"),
            )
    return None, None, None


def data_policy(ctx: RunContext, state: AgentState) -> None:
    """Scripted policy for data_agent: profile -> identify target -> clean -> check leakage."""
    # Act on critic feedback to drop columns if any
    for fb in state.get("feedback") or []:
        if fb.target_agent == "data_agent":
            leakage_evs = [e for e in ctx.ledger.all() if e.tool == "detect_leakage"]
            if leakage_evs and isinstance(leakage_evs[-1].result, dict):
                suspects = leakage_evs[-1].result.get("suspects", [])
                cols_to_drop = [s["feature"] for s in suspects if isinstance(s, dict) and "feature" in s]
                if cols_to_drop:
                    execute_tool("drop_columns", {"columns": cols_to_drop}, ctx)

    execute_tool("profile_dataset", {}, ctx)

    goal = state.get("goal") or ctx.goal or "Analyze the dataset"
    execute_tool("identify_target", {"goal": goal}, ctx)

    target, problem_type, _ = get_target_info(ctx)
    execute_tool("clean_data", {"target": target} if target else {}, ctx)

    if target:
        execute_tool("detect_leakage", {"target": target}, ctx)


def eda_policy(ctx: RunContext, state: AgentState) -> None:
    """Scripted policy for eda_agent: test_association -> top_correlations -> segment_rates."""
    target, problem_type, n_unique = get_target_info(ctx)

    assoc_res = None
    if target:
        assoc_res, _ = execute_tool("test_association", {"target": target}, ctx)

    execute_tool("top_correlations", {}, ctx)

    if (
        target
        and assoc_res
        and assoc_res.ok
        and isinstance(assoc_res.data, dict)
        and not (problem_type == "classification" and (n_unique or 0) > 2)
    ):
        results = assoc_res.data.get("results", [])
        for row in results[:3]:
            feature = row.get("feature")
            if feature:
                execute_tool("segment_rates", {"feature": feature, "target": target}, ctx)


def ml_policy(ctx: RunContext, state: AgentState) -> None:
    """Scripted policy for ml_agent: train_cv -> permutation_importance."""
    target, problem_type, _ = get_target_info(ctx)
    if target:
        execute_tool("train_cv", {"target": target}, ctx)
        execute_tool("permutation_importance", {"target": target}, ctx)


def viz_policy(ctx: RunContext, state: AgentState) -> None:
    """Scripted policy for viz_agent: plot_importance -> plot_target_by_feature -> plot_correlation."""
    target, problem_type, n_unique = get_target_info(ctx)
    if target:
        execute_tool("plot_importance", {"target": target}, ctx)

        top_features = []
        for ev in reversed(ctx.ledger.all()):
            if ev.tool == "test_association" and isinstance(ev.result, dict):
                top_features = [r["feature"] for r in ev.result.get("results", [])[:2] if "feature" in r]
                break

        for feat in top_features:
            execute_tool("plot_target_by_feature", {"feature": feat, "target": target}, ctx)

    execute_tool("plot_correlation", {}, ctx)

