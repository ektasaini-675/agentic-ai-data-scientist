"""LangGraph StateGraph for the Autonomous Data Scientist multi-agent system."""
from __future__ import annotations

import sys
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from ads.agents import critic, data, eda, insight, ml, orchestrator, viz
from ads.config import SETTINGS
from ads.report.render import render_report
from ads.state import AgentState


def _wrap_step(agent_fn):
    """Execute agent function and advance the plan pointer."""
    def node_fn(state: AgentState) -> Dict[str, Any]:
        update = agent_fn(state)
        # Advance plan index after worker step
        adv = orchestrator.advance(state)
        update.update(adv)
        return update
    return node_fn


def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Compile final executive report in HTML and Markdown with audit appendix."""
    ctx = state["ctx"]
    claims = state.get("claims") or []
    rejected = state.get("rejected_claims") or []
    feedback = state.get("feedback") or []
    confidence = state.get("confidence")
    goal = state.get("goal") or ctx.goal

    report_paths = render_report(
        claims=claims,
        rejected_claims=rejected,
        feedback=feedback,
        confidence=confidence,
        ctx=ctx,
        goal=goal,
    )
    ctx.save_ledger()

    new_trace = list(state.get("trace") or [])
    new_trace.append(f"[finalize] Executive reports saved to: {report_paths['html']}")

    return {
        "report_paths": report_paths,
        "trace": new_trace,
    }


def critic_route(state: AgentState) -> str:
    """Evaluate critic findings: trigger targeted replan loop or proceed to finalization."""
    status = state.get("validation_status")
    replans = state.get("replans", 0)
    retry_counts = dict(state.get("retry_counts") or {})
    feedbacks = state.get("feedback") or []

    if status == "needs_revision" and replans < SETTINGS.max_replans:
        blockers = [fb for fb in feedbacks if fb.severity == "blocker"]
        if blockers:
            target = blockers[0].target_agent
            current_retries = retry_counts.get(target, 0)
            if current_retries < SETTINGS.max_retries_per_agent:
                retry_counts[target] = current_retries + 1
                state["replans"] = replans + 1
                state["retry_counts"] = retry_counts

                # Reset plan index to point to target agent
                plan = state.get("plan") or []
                if target in plan:
                    state["plan_index"] = plan.index(target)

                new_trace = list(state.get("trace") or [])
                new_trace.append(
                    f"[critic_route] blocker encountered! Routing retry loop to '{target}' "
                    f"(replan {replans + 1}/{SETTINGS.max_replans})"
                )
                state["trace"] = new_trace
                return target

    return "finalize"


def build_graph():
    """Construct and compile the multi-agent execution graph."""
    g = StateGraph(AgentState)

    g.add_node("plan", orchestrator.plan_tasks)
    g.add_node("data_agent", _wrap_step(data.run))
    g.add_node("eda_agent", _wrap_step(eda.run))
    g.add_node("ml_agent", _wrap_step(ml.run))
    g.add_node("viz_agent", _wrap_step(viz.run))
    g.add_node("insight_agent", _wrap_step(insight.run))
    g.add_node("critic_agent", critic.run)  # Does not auto-advance; routes conditionally
    g.add_node("finalize", finalize_node)

    g.set_entry_point("plan")

    router_targets = {
        "data_agent": "data_agent",
        "eda_agent": "eda_agent",
        "ml_agent": "ml_agent",
        "viz_agent": "viz_agent",
        "insight_agent": "insight_agent",
        "critic_agent": "critic_agent",
        "finalize": "finalize",
    }

    g.add_conditional_edges("plan", orchestrator.route_next, router_targets)

    for worker in ["data_agent", "eda_agent", "ml_agent", "viz_agent", "insight_agent"]:
        g.add_conditional_edges(worker, orchestrator.route_next, router_targets)

    g.add_conditional_edges(
        "critic_agent",
        critic_route,
        {
            "data_agent": "data_agent",
            "eda_agent": "eda_agent",
            "ml_agent": "ml_agent",
            "viz_agent": "viz_agent",
            "insight_agent": "insight_agent",
            "finalize": "finalize",
        },
    )

    g.add_edge("finalize", END)

    return g.compile()

