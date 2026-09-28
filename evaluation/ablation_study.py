"""
Ablation Study — evaluates the empirical impact of the Critic Agent
in the multi-agent system architecture.

Compares:
  A. Full Multi-Agent System (with Critic Agent validation & retry loop)
  B. Ablated Multi-Agent System (WITHOUT Critic Agent validation)

Usage:
  python evaluation/ablation_study.py datasets/raw/sample.csv "predict whether tumor is malignant"
"""
import os
import sys
import json
import time
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState
from agents import orchestrator, data_agent, eda_agent, ml_agent, viz_agent, critic_agent, insight_agent
from agents.orchestrator_llm import plan_tasks_llm

USE_LLM_ORCHESTRATOR = os.environ.get("USE_LLM_ORCHESTRATOR", "0") == "1"


def build_full_graph():
    from graph import build_graph
    return build_graph()


def build_ablated_graph():
    """Builds pipeline without Critic Agent — workers route straight to insight_agent."""
    g = StateGraph(AgentState)

    planner = plan_tasks_llm if USE_LLM_ORCHESTRATOR else orchestrator.plan_tasks
    g.add_node("plan", planner)
    g.add_node("data_agent", lambda s: orchestrator.advance(data_agent.run(s)))
    g.add_node("eda_agent", lambda s: orchestrator.advance(eda_agent.run(s)))
    g.add_node("ml_agent", lambda s: orchestrator.advance(ml_agent.run(s)))
    g.add_node("viz_agent", lambda s: orchestrator.advance(viz_agent.run(s)))
    # Critic Agent is omitted from ablated graph
    g.add_node("insight_agent", lambda s: orchestrator.advance(insight_agent.run(s)))

    g.set_entry_point("plan")

    def route_ablated(state: AgentState) -> str:
        # If orchestrator suggests critic_agent, skip straight to insight_agent
        next_step = orchestrator.route_next(state)
        if next_step == "critic_agent":
            # Skip critic task in plan
            orchestrator.advance(state)
            return orchestrator.route_next(state)
        return next_step

    g.add_conditional_edges("plan", route_ablated, {
        "data_agent": "data_agent", "eda_agent": "eda_agent", "ml_agent": "ml_agent",
        "viz_agent": "viz_agent", "insight_agent": "insight_agent", "end": END,
    })

    for node in ["data_agent", "eda_agent", "ml_agent", "viz_agent"]:
        g.add_conditional_edges(node, route_ablated, {
            "data_agent": "data_agent", "eda_agent": "eda_agent", "ml_agent": "ml_agent",
            "viz_agent": "viz_agent", "insight_agent": "insight_agent", "end": END,
        })

    g.add_edge("insight_agent", END)
    return g.compile()


def run_experiment(dataset_path: str, goal: str):
    print("=" * 70)
    print("MULTI-AGENT ARCHITECTURE ABLATION STUDY")
    print("Target: Assessing Critic Agent Impact on Report Quality & Safety")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Goal:    {goal}\n")

    os.makedirs("evaluation/results", exist_ok=True)

    # 1. Full Pipeline (With Critic)
    print("[1/2] Running FULL Multi-Agent Pipeline (With Critic Agent)...")
    full_graph = build_full_graph()
    t0 = time.time()
    state_full = {"dataset_path": dataset_path, "user_goal": goal, "agent_trace": []}
    res_full = full_graph.invoke(state_full, config={"recursion_limit": 50})
    time_full = time.time() - t0

    # 2. Ablated Pipeline (Without Critic)
    print("[2/2] Running ABLATED Pipeline (WITHOUT Critic Agent)...")
    ablated_graph = build_ablated_graph()
    t1 = time.time()
    state_ablated = {"dataset_path": dataset_path, "user_goal": goal, "agent_trace": []}
    res_ablated = ablated_graph.invoke(state_ablated, config={"recursion_limit": 50})
    time_ablated = time.time() - t1

    # Analysis & Comparison Metrics
    full_trace_len = len(res_full.get("agent_trace", []))
    ablated_trace_len = len(res_ablated.get("agent_trace", []))
    full_conf = res_full.get("confidence_score", "N/A")
    full_val_status = res_full.get("validation_status", "N/A")

    critic_checked_overfitting = (res_full.get("model_report", {}).get("overfit_gap") is not None and "critic_agent" in [step.split()[0].replace("[", "").replace("]", "") for step in res_full.get("agent_trace", [])])
    critic_checked_sample = any("sample" in str(note).lower() or "adequate" in str(note).lower() for note in res_full.get("validation_notes", []))

    summary = {
        "dataset": dataset_path,
        "goal": goal,
        "full_system": {
            "execution_time_seconds": round(time_full, 2),
            "trace_steps": full_trace_len,
            "validation_status": full_val_status,
            "confidence_score": full_conf,
            "overfitting_verified": critic_checked_overfitting,
            "sample_size_verified": critic_checked_sample,
            "report_length_chars": len(res_full.get("final_report", "")),
        },
        "ablated_system": {
            "execution_time_seconds": round(time_ablated, 2),
            "trace_steps": ablated_trace_len,
            "validation_status": "NONE (Critic Bypassed)",
            "confidence_score": "None (No Critic)",
            "overfitting_verified": False,
            "sample_size_verified": False,
            "report_length_chars": len(res_ablated.get("final_report", "")),
        }
    }

    with open("evaluation/results/ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("evaluation/results/full_system_report.md", "w", encoding="utf-8") as f:
        f.write(res_full.get("final_report", ""))

    with open("evaluation/results/ablated_system_report.md", "w", encoding="utf-8") as f:
        f.write(res_ablated.get("final_report", ""))

    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Full (With Critic)':<20} | {'Ablated (No Critic)':<20}")
    print("-" * 74)
    print(f"{'Execution Time':<30} | {f'{time_full:.2f}s':<20} | {f'{time_ablated:.2f}s':<20}")
    print(f"{'Agent Trace Steps':<30} | {full_trace_len:<20} | {ablated_trace_len:<20}")
    print(f"{'Validation Status':<30} | {str(full_val_status):<20} | {'None (Bypassed)':<20}")
    print(f"{'Confidence Score':<30} | {f'{full_conf}%':<20} | {'Not Scored':<20}")
    print(f"{'Overfitting Gap Check':<30} | {str(critic_checked_overfitting):<20} | {'False':<20}")
    print(f"{'Sample Size Check':<30} | {str(critic_checked_sample):<20} | {'False':<20}")
    print("=" * 74)
    print("\nDetailed reports saved to:")
    print(" - evaluation/results/ablation_results.json")
    print(" - evaluation/results/full_system_report.md")
    print(" - evaluation/results/ablated_system_report.md")


if __name__ == "__main__":
    d_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw/sample.csv"
    u_goal = sys.argv[2] if len(sys.argv) > 2 else "predict whether the tumor is malignant based on measurements"
    run_experiment(d_path, u_goal)
