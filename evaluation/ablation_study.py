"""
Academic Architecture Ablation Study.
Empirically evaluates the value of the Critic Agent and Closed-Loop Reflexion.

Compares:
  A. Full Adaptive Multi-Agent System (Critic + Dynamic Remediation Feedback Loop)
  B. Passive Multi-Agent System (Critic audits without triggering adaptive remediation)
  C. Ablated Multi-Agent System (WITHOUT Critic Agent validation layer)

Usage:
    cmd.exe /c run.bat evaluation/ablation_study.py datasets/raw/sample.csv "predict whether tumor is malignant"
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


def build_full_system():
    """Builds full system with active reflex loop."""
    from graph import build_graph
    return build_graph()


def build_passive_critic_system():
    """Builds pipeline where Critic records notes and confidence score, but NEVER triggers retry."""
    g = StateGraph(AgentState)
    g.add_node("plan", orchestrator.plan_tasks)
    g.add_node("data_agent", lambda s: orchestrator.advance(data_agent.run(s)))
    g.add_node("eda_agent", lambda s: orchestrator.advance(eda_agent.run(s)))
    g.add_node("ml_agent", lambda s: orchestrator.advance(ml_agent.run(s)))
    g.add_node("viz_agent", lambda s: orchestrator.advance(viz_agent.run(s)))

    def passive_critic_node(state: AgentState) -> AgentState:
        # Run critic but force status to pass immediately
        s = critic_agent.run(state)
        s["validation_status"] = "pass"
        s["critic_directives"] = []
        return orchestrator.advance(s)

    g.add_node("critic_agent", passive_critic_node)
    g.add_node("insight_agent", lambda s: orchestrator.advance(insight_agent.run(s)))

    g.set_entry_point("plan")

    for node in ["plan", "data_agent", "eda_agent", "ml_agent", "viz_agent", "critic_agent"]:
        g.add_conditional_edges(node, orchestrator.route_next, {
            "data_agent": "data_agent", "eda_agent": "eda_agent", "ml_agent": "ml_agent",
            "viz_agent": "viz_agent", "critic_agent": "critic_agent",
            "insight_agent": "insight_agent", "end": END,
        })

    g.add_edge("insight_agent", END)
    return g.compile()


def build_ablated_system():
    """Builds pipeline WITHOUT Critic Agent node entirely."""
    g = StateGraph(AgentState)
    g.add_node("plan", orchestrator.plan_tasks)
    g.add_node("data_agent", lambda s: orchestrator.advance(data_agent.run(s)))
    g.add_node("eda_agent", lambda s: orchestrator.advance(eda_agent.run(s)))
    g.add_node("ml_agent", lambda s: orchestrator.advance(ml_agent.run(s)))
    g.add_node("viz_agent", lambda s: orchestrator.advance(viz_agent.run(s)))
    g.add_node("insight_agent", lambda s: orchestrator.advance(insight_agent.run(s)))

    g.set_entry_point("plan")

    def route_skip_critic(state: AgentState) -> str:
        nxt = orchestrator.route_next(state)
        if nxt == "critic_agent":
            orchestrator.advance(state)
            return orchestrator.route_next(state)
        return nxt

    for node in ["plan", "data_agent", "eda_agent", "ml_agent", "viz_agent"]:
        g.add_conditional_edges(node, route_skip_critic, {
            "data_agent": "data_agent", "eda_agent": "eda_agent", "ml_agent": "ml_agent",
            "viz_agent": "viz_agent", "insight_agent": "insight_agent", "end": END,
        })

    g.add_edge("insight_agent", END)
    return g.compile()


def run_ablation(dataset_path: str, goal: str):
    print("=" * 78)
    print("IIT BTP EMPIRICAL ARCHITECTURAL ABLATION STUDY")
    print("Assessing the Empirical Efficacy of Critic Agent & Closed-Loop Reflexion")
    print("=" * 78)
    print(f"Benchmark Dataset: {dataset_path}")
    print(f"Goal Objective:    {goal}\n")

    os.makedirs("evaluation/results", exist_ok=True)

    # 1. Full Adaptive System
    print("[1/3] Executing Architecture A: Full Adaptive System (Critic + Dynamic Remediation)...")
    app_a = build_full_system()
    t0 = time.time()
    res_a = app_a.invoke({"dataset_path": dataset_path, "user_goal": goal, "agent_trace": []})
    time_a = time.time() - t0

    # 2. Passive Critic System
    print("[2/3] Executing Architecture B: Passive Critic System (Audit Only, No Retry Loop)...")
    app_b = build_passive_critic_system()
    t1 = time.time()
    res_b = app_b.invoke({"dataset_path": dataset_path, "user_goal": goal, "agent_trace": []})
    time_b = time.time() - t1

    # 3. Ablated System
    print("[3/3] Executing Architecture C: Ablated Baseline (No Critic Agent Node)...")
    app_c = build_ablated_system()
    t2 = time.time()
    res_c = app_c.invoke({"dataset_path": dataset_path, "user_goal": goal, "agent_trace": []})
    time_c = time.time() - t2

    summary = {
        "dataset": dataset_path,
        "goal": goal,
        "full_adaptive_system": {
            "execution_time_seconds": round(time_a, 2),
            "trace_step_count": len(res_a.get("agent_trace", [])),
            "retries_triggered": res_a.get("retry_count", 0),
            "evidence_confidence_score": res_a.get("confidence_score", "N/A"),
            "validation_status": res_a.get("validation_status", "N/A"),
            "model_test_accuracy": res_a.get("model_report", {}).get("accuracy", "N/A"),
            "overfit_gap": res_a.get("model_report", {}).get("overfit_gap", "N/A"),
            "overfitting_audit": True,
            "multicollinearity_pruned": any("Pruned high-multicollinearity" in step for step in res_a.get("agent_trace", []))
        },
        "passive_critic_system": {
            "execution_time_seconds": round(time_b, 2),
            "trace_step_count": len(res_b.get("agent_trace", [])),
            "retries_triggered": 0,
            "evidence_confidence_score": res_b.get("confidence_score", "N/A"),
            "validation_status": "pass (passive)",
            "model_test_accuracy": res_b.get("model_report", {}).get("accuracy", "N/A"),
            "overfit_gap": res_b.get("model_report", {}).get("overfit_gap", "N/A"),
            "overfitting_audit": True,
            "multicollinearity_pruned": False
        },
        "ablated_system": {
            "execution_time_seconds": round(time_c, 2),
            "trace_step_count": len(res_c.get("agent_trace", [])),
            "retries_triggered": 0,
            "evidence_confidence_score": "Unscored (No Critic)",
            "validation_status": "Bypassed",
            "model_test_accuracy": res_c.get("model_report", {}).get("accuracy", "N/A"),
            "overfit_gap": res_c.get("model_report", {}).get("overfit_gap", "N/A"),
            "overfitting_audit": False,
            "multicollinearity_pruned": False
        }
    }

    # Save JSON results
    with open("evaluation/results/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Reports
    with open("evaluation/results/full_system_report.md", "w", encoding="utf-8") as f:
        f.write(res_a.get("final_report", ""))
    with open("evaluation/results/ablated_system_report.md", "w", encoding="utf-8") as f:
        f.write(res_c.get("final_report", ""))

    print("\n" + "=" * 78)
    print("EMPIRICAL ARCHITECTURAL ABLATION RESULTS")
    print("=" * 78)
    col_w = 24
    print(f"{'Evaluation Metric':<30} | {'(A) Full Adaptive':<{col_w}} | {'(B) Passive Audit':<{col_w}} | {'(C) Ablated':<{col_w}}")
    print("-" * 105)
    print(f"{'Execution Latency':<30} | {f'{time_a:.2f}s':<{col_w}} | {f'{time_b:.2f}s':<{col_w}} | {f'{time_c:.2f}s':<{col_w}}")
    print(f"{'Agent Trace Steps':<30} | {len(res_a.get('agent_trace', [])):<{col_w}} | {len(res_b.get('agent_trace', [])):<{col_w}} | {len(res_c.get('agent_trace', [])):<{col_w}}")
    print(f"{'Reflexion Retries':<30} | {res_a.get('retry_count', 0):<{col_w}} | {0:<{col_w}} | {0:<{col_w}}")
    print(f"{'Validation Certification':<30} | {str(res_a.get('validation_status')):<{col_w}} | {'Passive Pass':<{col_w}} | {'Bypassed':<{col_w}}")
    conf_a = f"{res_a.get('confidence_score')}%"
    conf_b = f"{res_b.get('confidence_score')}%"
    print(f"{'Evidence Confidence Score':<30} | {conf_a:<{col_w}} | {conf_b:<{col_w}} | {'Unscored':<{col_w}}")
    print(f"{'Overfitting Gap Audited':<30} | {'Guaranteed':<{col_w}} | {'Audited (No Action)':<{col_w}} | {'Silent Risk':<{col_w}}")
    print(f"{'Multicollinearity Remediation':<30} | {str(summary['full_adaptive_system']['multicollinearity_pruned']):<{col_w}} | {'False':<{col_w}} | {'False':<{col_w}}")
    print("=" * 105)
    print("\nSaved artifacts to evaluation/results/ablation_results.json")


if __name__ == "__main__":
    d_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw/sample.csv"
    u_goal = sys.argv[2] if len(sys.argv) > 2 else "predict whether the tumor is malignant based on measurements"
    run_ablation(d_path, u_goal)
