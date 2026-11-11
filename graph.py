"""
Autonomous Multi-Agent AI System: LangGraph StateGraph Architecture.
Implements cyclic reflexivity, dynamic edge routing, and adaptive remediation.

Usage via CLI:
    python graph.py datasets/raw/sample.csv "predict whether tumor is malignant based on measurements"
"""
import os
import sys
import time
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents import (
    orchestrator,
    data_agent,
    eda_agent,
    ml_agent,
    viz_agent,
    critic_agent,
    insight_agent
)
from agents.orchestrator_llm import plan_tasks_llm

USE_LLM_ORCHESTRATOR = os.environ.get("USE_LLM_ORCHESTRATOR", "0") == "1"


def build_graph():
    """Compiles the multi-agent StateGraph with adaptive cyclic reflection."""
    g = StateGraph(AgentState)

    planner = plan_tasks_llm if USE_LLM_ORCHESTRATOR else orchestrator.plan_tasks
    
    # Wrap node executions to capture latency telemetry
    def timed_node(node_name, func):
        def wrapper(state: AgentState) -> AgentState:
            t0 = time.time()
            res = func(state)
            elapsed = round(time.time() - t0, 3)
            res.setdefault("node_timings", {})[node_name] = elapsed
            return res
        return wrapper

    g.add_node("plan", timed_node("plan", planner))
    g.add_node("data_agent", timed_node("data_agent", lambda s: orchestrator.advance(data_agent.run(s))))
    g.add_node("eda_agent", timed_node("eda_agent", lambda s: orchestrator.advance(eda_agent.run(s))))
    g.add_node("ml_agent", timed_node("ml_agent", lambda s: orchestrator.advance(ml_agent.run(s))))
    g.add_node("viz_agent", timed_node("viz_agent", lambda s: orchestrator.advance(viz_agent.run(s))))
    g.add_node("critic_agent", timed_node("critic_agent", critic_agent.run))  # Does NOT auto-advance: handles retry state
    g.add_node("insight_agent", timed_node("insight_agent", lambda s: orchestrator.advance(insight_agent.run(s))))

    g.set_entry_point("plan")

    # Initial routing from plan
    g.add_conditional_edges("plan", orchestrator.route_next, {
        "data_agent": "data_agent",
        "eda_agent": "eda_agent",
        "ml_agent": "ml_agent",
        "viz_agent": "viz_agent",
        "critic_agent": "critic_agent",
        "insight_agent": "insight_agent",
        "end": END,
    })

    # Sequential routing across standard agent workers
    for node in ["data_agent", "eda_agent", "ml_agent", "viz_agent"]:
        g.add_conditional_edges(node, orchestrator.route_next, {
            "data_agent": "data_agent",
            "eda_agent": "eda_agent",
            "ml_agent": "ml_agent",
            "viz_agent": "viz_agent",
            "critic_agent": "critic_agent",
            "insight_agent": "insight_agent",
            "end": END,
        })

    # DYNAMIC CYCLIC REFLEXION EDGE
    def critic_route(state: AgentState) -> str:
        if state.get("validation_status") == "needs_revision":
            directives = state.get("critic_directives", [])
            for d in directives:
                if d.get("action") == "regularize_model":
                    return "ml_agent"
            return "eda_agent"

        # If validation passed: advance critic task pointer and proceed
        orchestrator.advance(state)
        return orchestrator.route_next(state)

    g.add_conditional_edges("critic_agent", critic_route, {
        "eda_agent": "eda_agent",
        "ml_agent": "ml_agent",
        "insight_agent": "insight_agent",
        "end": END,
    })

    g.add_edge("insight_agent", END)

    return g.compile()


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw/sample.csv"
    goal = sys.argv[2] if len(sys.argv) > 2 else "predict whether the tumor is malignant based on measurements"

    print("=" * 70)
    print("AUTONOMOUS MULTI-AGENT AI DATA SCIENTIST (LANGGRAPH STATEGRAPH)")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Goal:    {goal}")
    print("=" * 70 + "\n")

    app = build_graph()
    initial_state: AgentState = {
        "dataset_path": dataset_path,
        "user_goal": goal,
        "agent_trace": []
    }

    start_time = time.time()
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})
    total_time = round(time.time() - start_time, 2)

    print("\n" + "=" * 70)
    print(f"AGENT EXECUTION TRACE (Completed in {total_time}s)")
    print("=" * 70)
    for line in final_state.get("agent_trace", []):
        print(line)

    print("\n" + "=" * 70)
    print("FINAL EXECUTIVE REPORT")
    print("=" * 70)
    print(final_state.get("final_report", "No report generated."))

    print("\n" + "=" * 70)
    print(f"GENERATED ARTIFACTS:")
    print(f"  - PDF Report:  {final_state.get('report_pdf_path')}")
    print(f"  - HTML Report: {final_state.get('report_html_path')}")
    print(f"  - Charts:      {len(final_state.get('chart_paths', []))} PNG figures in reports/charts/")
    print("=" * 70)
