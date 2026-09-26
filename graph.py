"""
Wires all agents into a LangGraph StateGraph.

Run this directly for a smoke test (no API key needed yet, since
orchestrator/insight agents are currently rule-based/template-based):

    python graph.py datasets/raw/sample.csv "predict churn and explain key factors"
"""
import sys
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents import orchestrator, data_agent, eda_agent, ml_agent, viz_agent, critic_agent, insight_agent


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("plan", orchestrator.plan_tasks)
    g.add_node("data_agent", lambda s: orchestrator.advance(data_agent.run(s)))
    g.add_node("eda_agent", lambda s: orchestrator.advance(eda_agent.run(s)))
    g.add_node("ml_agent", lambda s: orchestrator.advance(ml_agent.run(s)))
    g.add_node("viz_agent", lambda s: orchestrator.advance(viz_agent.run(s)))
    g.add_node("critic_agent", critic_agent.run)  # does NOT auto-advance -- may loop back
    g.add_node("insight_agent", lambda s: orchestrator.advance(insight_agent.run(s)))

    g.set_entry_point("plan")

    # after planning, route to whatever the plan says is task index 0
    g.add_conditional_edges("plan", orchestrator.route_next, {
        "data_agent": "data_agent", "eda_agent": "eda_agent", "ml_agent": "ml_agent",
        "viz_agent": "viz_agent", "critic_agent": "critic_agent",
        "insight_agent": "insight_agent", "end": END,
    })

    # every worker node re-checks the plan and routes to the next step
    for node in ["data_agent", "eda_agent", "ml_agent", "viz_agent"]:
        g.add_conditional_edges(node, orchestrator.route_next, {
            "data_agent": "data_agent", "eda_agent": "eda_agent", "ml_agent": "ml_agent",
            "viz_agent": "viz_agent", "critic_agent": "critic_agent",
            "insight_agent": "insight_agent", "end": END,
        })

    # THE RETRY LOOP: critic can send the plan back to eda_agent instead of moving on
    def critic_route(state: AgentState) -> str:
        if state["validation_status"] == "needs_revision":
            return "eda_agent"          # send back for another pass
        orchestrator.advance(state)     # mark critic's own step done before checking what's next
        return orchestrator.route_next(state)

    g.add_conditional_edges("critic_agent", critic_route, {
        "eda_agent": "eda_agent", "insight_agent": "insight_agent", "end": END,
    })

    g.add_edge("insight_agent", END)

    return g.compile()


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw/sample.csv"
    goal = sys.argv[2] if len(sys.argv) > 2 else "understand the key patterns in this data"

    app = build_graph()
    initial_state: AgentState = {"dataset_path": dataset_path, "user_goal": goal, "agent_trace": []}

    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    print("\n" + "=" * 60)
    print("AGENT TRACE")
    print("=" * 60)
    for line in final_state["agent_trace"]:
        print(line)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(final_state["final_report"])
