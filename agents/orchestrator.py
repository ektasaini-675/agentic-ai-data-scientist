"""
Orchestrator Agent.

Right now this uses a simple rule-based planner so the graph is runnable
WITHOUT an API key (good for today's smoke test). Swap `plan_tasks()`'s
body for an LLM call once you've got your API key wired up — the
function signature and return type should not need to change.
"""
from agents.state import AgentState


def plan_tasks(state: AgentState) -> AgentState:
    goal = state["user_goal"].lower()

    plan = ["data_agent"]  # cleaning always runs first

    plan.append("eda_agent")  # always look at patterns

    # crude keyword-based routing -- replace with LLM reasoning later
    if any(k in goal for k in ["predict", "classif", "churn", "risk", "forecast"]):
        plan.append("ml_agent")

    plan.append("viz_agent")
    plan.append("critic_agent")
    plan.append("insight_agent")

    state["task_plan"] = plan
    state["current_task_index"] = 0
    state.setdefault("agent_trace", []).append(
        f"[orchestrator] goal='{state['user_goal']}' -> plan={plan}"
    )
    return state


def route_next(state: AgentState) -> str:
    """Conditional edge: decides which node runs next based on the plan."""
    idx = state.get("current_task_index", 0)
    plan = state.get("task_plan", [])
    if idx >= len(plan):
        return "end"
    return plan[idx]


def advance(state: AgentState) -> AgentState:
    state["current_task_index"] = state.get("current_task_index", 0) + 1
    return state
