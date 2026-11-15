"""Agent base runner. Handles LLM vs offline fallback cleanly for every specialist."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ads.config import SETTINGS
from ads.context import RunContext
from ads.llm.client import LLMClient, LLMUnavailable
from ads.llm.loop import run_agent_loop
from ads.state import AgentState


def run_specialist(
    state: AgentState,
    agent_name: str,
    system: str,
    task: str,
    tool_names: List[str],
    fallback: Callable[[RunContext, AgentState], None],
) -> Dict[str, Any]:
    """Execute a specialist agent either with an LLM loop or the scripted fallback."""
    ctx: RunContext = state["ctx"]
    ctx.agent_name = agent_name
    client: Optional[LLMClient] = state.get("client")

    new_trace = list(state.get("trace") or [])

    if client is not None and client.available:
        try:
            loop_res = run_agent_loop(
                client=client,
                ctx=ctx,
                agent_name=agent_name,
                system=system,
                task=task,
                tool_names=tool_names,
                max_steps=SETTINGS.max_agent_steps,
            )
            step_summary = (
                f"[{agent_name}] completed via LLM ({loop_res.steps} step(s), "
                f"{len(loop_res.trace)} tool call(s))"
            )
            new_trace.append(step_summary)
            return {"trace": new_trace}
        except Exception:
            pass

    # Fallback to deterministic scripted policy
    fallback(ctx, state)
    step_summary = f"[{agent_name}] executed scripted policy ({ctx.dataset_version})"
    new_trace.append(step_summary)
    return {"trace": new_trace}

