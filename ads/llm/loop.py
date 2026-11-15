"""The generic agent loop used by every specialist agent.

The LLM sees a task and a list of allowed tools. Each turn it either calls
tools (we run them through `execute_tool`, so every result lands in the
ledger) or answers in plain text, which ends the loop. A hard step limit
prevents runaway loops.

What the LLM sees after a tool call is the evidence ID, the tool's own
plain-English summary, and a compact copy of the data. The LLM never computes
statistics itself; it only decides which tool to call next.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ads.config import SETTINGS
from ads.context import RunContext
from ads.ledger import to_jsonable
from ads.llm.client import LLMClient
from ads.tools import execute_tool, list_tools, tool_schemas


@dataclass
class LoopResult:
    final_text: str
    steps: int
    stopped_reason: str                      # "done" | "max_steps"
    trace: List[Dict[str, Any]] = field(default_factory=list)


def _compact(data: Any, limit: int = 2500) -> str:
    text = json.dumps(to_jsonable(data), separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _block(tool_use_id: str, content: str, is_error: bool = False) -> Dict[str, Any]:
    block: Dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def run_agent_loop(
    client: LLMClient,
    ctx: RunContext,
    agent_name: str,
    system: str,
    task: str,
    tool_names: Optional[Sequence[str]] = None,
    max_steps: Optional[int] = None,
) -> LoopResult:
    max_steps = max_steps or SETTINGS.max_agent_steps
    ctx.agent_name = agent_name
    allowed = set(tool_names) if tool_names is not None else {t.name for t in list_tools(agent_name)}
    schemas = [s for s in tool_schemas(agent_name) if s["name"] in allowed]

    messages: List[Dict[str, Any]] = [{"role": "user", "content": task}]
    trace: List[Dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        response = client.complete(system, messages, tools=schemas)
        messages.append({"role": "assistant", "content": response.content})

        calls = response.tool_calls
        if not calls:
            return LoopResult(final_text=response.text, steps=step, stopped_reason="done", trace=trace)

        blocks: List[Dict[str, Any]] = []
        for call in calls:
            name, args = call["name"], call.get("input") or {}
            if name not in allowed:
                blocks.append(_block(call["id"], f"ERROR: tool '{name}' is not available to {agent_name}.", True))
                trace.append({"tool": name, "args": args, "ok": False, "evidence_id": None})
                continue

            result, evidence = execute_tool(name, args, ctx)
            if result.ok:
                eid = evidence.id if evidence else None
                blocks.append(_block(call["id"], f"Evidence {eid}: {result.summary}\nData: {_compact(result.data)}"))
            else:
                blocks.append(_block(call["id"], f"ERROR: {result.error}", True))
            trace.append(
                {"tool": name, "args": args, "ok": result.ok, "evidence_id": evidence.id if evidence else None}
            )
        messages.append({"role": "user", "content": blocks})

    return LoopResult(final_text="", steps=max_steps, stopped_reason="max_steps", trace=trace)
