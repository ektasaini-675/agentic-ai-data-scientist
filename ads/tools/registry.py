"""Tool registry + the single runner that every tool call goes through.

Agents never touch the ledger directly. They call `execute_tool(...)`, which
runs the tool, writes the Evidence entry, and handles caching and dataset
versioning. That is what guarantees nothing computed is ever unrecorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ads.context import RunContext
from ads.state import Evidence, ToolResult

_EMPTY_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., ToolResult]
    cacheable: bool = True            # False for tools that transform the data
    agents: Tuple[str, ...] = ()      # which agents may use it (empty = all)
    needs_ctx: bool = False           # True: runner passes ctx=RunContext (plots, fitted models)


_REGISTRY: Dict[str, ToolSpec] = {}


def register(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    cacheable: bool = True,
    agents: Tuple[str, ...] = (),
    needs_ctx: bool = False,
) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
    """Decorator: `@register("profile_dataset", "...")` over `def f(df, **args) -> ToolResult`."""

    def deco(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        _REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters or _EMPTY_SCHEMA,
            fn=fn,
            cacheable=cacheable,
            agents=agents,
            needs_ctx=needs_ctx,
        )
        return fn

    return deco


def get_tool(name: str) -> Optional[ToolSpec]:
    return _REGISTRY.get(name)


def list_tools(agent: Optional[str] = None) -> List[ToolSpec]:
    return [t for t in _REGISTRY.values() if agent is None or not t.agents or agent in t.agents]


def tool_schemas(agent: Optional[str] = None) -> List[Dict[str, Any]]:
    """Tool definitions in the shape the Anthropic Messages API expects."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in list_tools(agent)
    ]


def execute_tool(name: str, args: Optional[Dict[str, Any]], ctx: RunContext) -> Tuple[ToolResult, Optional[Evidence]]:
    """Run one tool call. Returns (result, evidence). Evidence is None if the call failed."""
    spec = _REGISTRY.get(name)
    if spec is None:
        return ToolResult(ok=False, error=f"unknown tool '{name}'"), None

    args = dict(args or {})

    if spec.cacheable:
        cached = ctx.ledger.find_cached(name, args, ctx.dataset_version)
        if cached is not None:
            return ToolResult(ok=True, data=cached.result, summary=cached.summary), cached

    step = ctx.next_step()
    version_it_ran_on = ctx.dataset_version
    try:
        call_args = dict(args)
        if spec.needs_ctx:
            call_args["ctx"] = ctx  # never written to the ledger (only `args` is)
        result = spec.fn(ctx.df, **call_args)
    except Exception as exc:  # a tool failure must never crash the run
        return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}"), None

    if not result.ok:
        return result, None

    if result.dataframe is not None:
        ctx.set_dataframe(result.dataframe)

    evidence = ctx.ledger.add(
        tool=name,
        args=args,
        result=result.data,
        summary=result.summary,
        created_by=ctx.agent_name,
        dataset_version=version_it_ran_on,
        step=step,
        cacheable=spec.cacheable,
    )
    return result, evidence
