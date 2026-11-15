"""High-level analysis pipeline entry point."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ads.context import RunContext
from ads.graph import build_graph
from ads.llm.client import LLMClient
from ads.state import AgentState, Claim


@dataclass
class RunResult:
    run_dir: Path
    ledger_path: Path
    report_paths: Dict[str, str]
    confidence: int
    claims: List[Claim]
    trace: List[str]
    ctx: RunContext


def run_analysis(
    dataset_path: str,
    goal: str = "",
    client: Optional[LLMClient] = None,
    base_dir: Optional[Path] = None,
) -> RunResult:
    """Run the complete autonomous multi-agent analysis pipeline on a dataset."""
    ctx = RunContext.create(dataset_path=dataset_path, goal=goal, base_dir=base_dir)
    client = client if client is not None else LLMClient()

    initial_state: AgentState = {
        "dataset_path": str(dataset_path),
        "goal": goal,
        "ctx": ctx,
        "client": client,
        "trace": [],
        "retry_counts": {},
        "replans": 0,
    }

    app = build_graph()
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    ledger_path = ctx.save_ledger()
    report_paths = final_state.get("report_paths") or {}
    confidence = final_state.get("confidence") or 0
    claims = final_state.get("claims") or []
    trace = final_state.get("trace") or []

    return RunResult(
        run_dir=ctx.run_dir,
        ledger_path=ledger_path,
        report_paths=report_paths,
        confidence=confidence,
        claims=claims,
        trace=trace,
        ctx=ctx,
    )

