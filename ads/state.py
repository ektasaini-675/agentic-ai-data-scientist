"""The shared contracts every part of the system agrees on.

Keep this file small and stable. If one of these shapes changes, everything
that touches it must change in the same commit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


@dataclass(frozen=True)
class Evidence:
    """One computed fact. Immutable: corrections are NEW entries, never edits."""

    id: str                      # "E0017"
    tool: str                    # which tool produced it
    args: Dict[str, Any]         # exact arguments the tool was called with
    result: Any                  # JSON-safe raw result (exact numbers live here)
    summary: str                 # plain-English sentence written by the tool
    created_by: str = "unknown"  # agent that called the tool
    dataset_version: str = "v0"  # which cleaned version of the data it ran on
    step: int = 0                # order within the run

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        return cls(**d)


@dataclass
class Claim:
    """One sentence of the report plus the evidence that must back it."""

    text: str
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class Feedback:
    """What the Critic/verifier sends back to a specific agent."""

    issue: str
    severity: str                 # "info" | "warning" | "blocker"
    target_agent: str             # e.g. "data_agent"
    action: str                   # what the target agent should do
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """What every tool returns.

    `dataframe` is in-memory only (never stored in the ledger). A tool that
    transforms the data returns the new frame here and the runner bumps the
    dataset version.
    """

    ok: bool
    data: Any = None
    summary: str = ""
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    dataframe: Any = None


class AgentState(TypedDict, total=False):
    """LangGraph state. Agents return partial updates, they do not mutate in place."""

    # input
    dataset_path: str
    goal: str

    # runtime handles (in-memory objects)
    ctx: Any                      # ads.context.RunContext
    client: Any                   # ads.llm.LLMClient

    # orchestration
    plan: List[str]
    plan_index: int
    replans: int
    retry_counts: Dict[str, int]

    # analysis outputs (small summaries; the real facts are in the ledger)
    target: Optional[str]
    problem_type: Optional[str]

    # insight + critic
    claims: List[Claim]
    rejected_claims: List[Dict[str, Any]]
    feedback: List[Feedback]
    confidence: Optional[int]
    validation_status: str        # "pass" | "needs_revision"

    # outputs
    report_paths: Dict[str, str]
    trace: List[str]
