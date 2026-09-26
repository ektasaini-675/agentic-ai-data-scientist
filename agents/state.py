"""
Shared state object passed between every agent node in the LangGraph.
This is the single source of truth the whole team reads/writes to —
it's what makes this "agentic" rather than a linear script.
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    # --- input ---
    dataset_path: str
    user_goal: str

    # --- orchestrator ---
    task_plan: List[str]          # ordered list of task names to execute
    current_task_index: int

    # --- data agent output ---
    schema_report: Dict[str, Any]
    cleaning_log: List[str]

    # --- eda / stats agent output ---
    eda_findings: Dict[str, Any]

    # --- ml agent output ---
    model_report: Dict[str, Any]

    # --- viz agent output ---
    chart_paths: List[str]

    # --- critic agent output ---
    validation_status: str        # "pass" | "needs_revision"
    validation_notes: List[str]
    retry_count: int

    # --- insight agent output ---
    final_report: Optional[str]

    # --- logging (for your evaluation/report section later) ---
    agent_trace: List[str]
