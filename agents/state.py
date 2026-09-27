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

    # --- insight & critic enrichments ---
    final_report: Optional[str]
    confidence_score: int
    hypothesis_tests: List[Dict[str, Any]]
    models_evaluated: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    key_drivers: List[Dict[str, Any]]
    report_pdf_path: Optional[str]
    report_html_path: Optional[str]

    # --- logging (for your evaluation/report section later) ---
    agent_trace: List[str]
