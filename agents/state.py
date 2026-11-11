"""
Shared State Schema for the Autonomous Multi-Agent AI Data Scientist System.
Built for LangGraph StateGraph.

This TypedDict serves as the single source of truth passed across all agent nodes:
Orchestrator -> Data Agent -> EDA Agent -> ML Agent -> Viz Agent -> Critic Agent <-> Insight Agent
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    # --- Input Specification ---
    dataset_path: str
    user_goal: str
    system_mode: str              # "heuristic" | "llm_gemini" | "llm_anthropic" | "llm_openai"

    # --- Orchestrator & Planning ---
    task_plan: List[str]          # Ordered list of agent node names
    current_task_index: int       # Execution pointer
    plan_rationale: Optional[str] # Explanation of task decomposition

    # --- Data Agent Output ---
    schema_report: Dict[str, Any]
    cleaning_log: List[str]
    data_quality_report: Dict[str, Any]  # Missing rate, duplicate count, high cardinality, ID columns dropped
    cleaned_dataset_path: Optional[str]

    # --- EDA / Statistical Agent Output ---
    eda_findings: Dict[str, Any]
    hypothesis_tests: List[Dict[str, Any]]  # t-test, ANOVA, Chi-Square, Mann-Whitney U, p-values, effect sizes
    vif_report: List[Dict[str, Any]]        # Variance Inflation Factor for multicollinearity
    outlier_report: Dict[str, Any]          # IQR / Z-score outlier detection

    # --- ML Agent Output ---
    model_report: Dict[str, Any]            # Target column, problem type, best model, CV scores, overfit gap
    models_evaluated: List[Dict[str, Any]]  # Cross-validation leaderboard (RF, GB, Logistic Regression, etc.)
    top_features: List[Dict[str, Any]]      # XAI feature importance, rank, directional impact (+/-)
    confusion_matrix: Optional[List[List[int]]]
    class_imbalance_ratio: Optional[float]

    # --- Visualization Agent Output ---
    chart_paths: List[str]                  # High-res static PNGs for PDF & Streamlit
    interactive_charts: Dict[str, Any]      # Plotly JSON / figures

    # --- Critic / Validation Agent Output ---
    validation_status: str                  # "pass" | "needs_revision"
    validation_notes: List[str]             # Detailed multi-pillar audit notes
    confidence_score: int                   # Empirical Evidence Confidence Score (0–100%)
    retry_count: int                        # Loop counter for LangGraph cyclic edge
    critic_directives: List[Dict[str, Any]] # Structured remediation directives (e.g., prune_features, regularize)
    pillar_scores: Dict[str, int]           # Scores for Data Quality, Generalization, Statistical Rigor, Plausibility

    # --- Insight & Synthesis Agent Output ---
    final_report: Optional[str]             # Markdown report
    recommendations: List[Dict[str, Any]]   # Strategic recommendations with impact rating
    key_drivers: List[Dict[str, Any]]       # Top causal/predictive factors
    report_pdf_path: Optional[str]          # Path to executive PDF
    report_html_path: Optional[str]         # Path to interactive HTML
    report_md_path: Optional[str]           # Path to Markdown report

    # --- Telemetry & Verification ---
    agent_trace: List[str]                  # Detailed chronological execution log
    node_timings: Dict[str, float]          # Latency in seconds per agent node
