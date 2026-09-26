"""
Insight Agent — turns validated numeric results into a readable report.
Currently template-based (no hallucination risk since every line pulls
straight from stored state). Swap the body for an LLM call later --
but keep passing it ONLY state contents, never let it invent numbers.
"""
from agents.state import AgentState


def run(state: AgentState) -> AgentState:
    lines = ["# Automated Data Analysis Report\n"]
    lines.append(f"**Objective:** {state['user_goal']}\n")

    schema = state.get("schema_report", {})
    lines.append("## Dataset Overview")
    lines.append(f"- Rows: {schema.get('n_rows')}, Columns: {schema.get('n_cols')}")
    for note in state.get("cleaning_log", []):
        lines.append(f"- {note}")
    lines.append("")

    eda = state.get("eda_findings", {})
    lines.append("## Key Patterns (EDA)")
    for pair in eda.get("top_correlations", []):
        lines.append(f"- {pair['col_a']} vs {pair['col_b']}: correlation = {pair['corr']:.2f}")
    lines.append("")

    model_report = state.get("model_report")
    if model_report and model_report.get("status") != "skipped":
        lines.append("## Predictive Modeling")
        lines.append(f"- Target: {model_report['target_column']}")
        lines.append(f"- Model: {model_report['model']}")
        lines.append(f"- Accuracy: {model_report['accuracy']}, F1: {model_report['f1_score']}")
        lines.append(f"- Most influential features: {', '.join(f[0] for f in model_report['top_features'])}")
        lines.append("")

    lines.append("## Validation Notes")
    for note in state.get("validation_notes", []):
        lines.append(f"- {note}")
    lines.append("")

    if state.get("chart_paths"):
        lines.append("## Charts Generated")
        for p in state["chart_paths"]:
            lines.append(f"- {p}")

    report = "\n".join(lines)
    state["final_report"] = report
    state.setdefault("agent_trace", []).append("[insight_agent] final report generated")
    return state
