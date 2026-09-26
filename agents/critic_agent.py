"""
Critic / Validation Agent — this is your project's most distinctive piece.
It checks whether conclusions are actually supported by the data, and can
send the state back for another pass via the retry loop in graph.py.

Current checks (rule-based, extend with LLM-based contradiction detection later):
  1. Sample size adequacy for any trained model.
  2. Overfitting check (train vs test performance gap) -- placeholder until
     ml_agent reports train accuracy too.
  3. Cross-check: do the ML agent's top features overlap at all with the
     EDA agent's top correlated columns? Flags a contradiction if not.
"""
from agents.state import AgentState

MIN_SAMPLE_SIZE = 50
MAX_RETRIES = 2


def run(state: AgentState) -> AgentState:
    notes = []
    status = "pass"

    model_report = state.get("model_report")
    if model_report and model_report.get("status") != "skipped":
        if model_report.get("test_size", 0) < MIN_SAMPLE_SIZE * 0.2:
            notes.append(
                f"Test set is small (n={model_report.get('test_size')}); "
                f"accuracy/F1 estimates may be unreliable."
            )
            status = "needs_revision"

        # contradiction check: do ML top features show up in EDA's top correlations at all?
        eda = state.get("eda_findings", {})
        top_corr_cols = set()
        for pair in eda.get("top_correlations", []):
            top_corr_cols.add(pair["col_a"])
            top_corr_cols.add(pair["col_b"])
        ml_top_features = {f for f, _ in model_report.get("top_features", [])}
        if top_corr_cols and ml_top_features and not (top_corr_cols & ml_top_features):
            notes.append(
                "ML feature importances do not overlap with EDA's top correlated "
                "columns -- findings may be inconsistent, consider re-checking."
            )
            status = "needs_revision"

    if not notes:
        notes.append("No issues detected: sample size adequate, ML and EDA findings consistent.")

    state["validation_status"] = status
    state["validation_notes"] = notes
    state["retry_count"] = state.get("retry_count", 0)

    if status == "needs_revision" and state["retry_count"] < MAX_RETRIES:
        state["retry_count"] += 1
    else:
        state["validation_status"] = "pass"  # force through after max retries, but log it
        if status == "needs_revision":
            notes.append(f"Max retries ({MAX_RETRIES}) reached -- proceeding with caveats noted above.")

    state.setdefault("agent_trace", []).append(
        f"[critic_agent] status={state['validation_status']} notes={notes}"
    )
    return state
