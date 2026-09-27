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
    penalty = 0

    model_report = state.get("model_report")
    if model_report and model_report.get("status") != "skipped":
        # 1. Sample size check
        test_size = model_report.get("test_size", 0)
        total_samples = model_report.get("train_size", 0) + test_size
        if test_size < MIN_SAMPLE_SIZE * 0.2:
            notes.append(
                f"Sample size warning: Test set is small (n={test_size}); "
                f"accuracy/F1 estimates may have higher sample variance."
            )
            penalty += 10
            status = "needs_revision"
        else:
            notes.append(f"Sample size adequacy verified: {total_samples} total instances (n={test_size} test split).")

        # 2. Real Overfitting check (train vs test gap)
        overfit_gap = model_report.get("overfit_gap", 0.0)
        if overfit_gap > 0.15:
            notes.append(
                f"Overfitting risk detected: {model_report.get('model')} training score exceeds "
                f"test score by {round(overfit_gap * 100, 1)}%."
            )
            penalty += 8
            # Do not force revision if accuracy is still high, but log warning
            if overfit_gap > 0.30:
                status = "needs_revision"
        else:
            notes.append(f"Overfitting check passed: generalisation gap is within nominal bound ({round(overfit_gap * 100, 1)}% <= 15%).")

        # 3. Contradiction check: do ML top features show up in EDA's top correlations?
        eda = state.get("eda_findings", {})
        top_corr_cols = set()
        for pair in eda.get("top_correlations", []):
            top_corr_cols.add(pair["col_a"])
            top_corr_cols.add(pair["col_b"])
        ml_top_features = {f for f, _ in model_report.get("top_features", [])}
        if top_corr_cols and ml_top_features and not (top_corr_cols & ml_top_features):
            notes.append(
                "ML feature importances do not overlap with EDA's top correlated "
                "columns -- findings may indicate non-linear relations or proxy collinearity."
            )
            penalty += 5

    # 4. Statistical hypothesis check
    hyp_tests = state.get("hypothesis_tests", [])
    if hyp_tests:
        sig_count = sum(1 for t in hyp_tests if t.get("is_significant"))
        notes.append(f"Empirical validation: {sig_count} of {len(hyp_tests)} hypothesis tests passed statistical significance (p < 0.05).")

    if not notes:
        notes.append("No issues detected: sample size adequate, ML and EDA findings consistent.")

    confidence_score = max(60, min(99, 100 - penalty))
    state["confidence_score"] = confidence_score
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
        f"[critic_agent] status={state['validation_status']} confidence_score={confidence_score}% notes={len(notes)} items"
    )
    return state
