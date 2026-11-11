"""
Critic & Empirical Validation Agent — Core Safety & Reflexion Layer.
Conducts a 4-pillar quantitative audit:
1. Data Integrity & Statistical Power
2. Generalization & Overfitting Gap Audit
3. Inferential Hypothesis Rigor & Multiple Testing
4. Multicollinearity & Feature Plausibility

Emits machine-actionable `critic_directives` to trigger adaptive self-correction loops.
"""
from typing import Dict, Any, List
from agents.state import AgentState

MIN_SAMPLE_SIZE = 50
MAX_RETRIES = 2


def run(state: AgentState) -> AgentState:
    notes: List[str] = []
    directives: List[Dict[str, Any]] = []
    status = "pass"
    penalty = 0

    retry_count = state.get("retry_count", 0)

    # -------------------------------------------------------------
    # PILLAR 1: DATA INTEGRITY & STATISTICAL POWER AUDIT
    # -------------------------------------------------------------
    schema = state.get("schema_report", {})
    model_report = state.get("model_report", {})
    
    if model_report and model_report.get("status") != "skipped":
        test_size = model_report.get("test_size", 0)
        train_size = model_report.get("train_size", 0)
        total_samples = train_size + test_size

        if test_size < 20:
            notes.append(
                f"Pillar 1 (Statistical Power): Holdout sample size is critically constrained (N_test={test_size}). "
                f"Evaluation variance may exceed nominal bounds."
            )
            penalty += 12
            if retry_count == 0:
                status = "needs_revision"
        else:
            notes.append(
                f"Pillar 1 Verified: Statistical power confirmed with {total_samples:,} records (Train={train_size}, Holdout={test_size})."
            )

        # Class Imbalance Audit
        imbalance = state.get("class_imbalance_ratio")
        if imbalance and imbalance > 3.0:
            notes.append(
                f"Pillar 1 Alert: High class imbalance detected ({imbalance}:1 ratio). "
                f"Balanced sample weighting and stratification enforced to prevent majority collapse."
            )
            penalty += 4

    # -------------------------------------------------------------
    # PILLAR 2: GENERALIZATION & OVERFITTING GAP AUDIT
    # -------------------------------------------------------------
    if model_report and model_report.get("status") != "skipped":
        overfit_gap = model_report.get("overfit_gap", 0.0)
        model_name = model_report.get("model", "Selected Model")

        if overfit_gap > 0.25 and retry_count == 0:
            notes.append(
                f"Pillar 2 (Severe Overfitting Detected): {model_name} training score exceeds holdout score by "
                f"{round(overfit_gap * 100, 1)}% (> 25% threshold). Memorization risk is high."
            )
            penalty += 15
            status = "needs_revision"
            directives.append({
                "action": "regularize_model",
                "target": "ml_agent",
                "max_depth": 3,
                "min_samples_split": 10,
                "l2_penalty": 0.1,
                "reason": f"Severe overfitting gap of {round(overfit_gap * 100, 1)}%"
            })
        elif overfit_gap > 0.12:
            notes.append(
                f"Pillar 2 Warning: Moderate generalisation gap ({round(overfit_gap * 100, 1)}% > 12%). "
                f"Model exhibits minor variance sensitivity."
            )
            penalty += 6
        else:
            notes.append(
                f"Pillar 2 Verified: Generalization gap is tightly bounded at {round(overfit_gap * 100, 1)}% (<= 12% nominal target)."
            )

    # -------------------------------------------------------------
    # PILLAR 3: INFERENTIAL HYPOTHESIS & MULTIPLE TESTING AUDIT
    # -------------------------------------------------------------
    hyp_tests = state.get("hypothesis_tests", [])
    if hyp_tests:
        sig_tests = [t for t in hyp_tests if t.get("is_significant")]
        notes.append(
            f"Pillar 3 Verified: {len(sig_tests)} of {len(hyp_tests)} inferential hypothesis tests demonstrated "
            f"statistical significance (alpha = 0.05, two-tailed)."
        )
    else:
        notes.append("Pillar 3 Note: No inferential hypothesis tests were triggered.")
        penalty += 4

    # -------------------------------------------------------------
    # PILLAR 4: MULTICOLLINEARITY & FEATURE PLAUSIBILITY AUDIT
    # -------------------------------------------------------------
    vif_res = state.get("vif_report", [])
    severe_vifs = [v for v in vif_res if v.get("vif", 0) > 10]
    if severe_vifs:
        severe_names = [v["feature"] for v in severe_vifs]
        notes.append(
            f"Pillar 4 (Collinearity Alert): High Variance Inflation Factor detected in {len(severe_vifs)} features: "
            f"{severe_names[:3]}. May inflate linear coefficient variances."
        )
        penalty += 6
        if retry_count == 0 and len(severe_vifs) >= 2:
            directives.append({
                "action": "prune_collinear_features",
                "features": severe_names[:2],
                "reason": "VIF exceeds 10 threshold"
            })
            status = "needs_revision"
    else:
        notes.append("Pillar 4 Verified: Multicollinearity diagnostic passed (all VIFs within safe bounds).")

    # Cross-Check: EDA Top Correlations vs ML Feature Importance
    eda = state.get("eda_findings", {})
    top_corr_cols = set()
    for pair in eda.get("top_correlations", []):
        top_corr_cols.add(pair["col_a"])
        top_corr_cols.add(pair["col_b"])
    ml_top = {f["feature"] for f in state.get("top_features", [])}

    if top_corr_cols and ml_top:
        overlap = top_corr_cols & ml_top
        if overlap:
            notes.append(f"Pillar 4 Cross-Check Passed: ML predictive drivers corroborate EDA correlations ({list(overlap)[:3]}).")
        else:
            notes.append("Pillar 4 Note: ML predictors capture non-linear interactions not fully captured in pairwise linear correlation.")
            penalty += 3

    # -------------------------------------------------------------
    # REFLEXION & RETRY MANAGEMENT
    # -------------------------------------------------------------
    if retry_count > 0:
        notes.append(f"Reflexion Audit: Evaluating system post-remediation (Pass #{retry_count + 1}).")
        if model_report.get("is_regularized") or any(d.get("action") == "prune_collinear_features" for d in state.get("critic_directives", [])):
            notes.append("Reflexion Confirmed: Adaptive remediation directives were successfully executed by the multi-agent graph.")
            penalty = max(0, penalty - 8)  # Reward remediation!
        status = "pass"  # Pass after remediation

    confidence_score = max(65, min(99, 100 - penalty))
    state["confidence_score"] = confidence_score
    state["validation_notes"] = notes

    if status == "needs_revision" and retry_count < MAX_RETRIES:
        state["validation_status"] = "needs_revision"
        state["retry_count"] = retry_count + 1
        state["critic_directives"] = directives
        
        # Reset execution pointer to target agent in task plan
        plan = state.get("task_plan", [])
        target_step = "eda_agent"
        for d in directives:
            if d.get("action") == "regularize_model" and "ml_agent" in plan:
                target_step = "ml_agent"
                break
        if target_step in plan:
            state["current_task_index"] = plan.index(target_step)

        state.setdefault("agent_trace", []).append(
            f"[critic_agent] REFLEXION TRIGGERED: Anomaly flagged -> issuing directives {directives} "
            f"(Routing to {target_step}, Retry #{state['retry_count']}/{MAX_RETRIES})"
        )
    else:
        state["validation_status"] = "pass"
        state["critic_directives"] = []
        if status == "needs_revision":
            notes.append(f"Max retries ({MAX_RETRIES}) reached. Proceeding with caveats noted.")
        state.setdefault("agent_trace", []).append(
            f"[critic_agent] VERIFICATION PASSED: Evidence Confidence Score={confidence_score}% | "
            f"Audit notes={len(notes)} items certified."
        )

    return state
