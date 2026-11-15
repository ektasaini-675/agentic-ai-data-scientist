"""Critic Agent: programmatic validity audits, statistical guards, and confidence scoring."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ads.config import CONFIDENCE_RUBRIC
from ads.context import RunContext
from ads.state import AgentState, Feedback

CAUSAL_REGEX = re.compile(r"\b(causes|caused|causing|will reduce|will increase|guarantee|proves)\b", re.IGNORECASE)


def run(state: AgentState) -> Dict[str, Any]:
    ctx: RunContext = state["ctx"]
    ctx.agent_name = "critic_agent"
    new_trace = list(state.get("trace") or [])

    feedback: List[Feedback] = []
    rubric_status: Dict[str, bool] = {k: True for k in CONFIDENCE_RUBRIC}

    # 1. Leakage check
    leakage_evs = [e for e in ctx.ledger.all() if e.tool == "detect_leakage"]
    has_leakage = False
    if leakage_evs:
        ev = leakage_evs[-1]
        suspects = ev.result.get("suspects", []) if isinstance(ev.result, dict) else []
        if suspects:
            has_leakage = True
            rubric_status["no_leakage"] = False
            suspect_names = [s.get("feature", "unknown") for s in suspects if isinstance(s, dict)]
            feedback.append(
                Feedback(
                    issue=f"Target leakage detected in {len(suspects)} feature(s): {', '.join(suspect_names)}",
                    severity="blocker",
                    target_agent="data_agent",
                    action=f"Drop suspect column(s) {suspect_names} using drop_columns.",
                    evidence_ids=[ev.id],
                )
            )

    # 2. Implausible score without leakage check
    cv_evs = [e for e in ctx.ledger.all() if e.tool == "train_cv"]
    if cv_evs and not leakage_evs:
        ev = cv_evs[-1]
        res = ev.result if isinstance(ev.result, dict) else {}
        score = res.get("test_score") or 0.0
        if score > 0.98:
            rubric_status["no_leakage"] = False
            feedback.append(
                Feedback(
                    issue=f"Implausibly high test score ({score:.3f}) observed without a leakage check.",
                    severity="warning",
                    target_agent="data_agent",
                    action="Run detect_leakage to confirm features are genuine.",
                    evidence_ids=[ev.id],
                )
            )

    # 3. Sample size check
    prof_evs = [e for e in ctx.ledger.all() if e.tool == "profile_dataset"]
    n_rows = 0
    if prof_evs:
        ev = prof_evs[-1]
        res = ev.result if isinstance(ev.result, dict) else {}
        n_rows = res.get("n_rows", 0)
        if n_rows < 200:
            rubric_status["adequate_sample"] = False
            feedback.append(
                Feedback(
                    issue=f"Small sample size ({n_rows} rows < 200 threshold).",
                    severity="warning",
                    target_agent="insight_agent",
                    action="Highlight sample size limitations and recommend collecting more data.",
                    evidence_ids=[ev.id],
                )
            )

    # 4. Overfitting & CV stability checks
    if cv_evs:
        ev = cv_evs[-1]
        res = ev.result if isinstance(ev.result, dict) else {}
        gap = res.get("train_vs_cv_gap")
        if gap is None:
            train_score = res.get("train_score")
            cv_score = res.get("cv_score")
            if train_score is not None and cv_score is not None:
                gap = abs(train_score - cv_score)

        if gap is not None and gap > 0.10:
            rubric_status["controlled_overfitting"] = False
            feedback.append(
                Feedback(
                    issue=f"Overfitting detected: generalization gap ({gap:.1%}) exceeds 10% threshold.",
                    severity="warning",
                    target_agent="ml_agent",
                    action="Increase regularization or reduce model complexity.",
                    evidence_ids=[ev.id],
                )
            )

        cv_std = res.get("cv_std")
        if cv_std is not None and cv_std > 0.05:
            feedback.append(
                Feedback(
                    issue=f"Cross-validation variance is elevated (std {cv_std:.3f} > 0.05).",
                    severity="warning",
                    target_agent="ml_agent",
                    action="Increase CV folds or evaluate simpler estimators.",
                    evidence_ids=[ev.id],
                )
            )

        test_score = res.get("test_score")
        baseline = res.get("majority_baseline")
        if test_score is not None and baseline is not None:
            if test_score <= baseline:
                rubric_status["beats_baseline"] = False
                feedback.append(
                    Feedback(
                        issue=f"Model performance ({test_score:.3f}) does not beat majority baseline ({baseline:.3f}).",
                        severity="warning",
                        target_agent="ml_agent",
                        action="Tune hyperparameters or engineer more informative features.",
                        evidence_ids=[ev.id],
                    )
                )

    # 5. Multiplicity check
    assoc_evs = [e for e in ctx.ledger.all() if e.tool == "test_association"]
    if assoc_evs:
        ev = assoc_evs[-1]
        feedback.append(
            Feedback(
                issue="Multiple hypothesis testing conducted.",
                severity="info",
                target_agent="insight_agent",
                action="Confirm Benjamini-Hochberg FDR correction is acknowledged in findings.",
                evidence_ids=[ev.id],
            )
        )
    else:
        rubric_status["multiplicity_correction"] = False

    # 6. Rejected claims check
    rejected = state.get("rejected_claims") or []
    if rejected:
        rubric_status["verified_claims"] = False
        feedback.append(
            Feedback(
                issue=f"{len(rejected)} claim(s) failed verification against the Evidence Ledger.",
                severity="blocker",
                target_agent="insight_agent",
                action="Rewrite claims using only verified numbers or exclude them from report.",
                evidence_ids=[],
            )
        )

    # 7. Causal language check
    claims = state.get("claims") or []
    for c in claims:
        if CAUSAL_REGEX.search(c.text):
            feedback.append(
                Feedback(
                    issue=f"Claim uses unsupported causal or predictive promise: '{c.text}'.",
                    severity="warning",
                    target_agent="insight_agent",
                    action="Rephrase using observational correlation or association phrasing.",
                    evidence_ids=list(c.evidence_ids),
                )
            )

    # Calculate confidence score via rubric
    total_weight = sum(v["weight"] for v in CONFIDENCE_RUBRIC.values())
    passed_weight = sum(
        v["weight"] for k, v in CONFIDENCE_RUBRIC.items() if rubric_status.get(k, False)
    )
    confidence = int((passed_weight / total_weight) * 100) if total_weight > 0 else 0

    has_blocker = any(fb.severity == "blocker" for fb in feedback)
    status = "needs_revision" if has_blocker else "pass"

    new_trace.append(
        f"[critic_agent] audit complete: status={status}, confidence={confidence}%, "
        f"{len(feedback)} feedback item(s) generated"
    )

    return {
        "feedback": feedback,
        "confidence": confidence,
        "validation_status": status,
        "trace": new_trace,
    }

