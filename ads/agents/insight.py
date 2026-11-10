"""Insight Agent: transforms ledger evidence into verified claims and recommendations."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ads.config import SETTINGS
from ads.context import RunContext
from ads.llm.client import LLMClient
from ads.state import AgentState, Claim
from ads.verify import ClaimCheck, verify_claims

INSIGHT_SYSTEM_PROMPT = """You are the Lead Insight Agent in an autonomous data science system.
Your job is to analyze the Evidence Ledger and generate precise, data-grounded claims for an executive report.

STRICT FACTUAL INTEGRITY RULES:
1. Every claim MUST cite the exact evidence ID(s) that substantiate it (e.g., ["E0001"]).
2. Every number, percentage, or metric MUST be strictly present in the cited evidence. Never invent or alter numbers.
3. Use associative and observational language ("is associated with", "correlates with"), NEVER causal claims ("causes", "will reduce").
4. Formulate actionable recommendations supported directly by segment outcome differences.
5. Return ONLY a valid JSON array of claim objects, with NO surrounding markdown or commentary:
[
  {"text": "The dataset contains 569 rows and 31 columns with 0 missing cells.", "evidence_ids": ["E0001"]},
  {"text": "Model LogisticRegression achieved a test ROC-AUC of 0.98 compared to baseline 0.63.", "evidence_ids": ["E0005"]}
]
"""


def _generate_offline_claims(ctx: RunContext) -> List[Claim]:
    """Deterministically construct verified claims from ledger evidence."""
    claims: List[Claim] = []

    for ev in ctx.ledger.all():
        res = ev.result if isinstance(ev.result, dict) else {}

        if ev.tool == "profile_dataset" and "n_rows" in res and "n_columns" in res:
            n_rows = res["n_rows"]
            n_cols = res["n_columns"]
            n_missing = res.get("missing_cells", 0)
            claims.append(
                Claim(
                    text=f"The dataset comprises {n_rows} rows and {n_cols} columns with {n_missing} missing cell(s).",
                    evidence_ids=[ev.id],
                )
            )

        elif ev.tool == "identify_target" and "target" in res:
            target = res["target"]
            prob = res.get("problem_type", "classification")
            claims.append(
                Claim(
                    text=f"Identified '{target}' as the analytical target for {prob} modeling.",
                    evidence_ids=[ev.id],
                )
            )

        elif ev.tool == "test_association" and "results" in res:
            results = res.get("results", [])
            for r in results[:2]:
                feat = r.get("feature")
                stat = r.get("test_name")
                eff = r.get("effect_size")
                if feat and eff is not None:
                    claims.append(
                        Claim(
                            text=f"Feature '{feat}' exhibited a significant relationship with the target ({stat}, effect size {eff:.2f}).",
                            evidence_ids=[ev.id],
                        )
                    )

        elif ev.tool == "segment_rates" and "highest_rate" in res and "lowest_rate" in res:
            feat = res.get("feature")
            high = res.get("highest_rate", 0)
            low = res.get("lowest_rate", 0)
            if feat:
                claims.append(
                    Claim(
                        text=f"Outcome rate for '{feat}' varied from {low:.1f}% in the lowest segment to {high:.1f}% in the highest segment.",
                        evidence_ids=[ev.id],
                    )
                )

        elif ev.tool == "train_cv" and "best_model" in res:
            model = res.get("best_model")
            test_score = res.get("test_score")
            baseline = res.get("majority_baseline")
            metric = res.get("metric", "score")
            if test_score is not None and baseline is not None:
                claims.append(
                    Claim(
                        text=f"The best model {model} achieved a test {metric} of {test_score:.2f} exceeding the baseline {baseline:.2f}.",
                        evidence_ids=[ev.id],
                    )
                )

        elif ev.tool == "permutation_importance" and "top_features" in res:
            top = res.get("top_features", [])
            if top:
                first = top[0]
                feat = first.get("feature")
                imp = first.get("importance")
                if feat and imp is not None:
                    claims.append(
                        Claim(
                            text=f"Permutation importance established '{feat}' as the top predictor with an importance score of {imp:.3f}.",
                            evidence_ids=[ev.id],
                        )
                    )

    return claims


def _parse_llm_claims(text: str) -> List[Claim]:
    """Parse JSON claims from LLM response."""
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"```$", "", clean, flags=re.MULTILINE).strip()
    match = re.search(r"\[\s*\{.*\}\s*\]", clean, re.DOTALL)
    if match:
        clean = match.group(0)

    try:
        data = json.loads(clean)
        claims = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "text" in item:
                    eids = item.get("evidence_ids") or []
                    if isinstance(eids, str):
                        eids = [eids]
                    claims.append(Claim(text=str(item["text"]).strip(), evidence_ids=list(eids)))
        return claims
    except Exception:
        return []


def run(state: AgentState) -> Dict[str, Any]:
    ctx: RunContext = state["ctx"]
    ctx.agent_name = "insight_agent"
    client: Optional[LLMClient] = state.get("client")
    new_trace = list(state.get("trace") or [])

    accepted: List[Claim] = []
    rejected: List[ClaimCheck] = []

    if client is not None and client.available:
        digest = ctx.ledger.digest()
        goal = state.get("goal") or ctx.goal
        task = (
            f"Analytical Goal: {goal}\n\n"
            f"Evidence Ledger:\n{digest}\n\n"
            "Generate verified executive claims and data-backed recommendations citing evidence IDs."
        )

        messages = [{"role": "user", "content": task}]
        max_retries = SETTINGS.max_retries_per_agent

        for attempt in range(max_retries + 1):
            try:
                resp = client.complete(system=INSIGHT_SYSTEM_PROMPT, messages=messages)
                candidate_claims = _parse_llm_claims(resp.text)
                if not candidate_claims:
                    candidate_claims = _generate_offline_claims(ctx)

                curr_accepted, curr_rejected = verify_claims(candidate_claims, ctx.ledger)
                accepted = curr_accepted
                rejected = curr_rejected

                if not curr_rejected or attempt == max_retries:
                    break

                # Feedback loop for rewrite attempt
                problems_summary = "\n".join(
                    f"- \"{r.claim.text}\" (Issues: {'; '.join(r.problems)})" for r in curr_rejected
                )
                retry_msg = (
                    f"The claim verifier rejected {len(curr_rejected)} of your claims:\n"
                    f"{problems_summary}\n\n"
                    "Please rewrite the claims ensuring EVERY cited number is strictly identical to the evidence, "
                    "or omit unsupported claims. Return only the revised JSON array."
                )
                messages.append({"role": "assistant", "content": resp.text})
                messages.append({"role": "user", "content": retry_msg})
            except Exception:
                accepted = _generate_offline_claims(ctx)
                rejected = []
                break

        new_trace.append(
            f"[insight_agent] synthesized {len(accepted)} verified claim(s) via LLM ({len(rejected)} rejected)"
        )
    else:
        candidate_claims = _generate_offline_claims(ctx)
        accepted, rejected = verify_claims(candidate_claims, ctx.ledger)
        new_trace.append(
            f"[insight_agent] synthesized {len(accepted)} verified claim(s) via offline template"
        )

    rejected_dicts = [
        {"claim": r.claim.text, "evidence_ids": r.claim.evidence_ids, "problems": r.problems}
        for r in rejected
    ]

    return {
        "claims": accepted,
        "rejected_claims": rejected_dicts,
        "trace": new_trace,
    }

