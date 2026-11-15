"""Claim verifier: checks that every claim in a report is backed by its evidence.

A claim passes only if:
  1. it cites at least one evidence ID,
  2. every cited ID exists in the ledger, and
  3. every meaningful number in the claim text appears in the cited evidence
     (within rounding tolerance implied by how many decimals the claim shows).

Heuristics (documented so the limits are honest):
  - Bare integers 0-10 are ignored (they are usually counts like "top 3").
  - "42.7%" matches evidence stored as 0.427 or as 42.7.
  - "p < 0.001" matches any cited number in [0, 0.001]; ">" is the mirror case.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from ads.ledger import Ledger
from ads.state import Claim, Evidence, Feedback

_NUM = re.compile(
    r"(?<![\w.])(?P<cmp>[<>\u2264\u2265]=?\s*)?(?P<num>[-+]?\d[\d,]*(?:\.\d+)?)(?P<pct>\s*(?:%|percent\b))?"
)


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: float
    decimals: int
    is_percent: bool
    cmp: str


def _scan(text: str) -> List[NumberToken]:
    tokens: List[NumberToken] = []
    for m in _NUM.finditer(text or ""):
        clean = m.group("num").replace(",", "")
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        tokens.append(
            NumberToken(
                raw=m.group(0).strip(),
                value=value,
                decimals=decimals,
                is_percent=bool(m.group("pct")),
                cmp=(m.group("cmp") or "").strip(),
            )
        )
    return tokens


def extract_numbers(text: str) -> List[NumberToken]:
    """Numbers in a claim that must be backed by evidence (small bare integers skipped)."""
    return [
        t
        for t in _scan(text)
        if t.is_percent or t.cmp or t.decimals > 0 or abs(t.value) > 10
    ]


def _walk(obj: Any, out: List[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, out)


def evidence_numbers(ev: Evidence) -> List[float]:
    """Every number an evidence entry can legitimately support."""
    nums: List[float] = []
    _walk(ev.result, nums)
    for t in _scan(ev.summary):
        nums.append(t.value)
        if t.is_percent:
            nums.append(t.value / 100.0)
    return nums


def _token_matches(tok: NumberToken, pool: List[float]) -> bool:
    if tok.is_percent:
        options = [(tok.value / 100.0, 0.5 * 10 ** -(tok.decimals + 2)), (tok.value, 0.5 * 10 ** -tok.decimals)]
    else:
        options = [(tok.value, 0.5 * 10 ** -tok.decimals)]
    for v, tol in options:
        for e in pool:
            if tok.cmp in ("<", "<=", "\u2264", "<=" ):
                if 0 <= e <= v + tol:
                    return True
            elif tok.cmp in (">", ">=", "\u2265"):
                if e >= v - tol:
                    return True
            elif abs(e - v) <= tol + 1e-12:
                return True
    return False


@dataclass
class ClaimCheck:
    claim: Claim
    ok: bool
    problems: List[str] = field(default_factory=list)


def verify_claim(claim: Claim, ledger: Ledger) -> ClaimCheck:
    problems: List[str] = []
    if not claim.evidence_ids:
        problems.append("cites no evidence")

    found: List[Evidence] = []
    for ev_id in claim.evidence_ids:
        if ev_id in ledger:
            found.append(ledger.get(ev_id))
        else:
            problems.append(f"unknown evidence id {ev_id}")

    if found:
        pool: List[float] = []
        for ev in found:
            pool.extend(evidence_numbers(ev))
        cited = ", ".join(claim.evidence_ids)
        for tok in extract_numbers(claim.text):
            if not _token_matches(tok, pool):
                problems.append(f"number '{tok.raw}' not found in cited evidence ({cited})")

    return ClaimCheck(claim=claim, ok=not problems, problems=problems)


def verify_claims(claims: List[Claim], ledger: Ledger) -> Tuple[List[Claim], List[ClaimCheck]]:
    """Split claims into (accepted, rejected-with-reasons)."""
    accepted: List[Claim] = []
    rejected: List[ClaimCheck] = []
    for claim in claims:
        check = verify_claim(claim, ledger)
        (accepted if check.ok else rejected).append(claim if check.ok else check)
    return accepted, rejected


def rejections_to_feedback(rejected: List[ClaimCheck]) -> List[Feedback]:
    """Turn failed claims into targeted feedback for the Insight agent."""
    return [
        Feedback(
            issue=f"Unsupported claim: \"{c.claim.text}\" ({'; '.join(c.problems)})",
            severity="blocker",
            target_agent="insight_agent",
            action="Rewrite using only numbers present in the cited evidence, or drop the claim.",
            evidence_ids=list(c.claim.evidence_ids),
        )
        for c in rejected
    ]
