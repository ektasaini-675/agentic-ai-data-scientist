from ads.ledger import Ledger
from ads.state import Claim
from ads.verify import extract_numbers, rejections_to_feedback, verify_claim, verify_claims


def _ledger():
    ledger = Ledger()
    ledger.add(
        "segment_rates",
        {"column": "Contract"},
        {"Month-to-month": 0.42748, "Two year": 0.0283, "n": 7043, "p_value": 1.2e-12},
        "Churn rate by Contract: month-to-month 42.7%, two year 2.8%.",
    )
    return ledger


def test_matching_percent_passes_in_both_notations():
    ledger = _ledger()
    assert verify_claim(Claim("Month-to-month customers churn at 42.7%.", ["E0001"]), ledger).ok
    assert verify_claim(Claim("Month-to-month customers churn at 43%.", ["E0001"]), ledger).ok


def test_wrong_number_is_rejected():
    check = verify_claim(Claim("Month-to-month customers churn at 47%.", ["E0001"]), _ledger())
    assert not check.ok
    assert "47%" in check.problems[0]


def test_unknown_evidence_id_is_rejected():
    check = verify_claim(Claim("Churn is high.", ["E0099"]), _ledger())
    assert not check.ok and "unknown evidence id E0099" in check.problems[0]


def test_claim_without_evidence_is_rejected_even_without_numbers():
    check = verify_claim(Claim("Churn is high.", []), _ledger())
    assert not check.ok and "cites no evidence" in check.problems


def test_thousands_separator_and_integer_counts():
    assert verify_claim(Claim("The data has 7,043 rows.", ["E0001"]), _ledger()).ok
    assert not verify_claim(Claim("The data has 7,100 rows.", ["E0001"]), _ledger()).ok


def test_p_value_upper_bound_matches_tiny_p():
    assert verify_claim(Claim("The difference is significant (p < 0.001).", ["E0001"]), _ledger()).ok


def test_small_bare_integers_are_ignored():
    assert extract_numbers("The top 3 drivers matter.") == []
    assert extract_numbers("Churn is 5% higher.")[0].is_percent


def test_ids_inside_text_are_not_mistaken_for_numbers():
    assert extract_numbers("See E0017 for details.") == []


def test_verify_claims_splits_and_builds_feedback():
    ledger = _ledger()
    good = Claim("Two-year contracts churn at 2.8%.", ["E0001"])
    bad = Claim("Two-year contracts churn at 9.9%.", ["E0001"])
    accepted, rejected = verify_claims([good, bad], ledger)
    assert accepted == [good] and len(rejected) == 1
    fb = rejections_to_feedback(rejected)[0]
    assert fb.target_agent == "insight_agent" and fb.severity == "blocker"
