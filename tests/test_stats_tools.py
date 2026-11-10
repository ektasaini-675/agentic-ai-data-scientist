import numpy as np
import pandas as pd
import pytest

import ads.tools  # noqa: F401
from ads.state import Claim
from ads.tools import execute_tool
from ads.tools.stats import _bh_adjust
from ads.verify import verify_claim


def test_bh_adjust_matches_hand_calculation():
    assert _bh_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_association_finds_real_drivers_and_ignores_noise(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    result, ev = execute_tool("test_association", {"target": "Churn"}, ctx)
    assert result.ok, result.error
    by = {r["feature"]: r for r in result.data["results"]}
    assert by["Contract"]["significant"] and by["tenure"]["significant"]
    assert by["noise"]["effect_size"] < 0.15 and by["city"]["effect_size"] < 0.15
    assert result.data["n_tests"] == 4
    assert "Tested 4 feature(s)" in result.summary
    assert ev.tool == "test_association"


def test_association_uses_spearman_for_a_numeric_target(make_ctx):
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "price": 5 * x1 + rng.normal(size=n)})
    result, _ = execute_tool("test_association", {"target": "price"}, make_ctx(df))
    by = {r["feature"]: r for r in result.data["results"]}
    assert result.data["problem_type"] == "regression"
    assert by["x1"]["test"] == "Spearman" and by["x1"]["significant"]
    assert by["x1"]["effect_size"] > 0.8


def test_missing_target_is_a_clean_failure(churn_df, make_ctx):
    result, ev = execute_tool("test_association", {"target": "nope"}, make_ctx(churn_df))
    assert not result.ok and ev is None


def test_segment_rates_ranks_contracts(churn_df, make_ctx):
    result, _ = execute_tool("segment_rates", {"feature": "Contract", "target": "Churn"}, make_ctx(churn_df))
    d = result.data
    assert d["positive"] == "Yes"
    assert d["highest"]["segment"] == "Month-to-month"
    assert 0 < d["overall"] < 1 and len(d["segments"]) == 3
    assert "Month-to-month" in result.summary


def test_numeric_feature_is_binned_into_quartiles(churn_df, make_ctx):
    result, _ = execute_tool("segment_rates", {"feature": "tenure", "target": "Churn"}, make_ctx(churn_df))
    segs = result.data["segments"]
    assert result.data["binned"] and len(segs) == 4
    assert segs[0]["value"] > segs[-1]["value"]          # short-tenure customers churn more


def test_a_claim_built_from_a_tool_result_verifies_and_a_wrong_one_does_not(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    result, ev = execute_tool("segment_rates", {"feature": "Contract", "target": "Churn"}, ctx)
    rate = result.data["highest"]["value"]
    good = Claim(f"Month-to-month customers churn at {rate:.1%}.", [ev.id])
    bad = Claim(f"Month-to-month customers churn at {rate + 0.10:.1%}.", [ev.id])
    assert verify_claim(good, ctx.ledger).ok
    assert not verify_claim(bad, ctx.ledger).ok


def test_top_correlations_finds_the_near_duplicate_pair(make_ctx):
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(scale=0.05, size=200), "c": rng.normal(size=200)})
    result, _ = execute_tool("top_correlations", {}, make_ctx(df))
    top = result.data["pairs"][0]
    assert {top["a"], top["b"]} == {"a", "b"} and top["rho"] > 0.95
    assert result.data["n_high"] >= 1
