import json

import numpy as np
import pandas as pd

import ads.tools  # noqa: F401
from ads.tools import execute_tool


def test_train_cv_reports_honest_metrics(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    result, ev = execute_tool("train_cv", {"target": "Churn"}, ctx)
    assert result.ok, result.error
    r = result.data
    assert r["problem_type"] == "classification" and r["positive_class"] == "Yes"
    assert r["best_model"] in r["models"]
    assert r["cv_best"]["roc_auc"] > 0.65 and r["test"]["roc_auc"] > 0.6
    assert r["n_train"] + r["n_test"] == r["n_rows_used"]
    assert "majority class" in result.summary
    json.dumps(ev.result)                                  # ledger entry is JSON-safe


def test_second_call_reuses_the_fitted_model(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    execute_tool("train_cv", {"target": "Churn"}, ctx)
    n_cached = len(ctx.cache)
    execute_tool("permutation_importance", {"target": "Churn", "n_repeats": 3}, ctx)
    # importance added one cache entry (its own) but did not refit the model
    assert len(ctx.cache) == n_cached + 1


def test_permutation_importance_ranks_real_drivers_first(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    result, _ = execute_tool("permutation_importance", {"target": "Churn", "n_repeats": 3}, ctx)
    assert result.ok, result.error
    names = [f["feature"] for f in result.data["features"]]
    assert names[0] in {"Contract", "tenure"}
    assert names.index("noise") > 1


def test_detect_leakage_flags_a_copy_of_the_target(churn_df_leaky, make_ctx):
    result, _ = execute_tool("detect_leakage", {"target": "Churn"}, make_ctx(churn_df_leaky))
    assert result.ok, result.error
    assert result.data["suspects"] == ["ChurnFlag"]
    assert "ChurnFlag" in result.summary


def test_detect_leakage_raises_no_false_alarm_on_clean_data(churn_df, make_ctx):
    result, _ = execute_tool("detect_leakage", {"target": "Churn"}, make_ctx(churn_df))
    assert result.ok, result.error
    assert result.data["suspects"] == []


def test_excluding_the_leaky_column_deflates_the_score(churn_df_leaky, make_ctx):
    ctx = make_ctx(churn_df_leaky)
    leaky, _ = execute_tool("train_cv", {"target": "Churn"}, ctx)
    honest, _ = execute_tool("train_cv", {"target": "Churn", "exclude": ["ChurnFlag"]}, ctx)
    assert leaky.data["test"]["roc_auc"] > 0.99
    assert honest.data["test"]["roc_auc"] < 0.9


def test_regression_training(make_ctx):
    rng = np.random.default_rng(5)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {"x1": x1, "x2": x2, "grp": rng.choice(["a", "b"], n), "price": 3 * x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n)}
    )
    result, _ = execute_tool("train_cv", {"target": "price"}, make_ctx(df))
    assert result.ok, result.error
    assert result.data["problem_type"] == "regression"
    assert result.data["test"]["r2"] > 0.8
    assert "R2" in result.summary
