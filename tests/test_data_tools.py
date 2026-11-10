import numpy as np
import pandas as pd
import pytest

import ads.tools  # noqa: F401  (registers tools)
from ads.context import RunContext
from ads.ledger import Ledger
from ads.tools import execute_tool, tool_schemas


def _telco_like() -> pd.DataFrame:
    n = 60
    rng = np.random.default_rng(0)
    total = [f"{x:.2f}" for x in rng.uniform(20, 3000, n)]
    total[3] = " "   # the classic Telco quirk: blank string instead of a number
    total[10] = " "
    df = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "tenure": rng.integers(1, 72, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "TotalCharges": total,
            "Churn": rng.choice(["Yes", "No"], n),
        }
    )
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one exact duplicate row


@pytest.fixture
def ctx(tmp_path):
    return RunContext(run_dir=tmp_path, ledger=Ledger(), df=_telco_like(), agent_name="data_agent")


def test_profile_finds_text_numbers_ids_and_duplicates(ctx):
    result, ev = execute_tool("profile_dataset", {}, ctx)
    assert result.ok and ev is not None
    assert result.data["text_numeric_columns"] == ["TotalCharges"]
    assert result.data["id_like_columns"] == ["customerID"]
    assert result.data["n_duplicates"] == 1
    assert result.data["total_missing"] == 2          # the two blank strings
    assert ev.created_by == "data_agent" and ev.dataset_version == "v0"


def test_repeat_call_is_served_from_the_ledger_cache(ctx):
    _, first = execute_tool("profile_dataset", {}, ctx)
    _, second = execute_tool("profile_dataset", {}, ctx)
    assert first.id == second.id and len(ctx.ledger) == 1


def test_clean_data_fixes_the_real_problems_and_bumps_version(ctx):
    result, ev = execute_tool("clean_data", {"target": "Churn"}, ctx)
    assert result.ok
    cleaned = ctx.df
    assert ctx.dataset_version == "v1" and ev.dataset_version == "v0"
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned.isna().sum().sum() == 0
    assert "customerID" not in cleaned.columns
    assert len(cleaned) == 60                            # duplicate removed
    assert result.data["duplicates_dropped"] == 1
    assert (ctx.run_dir / "data" / "v1.csv").exists()    # snapshot for the audit trail


def test_identify_target_uses_goal_words_and_skips_ids(ctx):
    result, _ = execute_tool(
        "identify_target", {"goal": "Identify the main factors affecting customer churn"}, ctx
    )
    assert result.ok
    assert result.data["target"] == "Churn"
    assert result.data["problem_type"] == "classification"
    assert result.data["confidence"] == "high"


def test_unknown_tool_and_bad_arguments_fail_gracefully(ctx):
    result, ev = execute_tool("does_not_exist", {}, ctx)
    assert not result.ok and ev is None
    result, ev = execute_tool("identify_target", {"wrong_arg": 1}, ctx)
    assert not result.ok and ev is None and len(ctx.ledger) == 0


def test_tool_schemas_match_the_anthropic_format():
    schemas = tool_schemas()
    names = {s["name"] for s in schemas}
    assert {"profile_dataset", "clean_data", "identify_target", "drop_columns"} <= names
    assert all({"name", "description", "input_schema"} <= set(s) for s in schemas)


def test_drop_columns_bumps_version_and_records_evidence(ctx):
    assert "customerID" in ctx.df.columns
    assert ctx.dataset_version == "v0"

    result, ev = execute_tool("drop_columns", {"columns": ["customerID"]}, ctx)

    assert result.ok
    assert "customerID" not in ctx.df.columns
    assert ctx.dataset_version == "v1"
    assert ev is not None
    assert ev.tool == "drop_columns"
    assert "customerID" in ev.summary
    assert (ctx.run_dir / "data" / "v1.csv").exists()
