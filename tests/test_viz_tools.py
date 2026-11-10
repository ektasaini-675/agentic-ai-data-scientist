import pytest

import ads.tools  # noqa: F401
from ads.tools import execute_tool


@pytest.mark.parametrize(
    "tool,args",
    [
        ("plot_target_by_feature", {"feature": "Contract", "target": "Churn"}),
        ("plot_target_by_feature", {"feature": "tenure", "target": "Churn"}),
        ("plot_importance", {"target": "Churn"}),
        ("plot_correlation", {}),
    ],
)
def test_plots_are_saved_and_recorded_in_the_ledger(tool, args, churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    result, ev = execute_tool(tool, args, ctx)
    assert result.ok, result.error
    assert (ctx.run_dir / result.data["path"]).exists()
    assert ev.tool == tool and result.data["path"] in ev.summary
    assert "ctx" not in ev.args                      # runtime handle never leaks into the ledger
