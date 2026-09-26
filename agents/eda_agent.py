"""EDA / Statistical Agent — finds patterns in the cleaned data."""
import pandas as pd
from agents.state import AgentState


def run(state: AgentState) -> AgentState:
    df = pd.read_csv(state["dataset_path"])
    numeric_df = df.select_dtypes(include="number")

    findings = {}
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        # top 3 absolute correlations excluding self-correlation
        pairs = (
            corr.where(~corr.isna())
            .unstack()
            .reset_index()
        )
        pairs.columns = ["col_a", "col_b", "corr"]
        pairs = pairs[pairs["col_a"] != pairs["col_b"]]
        pairs["abs_corr"] = pairs["corr"].abs()
        top = pairs.sort_values("abs_corr", ascending=False).head(3)
        findings["top_correlations"] = top[["col_a", "col_b", "corr"]].to_dict("records")
    else:
        findings["top_correlations"] = []

    findings["numeric_summary"] = numeric_df.describe().to_dict()

    state["eda_findings"] = findings
    state.setdefault("agent_trace", []).append(
        f"[eda_agent] top_correlations={findings['top_correlations']}"
    )
    return state
