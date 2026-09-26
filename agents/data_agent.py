"""
Data Agent — inspects and cleans the dataset.
This one is REAL (not a stub): it actually loads the CSV and reports on it.
"""
import pandas as pd
from agents.state import AgentState


def run(state: AgentState) -> AgentState:
    df = pd.read_csv(state["dataset_path"])

    schema_report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "n_duplicates": int(df.duplicated().sum()),
    }

    cleaning_log = []
    if schema_report["n_duplicates"] > 0:
        cleaning_log.append(f"Found {schema_report['n_duplicates']} duplicate rows (flagged, not auto-dropped).")
    missing_cols = {c: v for c, v in schema_report["missing_values"].items() if v > 0}
    if missing_cols:
        cleaning_log.append(f"Missing values detected in columns: {missing_cols}")
    else:
        cleaning_log.append("No missing values detected.")

    state["schema_report"] = schema_report
    state["cleaning_log"] = cleaning_log
    state.setdefault("agent_trace", []).append(
        f"[data_agent] rows={schema_report['n_rows']} cols={schema_report['n_cols']} "
        f"missing_cols={list(missing_cols.keys())}"
    )
    return state
