"""Visualization Agent — produces a handful of relevant charts, not every possible one."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agents.state import AgentState


def run(state: AgentState) -> AgentState:
    df = pd.read_csv(state["dataset_path"])
    numeric_df = df.select_dtypes(include="number")
    out_dir = "reports/charts"
    os.makedirs(out_dir, exist_ok=True)
    chart_paths = []

    # Only draw a correlation heatmap if there's enough numeric structure to justify it
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        corr = numeric_df.corr(numeric_only=True)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        fig.colorbar(im)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        path = os.path.join(out_dir, "correlation_heatmap.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        chart_paths.append(path)

    # distribution of first numeric column as a sanity-check plot
    if numeric_df.shape[1] >= 1:
        col = numeric_df.columns[0]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(numeric_df[col].dropna(), bins=30)
        ax.set_title(f"Distribution of {col}")
        path = os.path.join(out_dir, f"dist_{col}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        chart_paths.append(path)

    state["chart_paths"] = chart_paths
    state.setdefault("agent_trace", []).append(f"[viz_agent] generated {len(chart_paths)} charts")
    return state
