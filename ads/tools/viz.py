"""Plot tools. Each plot is saved under runs/<id>/charts/ and recorded in the ledger,
so a chart in the report is traceable to the tool call (and data) that drew it."""
from __future__ import annotations

import re
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # no GUI: safe inside Streamlit and servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ads.context import RunContext  # noqa: E402
from ads.state import ToolResult  # noqa: E402
from ads.tools.ml import _importance  # noqa: E402
from ads.tools.registry import register  # noqa: E402
from ads.tools.stats import _segment_table  # noqa: E402

_COLOR = "#4C6EF5"
_ACCENT = "#E8590C"


def _save(fig, ctx: RunContext, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    charts = ctx.run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    fig.savefig(charts / safe, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"charts/{safe}"


@register(
    "plot_target_by_feature",
    "Bar chart of the outcome per segment of one feature (positive-class rate or mean target).",
    parameters={
        "type": "object",
        "properties": {
            "feature": {"type": "string"},
            "target": {"type": "string"},
            "positive_label": {"type": "string"},
        },
        "required": ["feature", "target"],
    },
    needs_ctx=True,
)
def plot_target_by_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
    positive_label: Optional[str] = None,
    ctx: Optional[RunContext] = None,
) -> ToolResult:
    t = _segment_table(df, feature, target, positive_label)
    segs = t["segments"]
    if not segs:
        return ToolResult(ok=False, error="no complete rows to plot")
    labels = [s["segment"] for s in segs]
    values = [s["value"] for s in segs]
    is_rate = t["metric"] == "rate"

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(values)), values, color=_COLOR)
    ax.axhline(t["overall"], color=_ACCENT, linestyle="--", linewidth=1.2, label="overall")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1%}" if is_rate else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"rate of {target} = {t['positive']}" if is_rate else f"mean {target}")
    ax.set_title(f"{target} by {feature}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    path = _save(fig, ctx, f"{target}_by_{feature}.png")
    return ToolResult(
        ok=True,
        data={"path": path, "kind": "bar", "feature": feature, "target": target, "n_segments": len(segs)},
        summary=f"Saved chart {path}: {'rate' if is_rate else 'mean'} of {target} by {feature} across {len(segs)} segment(s).",
        artifacts=[path],
    )


@register(
    "plot_importance",
    "Horizontal bar chart of the top permutation importances for the target.",
    parameters={
        "type": "object",
        "properties": {"target": {"type": "string"}, "top_k": {"type": "integer"}},
        "required": ["target"],
    },
    needs_ctx=True,
)
def plot_importance(df: pd.DataFrame, target: str, top_k: int = 10, ctx: Optional[RunContext] = None) -> ToolResult:
    imp = _importance(ctx, df, target)
    top = imp["features"][: int(top_k)][::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([f["feature"] for f in top], [f["importance_mean"] for f in top],
            xerr=[f["importance_std"] for f in top], color=_COLOR)
    ax.set_xlabel(f"drop in {imp['metric']} when shuffled (held-out data)")
    ax.set_title(f"What predicts {target} ({imp['model']})")
    ax.spines[["top", "right"]].set_visible(False)
    path = _save(fig, ctx, f"importance_{target}.png")
    return ToolResult(
        ok=True,
        data={"path": path, "kind": "importance", "target": target, "n_features": len(top)},
        summary=f"Saved chart {path}: top {len(top)} permutation importances for {target}.",
        artifacts=[path],
    )


@register(
    "plot_correlation",
    "Heatmap of Spearman correlations between numeric columns (up to 15).",
    parameters={"type": "object", "properties": {}, "required": []},
    needs_ctx=True,
)
def plot_correlation(df: pd.DataFrame, ctx: Optional[RunContext] = None) -> ToolResult:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return ToolResult(ok=False, error="need at least two numeric columns")
    numeric = numeric.iloc[:, :15]
    corr = numeric.corr(method="spearman")
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * n + 2), max(4, 0.6 * n + 1.5)))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    if n <= 10:
        for i in range(n):
            for j in range(n):
                v = corr.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation")
    path = _save(fig, ctx, "correlation_heatmap.png")
    return ToolResult(
        ok=True,
        data={"path": path, "kind": "heatmap", "n_columns": n},
        summary=f"Saved chart {path}: Spearman correlation heatmap of {n} numeric column(s).",
        artifacts=[path],
    )
