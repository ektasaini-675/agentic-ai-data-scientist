"""Statistics tools: association tests, segment rates, correlations.

Design choices (each one fixes a flaw of the prototype):
  - The test is chosen from the variable types AND tested against the TARGET
    (the prototype tested the first binary column and arbitrary neighbours).
  - Every p-value is adjusted with Benjamini-Hochberg, because testing 30
    features at 5% would otherwise "find" 1-2 effects by pure chance.
  - Effect sizes are reported next to p-values (a tiny effect can be
    "significant" in a big dataset).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from ads.state import ToolResult
from ads.tools.registry import register

ALPHA = 0.05
_POSITIVE_WORDS = {
    "1", "yes", "y", "true", "t", "positive", "churn", "churned", "default",
    "fraud", "malignant", "disease",
}


# ---------------------------------------------------------------- helpers ---
def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_classification_target(s: pd.Series) -> bool:
    return s.nunique(dropna=True) <= 10 or not is_numeric(s)


def _norm(v: Any) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip().lower()


def positive_class(s: pd.Series) -> Any:
    """For a binary target pick the 'positive' class: a yes/true/1-like value, else the minority."""
    vals = list(s.dropna().unique())
    if len(vals) != 2:
        return None
    for v in vals:
        if _norm(v) in _POSITIVE_WORDS:
            return v
    return s.value_counts().idxmin()


def resolve_positive(y: pd.Series, label: Optional[Any]) -> Any:
    if label is None:
        return positive_class(y)
    for v in y.dropna().unique():
        if _norm(v) == _norm(label):
            return v
    raise ValueError(f"positive label '{label}' not found in the target values")


def _effect_label(x: float) -> str:
    if x < 0.1:
        return "negligible"
    if x < 0.3:
        return "small"
    if x < 0.5:
        return "medium"
    return "large"


def _bh_adjust(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, pvals[i] * m / (rank + 1))
        adjusted[i] = running
    return adjusted.tolist()


def _kruskal(groups: List[np.ndarray]):
    h, p = sps.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta = max(0.0, (h - k + 1) / (n - k)) if n > k else 0.0
    return "Kruskal-Wallis", float(h), float(p), float(eta)


def _stat_test(x: pd.Series, y: pd.Series, clf: bool):
    """One feature vs the target -> (test name, statistic, p-value, effect size)."""
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 10:
        raise ValueError("fewer than 10 complete rows")
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        raise ValueError("constant column")

    if clf:
        if is_numeric(x):
            groups = [g.to_numpy(dtype=float) for _, g in pair.groupby("y")["x"] if len(g) >= 2]
            if len(groups) < 2:
                raise ValueError("fewer than two usable classes")
            if len(groups) == 2:
                u, p = sps.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                effect = abs(1.0 - 2.0 * u / (len(groups[0]) * len(groups[1])))  # rank-biserial
                return "Mann-Whitney U", float(u), float(p), float(effect)
            return _kruskal(groups)
        if pair["x"].nunique() > 30:
            raise ValueError("too many categories")
        table = pd.crosstab(pair["x"].astype(str), pair["y"].astype(str))
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError("contingency table too small")
        chi2, p, _, _ = sps.chi2_contingency(table, correction=False)
        n = table.to_numpy().sum()
        cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
        return "chi-square", float(chi2), float(p), cramers_v

    if is_numeric(x):
        rho, p = sps.spearmanr(pair["x"].astype(float), pair["y"].astype(float))
        return "Spearman", float(rho), float(p), float(abs(rho))
    groups = [g.to_numpy(dtype=float) for _, g in pair.groupby(pair["x"].astype(str))["y"] if len(g) >= 2]
    if len(groups) < 2:
        raise ValueError("fewer than two usable groups")
    return _kruskal(groups)


# -------------------------------------------------------- test_association ---
@register(
    "test_association",
    "Test every feature against the target with the statistically appropriate test "
    "(Mann-Whitney / Kruskal-Wallis / chi-square / Spearman), with Benjamini-Hochberg "
    "multiple-comparison correction and effect sizes.",
    parameters={
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "max_features": {"type": "integer", "description": "Cap on features tested (default 40)."},
        },
        "required": ["target"],
    },
)
def test_association(df: pd.DataFrame, target: str, max_features: int = 40) -> ToolResult:
    if target not in df.columns:
        return ToolResult(ok=False, error=f"target column '{target}' not found")
    y = df[target]
    clf = is_classification_target(y)

    features = [c for c in df.columns if c != target]
    skipped: List[Dict[str, str]] = []
    if len(features) > max_features:
        skipped.extend({"feature": str(c), "reason": "over max_features cap"} for c in features[max_features:])
        features = features[:max_features]

    raw: List[Dict[str, Any]] = []
    for c in features:
        try:
            test, stat, p, effect = _stat_test(df[c], y, clf)
        except Exception as exc:
            skipped.append({"feature": str(c), "reason": str(exc)})
            continue
        if not np.isfinite(p) or not np.isfinite(effect):
            skipped.append({"feature": str(c), "reason": "test returned a non-finite value"})
            continue
        raw.append({"feature": str(c), "test": test, "statistic": stat, "p_value": p, "effect_size": effect})

    if not raw:
        return ToolResult(ok=False, error="no feature could be tested against the target")

    adjusted = _bh_adjust([r["p_value"] for r in raw])
    results: List[Dict[str, Any]] = []
    for r, p_adj in zip(raw, adjusted):
        results.append(
            {
                "feature": r["feature"],
                "test": r["test"],
                "statistic": round(r["statistic"], 4),
                "p_value": r["p_value"],
                "p_adjusted": p_adj,
                "effect_size": round(r["effect_size"], 4),
                "effect_label": _effect_label(r["effect_size"]),
                "significant": bool(p_adj < ALPHA),
            }
        )
    results.sort(key=lambda r: (r["p_adjusted"], -r["effect_size"]))

    significant = [r for r in results if r["significant"]]
    strongest = sorted(significant, key=lambda r: -r["effect_size"])[:3]
    summary = (
        f"Tested {len(results)} feature(s) against '{target}' with Benjamini-Hochberg "
        f"correction at the 5% level: {len(significant)} significant."
    )
    if strongest:
        parts = [f"{r['feature']} ({r['test']}, effect size {r['effect_size']:.2f})" for r in strongest]
        summary += " Strongest effects: " + "; ".join(parts) + "."
    if skipped:
        summary += f" Skipped {len(skipped)} feature(s)."

    data = {
        "target": target,
        "problem_type": "classification" if clf else "regression",
        "n_tests": len(results),
        "n_significant": len(significant),
        "alpha": ALPHA,
        "results": results,
        "skipped": skipped,
    }
    return ToolResult(ok=True, data=data, summary=summary)


# ------------------------------------------------------------ segment_rates ---
def _segment_table(
    df: pd.DataFrame,
    feature: str,
    target: str,
    positive: Optional[Any] = None,
    max_levels: int = 12,
    bins: int = 4,
) -> Dict[str, Any]:
    """Outcome per segment of a feature: positive-class rate (classification) or mean (regression)."""
    if feature not in df.columns:
        raise ValueError(f"feature '{feature}' not found")
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found")
    y, x = df[target], df[feature]

    if is_classification_target(y):
        pos = resolve_positive(y, positive)
        if pos is None:
            raise ValueError("target has more than two classes; pass positive_label")
        outcome = (y == pos).astype(float)
        metric = "rate"
    else:
        pos = None
        outcome = pd.to_numeric(y, errors="coerce")
        metric = "mean"

    binned, order = False, None
    if is_numeric(x) and x.nunique(dropna=True) > 8:
        cat = pd.qcut(x, q=bins, duplicates="drop")
        order = [str(c) for c in cat.cat.categories]
        seg = cat.astype(str).where(x.notna())
        binned = True
    else:
        seg = x.astype(str).where(x.notna())

    valid = y.notna()
    d = pd.DataFrame({"seg": seg, "o": outcome})[valid].dropna()
    g = d.groupby("seg")["o"].agg(value="mean", n="count").reset_index()
    if binned:
        position = {s: i for i, s in enumerate(order)}
        g["_ord"] = g["seg"].map(position)
        g = g.sort_values("_ord").drop(columns="_ord")
    else:
        g = g.sort_values("n", ascending=False).head(max_levels)

    segments = [
        {"segment": str(r["seg"]), "n": int(r["n"]), "value": float(r["value"])} for r in g.to_dict("records")
    ]
    return {
        "metric": metric,
        "positive": None if pos is None else str(pos),
        "overall": float(outcome[valid].mean()),
        "segments": segments,
        "binned": binned,
    }


@register(
    "segment_rates",
    "Outcome by segment of one feature: the positive-class rate per category (classification) "
    "or mean target (regression). Numeric features are split into quartile bins.",
    parameters={
        "type": "object",
        "properties": {
            "feature": {"type": "string"},
            "target": {"type": "string"},
            "positive_label": {"type": "string", "description": "Which class counts as positive (default: inferred)."},
            "bins": {"type": "integer", "description": "Quantile bins for numeric features (default 4)."},
        },
        "required": ["feature", "target"],
    },
)
def segment_rates(
    df: pd.DataFrame, feature: str, target: str, positive_label: Optional[str] = None, bins: int = 4
) -> ToolResult:
    t = _segment_table(df, feature, target, positive_label, bins=int(bins))
    segs = t["segments"]
    if not segs:
        return ToolResult(ok=False, error="no complete rows for this feature and target")
    hi = max(segs, key=lambda s: s["value"])
    lo = min(segs, key=lambda s: s["value"])
    ratio = hi["value"] / lo["value"] if lo["value"] > 0 and len(segs) > 1 else None

    if t["metric"] == "rate":
        summary = (
            f"Rate of '{target}' = '{t['positive']}' by {feature}: highest is {hi['segment']} at "
            f"{hi['value']:.1%} (n={hi['n']:,}), lowest is {lo['segment']} at {lo['value']:.1%} "
            f"(n={lo['n']:,}); overall {t['overall']:.1%}; {len(segs)} segment(s)."
        )
        if ratio is not None:
            summary += f" Highest-to-lowest ratio is {ratio:.1f}x."
    else:
        summary = (
            f"Mean '{target}' by {feature}: highest is {hi['segment']} at {hi['value']:.2f} "
            f"(n={hi['n']:,}), lowest is {lo['segment']} at {lo['value']:.2f} (n={lo['n']:,}); "
            f"overall mean {t['overall']:.2f}; {len(segs)} segment(s)."
        )

    data = {
        "feature": feature,
        "target": target,
        **t,
        "highest": hi,
        "lowest": lo,
        "ratio_highest_to_lowest": ratio,
    }
    return ToolResult(ok=True, data=data, summary=summary)


# --------------------------------------------------------- top_correlations ---
@register(
    "top_correlations",
    "Strongest Spearman rank correlations between numeric columns; flags near-duplicate "
    "(collinear) pairs with |rho| >= 0.90.",
    parameters={"type": "object", "properties": {"k": {"type": "integer"}}, "required": []},
)
def top_correlations(df: pd.DataFrame, k: int = 5) -> ToolResult:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return ToolResult(
            ok=True,
            data={"n_numeric": int(numeric.shape[1]), "pairs": [], "n_high": 0},
            summary="Fewer than two numeric columns, so no correlations were computed.",
        )
    corr = numeric.corr(method="spearman")
    cols = list(corr.columns)
    iu = np.triu_indices(len(cols), k=1)
    pairs = [
        {"a": str(cols[i]), "b": str(cols[j]), "rho": float(corr.iat[i, j])}
        for i, j in zip(*iu)
        if np.isfinite(corr.iat[i, j])
    ]
    pairs.sort(key=lambda p: -abs(p["rho"]))
    n_high = sum(1 for p in pairs if abs(p["rho"]) >= 0.9)
    top = pairs[: int(k)]
    parts = [f"{p['a']} vs {p['b']} ({p['rho']:.2f})" for p in top]
    summary = (
        f"Strongest rank correlations among {len(cols)} numeric columns: "
        + ("; ".join(parts) if parts else "none")
        + f". {n_high} pair(s) have absolute correlation of at least 0.90."
    )
    data = {"n_numeric": len(cols), "pairs": top, "n_high": n_high, "method": "spearman"}
    return ToolResult(ok=True, data=data, summary=summary)
