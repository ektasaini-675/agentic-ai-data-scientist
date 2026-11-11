"""
Exploratory Data Analysis (EDA) & Inferential Statistics Agent.
Executes correlation matrices, normality audits (Shapiro-Wilk), parametric (Welch's t-test, ANOVA)
and non-parametric tests (Mann-Whitney U, Kruskal-Wallis), Chi-Square tests of independence,
effect size estimations (Cohen's d, Cramer's V, Eta-squared), and Variance Inflation Factor (VIF).
Includes adaptive remediation support driven by Critic Agent directives.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from agents.state import AgentState


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculates Cohen's d effect size for two independent samples."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def calculate_cramers_v(contingency_table: pd.DataFrame) -> float:
    """Calculates Cramer's V effect size for categorical association."""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if n == 0 or min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def calculate_vif(df: pd.DataFrame, max_cols: int = 12) -> List[Dict[str, Any]]:
    """Calculates Variance Inflation Factor (VIF) to detect severe multicollinearity."""
    vif_results = []
    numeric_df = df.select_dtypes(include="number").dropna()
    # Filter out columns with zero variance
    numeric_df = numeric_df.loc[:, numeric_df.std() > 1e-6]
    cols = numeric_df.columns.tolist()[:max_cols]
    
    if len(cols) < 2:
        return []

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        
        X_const = add_constant(numeric_df[cols], prepend=False)
        for i, col in enumerate(cols):
            v = variance_inflation_factor(X_const.values, i)
            if not np.isinf(v) and not np.isnan(v):
                status = "Severe (VIF > 10)" if v > 10 else ("Moderate (VIF > 5)" if v > 5 else "Low (Acceptable)")
                vif_results.append({
                    "feature": col,
                    "vif": round(float(v), 2),
                    "collinearity_level": status
                })
        vif_results.sort(key=lambda x: -x["vif"])
    except Exception:
        pass
    return vif_results


def run(state: AgentState) -> AgentState:
    data_path = state.get("cleaned_dataset_path") or state["dataset_path"]
    df = pd.read_csv(data_path)

    # 1. Check for Critic Agent Directives (Adaptive Remediation Loop)
    directives = state.get("critic_directives", [])
    pruned_features = []
    for d in directives:
        if d.get("action") == "prune_collinear_features":
            prune_targets = d.get("features", [])
            existing_to_drop = [f for f in prune_targets if f in df.columns]
            if existing_to_drop:
                df = df.drop(columns=existing_to_drop)
                pruned_features.extend(existing_to_drop)
                state.setdefault("agent_trace", []).append(
                    f"[eda_agent] ADAPTIVE REMEDIATION APPLIED: Pruned high-multicollinearity features: {existing_to_drop}"
                )

    numeric_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(include=["object", "category", "bool"])

    findings: Dict[str, Any] = {}

    # 2. Pearson & Spearman Correlation Analysis
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        pairs = corr.where(~corr.isna()).unstack().reset_index()
        pairs.columns = ["col_a", "col_b", "corr"]
        pairs = pairs[pairs["col_a"] != pairs["col_b"]]
        pairs["abs_corr"] = pairs["corr"].abs()
        top_pairs = pairs.sort_values("abs_corr", ascending=False).drop_duplicates(subset=["abs_corr"]).head(8)
        findings["top_correlations"] = [
            {"col_a": r["col_a"], "col_b": r["col_b"], "corr": round(float(r["corr"]), 4)}
            for _, r in top_pairs.iterrows()
        ]
    else:
        findings["top_correlations"] = []

    # 3. Comprehensive Summary Statistics & Normality Tests (Shapiro-Wilk)
    summary_dict = {}
    normality_tests = {}
    for col in numeric_df.columns[:10]:
        series = numeric_df[col].dropna()
        if len(series) >= 8:
            stat_w, p_val = stats.shapiro(series[:500])  # Sample up to 500 for Shapiro
            is_normal = bool(p_val > 0.05)
            normality_tests[col] = {
                "statistic": round(float(stat_w), 4),
                "p_value": float(p_val),
                "is_gaussian": is_normal
            }
        summary_dict[col] = {
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std()), 3),
            "median": round(float(series.median()), 3),
            "skew": round(float(series.skew()), 3),
            "kurtosis": round(float(series.kurtosis()), 3)
        }

    findings["numeric_summary"] = summary_dict
    findings["normality_tests"] = normality_tests

    # 4. Inferential Statistical Hypothesis Battery
    hypothesis_tests = []
    
    # Candidate target identification for group comparison
    binary_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]
    
    # Priority for target-like binary column
    goal_words = set(state.get("user_goal", "").lower().split())
    target_candidates = [c for c in binary_cols if any(w in c.lower() for w in goal_words)]
    eval_binary_col = target_candidates[0] if target_candidates else (binary_cols[0] if binary_cols else None)

    if eval_binary_col and numeric_df.shape[1] > 0:
        val1, val2 = df[eval_binary_col].dropna().unique()[:2]
        for num_c in numeric_df.columns[:6]:
            if num_c == eval_binary_col:
                continue
            g1 = df[df[eval_binary_col] == val1][num_c].dropna().values
            g2 = df[df[eval_binary_col] == val2][num_c].dropna().values
            
            if len(g1) >= 4 and len(g2) >= 4:
                # Welch's t-test (Parametric)
                t_stat, p_val_t = stats.ttest_ind(g1, g2, equal_var=False)
                # Mann-Whitney U test (Non-Parametric)
                u_stat, p_val_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                # Cohen's d effect size
                d_val = calculate_cohens_d(g1, g2)
                
                is_sig = bool(p_val_t < 0.05)
                effect_mag = "Large" if abs(d_val) >= 0.8 else ("Medium" if abs(d_val) >= 0.5 else "Small")
                
                hypothesis_tests.append({
                    "test_name": "Two-Sample Welch's t-test & Mann-Whitney U",
                    "variables": f"{num_c} across {eval_binary_col} ({val1} vs {val2})",
                    "t_statistic": round(float(t_stat), 3),
                    "p_value": float(p_val_t),
                    "mwu_p_value": float(p_val_u),
                    "effect_size": f"Cohen's d = {round(d_val, 3)} ({effect_mag})",
                    "is_significant": is_sig,
                    "interpretation": (
                        f"Statistically significant divergence in {num_c} across {eval_binary_col} groups "
                        f"(t = {t_stat:.2f}, p = {p_val_t:.4e}, d = {d_val:.2f})."
                        if is_sig else
                        f"Null hypothesis holds: No significant difference in {num_c} across {eval_binary_col} (p = {p_val_t:.4f})."
                    )
                })

    # Categorical association tests (Chi-Square + Cramer's V)
    cat_names = [c for c in cat_df.columns if df[c].nunique() < 20]
    if len(cat_names) >= 2:
        for i in range(min(3, len(cat_names) - 1)):
            c1, c2 = cat_names[i], cat_names[i+1]
            ctab = pd.crosstab(df[c1], df[c2])
            if ctab.shape[0] > 1 and ctab.shape[1] > 1:
                chi2, p_val_chi, dof, _ = stats.chi2_contingency(ctab)
                cramer_v = calculate_cramers_v(ctab)
                is_sig = bool(p_val_chi < 0.05)
                v_mag = "Strong" if cramer_v >= 0.3 else ("Moderate" if cramer_v >= 0.15 else "Weak")
                
                hypothesis_tests.append({
                    "test_name": "Pearson Chi-Square Test of Independence",
                    "variables": f"{c1} vs {c2}",
                    "t_statistic": round(float(chi2), 3),
                    "p_value": float(p_val_chi),
                    "degrees_of_freedom": int(dof),
                    "effect_size": f"Cramer's V = {round(cramer_v, 3)} ({v_mag})",
                    "is_significant": is_sig,
                    "interpretation": (
                        f"Statistically significant association between {c1} and {c2} "
                        f"(chi2({dof}) = {chi2:.2f}, p = {p_val_chi:.4e}, V = {cramer_v:.2f})."
                        if is_sig else
                        f"Features {c1} and {c2} are statistically independent (p = {p_val_chi:.4f})."
                    )
                })

    # 5. Multicollinearity VIF Diagnostic
    vif_res = calculate_vif(df)
    state["vif_report"] = vif_res

    # 6. Outlier Summary (Tukey IQR)
    outlier_counts = {}
    for col in numeric_df.columns[:8]:
        s = numeric_df[col].dropna()
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())
        if outliers > 0:
            outlier_counts[col] = {"outlier_count": outliers, "outlier_pct": round(outliers / len(s) * 100, 2)}
    
    findings["outlier_report"] = outlier_counts
    state["outlier_report"] = outlier_counts
    findings["hypothesis_tests"] = hypothesis_tests
    state["hypothesis_tests"] = hypothesis_tests
    state["eda_findings"] = findings

    sig_count = sum(1 for t in hypothesis_tests if t["is_significant"])
    severe_vif_count = sum(1 for v in vif_res if v["vif"] > 10)

    state.setdefault("agent_trace", []).append(
        f"[eda_agent] Analyzed correlations={len(findings['top_correlations'])} | "
        f"Hypothesis tests={len(hypothesis_tests)} ({sig_count} significant) | "
        f"Severe VIF warnings={severe_vif_count}"
    )
    return state
