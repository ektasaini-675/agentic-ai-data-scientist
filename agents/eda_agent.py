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
        top = pairs.sort_values("abs_corr", ascending=False).head(5)
        findings["top_correlations"] = top[["col_a", "col_b", "corr"]].to_dict("records")
    else:
        findings["top_correlations"] = []

    findings["numeric_summary"] = numeric_df.describe().to_dict()

    # --- Automated Hypothesis Testing (t-tests, ANOVA, Chi-Square) ---
    hypothesis_tests = []
    try:
        from scipy import stats
        # Check if there is a target or binary column for two-sample testing
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = numeric_df.columns.tolist()

        # Check candidate target or first binary column
        binary_cols = [c for c in df.columns if df[c].nunique() == 2]
        if binary_cols and num_cols:
            bin_col = binary_cols[0]
            val1, val2 = df[bin_col].dropna().unique()
            for num_c in num_cols[:4]:
                if num_c == bin_col:
                    continue
                g1 = df[df[bin_col] == val1][num_c].dropna().values
                g2 = df[df[bin_col] == val2][num_c].dropna().values
                if len(g1) > 2 and len(g2) > 2:
                    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                    if not pd.isna(p_val):
                        is_sig = bool(p_val < 0.05)
                        hypothesis_tests.append({
                            "test_name": "Two-Sample Welch's t-test",
                            "variables": f"{num_c} across {bin_col} ({val1} vs {val2})",
                            "statistic": float(t_stat),
                            "p_value": float(p_val),
                            "is_significant": is_sig,
                            "interpretation": f"Mean {num_c} differs significantly across {bin_col} categories (p = {p_val:.4e} < 0.05)." if is_sig else f"No statistically significant difference in {num_c} across {bin_col} (p = {p_val:.4f})."
                        })

        # Chi-Square test between categorical columns
        if len(cat_cols) >= 2:
            for i in range(min(3, len(cat_cols) - 1)):
                c1, c2 = cat_cols[i], cat_cols[i+1]
                tab = pd.crosstab(df[c1], df[c2])
                if tab.shape[0] > 1 and tab.shape[1] > 1:
                    chi2, p_val, _, _ = stats.chi2_contingency(tab)
                    is_sig = bool(p_val < 0.05)
                    hypothesis_tests.append({
                        "test_name": "Chi-Square Test of Independence",
                        "variables": f"{c1} vs {c2}",
                        "statistic": float(chi2),
                        "p_value": float(p_val),
                        "is_significant": is_sig,
                        "interpretation": f"Significant dependency detected between {c1} and {c2} (χ² = {chi2:.2f}, p = {p_val:.4e})." if is_sig else f"Independent: No association detected between {c1} and {c2}."
                    })
    except Exception as e:
        pass

    findings["hypothesis_tests"] = hypothesis_tests
    state["hypothesis_tests"] = hypothesis_tests
    state["eda_findings"] = findings
    sig_count = sum(1 for t in hypothesis_tests if t["is_significant"])
    state.setdefault("agent_trace", []).append(
        f"[eda_agent] top_correlations={len(findings['top_correlations'])} pairs, "
        f"hypothesis_tests={len(hypothesis_tests)} ({sig_count} significant at alpha=0.05)"
    )
    return state
