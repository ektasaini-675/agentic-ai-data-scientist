"""
Comprehensive Academic Benchmark Suite.
Runs the Autonomous Multi-Agent AI Data Scientist across all 4 diverse domain datasets:
1. Breast Cancer Diagnostics (sample.csv)
2. UCI Heart Disease (heart_disease.csv)
3. Financial Loan Approval (loan_approval.csv)
4. Telco Customer Churn (telco_churn.csv)

Generates:
- evaluation/results/benchmark_summary.json
- evaluation/results/benchmark_table.md
- evaluation/results/benchmark_latex.tex (Ready for IEEE/ACM BTP Thesis LaTeX)

Usage:
    cmd.exe /c run.bat evaluation/benchmark_suite.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import build_graph
from agents.state import AgentState

BENCHMARK_TASKS = [
    {
        "domain": "Oncology / Pathology",
        "dataset_name": "Breast Cancer Diagnostic",
        "path": "datasets/raw/sample.csv",
        "goal": "predict whether tumor is malignant based on biometric cell measurements"
    },
    {
        "domain": "Cardiology / Clinical",
        "dataset_name": "UCI Heart Disease",
        "path": "datasets/raw/heart_disease.csv",
        "goal": "predict presence of heart disease based on clinical measurements"
    },
    {
        "domain": "Financial Risk",
        "dataset_name": "Loan Approval Underwriting",
        "path": "datasets/raw/loan_approval.csv",
        "goal": "predict loan approval status and evaluate applicant risk factors"
    },
    {
        "domain": "Telecommunications",
        "dataset_name": "Telco Customer Churn",
        "path": "datasets/raw/telco_churn.csv",
        "goal": "predict customer churn and explain key retention drivers"
    }
]


def run_benchmarks():
    print("=" * 82)
    print("IIT BTP COMPREHENSIVE MULTI-DOMAIN ACADEMIC BENCHMARK SUITE")
    print("Autonomous Multi-Agent AI Data Scientist across 4 Real-World Datasets")
    print("=" * 82 + "\n")

    os.makedirs("evaluation/results", exist_ok=True)
    app = build_graph()
    results = []

    for idx, task in enumerate(BENCHMARK_TASKS, 1):
        d_path = task["path"]
        if not os.path.exists(d_path):
            print(f"[{idx}/4] Dataset not found: {d_path}, skipping.")
            continue

        print(f"[{idx}/4] Running benchmark on '{task['dataset_name']}' ({task['domain']})...")
        initial_state: AgentState = {
            "dataset_path": d_path,
            "user_goal": task["goal"],
            "agent_trace": []
        }

        t0 = time.time()
        final_state = app.invoke(initial_state, config={"recursion_limit": 50})
        elapsed = round(time.time() - t0, 2)

        schema = final_state.get("schema_report", {})
        m_rep = final_state.get("model_report", {})
        hyp = final_state.get("hypothesis_tests", [])
        sig_count = sum(1 for t in hyp if t.get("is_significant"))

        row = {
            "domain": task["domain"],
            "dataset": task["dataset_name"],
            "rows": schema.get("n_rows", 0),
            "features": schema.get("n_cols", 0),
            "target": m_rep.get("target_column", "N/A"),
            "best_model": m_rep.get("model", "N/A"),
            "cv_accuracy": m_rep.get("cv_accuracy", "N/A"),
            "test_accuracy": m_rep.get("accuracy", "N/A"),
            "f1_score": m_rep.get("f1_score", "N/A"),
            "roc_auc": m_rep.get("roc_auc", "N/A"),
            "overfit_gap": f"{round(m_rep.get('overfit_gap', 0) * 100, 1)}%",
            "significant_hyp_tests": f"{sig_count}/{len(hyp)}",
            "critic_confidence": f"{final_state.get('confidence_score', 0)}%",
            "runtime_s": elapsed
        }
        results.append(row)
        print(f"      -> Completed in {elapsed}s | Best Model: {row['best_model']} | Test Acc: {row['test_accuracy']} | Confidence: {row['critic_confidence']}\n")

    # Save JSON summary
    with open("evaluation/results/benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Empirical Benchmark Evaluation Table Across Domains\n",
        "| Domain | Dataset | N (Rows) | D (Cols) | Best Model | 5-Fold CV Score | Holdout Acc/R² | F1-Score | ROC-AUC | Overfit Gap | Significant Tests | Confidence | Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in results:
        md_lines.append(
            f"| **{r['domain']}** | {r['dataset']} | {r['rows']:,} | {r['features']} | `{r['best_model']}` | {r['cv_accuracy']} | **{r['test_accuracy']}** | {r['f1_score']} | {r['roc_auc']} | {r['overfit_gap']} | {r['significant_hyp_tests']} | **{r['critic_confidence']}** | {r['runtime_s']}s |"
        )
    md_table = "\n".join(md_lines)
    with open("evaluation/results/benchmark_table.md", "w", encoding="utf-8") as f:
        f.write(md_table)

    # Generate LaTeX Table
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Multi-Agent Empirical Evaluation Across Benchmark Domains}",
        r"\label{tab:system_benchmarks}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccccc}",
        r"\hline",
        r"\textbf{Domain} & \textbf{Dataset} & \textbf{Instances} & \textbf{Features} & \textbf{Best Model} & \textbf{5-Fold CV} & \textbf{Holdout Acc} & \textbf{F1} & \textbf{ROC-AUC} & \textbf{Gap} & \textbf{Confidence} \\ \hline"
    ]
    for r in results:
        best_clean = r['best_model'].replace("Classifier", "").replace("Regressor", "")
        cv_clean = str(r['cv_accuracy']).replace("±", r"\pm")
        latex_lines.append(
            f"{r['domain']} & {r['dataset']} & {r['rows']:,} & {r['features']} & {best_clean} & {cv_clean} & \\textbf{{{r['test_accuracy']}}} & {r['f1_score']} & {r['roc_auc']} & {r['overfit_gap']} & \\textbf{{{r['critic_confidence']}}} \\\\"
        )
    latex_lines.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])
    latex_table = "\n".join(latex_lines)
    with open("evaluation/results/benchmark_latex.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("=" * 82)
    print("CONSOLIDATED BENCHMARK SUMMARY")
    print("=" * 82)
    print(md_table)
    print("\n" + "=" * 82)
    print("LaTeX table written to evaluation/results/benchmark_latex.tex")
    print("Markdown table written to evaluation/results/benchmark_table.md")
    print("JSON summary written to evaluation/results/benchmark_summary.json")
    print("=" * 82)


if __name__ == "__main__":
    run_benchmarks()

