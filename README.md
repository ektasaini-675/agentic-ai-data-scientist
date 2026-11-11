# Autonomous Multi-Agent AI System for Tabular Data Analysis & Insight Generation

[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph%200.2+-blue.svg)](https://github.com/langchain-ai/langgraph)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Web%20UI-Streamlit-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.4+-orange.svg)](https://scikit-learn.org/)

**B.Tech Major Project (BTP Capstone - 4th Year)**  
**Department of Computer Science & Engineering**  
**Team Members:**  
- **Ekta** (`B23CS1018`)  
- **Kashish Joshi** (`B23CS1025`)  
- **Kurra Hema** (`B23CS1031`)  

**GitHub Repository:** [ektasaini-675/agentic-ai-data-scientist](https://github.com/ektasaini-675/agentic-ai-data-scientist)  

---

## 📌 Project Overview
An autonomous, verifiable multi-agent data scientist framework built on **LangGraph StateGraph**. Given any raw tabular dataset (CSV) and a natural-language business/analytical goal, the system coordinates seven specialized collaborative agents sharing a centralized, typed state schema:

1. **Orchestrator Agent**: Decomposes natural language goals into an ordered task graph (supporting both deterministic heuristic planning and Multi-LLM planning via Gemini, Claude, or OpenAI).
2. **Data Agent**: Automated data hygiene, schema detection, primary key/ID isolation, missing data mechanism diagnostics, and statistical imputation (median/mean/mode).
3. **EDA & Statistical Agent**: Computes correlation matrices, runs **normality tests (Shapiro-Wilk)**, executes inferential hypothesis tests (**Two-sample Welch's t-test, Mann-Whitney U, One-Way ANOVA, Chi-Square $\chi^2$**), calculates exact effect sizes (**Cohen's $d$, Cramér's $V$**), and evaluates multicollinearity via **Variance Inflation Factor (VIF)**.
4. **Machine Learning Agent**: Automates multi-model training (Random Forest, Gradient Boosting, Extra Trees, Logistic Regression), runs **5-Fold Stratified Cross-Validation**, handles class imbalance, quantifies generalization/overfitting gaps ($Acc_{\text{train}} - Acc_{\text{test}}$), and computes **Explainable AI (XAI)** feature importance rankings with **Directional Risk Impact (+/-)**.
5. **Visualization Agent**: Generates six publication-ready 300-DPI charts (triangular correlation heatmap, XAI feature importance, model generalization comparison, multi-panel density distributions, normalized confusion matrix, and 4-pillar critic radar chart).
6. **Critic / Validation Agent**: Implements an empirical 4-pillar verification audit (Sample Size & Statistical Power, Overfitting Gap, Inferential Hypothesis Rigor, Multicollinearity/Plausibility) and computes a quantitative **Evidence Confidence Score (0–100%)**. Emits machine-actionable `critic_directives` to trigger an automatic **LangGraph Closed-Loop Reflexion Retry** if anomalies are detected.
7. **Insight Agent**: Synthesizes verified facts into an executive summary, prioritizes actionable business recommendations, and exports professional **PDF** (`reports/executive_report.pdf`) and **HTML** (`reports/executive_report.html`) reports.

---

## 🏗️ Multi-Agent Architecture
```
                         ┌─────────────────────────────┐
                         │   Raw Dataset (CSV) + Goal  │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │     Orchestrator Agent      │
                         │   Decomposes Goal into DAG  │
                         └──────────────┬──────────────┘
                                        │
        ┌───────────────────┬───────────┴───────────┬───────────────────┐
        ▼                   ▼                       ▼                   ▼
 ┌──────────────┐    ┌──────────────┐        ┌──────────────┐    ┌──────────────┐
 │  Data Agent  │    │  EDA Agent   │        │   ML Agent   │    │  Viz Agent   │
 │Hygiene/Nulls │    │t-test/MWU/VIF│        │5-Fold CV/XAI │    │6 x 300DPI PNG│
 └──────┬───────┘    └──────┬───────┘        └──────┬───────┘    └──────┬───────┘
        └───────────────────┼───────────────────────┘                   │
                            ▼                                           │
                 ┌─────────────────────┐                                │
                 │    Critic Agent     │ ── [needs_revision] ───────────┘
                 │ 4-Pillar Validation │    (Adaptive Closed-Loop Reflexion)
                 │  Confidence Score % │
                 └──────────┬──────────┘
                            │ [pass]
                            ▼
                 ┌─────────────────────┐
                 │    Insight Agent    │ ──→ Executive PDF Report
                 │ Executive Synthesis │ ──→ Standalone HTML Report
                 └─────────────────────┘ ──→ Markdown Audit Report
```

---

## 📊 Empirical Multi-Domain Benchmark Evaluation

The framework was benchmarked across four real-world datasets spanning distinct domain challenges:

| Domain | Dataset | Instances | Features | Optimal Model | 5-Fold CV Score | Holdout Acc/R² | F1-Score | ROC-AUC | Overfit Gap | Critic Confidence | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oncology / Pathology** | Breast Cancer Diagnostic | 569 | 31 | `LogisticRegression` | 0.978 ± 0.0098 | **98.25%** | 0.9825 | 0.9954 | 0.7% | **99%** | 9.61s |
| **Cardiology / Clinical** | UCI Heart Disease | 302 | 14 | `RandomForestClassifier` | 0.8299 ± 0.0804 | **80.33%** | 0.7997 | 0.9123 | 18.4% | **94%** | 3.55s |
| **Financial Risk** | Loan Approval Underwriting | 614 | 13 | `ExtraTreesClassifier` | 0.7698 ± 0.0472 | **83.74%** | 0.8296 | 0.8497 | 2.6% | **99%** | 3.61s |
| **Telecommunications** | Telco Customer Churn | 7,043 | 21 | `GradientBoostingClassifier` | 0.8005 ± 0.0084 | **80.77%** | 0.8001 | 0.8446 | 3.1% | **99%** | 8.95s |

---

## ⚖️ Architectural Ablation Study (Critic Efficacy)

Empirical evaluation demonstrating the impact of the **Critic Agent** and **Closed-Loop Reflexion**:

| Evaluation Metric | (A) Full Adaptive System | (B) Passive Critic System | (C) Ablated Baseline |
| :--- | :--- | :--- | :--- |
| **Execution Latency** | 9.78s | 9.14s | 4.75s |
| **Agent Trace Steps** | 12 discrete steps | 10 discrete steps | 6 discrete steps |
| **Reflexion Retries** | 1 adaptive retry | 0 retries | 0 retries |
| **Validation Certification** | **Formally Verified (`pass`)** | Passive Pass | Bypassed entirely |
| **Evidence Confidence Score** | **99% Certified** | 94% Certified | **Unscored** |
| **Overfitting Gap Audited** | **Guaranteed verified $\le 12\%$** | Audited (No action) | **Silent risk of overfit** |
| **Multicollinearity Pruning** | **Active ($VIF > 10$ pruned)** | Inactive | Inactive |

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ektasaini-675/agentic-ai-data-scientist.git
cd agentic-ai-data-scientist

# Create and activate virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Web UI
```bash
# Using the preconfigured runner (Windows):
cmd.exe /c run.bat -m streamlit run app.py

# Or standard command:
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Run via Command Line Interface (CLI)
```bash
# Breast cancer diagnostic dataset
cmd.exe /c run.bat graph.py datasets/raw/sample.csv "predict whether tumor is malignant based on measurements"

# Telco customer churn dataset
cmd.exe /c run.bat graph.py datasets/raw/telco_churn.csv "predict customer churn and explain key retention drivers"
```

### 4. Run Automated Benchmark Suite
```bash
cmd.exe /c run.bat evaluation/benchmark_suite.py
```
Outputs consolidated tables, `evaluation/results/benchmark_summary.json`, and LaTeX code in `evaluation/results/benchmark_latex.tex`.

### 5. Run Architectural Ablation Study
```bash
cmd.exe /c run.bat evaluation/ablation_study.py datasets/raw/sample.csv "predict whether tumor is malignant"
```

### 6. Run Automated Unit & Integration Tests
```bash
cmd.exe /c run.bat tests/test_system.py
```

---

## 📁 Repository Structure
```
agentic-ai-data-scientist/
├── agents/                       # Specialized autonomous agents
│   ├── state.py                  # Shared TypedDict state schema
│   ├── orchestrator.py           # Heuristic task planner & state router
│   ├── orchestrator_llm.py       # Multi-LLM planner (Gemini, Claude, OpenAI)
│   ├── data_agent.py             # Schema detection & hygiene imputation
│   ├── eda_agent.py              # Inferential hypothesis testing & VIF
│   ├── ml_agent.py               # 5-fold CV, multi-model benchmark & XAI
│   ├── viz_agent.py              # 6 publication-grade 300DPI charts
│   ├── critic_agent.py           # 4-pillar empirical validation & retry loop
│   └── insight_agent.py          # Executive report builder (PDF, HTML, MD)
├── datasets/raw/                 # Real-world benchmark evaluation datasets
│   ├── sample.csv                # Breast cancer diagnostic dataset (569 rows)
│   ├── heart_disease.csv         # UCI heart disease dataset (302 rows)
│   ├── loan_approval.csv         # Loan approval underwriting dataset (614 rows)
│   └── telco_churn.csv           # Telco customer churn dataset (7,043 rows)
├── evaluation/                   # Academic empirical testing suite
│   ├── benchmark_suite.py        # 4-dataset automated benchmark runner
│   ├── ablation_study.py         # 3-way architectural ablation benchmark
│   ├── baseline_comparison.py    # Multi-agent vs single-prompt LLM evaluation
│   └── results/                  # Stored empirical findings (JSON, MD, LaTeX)
├── tests/                        # Automated unit & integration tests
│   └── test_system.py            # Complete test suite (7 passing tests)
├── reports/                      # System-generated outputs
│   ├── executive_report.pdf      # Professional executive PDF report
│   ├── executive_report.html     # Standalone interactive HTML report
│   └── charts/                   # Generated 300DPI PNG visualizations
├── docs/                         # Academic documentation & thesis
│   ├── IIT_BTP_FINAL_THESIS_REPORT.md # Comprehensive 60-page caliber BTP thesis
│   ├── BTP_DEFENSE_PRESENTATION.md    # 30-slide presentation script & viva guide
│   ├── PROJECT_UNDERSTANDING_DOCUMENT.md # Technical reference manual
│   └── weekly_log.md             # Chronological progress logs
├── app.py                        # Streamlit web application & interactive dashboard
├── graph.py                      # LangGraph StateGraph pipeline wiring
├── run.bat                       # Universal environment execution runner
└── requirements.txt              # Production dependency specifications
```

---

## 👥 Contributors
- **Ekta** (`B23CS1018`)
- **Kashish Joshi** (`B23CS1025`)
- **Kurra Hema** (`B23CS1031`)
