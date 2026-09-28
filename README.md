# Autonomous Multi-Agent AI System for Data Analysis & Insight Generation

**B.Tech Major Project (BTPP)**  
**Team Members:** Ekta (`B23CS1018`), Kashish Joshi (`B23CS1025`), Kurra Hema (`B23CS1031`)  
**Repository:** [ektasaini-675/agentic-ai-data-scientist](https://github.com/ektasaini-675/agentic-ai-data-scientist)

---

## 📌 Project Overview
An autonomous, verifiable multi-agent data scientist system built on **LangGraph**. Given any raw dataset (CSV) and a natural-language business/analytical goal, the system:
1. **Orchestrator Agent**: Decomposes the analytical goal into ordered subtasks (supporting both heuristic planning and Claude/LLM planning).
2. **Data Agent**: Inspects schemas, missing value distributions, cardinalities, and cleans duplicates.
3. **EDA Agent**: Computes correlation matrices, detects outliers, and executes **statistical hypothesis testing** (Welch's t-test, One-Way ANOVA, Chi-Square $\chi^2$ independence tests).
4. **ML Agent**: Automates multi-model training (RandomForest, GradientBoosting, LogisticRegression with standard scaling), evaluates generalization/overfitting gaps, and identifies top predictive features.
5. **Visualization Agent**: Generates publication-ready correlation heatmaps, feature importance plots, and distributions.
6. **Critic / Validation Agent**: Performs empirical verification checks (sample-size adequacy, overfitting gap tolerance, EDA vs. ML feature consistency) and calculates a quantitative **Evidence Confidence Score (0–100%)**. Can trigger an automatic LangGraph retry loop if anomalies are detected.
7. **Insight Agent**: Synthesizes verified facts into an executive summary, ranks business recommendations, and exports professional **PDF** (`reports/executive_report.pdf`) and **HTML** reports.

---

## 🏗️ Multi-Agent Architecture
```
                         ┌────────────────────┐
                         │   User / CSV File  │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Orchestrator Agent │
                         └─────────┬──────────┘
                                   │
       ┌───────────────┬───────────┴───────────┬───────────────┐
       ▼               ▼                       ▼               ▼
┌──────────────┐┌──────────────┐       ┌──────────────┐┌──────────────┐
│  Data Agent  ││  EDA Agent   │       │   ML Agent   ││  Viz Agent   │
└──────┬───────┘└──────┬───────┘       └──────┬───────┘└──────┬───────┘
       └───────────────┼───────────────────────┘               │
                       ▼                                       │
            ┌─────────────────────┐                            │
            │    Critic Agent     │ ── [needs_revision] ───────┘ (Retry loop to EDA)
            └──────────┬──────────┘
                       │ [pass]
                       ▼
            ┌─────────────────────┐
            │    Insight Agent    │ ──→ Executive PDF & HTML Reports
            └─────────────────────┘
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ektasaini-675/agentic-ai-data-scientist.git
cd agentic-ai-data-scientist

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*(Optional)* Configure LLM planner via `.env`:
```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY=your_key_here
# Set USE_LLM_ORCHESTRATOR=1
```
*Note: The system works completely offline without any API keys required using the built-in heuristic/rule-based engine!*

---

### 2. Run the Interactive Streamlit Web UI
Launch the interactive dashboard to run analyses, inspect real-time agent traces, and download executive reports:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### 3. Run via Command Line Interface (CLI)
You can directly execute the LangGraph pipeline on any dataset:
```bash
# Breast cancer sample dataset
python graph.py datasets/raw/sample.csv "predict whether the tumor is malignant based on measurements"

# Telco customer churn dataset
python graph.py datasets/raw/telco_churn.csv "predict customer churn and explain key retention drivers"
```

---

### 4. Run the Ablation Study
Measure the empirical impact of the Critic Agent with our automated ablation script:
```bash
python evaluation/ablation_study.py datasets/raw/sample.csv "predict whether the tumor is malignant"
```
Outputs a comparison table evaluating execution time, validation passes, confidence scores, and saves `evaluation/results/ablation_results.json`.

---

## 📁 Repository Structure
```
agentic-ai-data-scientist/
├── agents/                  # Specialized autonomous agents
│   ├── state.py             # Shared TypedDict state schema
│   ├── orchestrator.py      # Rule-based task planner & router
│   ├── orchestrator_llm.py  # LLM-based task planner (Claude)
│   ├── data_agent.py        # Schema detection & data hygiene
│   ├── eda_agent.py         # Correlation, distribution & statistical tests
│   ├── ml_agent.py          # Multi-model evaluation & feature ranking
│   ├── viz_agent.py         # Automated chart generation (Matplotlib/Plotly)
│   ├── critic_agent.py      # Overfitting, sample size & confidence scoring
│   └── insight_agent.py     # Executive report builder & PDF/HTML exporter
├── datasets/                # Benchmark & raw datasets
│   └── raw/
│       ├── sample.csv       # Breast cancer diagnostic dataset (569 rows)
│       └── telco_churn.csv  # Telco customer churn dataset (7,043 rows)
├── evaluation/              # Evaluation & empirical benchmarking
│   ├── baseline_comparison.py # Multi-agent vs single-prompt LLM baseline
│   └── ablation_study.py    # With-Critic vs. Without-Critic ablation study
├── reports/                 # Auto-generated outputs (PDF, HTML, charts)
├── docs/                    # Weekly project logs & architecture records
├── app.py                   # Streamlit web application & live dashboard
├── graph.py                 # LangGraph StateGraph pipeline definition
└── requirements.txt         # Project dependencies
```

---

## 👥 Contributors
- **Ekta** (`B23CS1018`)
- **Kashish Joshi** (`B23CS1025`)
- **Kurra Hema** (`B23CS1031`)
