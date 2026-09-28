# Autonomous Multi-Agent AI Data Scientist: Simple Project Guide

**Project Name:** Autonomous Multi-Agent AI System for Automated Data Analysis and Insight Generation  
**Team Members:** Ekta (`B23CS1018`), Kashish Joshi (`B23CS1025`), Kurra Hema (`B23CS1031`)  
**Repository Branch:** [`kashish-dev`](https://github.com/ektasaini-675/agentic-ai-data-scientist/tree/kashish-dev)

---

## 1. Refined Problem Statement

### The Problem in the Real World
Organizations collect massive amounts of tabular data (spreadsheets, databases, customer logs). However, deriving trustworthy insights from raw data currently suffers from two major extremes:

1. **The Manual Bottleneck:** Traditional data analysis requires human data scientists to write repetitive code for data cleaning, calculating correlations, testing statistical significance, training models, and formatting slides. This process takes days or weeks.
2. **The "Single-Prompt LLM" Flaw (ChatGPT / Claude alone):** When users upload a raw CSV into a chatbot:
   - **Numerical Hallucination:** Chatbots frequently invent numbers, p-values, and correlation coefficients because they generate text rather than running code.
   - **No True Machine Learning:** Chatbots cannot train actual machine learning models on a computer to benchmark accuracy or feature weights.
   - **Silent Overfitting:** A single LLM cannot detect if a model memorized the training data or if sample sizes were too small to draw statistically valid conclusions.
   - **Token Limits & Privacy:** Large datasets exceed conversational memory and risk leaking sensitive enterprise data to external servers.

### Our Solution
We built an **Autonomous Multi-Agent AI System** that automates the complete data science lifecycle. Instead of asking one AI to do everything, our system acts like a **virtual data science team of 7 specialized AI agents**. 

Every computation (correlations, statistical tests, model training, overfitting audits) is performed **deterministically in real Python code** using established mathematical libraries. The AI orchestrates the workflow, self-corrects via a **Critic retry loop**, and delivers verified, publication-ready executive reports with an **Evidence Confidence Score**.

---

## 2. Types of Data Analysis Performed

Our system executes all 5 core tiers of modern data analysis:

```
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Descriptive  ── What happened? (Summary & Hygiene)       │
  │ 2. Diagnostic   ── Why did it happen? (Correlations)       │
  │ 3. Inferential  ── Is it statistically real? (p < 0.05)     │
  │ 4. Predictive   ── What will happen next? (ML Classifiers)  │
  │ 5. Prescriptive ── What action should we take? (Strategy)   │
  └─────────────────────────────────────────────────────────────┘
```

### 1. Descriptive Analysis (*"What does the data look like?"*)
- **Schema & Type Detection:** Distinguishes numerical, categorical, and identifier columns.
- **Hygiene Auditing:** Automatically scans for missing values, null rates, and duplicate rows.
- **Distribution Profiles:** Measures means, medians, standard deviations, and value ranges.

### 2. Diagnostic Analysis (*"What relationships exist in the data?"*)
- **Pearson Correlation Matrix:** Identifies positive and negative relationships between features (e.g., in heart disease, chest pain type correlates $+0.43$ with disease presence).
- **Outlier & Multicollinearity Detection:** Finds extreme anomalies that could skew findings.

### 3. Inferential / Statistical Hypothesis Testing (*"Are the patterns genuine or pure chance?"*)
- **Two-Sample Welch's t-test:** Compares continuous metrics across binary groups without assuming equal variances.
- **One-Way ANOVA ($F$-statistic):** Tests variance across multi-class target groups.
- **Chi-Square ($\chi^2$) Test of Independence:** Builds contingency tables to detect statistically significant relationships between categorical features (e.g., testing whether customer churn is independent of contract type).
- **Exact p-value verification:** Only findings meeting the scientific threshold ($\alpha = 0.05$) are confirmed as statistically significant.

### 4. Predictive Analysis (*"What will happen in future cases?"*)
- **Multi-Model Candidate Pool:** Concurrently trains and benchmarks:
  - **Random Forest Classifier** (Ensemble bagging trees)
  - **Gradient Boosting Classifier** (Sequential boosting)
  - **Logistic Regression** (Linear baseline with standard scaling)
- **Holdout Evaluation:** Splits data into 80% training / 20% testing sets to measure holdout Accuracy and F1-Score.
- **Feature Importance (Drivers):** Extracts Gini importance scores to rank the most decisive predictive factors.
- **Overfitting Gap Tracking:** Calculates:
  $$\text{Overfit Gap} = \text{Training Accuracy} - \text{Testing Accuracy}$$

### 5. Prescriptive Analysis (*"What decisions should business leaders make?"*)
- **Strategic Recommendations:** Translates model drivers into concrete, prioritized business/clinical actions (e.g., *"Incentivize annual contract migrations to reduce churn by an estimated 15-20%"*).

---

## 3. The 7 Specialized Agents

Think of our system as a professional data science department where each agent has one specific job:

| Agent Name | Real-World Role | What It Actually Does |
| :--- | :--- | :--- |
| **1. Orchestrator Agent** | *Project Manager / Team Lead* | Reads the user's natural language goal and dataset schema. Decomposes the goal into an ordered task pipeline. Runs offline using fast heuristics or uses Claude/Anthropic when an API key is set. |
| **2. Data Agent** | *Data Cleaning Engineer* | Audits the raw CSV file. Detects column types, flags primary key/ID columns, identifies duplicate rows, and checks missing value percentages. |
| **3. EDA Agent** | *Statistician & Detective* | Calculates the correlation matrix and runs formal statistical hypothesis tests (Welch's t-test, ANOVA, Chi-Square). Outputs exact p-values. |
| **4. ML Agent** | *Machine Learning Engineer* | Trains 3 candidate models (Random Forest, Gradient Boosting, Logistic Regression). Measures accuracy, F1-scores, feature importance rankings, and computes the generalization/overfitting gap. |
| **5. Visualization Agent** | *Graphic Designer* | Generates publication-quality charts: high-resolution Pearson correlation heatmaps and distribution histograms saved as PNG images. |
| **6. Critic Agent** | *Quality Assurance / Peer Reviewer* | **The Project's Unique Feature.** Audits the results for flaws: checks if sample size was large enough, checks if overfitting exceeds 15%, and ensures ML features agree with EDA correlations. If an anomaly is found, it can trigger a **retry loop** back to EDA. Computes an **Evidence Confidence Score (0–100%)**. |
| **7. Insight Agent** | *Business Consultant & Author* | Takes verified findings from all agents and synthesizes them into an executive summary with ranked business recommendations. Exports printable **PDF** and interactive **HTML** reports. |

---

## 4. Technology Stack

Our tech stack is divided into clear layers:

### 1. Multi-Agent Orchestration & Workflow
- **LangGraph (`langgraph>=0.2`):** Core framework used to build the state machine (`StateGraph`). It manages shared memory (`AgentState`), node execution, and the **conditional self-correction retry loop** from Critic back to EDA.
- **LangChain Core (`langchain-core>=0.3`):** Underlying message and state definitions.

### 2. Programming Language & Core Computation
- **Python 3.11:** Primary backend execution environment.
- **Pandas (`pandas>=2.0`):** Fast tabular data structures, data cleaning, and aggregation.
- **NumPy (`numpy>=1.26`):** Low-level numerical vector operations and array management.

### 3. Machine Learning & Statistical Engines
- **Scikit-Learn (`scikit-learn>=1.4`):** Real model training (`RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`), `StandardScaler`, and metrics calculation (`accuracy_score`, `f1_score`).
- **SciPy (`scipy>=1.11`):** Statistical inference calculations (`stats.ttest_ind` for Welch's t-test, `stats.f_oneway` for ANOVA, `stats.chi2_contingency` for independence tests).
- **Statsmodels (`statsmodels>=0.14`):** Supplemental statistical auditing.

### 4. User Interface & Dashboards
- **Streamlit (`streamlit>=1.35`):** Interactive web application displaying real-time agent execution traces, dataset selection, metric KPI cards, model leaderboards, and report download buttons.
- **Plotly (`plotly>=5.20`):** Interactive browser charts for feature distributions and correlations.
- **Matplotlib (`matplotlib>=3.8`):** Deterministic generation of chart artifacts saved to disk.

### 5. Document & Report Generation
- **FPDF2 (`fpdf2>=2.7`):** Generates publication-ready, multi-page executive PDF reports (`reports/executive_report.pdf`).
- **HTML5 & CSS3:** Generates responsive, standalone web reports (`reports/executive_report.html`).

### 6. Optional Generative AI Integration
- **Anthropic Claude API (`anthropic>=0.34`):** Optional LLM engine for reasoning-based task planning (with seamless automatic fallback to built-in rule heuristics if no key is provided).

---

## 5. End-to-End Implementation Flow

Here is exactly what happens when you click **"Run"**:

```
[1] User uploads CSV or selects benchmark (e.g. Heart Disease, Churn)
         │
[2] Orchestrator builds the task sequence: Data -> EDA -> ML -> Viz -> Critic -> Insight
         │
[3] Data Agent checks 303 rows, 14 columns, finds 1 duplicate, 0 nulls
         │
[4] EDA Agent finds chest pain correlates +0.43 with disease; runs Chi-Square tests
         │
[5] ML Agent trains 3 models: Random Forest wins with 85.2% accuracy (Overfit Gap: 13.1%)
         │
[6] Viz Agent generates Correlation Heatmap PNG and Age Distribution PNG
         │
[7] Critic Agent checks sample size (n=61 test split), confirms overfitting < 15%
    ├── Status = PASS (Confidence Score = 99%)
    └── (If FAILED, triggers retry loop back to step 4)
         │
[8] Insight Agent compiles executive summary, formats recommendations, builds PDF & HTML
         │
[9] Streamlit UI updates live with metrics, charts, tables, and download buttons!
```

---

## 6. How to Run It Locally (Quick Reference)

```powershell
# 1. Go to project directory
cd C:\Users\Lenovo\.gemini\antigravity-ide\scratch\agentic-ai-data-scientist

# 2. Launch the interactive Streamlit Web UI
.\.venv\Scripts\python.exe -m streamlit run app.py
# (Opens http://localhost:8501 in your browser)

# 3. Or run directly from terminal
.\.venv\Scripts\python.exe graph.py datasets/raw/sample.csv "predict whether the tumor is malignant"

# 4. Run the Critic Agent Ablation Study
.\.venv\Scripts\python.exe evaluation/ablation_study.py datasets/raw/sample.csv "predict whether tumor is malignant"
```
