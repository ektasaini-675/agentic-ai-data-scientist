# Autonomous Multi-Agent AI System for Data Analysis & Insight Generation

**B.Tech Major Project (BTPP)**  
**Team Members:** Ekta (`B23CS1018`), Kashish Joshi (`B23CS1025`), Kurra Hema (`B23CS1031`)  
**Repository:** [ektasaini-675/agentic-ai-data-scientist](https://github.com/ektasaini-675/agentic-ai-data-scientist)

---

## 📌 Project Overview
An autonomous, verifiable multi-agent data scientist system built on **LangGraph**. Given any raw dataset (CSV) and a natural-language business/analytical goal, the system:
1. **Orchestrator Agent**: Decomposes analytical goals into dynamic task sequences using LLMs (Google Gemini / Anthropic) with offline fallback.
2. **Data Agent**: Inspects schemas, missing value distributions, cardinalities, cleans data, checks for target leakage, and drops suspect columns (`drop_columns`).
3. **EDA / Statistics Agent**: Tests associations with Benjamini-Hochberg FDR correction, evaluates rank correlations, and measures segment rates.
4. **Machine Learning Agent**: Automates model cross-validation on train splits, held-out test scoring, baseline comparison, and permutation feature importance.
5. **Visualization Agent**: Generates publication-ready correlation heatmaps, feature importance plots, and outcome distribution charts.
6. **Insight Agent**: Synthesizes verified claims and actionable recommendations backed strictly by cited evidence IDs.
7. **Critic / Validation Agent**: Programmatically audits target leakage, overfitting gaps, sample sizes, and CV stability; calculates a 100-point **Confidence Rubric** score, and triggers backward retry loops on blockers.

### Core Architectural Invariant
> **A number may only appear in a report if a deterministic tool computed it and the Evidence Ledger recorded it.**

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
            │    Insight Agent    │ ── (Writes Claims)         │
            └──────────┬──────────┘                            │
                       ▼                                       │
            ┌─────────────────────┐                            │
            │  Critic/Validation  │ ── [needs_revision] ───────┘ (Re-plan / retry to target)
            └──────────┬──────────┘
                       │ [pass]
                       ▼
            ┌─────────────────────┐
            │   Executive Report  │ ──→ HTML, Markdown & Evidence Ledger
            └─────────────────────┘
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ektasaini-675/agentic-ai-data-scientist.git
cd agentic-ai-data-scientist

# Install dependencies
pip install -r requirements-dev.txt
```

### 2. Configure LLM Key (Free Google Gemini)
Copy `.env.example` to `.env` and insert your Gemini API key (from [Google AI Studio](https://aistudio.google.com)):
```bash
cp .env.example .env
# Edit .env:
# GEMINI_API_KEY=your_gemini_key_here
# ADS_MODEL=gemini-3.5-flash
```
*Note: The system also functions completely offline without any API keys required using deterministic scripted policies!*

### 3. Verify System Health
```bash
# Run unit & integration test suite (56 tests)
pytest

# Smoke test every tool across all benchmark datasets
python scripts/smoke_tools.py

# Verify live Gemini LLM connection and tool calling
python scripts/check_llm.py
```

---

## 💻 Running the System

### A. Interactive Streamlit Web UI
Launch the interactive dashboard to upload custom CSVs, monitor real-time agent execution traces, browse the Evidence Ledger, view charts, and download reports:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### B. Command Line Interface (CLI)
Run the full multi-agent pipeline on any dataset:
```bash
python -m ads.cli datasets/raw/sample.csv "predict diagnosis and explain key factors"
python -m ads.cli datasets/raw/telco_churn.csv "identify factors affecting customer churn"
```

### C. Academic Benchmark Evaluation
Run the comprehensive 10-case benchmark (B1 Single-Prompt vs. B2 Ablated vs. B3 Full System) measuring flaw detection, number grounding rates, and self-correction:
```bash
python evaluation/benchmark.py
```
Results and analysis are recorded in `docs/evaluation.md`.

---

## 📁 Repository Structure
```
agentic-ai-data-scientist/
├── ads/                     # Core multi-agent package
│   ├── agents/              # Specialist agents (orchestrator, data, eda, ml, viz, insight, critic)
│   ├── llm/                 # Unified LLM client (Gemini/Anthropic), cache, agent loop
│   ├── tools/               # Deterministic statistical, data, ML, and viz tools
│   ├── report/              # Executive report renderer (HTML/Markdown + audit appendix)
│   ├── context.py           # RunContext & per-run directory management
│   ├── ledger.py            # Append-only Evidence Ledger
│   ├── state.py             # Shared contracts (Evidence, Claim, Feedback, AgentState)
│   ├── verify.py            # Claim verifier (anti-hallucination guard)
│   ├── pipeline.py          # End-to-end pipeline executor
│   └── cli.py               # Command-line entry point
├── datasets/raw/            # Benchmark datasets (sample, telco_churn, heart_disease, loan_approval)
├── docs/                    # Architecture, evaluation benchmark, and handover specs
├── evaluation/              # Benchmark suites and comparative studies
├── scripts/                 # check_llm.py, smoke_tools.py
├── tests/                   # Pytest test suite (56 tests)
├── app.py                   # Streamlit web application & live dashboard
└── README.md                # Project documentation
```

---

## 👥 Contributors
- **Ekta** (`B23CS1018`)
- **Kashish Joshi** (`B23CS1025`)
- **Kurra Hema** (`B23CS1031`)
