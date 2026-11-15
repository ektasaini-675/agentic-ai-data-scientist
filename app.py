"""
Autonomous Multi-Agent AI System for Automated Data Analysis and Insight Generation
Streamlit Web Application built on LangGraph Multi-Agent Architecture.

Team Members:
- Ekta (B23CS1018)
- Kashish Joshi (B23CS1025)
- Kurra Hema (B23CS1031)
"""
import io
import os
import time
from pathlib import Path
import pandas as pd
import streamlit as st

from ads.config import CONFIDENCE_RUBRIC
from ads.pipeline import run_analysis

# ---------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Agentic AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 45%, #13122c 100%);
        color: #f1f5f9;
    }
    
    .hero-container {
        background: rgba(22, 29, 49, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 26px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f43f5e);
    }
    
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #cbd5e1, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 14.5px;
        margin-bottom: 14px;
    }
    
    .team-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .team-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #c7d2fe;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        text-align: center;
    }
    
    .metric-number {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    .trace-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        background: rgba(15, 23, 42, 0.7);
        border-left: 3px solid #38bdf8;
        padding: 8px 14px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
        color: #e2e8f0;
    }

    .claim-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 10px;
        font-size: 15px;
        line-height: 1.5;
    }
    
    .evidence-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("## ⚙️ Control Center")
    st.write("Autonomous Multi-Agent AI Data Scientist Configuration")

    mode = st.radio("Dataset Source:", ["Benchmark Datasets", "Upload Custom CSV"])

    dataset_path = None
    default_goal = "Analyze this dataset, predict the primary outcome, and uncover key drivers."

    if mode == "Benchmark Datasets":
        benchmark = st.selectbox(
            "Select Benchmark:",
            [
                "Breast Cancer (sample.csv)",
                "Telco Customer Churn (telco_churn.csv)",
                "Heart Disease Risk (heart_disease.csv)",
                "Loan Approval Risk (loan_approval.csv)",
            ],
        )
        if "sample" in benchmark:
            dataset_path = "datasets/raw/sample.csv"
            default_goal = "predict diagnosis and explain key factors"
        elif "telco" in benchmark:
            dataset_path = "datasets/raw/telco_churn.csv"
            default_goal = "identify the major factors affecting customer churn and suggest strategies to reduce it"
        elif "heart" in benchmark:
            dataset_path = "datasets/raw/heart_disease.csv"
            default_goal = "predict presence of heart disease and identify primary clinical risk factors"
        elif "loan" in benchmark:
            dataset_path = "datasets/raw/loan_approval.csv"
            default_goal = "identify factors that determine whether loan is approved"
    else:
        uploaded_file = st.file_uploader("Upload CSV dataset:", type=["csv"])
        if uploaded_file is not None:
            temp_dir = Path("datasets/uploaded")
            temp_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = str(temp_dir / uploaded_file.name)
            with open(dataset_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            default_goal = "Analyze key patterns, correlations, and predictive drivers in this data"

    if dataset_path and os.path.exists(dataset_path):
        preview_df = pd.read_csv(dataset_path)
        st.caption(f"📊 Rows: **{len(preview_df)}** | Columns: **{len(preview_df.columns)}**")

        st.markdown("---")
        st.markdown("### 🎯 Analytical Goal")
        user_goal = st.text_area("Natural Language Objective:", value=default_goal, height=90)
        st.markdown("---")
        run_btn = st.button("🚀 Run Multi-Agent System", type="primary", use_container_width=True)
    else:
        run_btn = False

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
<div class="hero-container">
    <div class="hero-title">🧠 Autonomous Agentic AI Data Scientist</div>
    <div class="hero-subtitle">Multi-Agent System with Grounded Evidence Ledger, Statistical Guardrails, and Self-Correction</div>
    <div class="team-badge-container">
        <span class="team-pill">🎓 B.Tech Final Year Project</span>
        <span class="team-pill">👤 Ekta (B23CS1018)</span>
        <span class="team-pill">👤 Kashish Joshi (B23CS1025)</span>
        <span class="team-pill">👤 Kurra Hema (B23CS1031)</span>
        <span class="team-pill">⚡ LangGraph StateGraph</span>
        <span class="team-pill">🛡️ Critic Self-Correction Loop</span>
        <span class="team-pill">🔒 Evidence Ledger Verifier</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------
if run_btn and dataset_path:
    with st.spinner("🤖 Orchestrating Multi-Agent Pipeline (Data -> EDA -> ML -> Viz -> Insight -> Critic)..."):
        start_time = time.time()
        result = run_analysis(dataset_path=dataset_path, goal=user_goal)
        elapsed = round(time.time() - start_time, 2)
        st.session_state.result = result
        st.success(f"🎉 Analysis Successfully Completed in {elapsed}s! All numbers verified against the Evidence Ledger.")

# ---------------------------------------------------------
# Results Display
# ---------------------------------------------------------
res = st.session_state.result

if res:
    ctx = res.ctx
    n_rows = ctx.df.shape[0]
    n_cols = ctx.df.shape[1]
    confidence = res.confidence

    # Identify target from ledger
    target_name = "N/A"
    problem_type = "N/A"
    best_model = "N/A"
    best_score = "N/A"

    for ev in ctx.ledger.all():
        if ev.tool == "identify_target" and isinstance(ev.result, dict):
            target_name = ev.result.get("target", "N/A")
            problem_type = ev.result.get("problem_type", "N/A")
        elif ev.tool == "train_cv" and isinstance(ev.result, dict):
            best_model = ev.result.get("best_model", "N/A")
            score = ev.result.get("test_score")
            best_score = f"{score:.3f}" if score is not None else "N/A"

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-number">{n_rows:,}</div><div class="metric-label">Dataset Rows ({ctx.dataset_version})</div></div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-number">{n_cols}</div><div class="metric-label">Features Profiled</div></div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-number" style="font-size: 20px;">{target_name}</div><div class="metric-label">Target ({problem_type})</div></div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-number" style="font-size: 20px;">{best_model} ({best_score})</div><div class="metric-label">Best ML Model</div></div>""",
            unsafe_allow_html=True,
        )
    with k5:
        color = "#10b981" if confidence >= 85 else "#f59e0b"
        st.markdown(
            f"""<div class="metric-card"><div class="metric-number" style="color: {color}; -webkit-text-fill-color: {color};">{confidence}%</div><div class="metric-label">Critic Confidence</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_claims, tab_trace, tab_ledger, tab_charts, tab_critic, tab_export = st.tabs(
        [
            "💡 Verified Findings",
            "💬 Agent Execution Trace",
            "📖 Evidence Ledger",
            "📈 Visual Analytics",
            "🛡️ Critic & Validity Audit",
            "📑 Download Reports",
        ]
    )

    # TAB 1: CLAIMS
    with tab_claims:
        st.markdown("### 📝 Verified Executive Claims & Actionable Insights")
        st.caption("Every claim below has been audited by the Claim Verifier. Numbers strictly match the Evidence Ledger.")
        for c in res.claims:
            tags = "".join(f"""<span class="evidence-tag">[{eid}]</span>""" for eid in c.evidence_ids)
            st.markdown(f"""<div class="claim-box">{c.text} {tags}</div>""", unsafe_allow_html=True)

    # TAB 2: TRACE
    with tab_trace:
        st.markdown("### 💬 Real-Time Agent Execution Trace")
        st.write("Dynamic plan decomposition, specialist agent execution, and self-correction loop:")
        for line in res.trace:
            st.markdown(f"""<div class="trace-item">{line}</div>""", unsafe_allow_html=True)

    # TAB 3: LEDGER
    with tab_ledger:
        st.markdown("### 📖 Append-Only Evidence Ledger")
        st.caption("Immutable record of all deterministic tool executions backing every reported number:")
        ledger_data = [
            {
                "Evidence ID": ev.id,
                "Tool": ev.tool,
                "Summary": ev.summary,
                "Data Version": ev.dataset_version,
                "Created By": ev.created_by,
            }
            for ev in ctx.ledger.all()
        ]
        st.dataframe(pd.DataFrame(ledger_data), use_container_width=True)

    # TAB 4: CHARTS
    with tab_charts:
        st.markdown("### 📈 Analytical Visualizations")
        chart_files = sorted(list((res.run_dir / "charts").glob("*.png"))) if (res.run_dir / "charts").exists() else []
        if chart_files:
            c_cols = st.columns(min(2, len(chart_files)))
            for i, p in enumerate(chart_files):
                with c_cols[i % len(c_cols)]:
                    st.image(str(p), caption=p.stem.replace("_", " ").title(), use_container_width=True)
        else:
            st.info("No visual charts generated.")

    # TAB 5: CRITIC
    with tab_critic:
        st.markdown(f"### 🛡️ Critic Agent Statistical Audit (Confidence Score: {confidence}%)")
        st.write("Validation rubric checklist and safety guardrails:")
        rubric_df = pd.DataFrame([{"Checklist Item": k, "Weight": v["weight"], "Description": v["desc"]} for k, v in CONFIDENCE_RUBRIC.items()])
        st.dataframe(rubric_df, use_container_width=True)

    # TAB 6: EXPORT
    with tab_export:
        st.markdown("### 📑 Download Executive Deliverables")
        html_file = Path(res.report_paths.get("html", ""))
        md_file = Path(res.report_paths.get("markdown", ""))
        ledger_file = Path(res.ledger_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            if html_file.exists():
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_file.read_text(encoding="utf-8"),
                    file_name="executive_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
        with col2:
            if md_file.exists():
                st.download_button(
                    label="📥 Download Markdown Report",
                    data=md_file.read_text(encoding="utf-8"),
                    file_name="executive_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        with col3:
            if ledger_file.exists():
                st.download_button(
                    label="📥 Download Evidence Ledger JSON",
                    data=ledger_file.read_text(encoding="utf-8"),
                    file_name="evidence_ledger.json",
                    mime="application/json",
                    use_container_width=True,
                )
else:
    st.info("👈 Select a benchmark dataset or upload a CSV in the sidebar, then click **'🚀 Run Multi-Agent System'** to begin.")
