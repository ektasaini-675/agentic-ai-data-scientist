"""
Autonomous Multi-Agent AI System for Automated Data Analysis and Insight Generation
Streamlit Web Application built on LangGraph StateGraph Architecture.

Team Members:
- Ekta (B23CS1018)
- Kashish Joshi (B23CS1025)
- Kurra Hema (B23CS1031)
"""
import os
import io
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from graph import build_graph
from agents.state import AgentState

# ---------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Agentic AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
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
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
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
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    .trace-item {
        background: rgba(26, 34, 53, 0.8);
        border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #e2e8f0;
    }
    
    .card-recommendation {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #10b981;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "final_state" not in st.session_state:
    st.session_state.final_state = None

# ---------------------------------------------------------
# Sidebar: Dataset & Goal Selection
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")
    
    # LLM Orchestrator Toggle
    use_llm = st.toggle("Use LLM Orchestrator (Claude / Anthropic)", value=False, help="Set USE_LLM_ORCHESTRATOR=1 when ANTHROPIC_API_KEY is available in .env")
    if use_llm:
        os.environ["USE_LLM_ORCHESTRATOR"] = "1"
    else:
        os.environ["USE_LLM_ORCHESTRATOR"] = "0"
        
    st.markdown("---")
    st.markdown("### 📁 Select Benchmark Dataset")
    
    data_source = st.radio(
        "Data Source:",
        ["Preloaded Benchmark Dataset", "Upload Custom CSV"],
        index=0
    )
    
    dataset_path = None
    if data_source == "Preloaded Benchmark Dataset":
        benchmark_choice = st.selectbox(
            "Choose Dataset:",
            [
                "Telco Customer Churn (7,043 rows) - E-commerce / Telecom",
                "UCI Heart Disease (303 rows) - Healthcare Diagnostics",
                "Loan Approval (614 rows) - Financial Risk",
                "Breast Cancer Diagnostic (569 rows) - Pathology"
            ]
        )
        dataset_map = {
            "Telco Customer Churn (7,043 rows) - E-commerce / Telecom": ("datasets/raw/telco_churn.csv", "Identify the major factors affecting customer churn and suggest strategies to reduce it"),
            "UCI Heart Disease (303 rows) - Healthcare Diagnostics": ("datasets/raw/heart_disease.csv", "Predict presence of heart disease based on clinical measurements"),
            "Loan Approval (614 rows) - Financial Risk": ("datasets/raw/loan_approval.csv", "Predict loan approval status and key determinants"),
            "Breast Cancer Diagnostic (569 rows) - Pathology": ("datasets/raw/sample.csv", "Predict whether the tumor is malignant based on measurements")
        }
        dataset_path, default_goal = dataset_map[benchmark_choice]
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            os.makedirs("datasets/raw", exist_ok=True)
            custom_path = os.path.join("datasets/raw", uploaded_file.name)
            with open(custom_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            dataset_path = custom_path
            default_goal = "Understand key patterns, correlations, and predictive drivers in this data"

    if dataset_path and os.path.exists(dataset_path):
        preview_df = pd.read_csv(dataset_path)
        st.caption(f"📊 Rows: **{len(preview_df)}** | Columns: **{len(preview_df.columns)}**")

        st.markdown("---")
        st.markdown("### 🎯 Analytical Goal")
        user_goal = st.text_area(
            "Natural Language Objective:",
            value=default_goal,
            height=90
        )
        
        st.markdown("---")
        run_btn = st.button("🚀 Run LangGraph Multi-Agent Pipeline", type="primary", use_container_width=True)
    else:
        run_btn = False

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 Autonomous Agentic AI Data Scientist</div>
    <div class="hero-subtitle">LangGraph Multi-Agent Architecture for Automated Data Cleaning, EDA, Predictive ML & Validated Insights</div>
    <div class="team-badge-container">
        <span class="team-pill">🎓 B.Tech Final Year Project</span>
        <span class="team-pill">👤 Ekta (B23CS1018)</span>
        <span class="team-pill">👤 Kashish Joshi (B23CS1025)</span>
        <span class="team-pill">👤 Kurra Hema (B23CS1031)</span>
        <span class="team-pill">⚡ LangGraph StateGraph</span>
        <span class="team-pill">🛡️ Critic Retry Loop</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------
if run_btn and dataset_path:
    with st.spinner("Compiling and Executing LangGraph Multi-Agent Pipeline..."):
        app = build_graph()
        initial_state: AgentState = {
            "dataset_path": dataset_path,
            "user_goal": user_goal,
            "agent_trace": []
        }
        
        start_time = time.time()
        final_state = app.invoke(initial_state, config={"recursion_limit": 50})
        elapsed = round(time.time() - start_time, 2)
        
        st.session_state.final_state = final_state
        st.success(f"🎉 LangGraph Pipeline Successfully Completed in {elapsed}s!")

# ---------------------------------------------------------
# Results Display
# ---------------------------------------------------------
final_state = st.session_state.final_state

if final_state:
    schema = final_state.get("schema_report", {})
    eda = final_state.get("eda_findings", {})
    model_report = final_state.get("model_report", {})
    critic_status = final_state.get("validation_status", "pass")
    confidence = final_state.get("confidence_score", 95)
    retries = final_state.get("retry_count", 0)
    
    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{schema.get('n_rows', 0):,}</div>
            <div class="metric-label">Dataset Records</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{schema.get('n_cols', 0)}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        best_model = model_report.get("model", "N/A") if model_report else "None"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="font-size: 19px; line-height: 26px;">{best_model}</div>
            <div class="metric-label">ML Model Selected</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        score_val = model_report.get("accuracy", 0) if model_report else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{score_val}</div>
            <div class="metric-label">Test Score (Acc/R²)</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        color = "#22c55e" if critic_status == "pass" else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: {color}; -webkit-text-fill-color: {color};">{confidence}%</div>
            <div class="metric-label">Critic Confidence ({critic_status.upper()})</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_report, tab_trace, tab_charts, tab_ml, tab_eda, tab_critic, tab_cleaning, tab_export = st.tabs([
        "💡 Final Report & Actions",
        "💬 Live Agent Trace",
        "📈 Generated Charts",
        "🤖 Predictive ML",
        "📊 Statistical EDA",
        "🛡️ Critic & Validation",
        "🧹 Data Cleaning",
        "📑 Export Reports"
    ])

    # TAB 1: FINAL REPORT
    with tab_report:
        st.markdown("### 📝 Synthesized Executive Report")
        if final_state.get("final_report"):
            st.markdown(final_state["final_report"])
        
        recs = final_state.get("recommendations", [])
        if recs:
            st.markdown("---")
            st.markdown("### 🚀 Strategic Action Plan")
            for r in recs:
                st.markdown(f"""
                <div class="card-recommendation">
                    <strong style="color: #34d399; font-size: 15px;">{r.get('title')}</strong>
                    <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 4px;">{r.get('detail')}</div>
                    <div style="font-size: 12px; color: #6ee7b7; margin-top: 4px;"><strong>Impact:</strong> {r.get('impact')}</div>
                </div>
                """, unsafe_allow_html=True)

    # TAB 2: AGENT TRACE
    with tab_trace:
        st.markdown("### 💬 LangGraph Agent Execution Trace")
        st.write("Real-time logging showing how specialized agents executed the plan, passed data, and coordinated decisions:")
        trace = final_state.get("agent_trace", [])
        for line in trace:
            st.markdown(f"""<div class="trace-item">{line}</div>""", unsafe_allow_html=True)

    # TAB 3: CHARTS
    with tab_charts:
        st.markdown("### 📈 Visual Analytics & Plots")
        chart_paths = final_state.get("chart_paths", [])
        if chart_paths:
            cols = st.columns(min(2, len(chart_paths)))
            for i, p in enumerate(chart_paths):
                if os.path.exists(p):
                    with cols[i % len(cols)]:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)
        else:
            st.info("No chart images generated.")

    # TAB 4: ML MODEL
    with tab_ml:
        st.markdown("### 🤖 Predictive Machine Learning Model Report")
        if model_report and model_report.get("status") != "skipped":
            st.write(f"Target Column: **`{model_report.get('target_column')}`**")
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Selected Model", model_report.get("model"))
            c_m2.metric("Accuracy / R²", model_report.get("accuracy"))
            c_m3.metric("F1 Score", model_report.get("f1_score"))

            # Models Evaluated Leaderboard
            models_eval = final_state.get("models_evaluated", [])
            if models_eval:
                st.markdown("#### Model Leaderboard")
                st.dataframe(pd.DataFrame(models_eval), use_container_width=True)

            # Feature Importances
            top_f = model_report.get("top_features", [])
            if top_f:
                st.markdown("#### Top Predictive Feature Importances")
                df_imp = pd.DataFrame(top_f, columns=["Feature", "Importance Score"])
                st.dataframe(df_imp, use_container_width=True)

                fig_imp = px.bar(
                    df_imp,
                    x="Importance Score",
                    y="Feature",
                    orientation="h",
                    title="<b>Feature Importance Rankings</b>",
                    color="Importance Score",
                    color_continuous_scale="Tealgrn"
                )
                fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.write("Modeling skipped or not applicable.")

    # TAB 5: STATISTICAL EDA
    with tab_eda:
        st.markdown("### 📊 Exploratory Patterns & Statistical Tests")
        
        # Hypothesis tests
        tests = final_state.get("hypothesis_tests", [])
        if tests:
            st.markdown("#### 🔬 Hypothesis Testing (alpha = 0.05)")
            st.dataframe(pd.DataFrame(tests), use_container_width=True)

        # Top Correlations
        top_corr = eda.get("top_correlations", [])
        if top_corr:
            st.markdown("#### 🔗 Strongest Feature Correlations")
            st.dataframe(pd.DataFrame(top_corr), use_container_width=True)

    # TAB 6: CRITIC & VALIDATION
    with tab_critic:
        st.markdown("### 🛡️ Critic Agent Validation & Audit Notes")
        st.write(f"Validation Status: **`{critic_status.upper()}`** | Retry Count: **`{retries}`** | Confidence: **`{confidence}%`**")
        
        notes = final_state.get("validation_notes", [])
        for n in notes:
            st.markdown(f"🔍 **Audit Note:** {n}")

    # TAB 7: DATA CLEANING
    with tab_cleaning:
        st.markdown("### 🧹 Data Hygiene & Cleaning Log")
        logs = final_state.get("cleaning_log", [])
        for log in logs:
            st.markdown(f"- {log}")
            
        if final_state.get("dataset_path") and os.path.exists(final_state["dataset_path"]):
            st.markdown("#### Raw Dataset Preview")
            st.dataframe(pd.read_csv(final_state["dataset_path"]).head(30), use_container_width=True)

    # TAB 8: EXPORT REPORTS
    with tab_export:
        st.markdown("### 📑 Download Executive Reports")
        pdf_path = final_state.get("report_pdf_path", "reports/executive_report.pdf")
        html_path = final_state.get("report_html_path", "reports/executive_report.html")

        e1, e2, e3 = st.columns(3)
        with e1:
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Executive PDF Report",
                        data=f.read(),
                        file_name="executive_data_science_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        with e2:
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=f.read(),
                        file_name="executive_data_science_report.html",
                        mime="text/html",
                        use_container_width=True
                    )
        with e3:
            report_text = final_state.get("final_report", "")
            st.download_button(
                label="📥 Download Markdown Report",
                data=report_text,
                file_name="report.md",
                mime="text/markdown",
                use_container_width=True
            )

else:
    st.info("👈 Select a benchmark dataset or upload a CSV in the sidebar, then click **'🚀 Run LangGraph Multi-Agent Pipeline'** to begin.")
