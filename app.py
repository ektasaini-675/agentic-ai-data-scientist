"""
Autonomous Multi-Agent AI System for Automated Data Science & Insight Generation
Interactive Web Application built on LangGraph StateGraph Architecture.

Major Project (B.Tech 4th Year - IIT BTP Capstone)
Team Members:
- Ekta (B23CS1018)
- Kashish Joshi (B23CS1025)
- Kurra Hema (B23CS1031)
"""
import os
import io
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from graph import build_graph
from agents.state import AgentState
from models.inference_engine import InferenceEngine

# ---------------------------------------------------------
# Page Configuration & Professional Academic Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Multi-Agent AI Data Scientist | IIT BTP",
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
        background: rgba(22, 29, 49, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 22px;
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
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #cbd5e1, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 12px;
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
        font-size: 11.5px;
        font-weight: 600;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-number {
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
        margin-top: 3px;
    }
    
    .trace-item {
        background: rgba(26, 34, 53, 0.8);
        border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0;
        padding: 9px 13px;
        margin-bottom: 7px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: #e2e8f0;
    }
    
    .card-recommendation {
        background: rgba(30, 41, 59, 0.55);
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
# Sidebar: Configuration & Dataset Selection
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Multi-Agent Settings")
    
    llm_provider = st.selectbox(
        "Orchestrator Intelligence:",
        [
            "Heuristic Semantic Engine (100% Offline / Zero API Key)",
            "Google Gemini API (gemini-1.5-flash)",
            "Anthropic Claude (claude-3-5-sonnet)",
            "OpenAI (gpt-4o-mini)"
        ],
        index=0
    )
    
    if "Heuristic" in llm_provider:
        os.environ["USE_LLM_ORCHESTRATOR"] = "0"
    else:
        os.environ["USE_LLM_ORCHESTRATOR"] = "1"
        
    st.markdown("---")
    st.markdown("### 📁 Select Benchmark Dataset")
    
    data_source = st.radio(
        "Data Source:",
        ["Preloaded Domain Benchmark", "Upload Custom CSV"],
        index=0
    )
    
    dataset_path = None
    if data_source == "Preloaded Domain Benchmark":
        benchmark_choice = st.selectbox(
            "Choose Benchmark Dataset:",
            [
                "Breast Cancer Diagnostic (569 rows) - Oncology / Pathology",
                "UCI Heart Disease (302 rows) - Cardiology / Diagnostics",
                "Financial Loan Approval (614 rows) - Credit Risk Underwriting",
                "Telco Customer Churn (7,043 rows) - Telecom / Customer Analytics"
            ]
        )
        dataset_map = {
            "Breast Cancer Diagnostic (569 rows) - Oncology / Pathology": (
                "datasets/raw/sample.csv",
                "predict whether the tumor is malignant based on cell measurements"
            ),
            "UCI Heart Disease (302 rows) - Cardiology / Diagnostics": (
                "datasets/raw/heart_disease.csv",
                "predict presence of heart disease based on clinical measurements"
            ),
            "Financial Loan Approval (614 rows) - Credit Risk Underwriting": (
                "datasets/raw/loan_approval.csv",
                "predict loan approval status and evaluate applicant risk factors"
            ),
            "Telco Customer Churn (7,043 rows) - Telecom / Customer Analytics": (
                "datasets/raw/telco_churn.csv",
                "predict customer churn and explain key retention drivers"
            )
        }
        dataset_path, default_goal = dataset_map[benchmark_choice]
    else:
        uploaded_file = st.file_uploader("Upload Tabular CSV", type=["csv"])
        if uploaded_file:
            os.makedirs("datasets/raw", exist_ok=True)
            custom_path = os.path.join("datasets/raw", uploaded_file.name)
            with open(custom_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            dataset_path = custom_path
            default_goal = "Understand key patterns, inferential correlations, and predictive drivers in this data"

    if dataset_path and os.path.exists(dataset_path):
        preview_df = pd.read_csv(dataset_path)
        st.caption(f"📊 Rows: **{len(preview_df):,}** | Columns: **{len(preview_df.columns)}**")

        st.markdown("---")
        st.markdown("### 🎯 Analytical Goal")
        user_goal = st.text_area(
            "Natural Language Objective:",
            value=default_goal,
            height=85
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
    <div class="hero-title">🧠 Autonomous Multi-Agent AI Data Scientist</div>
    <div class="hero-subtitle">LangGraph StateGraph Architecture with 5-Fold Stratified CV, Inferential Statistics, XAI & Closed-Loop Reflexion</div>
    <div class="team-badge-container">
        <span class="team-pill">🎓 B.Tech Final Year Major Project (IIT BTP Capstone)</span>
        <span class="team-pill">👤 Ekta (B23CS1018)</span>
        <span class="team-pill">👤 Kashish Joshi (B23CS1025)</span>
        <span class="team-pill">👤 Kurra Hema (B23CS1031)</span>
        <span class="team-pill">⚡ LangGraph StateGraph</span>
        <span class="team-pill">🛡️ Closed-Loop Critic Agent</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------
if run_btn and dataset_path:
    with st.spinner("Compiling and Executing LangGraph Multi-Agent StateGraph..."):
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
        st.success(f"🎉 LangGraph Multi-Agent Pipeline Successfully Completed in {elapsed}s!")

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
    overfit_gap = model_report.get("overfit_gap", 0.0) if model_report else 0.0
    
    # KPI Row (6 Key Metrics)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
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
        best_model = model_report.get("model", "N/A").replace("Classifier", "").replace("Regressor", "") if model_report else "None"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="font-size: 18px; line-height: 24px;">{best_model}</div>
            <div class="metric-label">Optimal Model</div>
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
        gap_color = "#22c55e" if overfit_gap <= 0.12 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: {gap_color}; -webkit-text-fill-color: {gap_color};">{round(overfit_gap*100, 1)}%</div>
            <div class="metric-label">Overfit Gap</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        color = "#22c55e" if critic_status == "pass" else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: {color}; -webkit-text-fill-color: {color};">{confidence}%</div>
            <div class="metric-label">Critic Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # In-Depth Tabs
    tab_report, tab_sim, tab_copilot, tab_trace, tab_ml, tab_eda, tab_charts, tab_critic, tab_cleaning, tab_benchmarks, tab_export, tab_defense = st.tabs([
        "💡 Executive Report",
        "🎯 What-If Simulation",
        "💬 Copilot / Ask Data",
        "📜 Live Agent Trace",
        "🤖 Predictive ML & CV",
        "📊 Inferential Statistics",
        "📈 Publication Charts",
        "🛡️ Critic 4-Pillar Audit",
        "🧹 Data Hygiene",
        "🔬 Benchmarks & Ablation",
        "📑 Export Center",
        "🎓 IIT BTP Viva Defense"
    ])

    # TAB 1: EXECUTIVE REPORT
    with tab_report:
        st.markdown("### 📝 Synthesized Executive Data Science Report")
        if final_state.get("final_report"):
            st.markdown(final_state["final_report"])
        
        recs = final_state.get("recommendations", [])
        if recs:
            st.markdown("---")
            st.markdown("### 🚀 Strategic Action Plan & Business Recommendations")
            for r in recs:
                st.markdown(f"""
                <div class="card-recommendation">
                    <strong style="color: #34d399; font-size: 15px;">[{r.get('priority', 'P0')}] {r.get('title')}</strong>
                    <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 4px;">{r.get('detail')}</div>
                    <div style="font-size: 12px; color: #6ee7b7; margin-top: 4px;"><strong>Expected Business Impact:</strong> {r.get('impact')}</div>
                </div>
                """, unsafe_allow_html=True)

    # TAB: WHAT-IF SIMULATION & COUNTERFACTUAL ENGINE
    with tab_sim:
        st.markdown("### 🎯 Live Model Inference & What-If Simulation Engine")
        st.write("Tweak feature inputs below in real-time to observe the predicted class outcome, probability distribution, and counterfactual sensitivity:")
        
        inf = InferenceEngine()
        if inf.is_ready():
            data_p = final_state.get("cleaned_dataset_path") or final_state.get("dataset_path")
            data_df = pd.read_csv(data_p) if (data_p and os.path.exists(data_p)) else pd.DataFrame()
            
            top_f_items = final_state.get("top_features", [])
            top_feature_names = [f["feature"] for f in top_f_items[:6]]
            if not top_feature_names:
                top_feature_names = inf.feature_names[:6]
                
            st.markdown("#### Configure Input Features")
            cols_input = st.columns(min(3, len(top_feature_names)))
            user_inputs = {}

            # Populate default values from medians
            for f in inf.feature_names:
                if f in data_df.columns and pd.api.types.is_numeric_dtype(data_df[f]):
                    user_inputs[f] = float(data_df[f].median())
                else:
                    user_inputs[f] = 0.0

            for idx, feat in enumerate(top_feature_names):
                c = cols_input[idx % len(cols_input)]
                with c:
                    if feat in data_df.columns and pd.api.types.is_numeric_dtype(data_df[feat]):
                        f_min = float(data_df[feat].min())
                        f_max = float(data_df[feat].max())
                        f_med = float(data_df[feat].median())
                        if f_min == f_max:
                            f_max = f_min + 1.0
                        step = round((f_max - f_min) / 100.0, 3) if f_max > f_min else 0.1
                        val = st.slider(f"{feat}:", min_value=f_min, max_value=f_max, value=f_med, step=step, key=f"sim_{feat}")
                        user_inputs[feat] = val
                    else:
                        val = st.number_input(f"{feat}:", value=float(user_inputs.get(feat, 0.0)), key=f"sim_num_{feat}")
                        user_inputs[feat] = val

            st.markdown("---")
            pred_res = inf.predict_one(user_inputs)
            st.markdown("#### Live Prediction Outcome")
            p_c1, p_c2 = st.columns([1, 1.5])
            with p_c1:
                pred_label = pred_res.get("prediction")
                conf_val = pred_res.get("confidence", 1.0)
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 2px solid #38bdf8; border-radius: 14px; padding: 20px; text-align: center;">
                    <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px; color: #94a3b8;">Predicted Target Outcome</div>
                    <div style="font-size: 32px; font-weight: 800; color: #38bdf8; margin: 8px 0;">{pred_label}</div>
                    <div style="font-size: 14px; color: #22c55e;">Model Confidence: <strong>{conf_val * 100:.1f}%</strong></div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">Engine: {pred_res.get('model_name')}</div>
                </div>
                """, unsafe_allow_html=True)
            with p_c2:
                probs = pred_res.get("probabilities", {})
                if probs:
                    prob_df = pd.DataFrame(list(probs.items()), columns=["Target Class", "Probability"])
                    fig_p = px.bar(
                        prob_df,
                        x="Probability",
                        y="Target Class",
                        orientation="h",
                        title="<b>Output Class Probability Distribution</b>",
                        color="Probability",
                        color_continuous_scale="Blues",
                        range_x=[0, 1]
                    )
                    fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), height=220)
                    st.plotly_chart(fig_p, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🔍 Counterfactual Sensitivity Analysis (What-If Recourse)")
            st.caption("Calculates how perturbing each top feature by ±20% shifts the output probability:")
            cf_list = inf.counterfactual_sensitivity(user_inputs, top_k=min(5, len(top_feature_names)))
            if cf_list:
                st.dataframe(pd.DataFrame(cf_list), use_container_width=True)
        else:
            st.info("Inference engine model artifact not ready. Run the pipeline first to train and serialize the model.")

    # TAB: COPILOT / ASK DATA
    with tab_copilot:
        st.markdown("### 💬 Data Scientist Copilot / Natural Language Inquiries")
        st.write("Ask any analytical or statistical question regarding the dataset, model, hypothesis tests, or recommendations:")
        
        sample_q = st.selectbox(
            "Suggested Analytical Inquiries:",
            [
                "Choose or type a custom question below...",
                "What is the single most influential predictive feature and why?",
                "Are any features exhibiting high multicollinearity?",
                "What was the holdout accuracy and 5-fold cross-validation score?",
                "What strategic business recommendations do the models support?",
                "Did the Critic Agent trigger any adaptive closed-loop remediations?"
            ]
        )
        
        custom_q = st.text_input("Enter your analytical question:", value="" if "Choose" in sample_q else sample_q)
        
        if custom_q:
            q_lower = custom_q.lower()
            st.markdown("#### 🤖 Copilot Response:")
            
            if any(k in q_lower for k in ["influential", "important", "feature", "driver"]):
                top_f = final_state.get("top_features", [])
                if top_f:
                    f0 = top_f[0]
                    st.success(f"**Top Predictive Feature:** `{f0['feature']}` with relative importance of **{round(f0['importance']*100, 1)}%**. It acts as a **{f0.get('impact_direction', 'Significant Driver')}** relative to target `{model_report.get('target_column')}`.")
                else:
                    st.info("Feature importance data is not available.")
            elif any(k in q_lower for k in ["collinear", "multicollinearity", "vif"]):
                vifs = final_state.get("vif_report", [])
                sev = [v for v in vifs if v.get("vif", 0) > 10]
                if sev:
                    st.warning(f"**Multicollinearity Warning:** {len(sev)} features exceed VIF threshold of 10: `{[s['feature'] for s in sev[:4]]}`. The Critic Agent flagged these for potential variance inflation.")
                else:
                    st.success("**Multicollinearity Status:** All evaluated numeric features exhibit safe Variance Inflation Factor (VIF <= 10) levels.")
            elif any(k in q_lower for k in ["accuracy", "cv", "score", "f1", "gap"]):
                st.info(f"**Selected Model:** `{model_report.get('model')}` achieved a holdout test score of **{model_report.get('accuracy')}** and a 5-fold CV score of **{model_report.get('cv_accuracy')}**. The generalization overfitting gap is bounded at **{round(model_report.get('overfit_gap', 0)*100, 1)}%**.")
            elif any(k in q_lower for k in ["recommend", "action", "strategy"]):
                recs = final_state.get("recommendations", [])
                for r in recs:
                    st.markdown(f"- **[{r['priority']}] {r['title']}**: {r['detail']} *(Impact: {r['impact']})*")
            elif any(k in q_lower for k in ["critic", "reflexion", "retry", "loop"]):
                retries = final_state.get("retry_count", 0)
                status = final_state.get("validation_status")
                conf = final_state.get("confidence_score")
                st.info(f"**Critic Status:** `{status.upper()}` with Evidence Confidence Score of **{conf}%**. The system executed **{retries}** adaptive reflexion retry loops.")
            else:
                st.markdown(f"**Data Scientist Copilot Findings:** Based on the validated dataset of `{schema.get('n_rows', 0):,}` instances, optimal performance was obtained by `{model_report.get('model')}` with score `{model_report.get('accuracy')}`. Consult the Executive Report tab for in-depth findings.")

    # TAB 2: LIVE AGENT TRACE
    with tab_trace:
        st.markdown("### 💬 LangGraph Agent Execution Trace & Reflexion Log")
        st.write("Chronological log capturing agent decomposition, state mutation, closed-loop retry loops, and adaptive remediations:")
        trace = final_state.get("agent_trace", [])
        for line in trace:
            if "REFLEXION" in line:
                st.markdown(f"""<div class="trace-item" style="border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.12);">{line}</div>""", unsafe_allow_html=True)
            elif "ADAPTIVE REMEDIATION" in line:
                st.markdown(f"""<div class="trace-item" style="border-left-color: #10b981; background: rgba(16, 185, 129, 0.12);">{line}</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="trace-item">{line}</div>""", unsafe_allow_html=True)

    # TAB 3: PREDICTIVE ML
    with tab_ml:
        st.markdown("### 🤖 Predictive Modeling Leaderboard & 5-Fold Stratified Cross-Validation")
        if model_report and model_report.get("status") != "skipped":
            st.write(f"Target Feature: **`{model_report.get('target_column')}`** | Problem Formulation: **`{model_report.get('problem_type')}`** | Class Imbalance: **`{final_state.get('class_imbalance_ratio', 1.0)}:1`**")
            
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            c_m1.metric("Selected Model", model_report.get("model"))
            c_m2.metric("Holdout Score (Acc/R²)", model_report.get("accuracy"))
            c_m3.metric("5-Fold CV Score", model_report.get("cv_accuracy"))
            c_m4.metric("Generalization Gap", f"{round(model_report.get('overfit_gap', 0)*100, 1)}%")

            # Models Evaluated Leaderboard
            models_eval = final_state.get("models_evaluated", [])
            if models_eval:
                st.markdown("#### Candidate Model Benchmark Leaderboard")
                st.dataframe(pd.DataFrame(models_eval), use_container_width=True)

            # Feature Importances with Plotly
            top_f = final_state.get("top_features", [])
            if top_f:
                st.markdown("#### Explainable AI (XAI): Feature Importance Attribution & Directional Impact")
                df_imp = pd.DataFrame(top_f)
                st.dataframe(df_imp, use_container_width=True)

                fig_imp = px.bar(
                    df_imp,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="<b>Ranked Feature Importance Drivers with Directional Impact</b>",
                    color="impact_direction",
                    color_discrete_map={
                        "Positive Risk (+)": "#ef4444",
                        "Protective / Negative (-)": "#10b981",
                        "Neutral": "#6366f1",
                        "Positive Driver (+)": "#ef4444",
                        "Inverse Driver (-)": "#10b981"
                    }
                )
                fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
                st.plotly_chart(fig_imp, use_container_width=True)

            # Confusion Matrix
            cm = final_state.get("confusion_matrix")
            if cm:
                st.markdown("#### Holdout Confusion Matrix")
                cm_df = pd.DataFrame(cm)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted Label", y="True Label", color="Instance Count"),
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Predictive machine learning modeling was skipped or no features were available.")

    # TAB 4: STATISTICAL EDA
    with tab_eda:
        st.markdown("### 📊 Inferential Hypothesis Testing & Exploratory Patterns")
        
        # Hypothesis tests
        tests = final_state.get("hypothesis_tests", [])
        if tests:
            st.markdown("#### 🔬 Inferential Hypothesis Battery (alpha = 0.05)")
            st.dataframe(pd.DataFrame(tests), use_container_width=True)

        # Multicollinearity VIF Diagnostic
        vif_res = final_state.get("vif_report", [])
        if vif_res:
            st.markdown("#### 🔗 Variance Inflation Factor (VIF) Multicollinearity Audit")
            st.dataframe(pd.DataFrame(vif_res), use_container_width=True)

        # Top Correlations
        top_corr = eda.get("top_correlations", [])
        if top_corr:
            st.markdown("#### 📈 Strongest Pairwise Feature Correlations")
            st.dataframe(pd.DataFrame(top_corr), use_container_width=True)

        # Normality tests
        norm_tests = eda.get("normality_tests", {})
        if norm_tests:
            st.markdown("#### 🧬 Shapiro-Wilk Normality Audit")
            st.dataframe(pd.DataFrame(norm_tests).T, use_container_width=True)

    # TAB 5: PUBLICATION CHARTS
    with tab_charts:
        st.markdown("### 📈 Publication-Quality Visual Analytics (300 DPI)")
        chart_paths = final_state.get("chart_paths", [])
        if chart_paths:
            cols = st.columns(min(2, len(chart_paths)))
            for i, p in enumerate(chart_paths):
                if os.path.exists(p):
                    with cols[i % len(cols)]:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)
        else:
            st.info("No visual chart artifacts generated.")

    # TAB 6: CRITIC 4-PILLAR AUDIT
    with tab_critic:
        st.markdown("### 🛡️ Critic Agent 4-Pillar Empirical Validation Audit")
        st.write(f"Validation Status: **`{critic_status.upper()}`** | Reflexion Loops: **`{retries}`** | Final Confidence: **`{confidence}%`**")
        
        # Radar Chart Display
        p_radar = "reports/charts/critic_radar.png"
        if os.path.exists(p_radar):
            col_r1, col_r2 = st.columns([1, 1.2])
            with col_r1:
                st.image(p_radar, caption="4-Pillar Validation Radar", use_container_width=True)
            with col_r2:
                notes = final_state.get("validation_notes", [])
                st.markdown("#### Empirical Audit Trail")
                for n in notes:
                    st.markdown(f"🔍 **Audit Note:** {n}")
        else:
            notes = final_state.get("validation_notes", [])
            for n in notes:
                st.markdown(f"🔍 **Audit Note:** {n}")

    # TAB 7: DATA HYGIENE
    with tab_cleaning:
        st.markdown("### 🧹 Data Hygiene & Preprocessing Profile")
        st.metric("Data Quality Health Score", f"{schema.get('data_quality_score', 100)}%")
        logs = final_state.get("cleaning_log", [])
        for log in logs:
            st.markdown(f"- {log}")
            
        cleaned_path = final_state.get("cleaned_dataset_path") or final_state.get("dataset_path")
        if cleaned_path and os.path.exists(cleaned_path):
            st.markdown("#### Cleaned Dataset Preview")
            st.dataframe(pd.read_csv(cleaned_path).head(30), use_container_width=True)

    # TAB 8: BENCHMARKS & ABLATION
    with tab_benchmarks:
        st.markdown("### 🔬 Empirical Academic Benchmarks & Architectural Ablation")
        
        # Benchmark Table
        bench_path = "evaluation/results/benchmark_table.md"
        if os.path.exists(bench_path):
            with open(bench_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        
        st.markdown("---")
        st.markdown("### ⚖️ Architectural Ablation Study (Critic Efficacy)")
        ablation_path = "evaluation/results/ablation_results.json"
        if os.path.exists(ablation_path):
            with open(ablation_path, "r", encoding="utf-8") as f:
                ab_data = json.load(f)
            st.json(ab_data)

    # TAB 9: EXPORT CENTER
    with tab_export:
        st.markdown("### 📑 Download Official Project Artifacts")
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
                        label="📥 Download Standalone HTML Report",
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

    # TAB 10: VIVA DEFENSE GUIDE
    with tab_defense:
        st.markdown("### 🎓 IIT B.Tech Major Project Viva Voce Defense Reference")
        st.markdown("""
        #### 1. Why LangGraph over AutoGPT or CrewAI?
        **Answer:** In tabular data science, deterministic state tracking and cyclic edge execution are required. LangGraph provides a formal Directed Acyclic Graph (DAG) with explicit conditional edges (`critic_route`), enabling closed-loop remediation when overfitting or multicollinearity is detected.
        
        #### 2. How do you guarantee zero numerical hallucination?
        **Answer:** Complete separation of concerns: all statistical, mathematical, and machine learning computations are executed deterministically by Python engines (`scikit-learn`, `scipy`, `numpy`, `statsmodels`). Language models are used strictly for intent decomposition and strategic synthesis, strictly referencing pre-calculated numbers in `AgentState`.
        
        #### 3. What does your Closed-Loop Reflexion actually adapt?
        **Answer:** Unlike linear pipelines, when the Critic flags severe overfitting (gap > 25%), it issues machine-actionable directives that dynamically reduce tree depth (`max_depth=3`), increase regularization, or prune high-VIF collinear features. The graph then re-evaluates and validates the improvement.
        
        #### 4. How is the Evidence Confidence Score calculated?
        **Answer:** Formal penalty deduction:
        $$\\text{Confidence Score} = \\max(60, \\min(99, 100 - \\sum \\text{Penalties}))$$
        Penalties are assessed across 4 pillars: Data Integrity (-12%), Overfitting Gap (-15%), Statistical Power (-5%), and Multicollinearity (-6%), with bonuses awarded upon verified remediation.
        """)

else:
    st.info("👈 Select a benchmark dataset or upload a CSV in the sidebar, then click **'🚀 Run LangGraph Multi-Agent Pipeline'** to begin.")
