"""
Production REST API for the Autonomous Multi-Agent AI Data Scientist System.
Built with Starlette & Uvicorn (Ultra-High Performance ASGI).

Endpoints:
- GET  /              -> Interactive Swagger-like API Documentation & Capabilities Portal
- GET  /health        -> System Status, Hardware Telemetry, Model Artifact Health
- POST /analyze       -> Triggers LangGraph StateGraph pipeline on dataset with goal
- POST /predict       -> Real-time sub-millisecond inference on trained model
- POST /whatif        -> Counterfactual sensitivity analysis
- GET  /benchmarks    -> Academic benchmark evaluation summary
- GET  /reports/{fmt} -> Direct download of executive PDF, HTML, or Markdown reports

Run server via:
    cmd.exe /c run.bat api.py
"""
import os
import sys
import json
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, FileResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import build_graph
from agents.state import AgentState
from models.inference_engine import InferenceEngine

inference_engine = InferenceEngine()


async def root_view(request):
    """Interactive API portal and documentation."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autonomous Agentic AI Data Scientist - Production REST API</title>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; padding: 40px; margin: 0; line-height: 1.6; }
    .container { max-width: 900px; margin: 0 auto; }
    .header { background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 30px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    h1 { margin: 0 0 10px 0; font-size: 26px; color: #38bdf8; }
    .badge { display: inline-block; background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; padding: 4px 12px; border-radius: 9999px; font-size: 13px; font-weight: bold; margin-right: 8px; }
    .endpoint-card { background: #131d33; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
    .method { display: inline-block; padding: 3px 8px; border-radius: 6px; font-weight: bold; font-size: 12px; margin-right: 10px; }
    .post { background: #059669; color: white; }
    .get { background: #2563eb; color: white; }
    code { font-family: 'JetBrains Mono', monospace; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; color: #e2e8f0; font-size: 13px; }
    pre { background: #080d1a; padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 13px; border: 1px solid rgba(255,255,255,0.06); }
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Autonomous Multi-Agent AI Data Scientist: REST API</h1>
        <div style="color: #94a3b8; font-size: 14px; margin-bottom: 12px;">IIT B.Tech Final Year Major Project | Production ASGI Server</div>
        <div>
            <span class="badge">Architecture: LangGraph StateGraph</span>
            <span class="badge" style="border-color: #22c55e; color: #22c55e;">Status: Healthy & Active</span>
        </div>
    </div>

    <h2>API Reference Endpoints</h2>

    <div class="endpoint-card">
        <span class="method get">GET</span> <code>/health</code>
        <p>Checks system status, hardware availability, active model artifact, and available benchmark datasets.</p>
    </div>

    <div class="endpoint-card">
        <span class="method post">POST</span> <code>/analyze</code>
        <p>Triggers the full multi-agent pipeline. Expects JSON body with <code>dataset_path</code> and <code>user_goal</code>.</p>
        <pre>POST /analyze
Content-Type: application/json

{
  "dataset_path": "datasets/raw/sample.csv",
  "user_goal": "predict whether tumor is malignant based on measurements"
}</pre>
    </div>

    <div class="endpoint-card">
        <span class="method post">POST</span> <code>/predict</code>
        <p>Executes sub-millisecond inference on the currently loaded trained model artifact.</p>
        <pre>POST /predict
Content-Type: application/json

{
  "features": {
    "mean radius": 14.5,
    "worst area": 880.0,
    "worst concave points": 0.15
  }
}</pre>
    </div>

    <div class="endpoint-card">
        <span class="method post">POST</span> <code>/whatif</code>
        <p>Runs counterfactual sensitivity analysis on the input feature values.</p>
    </div>

    <div class="endpoint-card">
        <span class="method get">GET</span> <code>/benchmarks</code>
        <p>Returns the multi-domain benchmark evaluation results across all 4 datasets.</p>
    </div>

    <div class="endpoint-card">
        <span class="method get">GET</span> <code>/reports/{format}</code>
        <p>Downloads generated report artifacts. Format options: <code>pdf</code>, <code>html</code>, <code>markdown</code>.</p>
    </div>
</div>
</body>
</html>"""
    return HTMLResponse(html)


async def health_view(request):
    """System health and model readiness check."""
    inference_engine.load()
    return JSONResponse({
        "status": "healthy",
        "system": "Autonomous Multi-Agent AI Data Scientist",
        "academic_project": "IIT B.Tech Major Project (BTP Capstone)",
        "model_artifact_loaded": inference_engine.is_ready(),
        "model_name": inference_engine.artifact.get("model_name") if inference_engine.is_ready() else None,
        "features_count": len(inference_engine.feature_names) if inference_engine.is_ready() else 0,
        "available_datasets": [
            "datasets/raw/sample.csv",
            "datasets/raw/heart_disease.csv",
            "datasets/raw/loan_approval.csv",
            "datasets/raw/telco_churn.csv"
        ]
    })


async def analyze_view(request):
    """Executes the full multi-agent LangGraph workflow."""
    try:
        body = await request.json()
        d_path = body.get("dataset_path", "datasets/raw/sample.csv")
        goal = body.get("user_goal", "analyze this dataset and identify predictive patterns")

        if not os.path.exists(d_path):
            return JSONResponse({"error": f"Dataset file not found: {d_path}"}, status_code=404)

        app = build_graph()
        initial_state: AgentState = {
            "dataset_path": d_path,
            "user_goal": goal,
            "agent_trace": []
        }

        final_state = app.invoke(initial_state, config={"recursion_limit": 50})
        inference_engine.load()

        return JSONResponse({
            "status": "success",
            "user_goal": goal,
            "dataset": d_path,
            "validation_status": final_state.get("validation_status"),
            "confidence_score": final_state.get("confidence_score"),
            "model_report": final_state.get("model_report"),
            "models_evaluated": final_state.get("models_evaluated"),
            "top_features": final_state.get("top_features"),
            "hypothesis_tests": final_state.get("hypothesis_tests"),
            "recommendations": final_state.get("recommendations"),
            "validation_notes": final_state.get("validation_notes"),
            "agent_trace": final_state.get("agent_trace"),
            "report_pdf_path": final_state.get("report_pdf_path"),
            "report_html_path": final_state.get("report_html_path")
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def predict_view(request):
    """Real-time inference endpoint."""
    inference_engine.load()
    if not inference_engine.is_ready():
        return JSONResponse({"error": "No trained model artifact found. Run /analyze first to train a model."}, status_code=400)

    try:
        body = await request.json()
        features = body.get("features", {})
        result = inference_engine.predict_one(features)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def whatif_view(request):
    """Counterfactual what-if sensitivity analysis."""
    inference_engine.load()
    if not inference_engine.is_ready():
        return JSONResponse({"error": "No trained model artifact found."}, status_code=400)

    try:
        body = await request.json()
        features = body.get("features", {})
        top_k = body.get("top_k", 5)
        sensitivities = inference_engine.counterfactual_sensitivity(features, top_k=top_k)
        return JSONResponse({"sensitivities": sensitivities})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def benchmarks_view(request):
    """Returns multi-dataset academic benchmarks."""
    bench_path = "evaluation/results/benchmark_summary.json"
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse({"benchmarks": data})
    return JSONResponse({"error": "Benchmark data not yet generated. Run benchmark_suite.py first."}, status_code=404)


async def reports_view(request):
    """Downloads report files."""
    fmt = request.path_params["fmt"].lower()
    if fmt == "pdf":
        path = "reports/executive_report.pdf"
        if os.path.exists(path):
            return FileResponse(path, media_type="application/pdf", filename="executive_report.pdf")
    elif fmt == "html":
        path = "reports/executive_report.html"
        if os.path.exists(path):
            return FileResponse(path, media_type="text/html", filename="executive_report.html")
    elif fmt in ["md", "markdown"]:
        path = "evaluation/results/full_system_report.md"
        if os.path.exists(path):
            return FileResponse(path, media_type="text/markdown", filename="executive_report.md")

    return JSONResponse({"error": f"Report format '{fmt}' not found."}, status_code=404)


routes = [
    Route("/", root_view, methods=["GET"]),
    Route("/health", health_view, methods=["GET"]),
    Route("/analyze", analyze_view, methods=["POST"]),
    Route("/predict", predict_view, methods=["POST"]),
    Route("/whatif", whatif_view, methods=["POST"]),
    Route("/benchmarks", benchmarks_view, methods=["GET"]),
    Route("/reports/{fmt}", reports_view, methods=["GET"]),
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

app = Starlette(debug=True, routes=routes, middleware=middleware)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 70)
    print(f"STARTING PRODUCTION REST API SERVER ON http://localhost:{port}")
    print("=" * 70)
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

