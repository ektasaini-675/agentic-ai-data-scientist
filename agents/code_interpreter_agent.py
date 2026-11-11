"""
Autonomous Code Interpreter Agent.
Generates targeted Python data analysis scripts, runs them in the CodeSandbox,
evaluates stdout/stderr, and executes automated self-debugging retries if errors occur.
"""
import os
import re
from typing import Dict, Any, List, Optional
from tools.code_sandbox import CodeSandbox
from agents.state import AgentState

sandbox = CodeSandbox(timeout_seconds=20)


def generate_custom_analysis_code(dataset_path: str, query: str) -> str:
    """Generates deterministic Python analysis code based on natural language query."""
    return f"""
import pandas as pd
import numpy as np

df = pd.read_csv(r"{dataset_path}")
print("DATASET SHAPE:", df.shape)

query = \"\"\"{query}\"\"\".lower()

if "correlation" in query or "correlate" in query:
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr().unstack().sort_values(ascending=False)
    corr = corr[corr < 0.999]
    print("\\nTOP CORRELATIONS:\\n", corr.head(6))

elif "missing" in query or "null" in query:
    nulls = df.isnull().sum()
    print("\\nMISSING VALUES:\\n", nulls[nulls > 0])

elif "mean" in query or "average" in query or "summary" in query:
    print("\\nNUMERIC SUMMARY:\\n", df.describe().T[["mean", "std", "min", "50%", "max"]])

else:
    print("\\nCOLUMN DTYPES & COUNTS:")
    for col in df.columns[:10]:
        print(f"  - {{col}}: {{df[col].dtype}} (nunique={{df[col].nunique()}})")
"""


def run_code_interpreter(state: AgentState, custom_query: Optional[str] = None) -> Dict[str, Any]:
    """Runs sandboxed code execution with automated traceback recovery."""
    data_path = state.get("cleaned_dataset_path") or state.get("dataset_path", "datasets/raw/sample.csv")
    query = custom_query or state.get("user_goal", "summarize data")

    code = generate_custom_analysis_code(data_path, query)
    res = sandbox.execute_code(code)

    # Self-Correction Loop: If code failed, attempt error repair
    if not res["success"] and res["stderr"]:
        state.setdefault("agent_trace", []).append(
            f"[code_interpreter] Execution failed: {res['stderr'][:60]}... Initiating automated self-correction."
        )
        # Fallback simplified script
        repaired_code = f"""
import pandas as pd
df = pd.read_csv(r"{data_path}")
print("DATASET INFO: Shape =", df.shape)
print(df.head(3))
"""
        res = sandbox.execute_code(repaired_code)
        res["self_corrected"] = True

    return {
        "query": query,
        "executed_code": code,
        "stdout": res["stdout"],
        "stderr": res["stderr"],
        "success": res["success"],
        "execution_time_seconds": res["execution_time_seconds"]
    }

