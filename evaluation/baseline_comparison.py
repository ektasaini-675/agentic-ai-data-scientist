"""
Baseline Comparison — the evidence that justifies building a multi-agent
system instead of just asking an LLM to "analyze this CSV" in one shot.

Run AFTER you have ANTHROPIC_API_KEY set:
    python evaluation/baseline_comparison.py datasets/raw/loan_approval.csv "predict loan approval status"

What it does:
  1. Runs your full multi-agent graph -> saves the report.
  2. Sends the raw dataset (as a text summary, not the full CSV) + goal to a
     single LLM call with NO tools, NO multi-step reasoning -> saves that report.
  3. Prints both side-by-side so you (and your prof) can compare on:
     correctness of numbers, depth, actionability, and hallucination risk.

This does NOT auto-score the comparison -- deliberately. Read both reports
yourself (or with 2-3 classmates, blind) and score them on a simple rubric:
correctness (1-5), completeness (1-5), actionability (1-5). That scored
comparison is your evaluation section, not this script's output.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from graph import build_graph


def run_multi_agent(dataset_path: str, goal: str) -> str:
    app = build_graph()
    state = {"dataset_path": dataset_path, "user_goal": goal, "agent_trace": []}
    final_state = app.invoke(state, config={"recursion_limit": 50})
    return final_state["final_report"]


def run_single_prompt_baseline(dataset_path: str, goal: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[SKIPPED: set ANTHROPIC_API_KEY to run the baseline comparison]"

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(dataset_path)
    # Give it a fair shot: schema + a small sample + basic stats, same info
    # your Data/EDA agents start from -- NOT the full dataset (that would be
    # a different, much more expensive kind of baseline).
    summary = (
        f"Columns: {list(df.columns)}\n"
        f"Shape: {df.shape}\n"
        f"Sample rows:\n{df.head(5).to_string()}\n"
        f"Basic stats:\n{df.describe(include='all').to_string()}"
    )
    prompt = (
        f"Goal: {goal}\n\nDataset summary:\n{summary}\n\n"
        "Analyze this dataset and produce a report with key findings, "
        "any predictive insight you can offer, and recommendations."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw/loan_approval.csv"
    goal = sys.argv[2] if len(sys.argv) > 2 else "predict loan approval status and explain key factors"

    print("Running multi-agent pipeline...")
    multi_agent_report = run_multi_agent(dataset_path, goal)

    print("Running single-prompt baseline...")
    baseline_report = run_single_prompt_baseline(dataset_path, goal)

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/multi_agent_report.md", "w") as f:
        f.write(multi_agent_report)
    with open("evaluation/results/baseline_report.md", "w") as f:
        f.write(baseline_report)

    print("\n" + "=" * 70)
    print("MULTI-AGENT REPORT (saved to evaluation/results/multi_agent_report.md)")
    print("=" * 70)
    print(multi_agent_report)

    print("\n" + "=" * 70)
    print("SINGLE-PROMPT BASELINE (saved to evaluation/results/baseline_report.md)")
    print("=" * 70)
    print(baseline_report)

    print("\n" + "=" * 70)
    print("NEXT STEP: read both reports and score them yourself (or with teammates,")
    print("blind) on correctness, completeness, and actionability, 1-5 each.")
    print("That scored comparison is your evaluation section -- not this script.")
    print("=" * 70)
