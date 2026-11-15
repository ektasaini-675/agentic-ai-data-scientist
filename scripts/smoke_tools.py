"""Smoke test: run every tool on every real dataset and print what each one found.

Usage (from the project folder):
    python scripts/smoke_tools.py                 # all datasets
    python scripts/smoke_tools.py telco_churn     # just one

Exit code is 1 if any tool failed, so it can be used as a quick regression check.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ads.tools  # noqa: E402,F401  (registers tools)
from ads.config import SETTINGS  # noqa: E402
from ads.context import RunContext  # noqa: E402
from ads.tools import execute_tool  # noqa: E402

DATASETS = {
    "telco_churn": "Identify the major factors affecting customer churn and suggest strategies to reduce it",
    "heart_disease": "Predict the presence of heart disease and identify the key risk factors",
    "loan_approval": "Identify the factors that determine whether a loan is approved",
    "sample": "Analyze this dataset and identify the key patterns",
}


def step(ctx, report, tool, args):
    t0 = time.time()
    result, _ = execute_tool(tool, args, ctx)
    dt = time.time() - t0
    report.append((tool, result.ok, dt, result.error))
    print(f"  [{'ok' if result.ok else 'FAIL'}] {tool} ({dt:.1f}s)")
    print(f"        {result.summary if result.ok else result.error}")
    return result


def run_one(name: str, goal: str):
    path = ROOT / "datasets" / "raw" / f"{name}.csv"
    if not path.exists():
        print(f"\n=== {name}: file not found, skipping")
        return None
    print(f"\n=== {name}  |  goal: {goal}")
    ctx = RunContext.create(str(path), goal=goal, base_dir=SETTINGS.runs_dir / "smoke")
    ctx.agent_name = "smoke"
    report = []

    step(ctx, report, "profile_dataset", {})
    ident = step(ctx, report, "identify_target", {"goal": goal})
    if not ident.ok:
        return report
    target = ident.data["target"]
    problem = ident.data["problem_type"]
    n_classes = ident.data["n_unique"]

    step(ctx, report, "clean_data", {"target": target})
    step(ctx, report, "detect_leakage", {"target": target})
    assoc = step(ctx, report, "test_association", {"target": target})
    step(ctx, report, "top_correlations", {})

    if assoc.ok and not (problem == "classification" and n_classes > 2):
        for row in assoc.data["results"][:2]:
            step(ctx, report, "segment_rates", {"feature": row["feature"], "target": target})
            step(ctx, report, "plot_target_by_feature", {"feature": row["feature"], "target": target})
    else:
        print("  (skipping segment_rates: target is not binary or association failed)")

    step(ctx, report, "train_cv", {"target": target})
    step(ctx, report, "permutation_importance", {"target": target})
    step(ctx, report, "plot_importance", {"target": target})
    step(ctx, report, "plot_correlation", {})

    ctx.save_ledger()
    print(f"  ledger: {len(ctx.ledger)} evidence entries -> {ctx.run_dir}")
    return report


def main():
    wanted = sys.argv[1:] or list(DATASETS)
    failures = []
    for name in wanted:
        report = run_one(name, DATASETS.get(name, "Analyze this dataset and identify the key patterns"))
        for tool, ok, _, err in report or []:
            if not ok:
                failures.append((name, tool, err))

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} tool call(s) failed:")
        for name, tool, err in failures:
            print(f"  - {name}: {tool}: {err}")
        sys.exit(1)
    print("All tool calls succeeded on all datasets.")


if __name__ == "__main__":
    main()
