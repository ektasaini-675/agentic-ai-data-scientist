"""Command-line interface for the Autonomous AI Data Scientist."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ads.pipeline import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous Multi-Agent AI Data Scientist")
    parser.add_argument("dataset", help="Path to the input CSV dataset")
    parser.add_argument("goal", nargs="?", default="Analyze this dataset and uncover key drivers and patterns",
                        help="Analytical goal or question")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file '{dataset_path}' not found.", file=sys.stderr)
        return 1

    print(f"Starting Autonomous Analysis...")
    print(f"Dataset: {dataset_path}")
    print(f"Goal: {args.goal}")
    print("=" * 60)

    result = run_analysis(str(dataset_path), goal=args.goal)

    print("\n" + "=" * 60)
    print("AGENT EXECUTION TRACE")
    print("=" * 60)
    for line in result.trace:
        print(line)

    print("\n" + "=" * 60)
    print(f"EXECUTIVE SUMMARY (Confidence: {result.confidence}%)")
    print("=" * 60)
    for c in result.claims:
        eids = f"[{', '.join(c.evidence_ids)}]" if c.evidence_ids else ""
        print(f"- {c.text} {eids}")

    print("\n" + "=" * 60)
    print("ARTIFACTS GENERATED")
    print("=" * 60)
    print(f"HTML Report:     {result.report_paths.get('html')}")
    print(f"Markdown Report: {result.report_paths.get('markdown')}")
    print(f"Evidence Ledger: {result.ledger_path}")
    print(f"Run Directory:   {result.run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

