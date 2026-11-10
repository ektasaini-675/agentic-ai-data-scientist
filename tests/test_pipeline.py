"""Integration tests for the complete multi-agent pipeline."""
from __future__ import annotations

from pathlib import Path

from ads.llm.client import LLMClient
from ads.pipeline import run_analysis
from ads.verify import verify_claims


def test_full_pipeline_offline(tmp_path, churn_df):
    csv_path = tmp_path / "test_churn.csv"
    churn_df.to_csv(csv_path, index=False)

    offline_client = LLMClient(mode="offline")
    result = run_analysis(
        dataset_path=str(csv_path),
        goal="Identify factors affecting customer churn",
        client=offline_client,
        base_dir=tmp_path / "runs",
    )

    assert result.run_dir.exists()
    assert result.ledger_path.exists()
    assert Path(result.report_paths["html"]).exists()
    assert Path(result.report_paths["markdown"]).exists()

    assert len(result.claims) >= 1
    # Verify every claim against the resulting ledger
    accepted, rejected = verify_claims(result.claims, result.ctx.ledger)
    assert len(accepted) == len(result.claims)
    assert len(rejected) == 0

    assert 0 <= result.confidence <= 100
    assert any("[data_agent]" in step for step in result.trace)
    assert any("[critic_agent]" in step for step in result.trace)

