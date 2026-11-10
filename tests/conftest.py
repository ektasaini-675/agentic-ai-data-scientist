"""Shared synthetic data and helpers for the tool tests."""
import numpy as np
import pandas as pd
import pytest

from ads.context import RunContext
from ads.ledger import Ledger


def _churn_df(n: int = 800, seed: int = 7, with_leak: bool = False) -> pd.DataFrame:
    """Churn depends on Contract and tenure; `noise` and `city` are pure noise."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.3, 0.2])
    tenure = rng.integers(1, 72, n)
    logit = -0.5 + (contract == "Month-to-month") * 1.6 - tenure * 0.04
    p = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.random(n) < p, "Yes", "No")
    df = pd.DataFrame(
        {
            "Contract": contract,
            "tenure": tenure,
            "noise": rng.normal(size=n),
            "city": rng.choice(list("ABCDE"), n),
            "Churn": churn,
        }
    )
    if with_leak:
        df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)  # a copy of the target
    return df


@pytest.fixture
def churn_df() -> pd.DataFrame:
    return _churn_df()


@pytest.fixture
def churn_df_leaky() -> pd.DataFrame:
    return _churn_df(with_leak=True)


@pytest.fixture
def make_ctx(tmp_path):
    def _make(df: pd.DataFrame) -> RunContext:
        return RunContext(run_dir=tmp_path, ledger=Ledger(), df=df, agent_name="test_agent")

    return _make
