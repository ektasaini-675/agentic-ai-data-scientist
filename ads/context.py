"""RunContext: everything one analysis run needs, in one object.

Holds the ledger, the current working dataframe, its version label, and the
per-run output folder (runs/<timestamp>/...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ads.config import SETTINGS
from ads.ledger import Ledger


@dataclass
class RunContext:
    run_dir: Path
    ledger: Ledger
    df: pd.DataFrame
    dataset_version: str = "v0"
    agent_name: str = "system"
    goal: str = ""
    step: int = 0
    cache: Dict[str, Any] = field(default_factory=dict)  # in-memory objects (fitted models), never persisted

    @classmethod
    def create(
        cls,
        dataset_path: str,
        goal: str = "",
        base_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> "RunContext":
        df = pd.read_csv(dataset_path)
        base = Path(base_dir) if base_dir else SETTINGS.runs_dir
        run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = base / run_id
        n = 2
        while run_dir.exists():  # never overwrite another run
            run_dir = base / f"{run_id}_{n}"
            n += 1
        (run_dir / "data").mkdir(parents=True)
        (run_dir / "charts").mkdir(parents=True)
        return cls(run_dir=run_dir, ledger=Ledger(), df=df, goal=goal)

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def set_dataframe(self, df: pd.DataFrame) -> str:
        """Install a transformed dataframe as the next dataset version and snapshot it."""
        number = int(self.dataset_version[1:]) + 1
        self.dataset_version = f"v{number}"
        self.df = df
        data_dir = self.run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_dir / f"{self.dataset_version}.csv", index=False)
        return self.dataset_version

    def save_ledger(self) -> Path:
        path = self.run_dir / "ledger.json"
        self.ledger.save(path)
        return path
