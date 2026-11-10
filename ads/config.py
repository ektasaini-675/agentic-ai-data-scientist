"""Central configuration. Everything tunable lives here, nothing is hard-coded elsewhere."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

try:  # the old prototype never loaded .env; do it here once, for every entry point
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 1500
    temperature: float = 0.0
    # "auto": live if ANTHROPIC_API_KEY is set, otherwise offline (scripted) mode
    # "live" | "offline" | "fake" (fake = replay scripted responses in tests)
    llm_mode: str = "auto"

    # --- agent loop budgets (stop runaway loops) ---
    max_agent_steps: int = 8
    max_retries_per_agent: int = 2
    max_replans: int = 2

    # --- reproducibility ---
    random_seed: int = 42

    # --- paths ---
    runs_dir: Path = ROOT / "runs"
    cache_dir: Path = ROOT / ".llm_cache"


def load_settings() -> Settings:
    """Read environment overrides at call time (keeps tests easy)."""
    return Settings(
        model=os.environ.get("ADS_MODEL", Settings.model),
        llm_mode=os.environ.get("ADS_LLM_MODE", Settings.llm_mode),
        max_agent_steps=int(os.environ.get("ADS_MAX_AGENT_STEPS", Settings.max_agent_steps)),
        random_seed=int(os.environ.get("ADS_SEED", Settings.random_seed)),
    )


SETTINGS = load_settings()

CONFIDENCE_RUBRIC = {
    "no_leakage": {"weight": 25, "desc": "No target leakage detected among features"},
    "adequate_sample": {"weight": 15, "desc": "Dataset size exceeds minimum threshold (>= 200 rows)"},
    "controlled_overfitting": {"weight": 20, "desc": "Train vs CV generalization gap within bound (<= 10%)"},
    "beats_baseline": {"weight": 15, "desc": "Model performance exceeds majority or naive baseline"},
    "verified_claims": {"weight": 15, "desc": "All report claims backed by evidence ledger"},
    "multiplicity_correction": {"weight": 10, "desc": "Benjamini-Hochberg multiple hypothesis correction applied"},
}


def resolve_provider() -> Optional[str]:
    """Which LLM provider to use: 'gemini', 'anthropic', or None (offline).

    ADS_PROVIDER forces a choice; otherwise whichever key is present (Gemini first, it is free).
    """
    forced = os.environ.get("ADS_PROVIDER", "").strip().lower()
    if forced in ("gemini", "anthropic"):
        return forced
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def resolve_llm_mode(settings: Settings = SETTINGS) -> str:
    """Return 'live' or 'offline' (or 'fake' if explicitly requested)."""
    if settings.llm_mode != "auto":
        return settings.llm_mode
    return "live" if resolve_provider() else "offline"
