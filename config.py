"""
Typed configuration for the Darwin Voice Agent system.

All configuration flows through the Settings singleton. Environment variables
override defaults via pydantic-settings. Every numeric constant that controls
system behavior lives here — no magic numbers in business logic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-configs (frozen value objects — no mutation after init)
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel, frozen=True):
    """Pinned LLM model identifiers. Pinning avoids silent behavior changes."""
    sim: str = "gpt-4o-mini-2024-07-18"
    eval: str = "gpt-4o-mini-2024-07-18"
    judge: str = "gpt-4o-2024-08-06"
    rewrite: str = "gpt-4o-2024-08-06"


class TemperatureConfig(BaseModel, frozen=True):
    eval: float = 0.0       # Deterministic for reproducibility
    sim: float = 0.7        # Variance in borrower simulation


class TokenBudget(BaseModel, frozen=True):
    """Hard token constraints — enforced at runtime, not aspirational."""
    max_agent: int = 2000
    max_handoff: int = 500
    agent1_prompt: int = 2000   # Full budget (no prior context)
    agent2_prompt: int = 1500   # 2000 - 500 handoff
    agent3_prompt: int = 1500   # 2000 - 500 handoff


class SettlementPolicy(BaseModel, frozen=True):
    """Policy-defined ranges for compliance Rule 4. Offers outside = violation."""
    lump_sum_discount_min: float = 0.60
    lump_sum_discount_max: float = 0.80
    payment_plan_months_min: int = 3
    payment_plan_months_max: int = 12


class SimulationConfig(BaseModel, frozen=True):
    conversation_turns: int = 10
    personas_per_eval: int = 5
    runs_per_persona: int = 5

    @property
    def total_convos(self) -> int:
        return self.personas_per_eval * self.runs_per_persona


class EvolutionConfig(BaseModel, frozen=True):
    max_generations: int = 15
    children_per_generation: int = 2
    success_threshold: float = 8.5
    plateau_generations: int = 3
    confidence_level: float = 0.95
    bootstrap_n: int = 1000


class MetaEvalConfig(BaseModel, frozen=True):
    frequency: int = 3                  # Every N generations
    scorer_consistency_runs: int = 3
    scorer_variance_threshold: float = 1.5


class CostConfig(BaseModel, frozen=True):
    budget_limit: float = 20.0
    budget_reserve: float = 2.0         # Stop evolution below this


class ModelPricing(BaseModel, frozen=True):
    """Per-1M-token pricing. Used by cost tracker for accounting."""
    input: float
    output: float


# ---------------------------------------------------------------------------
# Root Settings — the single source of truth
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Application settings. Loads OPENAI_API_KEY from environment / .env file.
    All other config uses sensible defaults that match the spec constraints.
    """
    model_config_pydantic: dict[str, Any] = {}  # reserved for pydantic-settings

    # Secrets (from env)
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # Project root (derived at init)
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)

    # Sub-configs
    models: ModelConfig = Field(default_factory=ModelConfig)
    temperature: TemperatureConfig = Field(default_factory=TemperatureConfig)
    tokens: TokenBudget = Field(default_factory=TokenBudget)
    settlement: SettlementPolicy = Field(default_factory=SettlementPolicy)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    meta_eval: MetaEvalConfig = Field(default_factory=MetaEvalConfig)
    cost: CostConfig = Field(default_factory=CostConfig)

    # Pricing lookup
    pricing: dict[str, ModelPricing] = Field(default_factory=lambda: {
        "gpt-4o-mini-2024-07-18": ModelPricing(input=0.15, output=0.60),
        "gpt-4o-2024-08-06": ModelPricing(input=2.50, output=10.00),
    })

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def _ensure_api_key(self) -> "Settings":
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set in environment or .env file"
            )
        return self

    # --- Derived paths (all under project_root/logs/) ---

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def archive_file(self) -> Path:
        return self.logs_dir / "archive.json"

    @property
    def costs_file(self) -> Path:
        return self.logs_dir / "costs.json"

    @property
    def token_budgets_file(self) -> Path:
        return self.logs_dir / "token_budgets.json"

    @property
    def raw_scores_file(self) -> Path:
        return self.logs_dir / "raw_scores.csv"

    @property
    def eval_versions_dir(self) -> Path:
        return self.logs_dir / "eval_versions"

    @property
    def conversations_dir(self) -> Path:
        return self.logs_dir / "conversations"

    def ensure_dirs(self) -> None:
        """Create all log directories. Idempotent."""
        for d in [
            self.logs_dir,
            self.eval_versions_dir,
            self.conversations_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def get_pricing(self, model: str) -> ModelPricing:
        """Look up pricing for a model. Raises KeyError with clear message."""
        if model not in self.pricing:
            raise KeyError(
                f"No pricing configured for model '{model}'. "
                f"Known models: {list(self.pricing.keys())}"
            )
        return self.pricing[model]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor. Cached after first call."""
    return Settings()
