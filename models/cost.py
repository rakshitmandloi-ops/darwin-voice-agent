"""Cost tracking models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from models.enums import CostCategory


class CostEntry(BaseModel, frozen=True):
    """Single LLM API call cost record."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str
    category: CostCategory
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class CostBreakdown(BaseModel, frozen=True):
    """Aggregated cost report."""
    total_usd: float
    by_category: dict[str, float]
    by_model: dict[str, float]
    call_counts: dict[str, int]     # conversations_simulated, evaluation_calls, etc.
    remaining_budget: float
