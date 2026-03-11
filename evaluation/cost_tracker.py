"""
Cost tracking for all LLM API calls.

Every OpenAI call in the system goes through `tracked_completion()`. This
ensures we have:
1. Accurate per-call cost accounting (input + output tokens × model pricing)
2. Running budget enforcement (hard stop when reserve is hit)
3. Call counts by category for the spec-required cost breakdown
4. Persistent log to disk for audit trail

Thread-safe via threading.Lock — safe for concurrent Temporal activities.

Usage:
    tracker = CostTracker(settings)
    response = await tracker.tracked_completion(
        model=settings.models.sim,
        messages=[...],
        category=CostCategory.SIMULATION,
    )
    # response is the standard OpenAI ChatCompletion object
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from config import Settings, get_settings
from models import CostBreakdown, CostCategory, CostEntry


class BudgetExhaustedError(Exception):
    """Raised when remaining budget is below the reserve threshold."""


class CostTracker:
    """
    Wraps all OpenAI API calls with cost accounting and budget enforcement.

    Not a singleton — instantiate once at app startup and pass around.
    The internal state (entries list + totals) is protected by a lock.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._entries: list[CostEntry] = []
        self._total_usd: float = 0.0
        self._counts: dict[str, int] = {c.value: 0 for c in CostCategory}
        self._by_category: dict[str, float] = {c.value: 0.0 for c in CostCategory}
        self._by_model: dict[str, float] = {}
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

        # Load existing costs from disk (resume after restart)
        self._load_from_disk()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def tracked_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        category: CostCategory,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Make an OpenAI chat completion call with cost tracking.

        Raises BudgetExhaustedError if remaining budget < reserve BEFORE the call.
        Returns the raw OpenAI ChatCompletion response.
        """
        self._check_budget()

        if temperature is None:
            temperature = self._settings.temperature.eval

        # Retry on timeout/transient errors
        import asyncio as _asyncio
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=60.0,
                    **kwargs,
                )
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await _asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise last_error

        # Extract token usage from response
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Calculate cost
        pricing = self._settings.get_pricing(model)
        cost_usd = (
            (input_tokens * pricing.input / 1_000_000)
            + (output_tokens * pricing.output / 1_000_000)
        )

        entry = CostEntry(
            model=model,
            category=category,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

        self._record(entry)
        return response

    def get_remaining_budget(self) -> float:
        with self._lock:
            return self._settings.cost.budget_limit - self._total_usd

    def get_breakdown(self) -> CostBreakdown:
        with self._lock:
            return CostBreakdown(
                total_usd=self._total_usd,
                by_category=dict(self._by_category),
                by_model=dict(self._by_model),
                call_counts=dict(self._counts),
                remaining_budget=self._settings.cost.budget_limit - self._total_usd,
            )

    def is_budget_exhausted(self) -> bool:
        return self.get_remaining_budget() < self._settings.cost.budget_reserve

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _check_budget(self) -> None:
        if self.is_budget_exhausted():
            breakdown = self.get_breakdown()
            raise BudgetExhaustedError(
                f"Budget exhausted: ${breakdown.total_usd:.2f} spent of "
                f"${self._settings.cost.budget_limit:.2f} limit "
                f"(reserve: ${self._settings.cost.budget_reserve:.2f}). "
                f"Breakdown: {breakdown.by_category}"
            )

    def _record(self, entry: CostEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            self._total_usd += entry.cost_usd
            self._counts[entry.category.value] = (
                self._counts.get(entry.category.value, 0) + 1
            )
            self._by_category[entry.category.value] = (
                self._by_category.get(entry.category.value, 0.0) + entry.cost_usd
            )
            self._by_model[entry.model] = (
                self._by_model.get(entry.model, 0.0) + entry.cost_usd
            )

        self._persist(entry)

    def _persist(self, entry: CostEntry) -> None:
        """Append entry to the costs JSON file. One JSON object per line (JSONL)."""
        costs_file = self._settings.costs_file
        costs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(costs_file, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def _load_from_disk(self) -> None:
        """Restore state from existing cost log (idempotent on restart)."""
        costs_file = self._settings.costs_file
        if not costs_file.exists():
            return
        with open(costs_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = CostEntry.model_validate_json(line)
                # Update counters without re-persisting
                self._entries.append(entry)
                self._total_usd += entry.cost_usd
                self._counts[entry.category.value] = (
                    self._counts.get(entry.category.value, 0) + 1
                )
                self._by_category[entry.category.value] = (
                    self._by_category.get(entry.category.value, 0.0) + entry.cost_usd
                )
                self._by_model[entry.model] = (
                    self._by_model.get(entry.model, 0.0) + entry.cost_usd
                )
