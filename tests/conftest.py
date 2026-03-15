"""
Shared test fixtures. Provides a mock CostTracker and settings that
do NOT require a real OpenAI API key or make real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

from config import Settings
from models import CostCategory, CostEntry


class MockCostTracker:
    """
    Drop-in replacement for CostTracker that returns canned responses
    instead of calling OpenAI. Thread-safe matching the real tracker.
    """

    def __init__(self, response_text: str = "Hello, I am an AI agent.") -> None:
        self._response_text = response_text
        self._calls: list[dict[str, Any]] = []

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
        self._calls.append({
            "model": model,
            "messages": messages,
            "category": category,
            "temperature": temperature,
            "metadata": metadata,
        })

        # Build a mock response matching OpenAI's ChatCompletion shape
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = self._response_text
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        return mock_response

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_call(self) -> dict[str, Any] | None:
        return self._calls[-1] if self._calls else None


@pytest.fixture
def mock_tracker() -> MockCostTracker:
    return MockCostTracker()


@pytest.fixture
def mock_settings(tmp_path) -> Settings:
    """Settings that work without .env or real API key."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-not-real"}):
        s = Settings(
            openai_api_key="test-key-not-real",
            project_root=tmp_path,
        )
        s.ensure_dirs()
        return s
