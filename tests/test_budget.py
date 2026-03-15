"""
Tests for token budget enforcement.
"""

from __future__ import annotations

import pytest

from agents.prompts import TokenBudgetExceeded, count_tokens, enforce_budget
from agents.core import agent_respond
from models import AgentType


def test_count_tokens_basic():
    """count_tokens returns a positive integer for non-empty text."""
    tokens = count_tokens("Hello, world!")
    assert isinstance(tokens, int)
    assert tokens > 0


def test_count_tokens_empty():
    """count_tokens returns 0 for empty string."""
    assert count_tokens("") == 0


def test_enforce_budget_passes():
    """enforce_budget returns usage dict when within limit."""
    usage = enforce_budget(
        prompt="Short prompt",
        handoff=None,
        limit=2000,
        agent_name="test",
    )
    assert usage["passed"] is True
    assert usage["total"] <= 2000


def test_enforce_budget_with_handoff():
    """enforce_budget counts both prompt and handoff tokens."""
    usage = enforce_budget(
        prompt="Short prompt",
        handoff="Short handoff",
        limit=2000,
        agent_name="test",
    )
    assert usage["passed"] is True
    assert usage["handoff_tokens"] > 0
    assert usage["total"] == usage["prompt_tokens"] + usage["handoff_tokens"]


def test_enforce_budget_raises_when_exceeded():
    """enforce_budget raises TokenBudgetExceeded when over limit."""
    # Create a prompt that's definitely over 10 tokens
    long_prompt = "This is a very long prompt that will exceed " * 100

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        enforce_budget(
            prompt=long_prompt,
            handoff=None,
            limit=10,
            agent_name="test_agent",
        )

    assert "test_agent" in str(exc_info.value)
    assert "exceeds limit" in str(exc_info.value)


def test_enforce_budget_raises_with_handoff_overflow():
    """enforce_budget raises when prompt + handoff combined exceed limit."""
    prompt = "A reasonable prompt " * 50  # ~200 tokens
    handoff = "A handoff context " * 200  # ~800 tokens

    # Set limit below combined total
    with pytest.raises(TokenBudgetExceeded):
        enforce_budget(
            prompt=prompt,
            handoff=handoff,
            limit=100,
            agent_name="test",
        )


@pytest.mark.asyncio
async def test_agent_respond_enforces_budget(mock_tracker, mock_settings):
    """agent_respond() raises TokenBudgetExceeded for oversized prompts."""
    huge_prompt = "word " * 5000  # Way over 2000 tokens

    with pytest.raises(TokenBudgetExceeded):
        await agent_respond(
            system_prompt=huge_prompt,
            handoff_context=None,
            conversation_history=[],
            agent_type=AgentType.ASSESSMENT,
            tracker=mock_tracker,
            settings=mock_settings,
        )
