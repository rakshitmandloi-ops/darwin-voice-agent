"""
Tests for agents/core.py — the single agent_respond() code path.
"""

from __future__ import annotations

import pytest

from agents.core import agent_respond
from agents.prompts import TokenBudgetExceeded
from models import AgentType


@pytest.mark.asyncio
async def test_agent_respond_returns_text(mock_tracker, mock_settings):
    """agent_respond() should return a non-empty string."""
    result = await agent_respond(
        system_prompt="You are a test agent.",
        handoff_context=None,
        conversation_history=[],
        agent_type=AgentType.ASSESSMENT,
        tracker=mock_tracker,
        settings=mock_settings,
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_agent_respond_with_handoff(mock_tracker, mock_settings):
    """agent_respond() works with handoff context."""
    result = await agent_respond(
        system_prompt="You are a test agent.",
        handoff_context="Prior summary: borrower is cooperative.",
        conversation_history=[],
        agent_type=AgentType.RESOLUTION,
        tracker=mock_tracker,
        settings=mock_settings,
    )
    assert isinstance(result, str)

    # Verify the LLM call included handoff in messages
    last_call = mock_tracker.last_call
    assert last_call is not None
    system_messages = [m for m in last_call["messages"] if m["role"] == "system"]
    assert len(system_messages) == 2  # system prompt + handoff
    assert "HANDOFF CONTEXT" in system_messages[1]["content"]


@pytest.mark.asyncio
async def test_agent_respond_with_history(mock_tracker, mock_settings):
    """agent_respond() passes conversation history to the LLM."""
    history = [
        {"role": "assistant", "content": "Hello, how can I help?"},
        {"role": "user", "content": "I need to discuss my account."},
    ]
    result = await agent_respond(
        system_prompt="You are a test agent.",
        handoff_context=None,
        conversation_history=history,
        agent_type=AgentType.ASSESSMENT,
        tracker=mock_tracker,
        settings=mock_settings,
    )
    assert isinstance(result, str)

    # Messages should include system + 2 history entries
    last_call = mock_tracker.last_call
    assert len(last_call["messages"]) == 3  # 1 system + 2 history


@pytest.mark.asyncio
async def test_agent_respond_uses_simulation_category(mock_tracker, mock_settings):
    """agent_respond() should use SIMULATION cost category."""
    await agent_respond(
        system_prompt="You are a test agent.",
        handoff_context=None,
        conversation_history=[],
        agent_type=AgentType.ASSESSMENT,
        tracker=mock_tracker,
        settings=mock_settings,
    )
    from models import CostCategory
    assert mock_tracker.last_call["category"] == CostCategory.SIMULATION


@pytest.mark.asyncio
async def test_agent_respond_same_for_all_agent_types(mock_tracker, mock_settings):
    """All agent types use the same code path — just different prompts."""
    for agent_type in AgentType:
        result = await agent_respond(
            system_prompt=f"You are {agent_type.value}.",
            handoff_context="context" if agent_type != AgentType.ASSESSMENT else None,
            conversation_history=[],
            agent_type=agent_type,
            tracker=mock_tracker,
            settings=mock_settings,
        )
        assert isinstance(result, str)
