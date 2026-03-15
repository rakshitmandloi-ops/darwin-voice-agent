"""
Single agent response function — the ONE code path for both simulation and production.

`agent_respond()` is stateless: it takes everything needed to produce one agent
response, makes one LLM call, and returns the text. The caller (simulator,
Temporal activity, Chainlit handler, Pipecat processor) owns the conversation
loop, history management, and state.
"""

from __future__ import annotations

from typing import Any

from agents.prompts import enforce_budget, log_token_usage
from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from models import AgentType, CostCategory


async def agent_respond(
    *,
    system_prompt: str,
    handoff_context: str | None,
    conversation_history: list[dict[str, str]],
    agent_type: AgentType,
    tracker: CostTracker,
    settings: Settings | None = None,
    temperature: float | None = None,
    metadata: dict[str, Any] | None = None,
    borrower_context: str | None = None,
) -> str:
    """
    Generate a single agent response. This is the ONLY function that turns
    agent config + conversation state into an LLM response.

    Args:
        system_prompt: The agent's system prompt (from AgentConfig).
        handoff_context: Prior-stage summary text, or None for Agent 1.
        conversation_history: List of {"role": ..., "content": ...} dicts
            representing the conversation so far (assistant + user turns).
        agent_type: Which agent is responding (for budget + metadata).
        tracker: Cost tracker for the LLM call.
        settings: App settings (uses singleton if None).
        temperature: Override temperature (defaults to sim temperature).
        metadata: Extra metadata for cost tracking.
        borrower_context: Optional borrower data block to prepend to the
            system prompt. Counted toward the token budget.

    Returns:
        The agent's response text.

    Raises:
        TokenBudgetExceeded: If system_prompt + handoff_context exceeds budget.
        BudgetExhaustedError: If the cost budget is exhausted.
    """
    s = settings or get_settings()

    # Prepend borrower context to system prompt if provided
    effective_prompt = system_prompt
    if borrower_context:
        effective_prompt = f"{borrower_context}\n\n{system_prompt}"

    # Enforce token budget on system prompt + handoff (hard constraint)
    usage = enforce_budget(
        prompt=effective_prompt,
        handoff=handoff_context,
        limit=s.tokens.max_agent,
        agent_name=agent_type.value,
    )
    log_token_usage(usage, s)

    # Build messages for the LLM call
    messages: list[dict[str, str]] = [{"role": "system", "content": effective_prompt}]

    if handoff_context:
        messages.append({
            "role": "system",
            "content": f"HANDOFF CONTEXT FROM PRIOR STAGE:\n{handoff_context}",
        })

    # Append conversation history
    messages.extend(conversation_history)

    # Single LLM call
    response = await tracker.tracked_completion(
        model=s.models.sim,
        messages=messages,
        category=CostCategory.SIMULATION,
        temperature=temperature if temperature is not None else s.temperature.sim,
        metadata=metadata or {"agent": agent_type.value},
    )

    return response.choices[0].message.content or ""
