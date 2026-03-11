"""
Handoff summarizer: compresses agent transcripts into structured context
that fits within the 500-token budget.

The summarizer is itself an evolvable component — the prompt can be
mutated by the evolution loop. This module handles the mechanics:
calling the LLM, enforcing the token budget, and retrying if over.
"""

from __future__ import annotations

from agents.prompts import count_tokens
from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from models import (
    AgentType,
    CostCategory,
    HandoffSummary,
    Transcript,
)


class HandoffBudgetExceeded(Exception):
    """Raised when summarizer cannot compress within budget after retries."""


async def summarize_for_handoff(
    *,
    transcript: Transcript,
    prior_summary: str | None,
    summarizer_prompt: str,
    source_agent: AgentType,
    target_agent: AgentType,
    tracker: CostTracker,
    settings: Settings | None = None,
    max_retries: int = 2,
) -> HandoffSummary:
    """
    Summarize a transcript (+ optional prior summary) into <=500 tokens.

    For Agent 1→2: transcript is Agent 1's conversation, no prior summary.
    For Agent 2→3: transcript is Agent 2's conversation, prior_summary is
    the Agent 1 handoff. Both must be compressed into a single 500-token summary.

    If the first attempt exceeds budget, retries with a stricter instruction.
    Raises HandoffBudgetExceeded if it still can't fit after max_retries.
    """
    s = settings or get_settings()
    budget = s.tokens.max_handoff

    # Build the input for the summarizer
    input_text = ""
    if prior_summary:
        input_text += f"PRIOR STAGE SUMMARY:\n{prior_summary}\n\n"
    input_text += f"CONVERSATION TRANSCRIPT:\n{transcript.text}"

    summary_text = await _generate_summary(
        summarizer_prompt=summarizer_prompt,
        input_text=input_text,
        budget=budget,
        tracker=tracker,
        settings=s,
    )

    token_count = count_tokens(summary_text)

    # Retry with stricter prompt if over budget
    for attempt in range(max_retries):
        if token_count <= budget:
            break

        summary_text = await _generate_summary(
            summarizer_prompt=summarizer_prompt,
            input_text=input_text,
            budget=budget,
            tracker=tracker,
            settings=s,
            strict_mode=True,
            previous_summary=summary_text,
            previous_tokens=token_count,
        )
        token_count = count_tokens(summary_text)

    if token_count > budget:
        # Last resort: hard truncate at token level
        summary_text = _truncate_to_budget(summary_text, budget)
        token_count = count_tokens(summary_text)

    return HandoffSummary(
        text=summary_text,
        token_count=token_count,
        source_agent=source_agent,
        target_agent=target_agent,
    )


async def _generate_summary(
    *,
    summarizer_prompt: str,
    input_text: str,
    budget: int,
    tracker: CostTracker,
    settings: Settings,
    strict_mode: bool = False,
    previous_summary: str | None = None,
    previous_tokens: int | None = None,
) -> str:
    """Call the LLM to generate a summary."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": summarizer_prompt},
    ]

    if strict_mode and previous_summary and previous_tokens:
        messages.append({
            "role": "user",
            "content": (
                f"Your previous summary was {previous_tokens} tokens, "
                f"which exceeds the hard limit of {budget} tokens. "
                f"Compress this further — remove less critical details, "
                f"use shorter phrasing, drop examples. Keep only what the "
                f"next agent absolutely needs.\n\n"
                f"Previous summary to compress:\n{previous_summary}"
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": (
                f"Summarize the following in UNDER {budget} tokens. "
                f"Be extremely concise.\n\n{input_text}"
            ),
        })

    response = await tracker.tracked_completion(
        model=settings.models.eval,
        messages=messages,
        category=CostCategory.SUMMARIZATION,
        temperature=0.0,
        metadata={
            "strict_mode": strict_mode,
            "budget": budget,
        },
    )

    return response.choices[0].message.content or ""


def _truncate_to_budget(text: str, budget: int) -> str:
    """
    Hard truncate text to fit within token budget.
    Last resort — loses information but guarantees budget compliance.
    """
    import tiktoken
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoder.encode(text)
    if len(tokens) <= budget:
        return text
    truncated_tokens = tokens[:budget]
    return encoder.decode(truncated_tokens)
