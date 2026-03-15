"""
Temporal workflow: 3-agent debt collection pipeline with outcome-based transitions.

Each stage runs a FULL multi-turn conversation (same loop as simulator).
Matches the spec flowchart for outcome-based routing.

Uses the same `agent_respond()` + `summarize_for_handoff()` code path as simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from workflow.activities import (
        AgentInput,
        AgentOutput,
        ConversationInput,
        ConversationOutput,
        HandoffInput,
        HandoffOutput,
        run_agent,
        run_agent_conversation,
        run_handoff,
    )


class PipelineOutcome(str, Enum):
    DEAL_AGREED = "deal_agreed"
    RESOLVED = "resolved"
    NO_RESOLUTION = "no_resolution"
    BORROWER_REFUSED = "borrower_refused"
    NO_RESPONSE = "no_response"


@dataclass
class PipelineInput:
    """Input to start a collection pipeline workflow."""
    agent1_prompt: str
    agent2_prompt: str
    agent3_prompt: str
    summarizer_prompt: str
    borrower_name: str = "Borrower"
    max_turns_per_stage: int = 10
    borrower_context_agent1: str | None = None
    borrower_context_agent2: str | None = None
    borrower_context_agent3: str | None = None
    borrower_system_prompt: str = ""  # For simulated borrower in automated mode
    borrower_voice_prompt: str = ""   # Voice persona for Agent 2
    max_assessment_retries: int = 3
    seed: int = 42
    automated: bool = True  # True = simulated borrower, False = real user (single-turn)


@dataclass
class PipelineResult:
    """Final result of the pipeline workflow."""
    outcome: str = PipelineOutcome.NO_RESOLUTION.value
    stage1_messages: list[dict[str, str]] = field(default_factory=list)
    stage2_messages: list[dict[str, str]] = field(default_factory=list)
    stage3_messages: list[dict[str, str]] = field(default_factory=list)
    handoff1_summary: str = ""
    handoff2_summary: str = ""
    completed_stages: int = 0
    assessment_attempts: int = 0


_ASSESSMENT_RETRY = RetryPolicy(
    maximum_attempts=1,  # We handle retries in workflow logic
    initial_interval=timedelta(seconds=1),
)

_DEFAULT_RETRY = RetryPolicy(
    maximum_attempts=2,
    initial_interval=timedelta(seconds=1),
)

# Outcome detection
_DEAL_SIGNALS = ["i agree", "i accept", "let's do it", "sounds good", "i'll take"]
_REFUSAL_SIGNALS = ["stop contacting", "don't call", "leave me alone", "do not contact"]
_RESOLUTION_SIGNALS = ["i agree", "i accept", "i'll pay", "sounds good", "deal"]


def _has_deal(messages: list[dict[str, str]]) -> bool:
    all_text = " ".join(m["content"].lower() for m in messages)
    return any(s in all_text for s in _DEAL_SIGNALS)


def _has_refusal(messages: list[dict[str, str]]) -> bool:
    borrower_msgs = [m["content"].lower() for m in messages if m["role"] == "user"]
    return any(s in " ".join(borrower_msgs) for s in _REFUSAL_SIGNALS)


@workflow.defn
class CollectionPipelineWorkflow:
    """
    Orchestrates the 3-agent pipeline with FULL multi-turn conversations.

    In automated mode: each stage runs a complete agent-borrower conversation
    loop (same as simulator). In production mode: single-turn per activity,
    conversation loop managed by UI/voice transport.
    """

    @workflow.run
    async def run(self, input: PipelineInput) -> PipelineResult:
        result = PipelineResult()

        # ==================================================================
        # Stage 1: Assessment (Chat) — with retry on no response
        # ==================================================================
        stage1_messages: list[dict[str, str]] = []

        for attempt in range(input.max_assessment_retries):
            result.assessment_attempts = attempt + 1

            if input.automated:
                # Full multi-turn conversation with simulated borrower
                conv_output = await workflow.execute_activity(
                    run_agent_conversation,
                    ConversationInput(
                        system_prompt=input.agent1_prompt,
                        handoff_context=None,
                        agent_type="agent1",
                        borrower_system_prompt=input.borrower_system_prompt,
                        max_turns=input.max_turns_per_stage,
                        borrower_context=input.borrower_context_agent1,
                        seed=input.seed + attempt,
                    ),
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=_ASSESSMENT_RETRY,
                )
                stage1_messages = conv_output.messages
            else:
                # Single turn — real user provides responses via UI
                agent_output = await workflow.execute_activity(
                    run_agent,
                    AgentInput(
                        system_prompt=input.agent1_prompt,
                        handoff_context=None,
                        conversation_history=[],
                        agent_type="agent1",
                        borrower_context=input.borrower_context_agent1,
                    ),
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=_ASSESSMENT_RETRY,
                )
                stage1_messages = [{"role": "assistant", "content": agent_output.response_text}]

            # Check if borrower responded (automated mode)
            borrower_msgs = [m for m in stage1_messages if m["role"] == "user"]
            if borrower_msgs:
                break  # Got responses, proceed

        result.stage1_messages = stage1_messages
        result.completed_stages = 1

        if _has_refusal(stage1_messages):
            result.outcome = PipelineOutcome.BORROWER_REFUSED.value
            return result

        # ==================================================================
        # Handoff 1: Assessment → Resolution
        # ==================================================================
        handoff1 = await workflow.execute_activity(
            run_handoff,
            HandoffInput(
                transcript_messages=stage1_messages,
                prior_summary=None,
                summarizer_prompt=input.summarizer_prompt,
                source_agent="agent1",
                target_agent="agent2",
            ),
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=_DEFAULT_RETRY,
        )
        result.handoff1_summary = handoff1.summary_text

        # ==================================================================
        # Stage 2: Resolution (Voice) — full conversation
        # ==================================================================
        stage2_handoff = handoff1.summary_text
        if input.borrower_name and input.borrower_name != "Borrower":
            stage2_handoff = f"BORROWER_ID: {input.borrower_name}\n{handoff1.summary_text}"

        if input.automated:
            conv_output = await workflow.execute_activity(
                run_agent_conversation,
                ConversationInput(
                    system_prompt=input.agent2_prompt,
                    handoff_context=stage2_handoff,
                    agent_type="agent2",
                    borrower_system_prompt=input.borrower_voice_prompt or input.borrower_system_prompt,
                    max_turns=input.max_turns_per_stage,
                    borrower_context=input.borrower_context_agent2,
                    seed=input.seed,
                ),
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=_DEFAULT_RETRY,
            )
            stage2_messages = conv_output.messages
        else:
            agent_output = await workflow.execute_activity(
                run_agent,
                AgentInput(
                    system_prompt=input.agent2_prompt,
                    handoff_context=stage2_handoff,
                    conversation_history=[],
                    agent_type="agent2",
                    borrower_context=input.borrower_context_agent2,
                ),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=_DEFAULT_RETRY,
            )
            stage2_messages = [{"role": "assistant", "content": agent_output.response_text}]

        result.stage2_messages = stage2_messages
        result.completed_stages = 2

        if _has_refusal(stage2_messages):
            result.outcome = PipelineOutcome.BORROWER_REFUSED.value
            return result

        if _has_deal(stage2_messages):
            result.outcome = PipelineOutcome.DEAL_AGREED.value
            return result

        # ==================================================================
        # Handoff 2: Resolution → Final Notice
        # ==================================================================
        handoff2 = await workflow.execute_activity(
            run_handoff,
            HandoffInput(
                transcript_messages=stage2_messages,
                prior_summary=handoff1.summary_text,
                summarizer_prompt=input.summarizer_prompt,
                source_agent="agent2",
                target_agent="agent3",
            ),
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=_DEFAULT_RETRY,
        )
        result.handoff2_summary = handoff2.summary_text

        # ==================================================================
        # Stage 3: Final Notice (Chat) — full conversation
        # ==================================================================
        if input.automated:
            conv_output = await workflow.execute_activity(
                run_agent_conversation,
                ConversationInput(
                    system_prompt=input.agent3_prompt,
                    handoff_context=handoff2.summary_text,
                    agent_type="agent3",
                    borrower_system_prompt=input.borrower_system_prompt,
                    max_turns=input.max_turns_per_stage,
                    borrower_context=input.borrower_context_agent3,
                    seed=input.seed,
                ),
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=_DEFAULT_RETRY,
            )
            stage3_messages = conv_output.messages
        else:
            agent_output = await workflow.execute_activity(
                run_agent,
                AgentInput(
                    system_prompt=input.agent3_prompt,
                    handoff_context=handoff2.summary_text,
                    conversation_history=[],
                    agent_type="agent3",
                    borrower_context=input.borrower_context_agent3,
                ),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=_DEFAULT_RETRY,
            )
            stage3_messages = [{"role": "assistant", "content": agent_output.response_text}]

        result.stage3_messages = stage3_messages
        result.completed_stages = 3

        if _has_refusal(stage3_messages):
            result.outcome = PipelineOutcome.BORROWER_REFUSED.value
            return result

        if _has_deal(stage3_messages):
            result.outcome = PipelineOutcome.RESOLVED.value
            return result

        result.outcome = PipelineOutcome.NO_RESOLUTION.value
        return result
