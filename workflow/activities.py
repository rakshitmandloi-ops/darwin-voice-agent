"""
Temporal activities for the 3-agent debt collection pipeline.

Each activity wraps `agent_respond()` + `summarize_for_handoff()` — the same
code path used by simulation. Activities are stateless; the workflow manages
conversation state and handoff context.

Two modes:
- `run_agent`: Single turn (for production — real user provides responses)
- `run_agent_conversation`: Full multi-turn with simulated borrower (for automated pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from temporalio import activity

from agents.core import agent_respond
from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from handoff.summarizer import summarize_for_handoff
from models import AgentType, CostCategory, HandoffSummary, Message, Transcript


@dataclass
class AgentInput:
    """Input for a single agent turn."""
    system_prompt: str
    handoff_context: str | None
    conversation_history: list[dict[str, str]]
    agent_type: str  # AgentType.value — serializable
    borrower_context: str | None = None


@dataclass
class AgentOutput:
    """Output from a single agent turn."""
    response_text: str
    agent_type: str


@dataclass
class ConversationInput:
    """Input for a full multi-turn agent conversation."""
    system_prompt: str
    handoff_context: str | None
    agent_type: str
    borrower_system_prompt: str
    max_turns: int = 10
    borrower_context: str | None = None
    seed: int = 42


@dataclass
class ConversationOutput:
    """Output from a full multi-turn conversation."""
    messages: list[dict[str, str]] = field(default_factory=list)
    agent_type: str = ""
    turns_completed: int = 0
    outcome: str = "no_deal"


@dataclass
class HandoffInput:
    """Input for a handoff summarization activity."""
    transcript_messages: list[dict[str, str]]
    prior_summary: str | None
    summarizer_prompt: str
    source_agent: str
    target_agent: str


@dataclass
class HandoffOutput:
    """Output from a handoff activity."""
    summary_text: str
    token_count: int


# Module-level tracker and settings — initialized by the worker before starting
_tracker: CostTracker | None = None
_settings: Settings | None = None


def init_activity_context(tracker: CostTracker, settings: Settings) -> None:
    """Called by the worker to inject shared dependencies."""
    global _tracker, _settings
    _tracker = tracker
    _settings = settings


def _get_tracker() -> CostTracker:
    if _tracker is None:
        raise RuntimeError("Activity context not initialized — call init_activity_context() first")
    return _tracker


def _get_settings() -> Settings:
    return _settings or get_settings()


# Conversation ending signals (same as simulator)
_ENDING_PHRASES = [
    "thank you for your time", "thank you for your cooperation",
    "we'll be in touch", "this concludes", "have a good day",
    "goodbye", "end this conversation", "take care",
]

_STOP_CONTACT_PHRASES = [
    "stop contacting me", "don't call me", "do not contact",
    "leave me alone", "stop calling", "remove me", "cease contact",
]


def _is_ending(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _ENDING_PHRASES)


def _wants_stop(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _STOP_CONTACT_PHRASES)


@activity.defn
async def run_agent(input: AgentInput) -> AgentOutput:
    """
    Run a single agent turn. Same code path as simulation.
    Used in production where the real user provides responses via UI/voice.
    """
    tracker = _get_tracker()
    settings = _get_settings()
    agent_type = AgentType(input.agent_type)

    response_text = await agent_respond(
        system_prompt=input.system_prompt,
        handoff_context=input.handoff_context,
        conversation_history=input.conversation_history,
        agent_type=agent_type,
        tracker=tracker,
        settings=settings,
        borrower_context=input.borrower_context,
    )

    return AgentOutput(
        response_text=response_text,
        agent_type=input.agent_type,
    )


@activity.defn
async def run_agent_conversation(input: ConversationInput) -> ConversationOutput:
    """
    Run a full multi-turn conversation between agent and simulated borrower.
    Same conversation loop as simulation/simulator.py::_run_agent_conversation().

    Used in automated Temporal pipeline runs (testing, demo, evolution).
    Agent responses go through agent_respond() — the single code path.
    Borrower responses are LLM-simulated.
    """
    tracker = _get_tracker()
    settings = _get_settings()
    agent_type = AgentType(input.agent_type)

    # Agent conversation history (for agent_respond)
    conversation_history: list[dict[str, str]] = []

    # Borrower simulation (separate context, not budget-constrained)
    borrower_messages: list[dict[str, str]] = [
        {"role": "system", "content": input.borrower_system_prompt},
    ]

    # Full transcript
    all_messages: list[dict[str, str]] = []

    for turn in range(input.max_turns):
        # --- Agent turn (via agent_respond — single code path) ---
        agent_text = await agent_respond(
            system_prompt=input.system_prompt,
            handoff_context=input.handoff_context,
            conversation_history=conversation_history,
            agent_type=agent_type,
            tracker=tracker,
            settings=settings,
            borrower_context=input.borrower_context,
        )

        conversation_history.append({"role": "assistant", "content": agent_text})
        all_messages.append({"role": "assistant", "content": agent_text})
        borrower_messages.append({"role": "user", "content": agent_text})

        if _is_ending(agent_text):
            break

        # --- Borrower turn (simulated via LLM) ---
        if turn < input.max_turns - 1:
            borrower_response = await tracker.tracked_completion(
                model=settings.models.sim,
                messages=borrower_messages,
                category=CostCategory.SIMULATION,
                temperature=settings.temperature.sim,
                metadata={"persona": "borrower", "turn": turn, "seed": input.seed},
            )
            borrower_text = borrower_response.choices[0].message.content or ""

            borrower_messages.append({"role": "assistant", "content": borrower_text})
            conversation_history.append({"role": "user", "content": borrower_text})
            all_messages.append({"role": "user", "content": borrower_text})

            if _wants_stop(borrower_text):
                break

    # Determine simple outcome
    all_text = " ".join(m["content"].lower() for m in all_messages)
    if any(s in all_text for s in _STOP_CONTACT_PHRASES):
        outcome = "borrower_refused"
    elif any(s in all_text for s in ["i agree", "i accept", "let's do it", "sounds good"]):
        outcome = "deal_agreed"
    elif "hardship" in all_text and any(s in all_text for s in ["yes", "please", "that would help"]):
        outcome = "hardship_referral"
    else:
        outcome = "no_deal"

    return ConversationOutput(
        messages=all_messages,
        agent_type=input.agent_type,
        turns_completed=len([m for m in all_messages if m["role"] == "assistant"]),
        outcome=outcome,
    )


@activity.defn
async def run_handoff(input: HandoffInput) -> HandoffOutput:
    """
    Summarize a transcript for handoff to the next agent.
    Same summarizer used by simulation.
    """
    tracker = _get_tracker()
    settings = _get_settings()

    messages = tuple(
        Message(role=m["role"], content=m["content"])
        for m in input.transcript_messages
    )
    transcript = Transcript(messages=messages)

    summary = await summarize_for_handoff(
        transcript=transcript,
        prior_summary=input.prior_summary,
        summarizer_prompt=input.summarizer_prompt,
        source_agent=AgentType(input.source_agent),
        target_agent=AgentType(input.target_agent),
        tracker=tracker,
        settings=settings,
    )

    return HandoffOutput(
        summary_text=summary.text,
        token_count=summary.token_count,
    )
