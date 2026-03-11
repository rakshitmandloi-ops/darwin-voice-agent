"""
Conversation simulator for the 3-agent debt collection pipeline.

Runs agents against LLM-simulated borrowers. Each agent stage produces a
transcript, which is summarized into a handoff for the next stage.

Full pipeline: Agent 1 (chat) → summarize → Agent 2 (voice) → summarize → Agent 3 (chat)
"""

from __future__ import annotations

import uuid
from typing import Any

from agents.prompts import count_tokens, enforce_budget, log_token_usage
from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from handoff.summarizer import summarize_for_handoff
from models import (
    AgentConfig,
    AgentType,
    CostCategory,
    Conversation,
    HandoffSummary,
    Message,
    Outcome,
    Persona,
    Transcript,
)


async def _run_agent_conversation(
    *,
    agent_type: AgentType,
    system_prompt: str,
    handoff_context: str | None,
    persona: Persona,
    max_turns: int,
    tracker: CostTracker,
    settings: Settings,
    seed: int,
) -> Transcript:
    """
    Run a multi-turn conversation between an agent and a simulated borrower.

    The agent uses `system_prompt` + optional `handoff_context`.
    The borrower uses the persona's system prompt (chat or voice variant).

    Returns the full transcript (excluding system messages for cleanliness,
    but the system prompt is counted in the token budget).
    """
    # --- Build agent messages ---
    agent_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if handoff_context:
        agent_messages.append({
            "role": "system",
            "content": f"HANDOFF CONTEXT FROM PRIOR STAGE:\n{handoff_context}",
        })

    # --- Build borrower messages ---
    is_voice = agent_type == AgentType.RESOLUTION
    borrower_prompt = persona.voice_system_prompt if is_voice else persona.system_prompt
    borrower_messages: list[dict[str, str]] = [
        {"role": "system", "content": borrower_prompt},
    ]

    # --- Token budget enforcement ---
    prompt_budget = {
        AgentType.ASSESSMENT: settings.tokens.agent1_prompt,
        AgentType.RESOLUTION: settings.tokens.agent2_prompt,
        AgentType.FINAL_NOTICE: settings.tokens.agent3_prompt,
    }[agent_type]

    usage = enforce_budget(
        prompt=system_prompt,
        handoff=handoff_context,
        limit=settings.tokens.max_agent,
        agent_name=agent_type.value,
    )
    log_token_usage(usage, settings)

    # --- Conversation loop ---
    transcript_messages: list[Message] = []

    # Agent speaks first
    for turn in range(max_turns):
        # --- Agent turn ---
        agent_response = await tracker.tracked_completion(
            model=settings.models.sim,
            messages=agent_messages,
            category=CostCategory.SIMULATION,
            temperature=settings.temperature.sim,
            metadata={"agent": agent_type.value, "turn": turn, "seed": seed},
        )
        agent_text = agent_response.choices[0].message.content or ""
        agent_messages.append({"role": "assistant", "content": agent_text})
        transcript_messages.append(Message(role="assistant", content=agent_text))

        # Add agent message to borrower's view (as "user" from borrower's perspective)
        borrower_messages.append({"role": "user", "content": agent_text})

        # Check for conversation-ending signals from agent
        if _is_conversation_ending(agent_text):
            break

        # --- Borrower turn ---
        if turn < max_turns - 1:  # Don't generate borrower response on last turn
            borrower_response = await tracker.tracked_completion(
                model=settings.models.sim,
                messages=borrower_messages,
                category=CostCategory.SIMULATION,
                temperature=settings.temperature.sim,
                metadata={"persona": persona.name, "turn": turn, "seed": seed},
            )
            borrower_text = borrower_response.choices[0].message.content or ""
            borrower_messages.append({"role": "assistant", "content": borrower_text})
            transcript_messages.append(Message(role="user", content=borrower_text))

            # Add borrower message to agent's view
            agent_messages.append({"role": "user", "content": borrower_text})

            # Check for borrower ending signals
            if _borrower_wants_to_stop(borrower_text):
                break

    return Transcript(messages=tuple(transcript_messages))


def _is_conversation_ending(text: str) -> bool:
    """Detect if agent is wrapping up the conversation."""
    lower = text.lower()
    endings = [
        "thank you for your time",
        "we'll be in touch",
        "i'll end this conversation",
        "this concludes our",
        "have a good day",
        "goodbye",
    ]
    return any(e in lower for e in endings)


def _borrower_wants_to_stop(text: str) -> bool:
    """Detect if borrower wants to stop contact."""
    lower = text.lower()
    stops = [
        "stop contacting me",
        "don't call me again",
        "leave me alone",
        "i refuse to",
        "remove me from",
        "do not contact",
    ]
    return any(s in lower for s in stops)


def _determine_outcome(
    agent1_transcript: Transcript,
    agent2_transcript: Transcript | None = None,
    agent3_transcript: Transcript | None = None,
) -> Outcome:
    """
    Determine conversation outcome from transcripts.

    Scans for agreement signals, refusals, hardship, or non-response.
    """
    all_text = agent1_transcript.text
    if agent2_transcript:
        all_text += "\n" + agent2_transcript.text
    if agent3_transcript:
        all_text += "\n" + agent3_transcript.text

    lower = all_text.lower()

    if any(s in lower for s in ["stop contacting", "don't call", "leave me alone", "do not contact"]):
        return Outcome.BORROWER_REFUSED
    if any(s in lower for s in ["hardship program", "hardship referral", "connect you with hardship"]):
        if any(s in lower for s in ["yes", "please", "that would help", "i'd like that"]):
            return Outcome.HARDSHIP_REFERRAL
    if any(s in lower for s in ["i agree", "i accept", "let's do it", "sounds good", "i'll take"]):
        return Outcome.DEAL_AGREED
    if agent1_transcript.turn_count < 2:
        return Outcome.NO_RESPONSE

    return Outcome.NO_DEAL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def simulate_pipeline(
    *,
    agent_config: AgentConfig,
    persona: Persona,
    seed: int,
    tracker: CostTracker,
    settings: Settings | None = None,
) -> Conversation:
    """
    Run the full 3-agent pipeline as text simulation.

    Agent 1 (Assessment/chat) → summarize → Agent 2 (Resolution/voice-simulated)
    → summarize → Agent 3 (Final Notice/chat)

    If the borrower requests stop-contact at any stage, subsequent stages
    are skipped. The Conversation object captures the full state.
    """
    s = settings or get_settings()
    conv_id = f"conv-{uuid.uuid4().hex[:8]}"

    # ---- Stage 1: Assessment (Chat) ----
    agent1_transcript = await _run_agent_conversation(
        agent_type=AgentType.ASSESSMENT,
        system_prompt=agent_config.agent1_prompt,
        handoff_context=None,
        persona=persona,
        max_turns=s.simulation.conversation_turns,
        tracker=tracker,
        settings=s,
        seed=seed,
    )

    # Check for early exit
    stop_contact = _borrower_refused_in_transcript(agent1_transcript)
    if stop_contact:
        return Conversation(
            conversation_id=conv_id,
            persona=persona,
            seed=seed,
            agent1_transcript=agent1_transcript,
            outcome=Outcome.BORROWER_REFUSED,
            stop_contact=True,
        )

    # ---- Handoff 1: Agent 1 → Agent 2 ----
    handoff_1 = await summarize_for_handoff(
        transcript=agent1_transcript,
        prior_summary=None,
        summarizer_prompt=agent_config.summarizer_prompt,
        source_agent=AgentType.ASSESSMENT,
        target_agent=AgentType.RESOLUTION,
        tracker=tracker,
        settings=s,
    )

    # ---- Stage 2: Resolution (Voice-simulated) ----
    agent2_transcript = await _run_agent_conversation(
        agent_type=AgentType.RESOLUTION,
        system_prompt=agent_config.agent2_prompt,
        handoff_context=handoff_1.text,
        persona=persona,
        max_turns=s.simulation.conversation_turns,
        tracker=tracker,
        settings=s,
        seed=seed,
    )

    # Check if deal agreed during Resolution — exit early
    stage2_outcome = _determine_outcome(agent1_transcript, agent2_transcript)
    if stage2_outcome == Outcome.DEAL_AGREED:
        return Conversation(
            conversation_id=conv_id,
            persona=persona,
            seed=seed,
            agent1_transcript=agent1_transcript,
            agent2_transcript=agent2_transcript,
            handoff_1=handoff_1,
            outcome=Outcome.DEAL_AGREED,
            stop_contact=False,
        )

    stop_contact = _borrower_refused_in_transcript(agent2_transcript)
    if stop_contact:
        return Conversation(
            conversation_id=conv_id,
            persona=persona,
            seed=seed,
            agent1_transcript=agent1_transcript,
            agent2_transcript=agent2_transcript,
            handoff_1=handoff_1,
            outcome=Outcome.BORROWER_REFUSED,
            stop_contact=True,
        )

    # ---- Handoff 2: Agent 1+2 → Agent 3 (combined into single ≤500 token summary) ----
    handoff_2 = await summarize_for_handoff(
        transcript=agent2_transcript,
        prior_summary=handoff_1.text,
        summarizer_prompt=agent_config.summarizer_prompt,
        source_agent=AgentType.RESOLUTION,
        target_agent=AgentType.FINAL_NOTICE,
        tracker=tracker,
        settings=s,
    )

    # ---- Stage 3: Final Notice (Chat) ----
    agent3_transcript = await _run_agent_conversation(
        agent_type=AgentType.FINAL_NOTICE,
        system_prompt=agent_config.agent3_prompt,
        handoff_context=handoff_2.text,
        persona=persona,
        max_turns=s.simulation.conversation_turns,
        tracker=tracker,
        settings=s,
        seed=seed,
    )

    # ---- Final outcome ----
    outcome = _determine_outcome(agent1_transcript, agent2_transcript, agent3_transcript)
    stop_contact = _borrower_refused_in_transcript(agent3_transcript)

    return Conversation(
        conversation_id=conv_id,
        persona=persona,
        seed=seed,
        agent1_transcript=agent1_transcript,
        agent2_transcript=agent2_transcript,
        agent3_transcript=agent3_transcript,
        handoff_1=handoff_1,
        handoff_2=handoff_2,
        outcome=outcome,
        stop_contact=stop_contact,
    )


def _borrower_refused_in_transcript(transcript: Transcript) -> bool:
    """Check if borrower explicitly asked to stop contact in this transcript."""
    for msg in transcript.messages:
        if msg.role == "user" and _borrower_wants_to_stop(msg.content):
            return True
    return False


async def simulate_agent1(
    *,
    agent_config: AgentConfig,
    persona: Persona,
    seed: int,
    tracker: CostTracker,
    settings: Settings | None = None,
) -> Conversation:
    """
    Run Agent 1 (Assessment) conversation only.
    Useful for testing Agent 1 in isolation.
    """
    s = settings or get_settings()

    transcript = await _run_agent_conversation(
        agent_type=AgentType.ASSESSMENT,
        system_prompt=agent_config.agent1_prompt,
        handoff_context=None,
        persona=persona,
        max_turns=s.simulation.conversation_turns,
        tracker=tracker,
        settings=s,
        seed=seed,
    )

    outcome = _determine_outcome(transcript)
    stop_contact = outcome == Outcome.BORROWER_REFUSED

    return Conversation(
        conversation_id=f"conv-{uuid.uuid4().hex[:8]}",
        persona=persona,
        seed=seed,
        agent1_transcript=transcript,
        outcome=outcome,
        stop_contact=stop_contact,
    )
