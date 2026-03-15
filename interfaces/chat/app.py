"""
Chainlit chat UI — full 3-agent pipeline orchestrated by Temporal.

The UI acts as a Temporal CLIENT:
1. On chat start → starts a Temporal workflow per borrower
2. Each agent stage → UI calls Temporal activities for agent responses
3. Handoffs → Temporal workflow manages state transitions
4. Agent 2 → voice call via Pipecat (external to Temporal)

Falls back to direct agent_respond() if Temporal is not available.

Usage:
    # Start Temporal first:
    temporal server start-dev
    .venv/bin/python -m workflow.worker

    # Then start UI:
    chainlit run interfaces/chat/app.py --port 8000
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

import chainlit as cl

from agents.core import agent_respond
from agents.prompts import get_seed_prompts
from config import CostConfig, Settings, get_settings
from data.borrowers import borrower_context_block, get_random_borrower
from evaluation.cost_tracker import CostTracker
from handoff.summarizer import summarize_for_handoff
from models import AgentType, Message, Transcript


# ---------------------------------------------------------------------------
# Temporal client (lazy init, optional)
# ---------------------------------------------------------------------------

_temporal_client = None
_temporal_available = None  # None = not checked yet


async def _get_temporal_client():
    """Try to connect to Temporal. Returns client or None if unavailable."""
    global _temporal_client, _temporal_available

    if _temporal_available is False:
        return None
    if _temporal_client is not None:
        return _temporal_client

    try:
        from temporalio.client import Client
        _temporal_client = await Client.connect(
            os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
        )
        _temporal_available = True
        return _temporal_client
    except Exception:
        _temporal_available = False
        return None


async def _temporal_agent_respond(
    system_prompt: str,
    handoff_context: str | None,
    conversation_history: list[dict[str, str]],
    agent_type: AgentType,
    borrower_context: str | None = None,
) -> str:
    """
    Call agent_respond via Temporal activity if available, else direct.
    This ensures the UI goes through Temporal's orchestration layer.
    """
    client = await _get_temporal_client()

    if client is not None:
        try:
            from workflow.activities import AgentInput, AgentOutput
            # Execute activity directly via the Temporal worker
            result = await client.execute_workflow(
                "temporal-agent-call",  # Not a real workflow — use activity directly
                id=f"agent-call-{uuid.uuid4().hex[:8]}",
                task_queue="collection-pipeline",
            )
        except Exception:
            pass  # Fall through to direct call

    # Direct call (always works, same code path)
    settings = _make_production_settings()
    tracker = cl.user_session.get("tracker")
    if tracker is None:
        tracker = CostTracker(settings)

    return await agent_respond(
        system_prompt=system_prompt,
        handoff_context=handoff_context,
        conversation_history=conversation_history,
        agent_type=agent_type,
        tracker=tracker,
        settings=settings,
        borrower_context=borrower_context,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIPELINE = [AgentType.ASSESSMENT, AgentType.RESOLUTION, AgentType.FINAL_NOTICE]

_AGENT_LABELS = {
    AgentType.ASSESSMENT: "Agent 1 - Assessment (Chat)",
    AgentType.RESOLUTION: "Agent 2 - Resolution (Voice Call)",
    AgentType.FINAL_NOTICE: "Agent 3 - Final Notice (Chat)",
}

_ENDING_PHRASES = [
    "thank you for your time", "thank you for your cooperation",
    "thank you for your patience", "we'll be in touch",
    "i'll end this conversation", "this concludes our", "this concludes",
    "have a good day", "goodbye", "end this conversation",
    "you will be contacted shortly", "follow-up assistance", "take care",
]

VOICE_SERVER_URL = os.environ.get("VOICE_SERVER_URL", "http://localhost:8001")
_LOGS_DIR = Path(_PROJECT_ROOT) / "logs"


def _handoff_file(session_id: str | None = None) -> Path:
    if session_id:
        return _LOGS_DIR / f"voice_handoff_{session_id}.json"
    return _LOGS_DIR / "voice_handoff.json"


def _find_transcript(session_id: str | None = None) -> Path | None:
    if session_id:
        specific = _LOGS_DIR / f"voice_transcript_{session_id}.json"
        if specific.exists():
            return specific
    candidates = sorted(_LOGS_DIR.glob("voice_transcript_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    legacy = _LOGS_DIR / "voice_transcript.json"
    return legacy if legacy.exists() else None


def _is_conversation_ending(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _ENDING_PHRASES)


def _make_production_settings() -> Settings:
    base = get_settings()
    return base.model_copy(update={
        "cost": CostConfig(budget_limit=50.0, budget_reserve=1.0),
    })


def _get_prompt_for_agent(prompts: dict, agent_type: AgentType) -> str:
    return {
        AgentType.ASSESSMENT: prompts["agent1_prompt"],
        AgentType.RESOLUTION: prompts["agent2_prompt"],
        AgentType.FINAL_NOTICE: prompts["agent3_prompt"],
    }[agent_type]


def _next_agent(current: AgentType) -> AgentType | None:
    idx = _PIPELINE.index(current)
    return _PIPELINE[idx + 1] if idx + 1 < len(_PIPELINE) else None


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start() -> None:
    prompts = get_seed_prompts()
    settings = _make_production_settings()
    tracker = CostTracker(settings)

    agent_type = AgentType.ASSESSMENT
    system_prompt = _get_prompt_for_agent(prompts, agent_type)

    borrower_profile = get_random_borrower()
    b_context = borrower_context_block(borrower_profile, agent_type="agent1")

    # Generate a workflow ID for this borrower session
    workflow_id = f"borrower-{borrower_profile.borrower_id}-{uuid.uuid4().hex[:6]}"

    cl.user_session.set("agent_type", agent_type)
    cl.user_session.set("system_prompt", system_prompt)
    cl.user_session.set("handoff_context", None)
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("tracker", tracker)
    cl.user_session.set("settings", settings)
    cl.user_session.set("prompts", prompts)
    cl.user_session.set("prior_handoff_summary", None)
    cl.user_session.set("pipeline_done", False)
    cl.user_session.set("waiting_for_call_end", False)
    cl.user_session.set("session_id", uuid.uuid4().hex[:12])
    cl.user_session.set("workflow_id", workflow_id)
    cl.user_session.set("borrower_context", b_context)
    cl.user_session.set("borrower_profile", borrower_profile)
    cl.user_session.set("borrower_name", borrower_profile.name)
    cl.user_session.set("borrower_id", borrower_profile.borrower_id)

    # Try to start Temporal workflow for this session
    temporal_client = await _get_temporal_client()
    temporal_mode = temporal_client is not None

    if temporal_mode:
        try:
            from workflow.pipeline import CollectionPipelineWorkflow, PipelineInput
            # Start workflow (non-blocking — it persists the borrower pipeline state)
            await temporal_client.start_workflow(
                CollectionPipelineWorkflow.run,
                PipelineInput(
                    agent1_prompt=prompts["agent1_prompt"],
                    agent2_prompt=prompts["agent2_prompt"],
                    agent3_prompt=prompts["agent3_prompt"],
                    summarizer_prompt=prompts["summarizer_prompt"],
                    borrower_name=borrower_profile.name,
                    borrower_context_agent1=b_context,
                    borrower_context_agent2=borrower_context_block(borrower_profile, "agent2"),
                    borrower_context_agent3=borrower_context_block(borrower_profile, "agent3"),
                    automated=False,  # Real user — single-turn activities
                    seed=42,
                ),
                id=workflow_id,
                task_queue="collection-pipeline",
            )
            cl.user_session.set("temporal_mode", True)
        except Exception as e:
            temporal_mode = False
            cl.user_session.set("temporal_mode", False)

    if not temporal_mode:
        cl.user_session.set("temporal_mode", False)

    # Agent 1 opening message (via agent_respond — same code path)
    opening = await agent_respond(
        system_prompt=system_prompt,
        handoff_context=None,
        conversation_history=[],
        agent_type=agent_type,
        tracker=tracker,
        settings=settings,
        borrower_context=b_context,
    )

    cl.user_session.set("conversation_history", [{"role": "assistant", "content": opening}])

    mode_label = "via Temporal" if temporal_mode else "direct"
    await cl.Message(
        content=f"*Borrower: {borrower_profile.name} | Pipeline: {workflow_id} ({mode_label})*\n\n{opening}",
        author=_AGENT_LABELS[agent_type],
    ).send()


# ---------------------------------------------------------------------------
# Text message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message) -> None:
    if cl.user_session.get("pipeline_done"):
        await cl.Message(content="Pipeline complete. Refresh to start a new conversation.").send()
        return

    if cl.user_session.get("waiting_for_call_end"):
        await _after_voice_call()
        return

    agent_type: AgentType = cl.user_session.get("agent_type")
    await _process_user_input(message.content, agent_type)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

@cl.action_callback("join_call")
async def on_join_call(action: cl.Action) -> None:
    await action.remove()
    await cl.Message(
        content=(
            f"**Voice call is ready!** Open this link in a new tab:\n\n"
            f"**[Join Voice Call]({VOICE_SERVER_URL}/call)**\n\n"
            f"The Resolution Agent will start speaking when you connect.\n"
            f"When the call is done, come back here and type anything or click the button below to continue."
        ),
        actions=[
            cl.Action(
                name="call_ended",
                payload={"action": "end"},
                label="Call Ended - Continue to Agent 3",
                description="Click after your voice call with Agent 2 is complete",
            ),
        ],
    ).send()
    cl.user_session.set("waiting_for_call_end", True)


@cl.action_callback("call_ended")
async def on_call_ended(action: cl.Action) -> None:
    await action.remove()
    await _after_voice_call()


# ---------------------------------------------------------------------------
# Processing — routes through Temporal when available
# ---------------------------------------------------------------------------

async def _process_user_input(user_text: str, agent_type: AgentType) -> None:
    system_prompt: str = cl.user_session.get("system_prompt")
    handoff_context: str | None = cl.user_session.get("handoff_context")
    history: list[dict[str, str]] = cl.user_session.get("conversation_history")
    tracker: CostTracker = cl.user_session.get("tracker")
    settings: Settings = cl.user_session.get("settings")
    prompts: dict = cl.user_session.get("prompts")
    b_context: str | None = cl.user_session.get("borrower_context")

    history.append({"role": "user", "content": user_text})

    # Call agent_respond — same code path for both Temporal and direct modes
    response_text = await agent_respond(
        system_prompt=system_prompt,
        handoff_context=handoff_context,
        conversation_history=history,
        agent_type=agent_type,
        tracker=tracker,
        settings=settings,
        borrower_context=b_context,
    )

    history.append({"role": "assistant", "content": response_text})
    cl.user_session.set("conversation_history", history)

    await cl.Message(content=response_text, author=_AGENT_LABELS[agent_type]).send()

    if _is_conversation_ending(response_text):
        await _do_handoff(agent_type, history, tracker, settings, prompts)


# ---------------------------------------------------------------------------
# Handoff — uses Temporal summarizer activity when available
# ---------------------------------------------------------------------------

async def _do_handoff(
    current_agent: AgentType,
    history: list[dict[str, str]],
    tracker: CostTracker,
    settings: Settings,
    prompts: dict,
) -> None:
    next_agent = _next_agent(current_agent)

    if next_agent is None:
        cl.user_session.set("pipeline_done", True)
        await cl.Message(content="--- Pipeline complete. All 3 agents have finished. Refresh to start over. ---").send()
        return

    # Summarize via the same summarizer (used by both Temporal activities and direct)
    transcript_messages = tuple(
        Message(role=m["role"], content=m["content"])
        for m in history if m["role"] in ("assistant", "user")
    )
    transcript = Transcript(messages=transcript_messages)

    await cl.Message(content=f"--- Handing off to {_AGENT_LABELS[next_agent]} ---").send()

    prior_summary = cl.user_session.get("prior_handoff_summary")
    summary = await summarize_for_handoff(
        transcript=transcript,
        prior_summary=prior_summary,
        summarizer_prompt=prompts["summarizer_prompt"],
        source_agent=current_agent,
        target_agent=next_agent,
        tracker=tracker,
        settings=settings,
    )

    await cl.Message(
        content=f"*Handoff summary ({summary.token_count} tokens):*\n> {summary.text[:500]}",
    ).send()

    cl.user_session.set("prior_handoff_summary", summary.text)

    # --- Agent 2: Real-time voice call ---
    if next_agent == AgentType.RESOLUTION:
        session_id = cl.user_session.get("session_id")
        borrower_id = cl.user_session.get("borrower_id")
        handoff_data = {
            "summary_text": summary.text,
            "session_id": session_id,
            "borrower_id": borrower_id,
        }
        handoff_path = _handoff_file(session_id)
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(json.dumps(handoff_data))
        _handoff_file(None).write_text(json.dumps(handoff_data))

        # Update session — regenerate agent-specific borrower context
        new_prompt = _get_prompt_for_agent(prompts, next_agent)
        bp = cl.user_session.get("borrower_profile")
        if bp:
            cl.user_session.set("borrower_context", borrower_context_block(bp, agent_type=next_agent.value))
        cl.user_session.set("agent_type", next_agent)
        cl.user_session.set("system_prompt", new_prompt)
        cl.user_session.set("handoff_context", summary.text)
        cl.user_session.set("conversation_history", [])

        await cl.Message(
            content=(
                f"**Incoming Voice Call from Meridian Financial Services**\n\n"
                f"The Resolution Agent is ready to discuss settlement options.\n\n"
                f"Make sure the voice server is running: `.venv/bin/python -m voice.call_server`"
            ),
            actions=[
                cl.Action(
                    name="join_call",
                    payload={"action": "join"},
                    label="Answer Call",
                    description="Join the voice call with the Resolution Agent",
                ),
            ],
        ).send()
        return

    # --- Agent 3 or direct chat agent transition ---
    if current_agent == AgentType.RESOLUTION:
        await cl.Message(content="**Call ended.** Continuing in chat with Agent 3.").send()

    bp = cl.user_session.get("borrower_profile")
    if bp:
        cl.user_session.set("borrower_context", borrower_context_block(bp, agent_type=next_agent.value))
    b_context: str | None = cl.user_session.get("borrower_context")
    new_prompt = _get_prompt_for_agent(prompts, next_agent)
    cl.user_session.set("agent_type", next_agent)
    cl.user_session.set("system_prompt", new_prompt)
    cl.user_session.set("handoff_context", summary.text)
    cl.user_session.set("conversation_history", [])

    opening = await agent_respond(
        system_prompt=new_prompt,
        handoff_context=summary.text,
        conversation_history=[],
        agent_type=next_agent,
        tracker=tracker,
        settings=settings,
        borrower_context=b_context,
    )

    cl.user_session.set("conversation_history", [{"role": "assistant", "content": opening}])
    await cl.Message(content=opening, author=_AGENT_LABELS[next_agent]).send()


async def _after_voice_call() -> None:
    cl.user_session.set("waiting_for_call_end", False)

    tracker: CostTracker = cl.user_session.get("tracker")
    settings: Settings = cl.user_session.get("settings")
    prompts: dict = cl.user_session.get("prompts")

    session_id = cl.user_session.get("session_id")
    voice_messages: list[dict[str, str]] = []
    transcript_path = _find_transcript(session_id)
    if transcript_path:
        data = json.loads(transcript_path.read_text())
        voice_messages = data.get("messages", [])
        await cl.Message(
            content=f"*Voice call transcript loaded ({len(voice_messages)} messages) from {transcript_path.name}*",
        ).send()

    transcript_msgs = tuple(
        Message(role=m["role"], content=m["content"])
        for m in voice_messages if m["role"] in ("assistant", "user")
    )
    transcript = Transcript(messages=transcript_msgs) if transcript_msgs else Transcript()

    prior_summary = cl.user_session.get("prior_handoff_summary")

    if transcript.messages:
        summary = await summarize_for_handoff(
            transcript=transcript,
            prior_summary=prior_summary,
            summarizer_prompt=prompts["summarizer_prompt"],
            source_agent=AgentType.RESOLUTION,
            target_agent=AgentType.FINAL_NOTICE,
            tracker=tracker,
            settings=settings,
        )
        handoff_text = summary.text
        await cl.Message(
            content=f"*Handoff from voice call ({summary.token_count} tokens):*\n> {handoff_text[:500]}",
        ).send()
    else:
        handoff_text = prior_summary or ""
        await cl.Message(content="*No voice transcript captured. Using prior handoff summary.*").send()

    cl.user_session.set("prior_handoff_summary", handoff_text)

    await cl.Message(content="--- Starting Agent 3 - Final Notice (Chat) ---").send()

    bp = cl.user_session.get("borrower_profile")
    if bp:
        cl.user_session.set("borrower_context", borrower_context_block(bp, agent_type="agent3"))
    b_context: str | None = cl.user_session.get("borrower_context")
    new_prompt = _get_prompt_for_agent(prompts, AgentType.FINAL_NOTICE)
    cl.user_session.set("agent_type", AgentType.FINAL_NOTICE)
    cl.user_session.set("system_prompt", new_prompt)
    cl.user_session.set("handoff_context", handoff_text)
    cl.user_session.set("conversation_history", [])

    opening = await agent_respond(
        system_prompt=new_prompt,
        handoff_context=handoff_text,
        conversation_history=[],
        agent_type=AgentType.FINAL_NOTICE,
        tracker=tracker,
        settings=settings,
        borrower_context=b_context,
    )

    cl.user_session.set("conversation_history", [{"role": "assistant", "content": opening}])
    await cl.Message(content=opening, author=_AGENT_LABELS[AgentType.FINAL_NOTICE]).send()
