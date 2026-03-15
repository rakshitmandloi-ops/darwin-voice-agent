"""
Real-time voice call server for Agent 2 (Resolution).

Uses Pipecat SmallWebRTC for peer-to-peer WebRTC voice calls in the browser.
Pipeline: User mic -> VAD -> OpenAI STT -> agent_respond() -> OpenAI TTS -> User speaker

CRITICAL: The LLM call goes through agent_respond() — the SAME code path
used by simulation, evaluation, Temporal, and Chainlit. This ensures that
production voice behavior is identical to what was tested during evolution.

Usage:
    .venv/bin/python -m interfaces.voice.call_server
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TextFrame,
    TTSSpeakFrame,
    EndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
    SmallWebRTCPatchRequest,
    IceCandidate,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from agents.core import agent_respond
from agents.prompts import get_seed_prompts
from config import CostConfig, get_settings
from data.borrowers import borrower_context_block, get_borrower, get_random_borrower
from evaluation.cost_tracker import CostTracker
from models import AgentType

# ---------------------------------------------------------------------------
# Shared state — per-session isolation
# ---------------------------------------------------------------------------

_LOGS_DIR = Path(_ROOT) / "logs"

# Maps pc_id -> {"messages": [...], "borrower_id": str | None}
_active_sessions: dict[str, dict] = {}


def _handoff_path(borrower_id: str | None = None) -> Path:
    """Return the handoff file path, optionally scoped to a borrower."""
    if borrower_id:
        return _LOGS_DIR / f"voice_handoff_{borrower_id}.json"
    return _LOGS_DIR / "voice_handoff.json"


def _transcript_path(pc_id: str) -> Path:
    """Return the transcript file path scoped to a WebRTC peer connection."""
    return _LOGS_DIR / f"voice_transcript_{pc_id}.json"


def _load_handoff_data(borrower_id: str | None = None) -> dict:
    """Load full handoff data (summary_text + borrower_id) from file."""
    path = _handoff_path(borrower_id)
    if path.exists():
        return json.loads(path.read_text())
    # Fall back to the generic file if borrower-specific one is missing
    if borrower_id:
        generic = _handoff_path(None)
        if generic.exists():
            return json.loads(generic.read_text())
    return {}


def _load_handoff_context(borrower_id: str | None = None) -> str | None:
    data = _load_handoff_data(borrower_id)
    return data.get("summary_text", "") or None


def _save_transcript(messages: list[dict[str, str]], pc_id: str) -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _transcript_path(pc_id)
    path.write_text(json.dumps({"messages": messages}, indent=2))


# ---------------------------------------------------------------------------
# Custom Pipecat processor: wraps agent_respond()
# ---------------------------------------------------------------------------

class AgentRespondProcessor(FrameProcessor):
    """
    Pipecat processor that calls agent_respond() — the SINGLE code path
    shared with simulation, evaluation, Temporal, and Chainlit.

    Receives TranscriptionFrame (user speech text from STT),
    calls agent_respond(), emits TTSSpeakFrame (for TTS to speak).
    """

    def __init__(
        self,
        system_prompt: str,
        handoff_context: str | None,
        tracker: CostTracker,
        conversation_log: list[dict[str, str]],
        borrower_context: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._system_prompt = system_prompt
        self._handoff_context = handoff_context
        self._tracker = tracker
        # Use the tracker's settings (already has production budget)
        self._settings = tracker._settings
        self._conversation_history: list[dict[str, str]] = []
        self._conversation_log = conversation_log  # shared ref for transcript saving
        self._borrower_context = borrower_context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            user_text = frame.text.strip()

            # Add user message to history
            self._conversation_history.append({"role": "user", "content": user_text})
            self._conversation_log.append({"role": "user", "content": user_text})

            # Call agent_respond() — SAME code path as simulation
            response_text = await agent_respond(
                system_prompt=self._system_prompt,
                handoff_context=self._handoff_context,
                conversation_history=self._conversation_history,
                agent_type=AgentType.RESOLUTION,
                tracker=self._tracker,
                settings=self._settings,
                borrower_context=self._borrower_context,
            )

            # Update history
            self._conversation_history.append({"role": "assistant", "content": response_text})
            self._conversation_log.append({"role": "assistant", "content": response_text})

            # Emit for TTS
            await self.push_frame(TTSSpeakFrame(text=response_text))
        else:
            # Pass through all other frames
            await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

request_handler = SmallWebRTCRequestHandler()


@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
    <head><title>Agent 2 - Voice Call</title></head>
    <body style="font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #1a1a2e;">
        <div style="text-align: center; color: white;">
            <h1>Meridian Financial Services</h1>
            <h2>Resolution Agent - Voice Call</h2>
            <p>Click below to start the call</p>
            <a href="/call" style="display: inline-block; padding: 20px 40px; background: #16a34a; color: white; text-decoration: none; border-radius: 50px; font-size: 20px; margin-top: 20px;">
                Start Call
            </a>
        </div>
    </body>
    </html>
    """)


@app.get("/status")
async def status(request: Request):
    borrower_id = request.query_params.get("borrower_id")
    return JSONResponse({
        "status": "ready",
        "has_handoff": _handoff_path(borrower_id).exists(),
        "active_sessions": len(_active_sessions),
    })


@app.post("/start")
async def start_bot(request: Request):
    """Step 1: Bot init. Accepts optional borrower_id/workflow_id. Returns WebRTC connection params."""
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass  # Body may be empty for default calls

    # Store borrower_id so /offer can pick it up via request_data
    borrower_id = body.get("borrower_id") or body.get("workflow_id")
    return JSONResponse({
        "webrtcRequestParams": {
            "endpoint": "/offer",
            "requestData": {"borrower_id": borrower_id} if borrower_id else {},
        },
    })


@app.post("/offer")
async def handle_offer(request: Request):
    """Step 2: Receive SDP offer, create pipeline, return SDP answer."""
    body = await request.json()
    allowed_keys = {"sdp", "type", "pc_id", "pcId", "restart_pc", "restartPc", "request_data", "requestData"}
    filtered = {k: v for k, v in body.items() if k in allowed_keys}
    webrtc_request = SmallWebRTCRequest.from_dict(filtered)
    answer = await request_handler.handle_web_request(
        webrtc_request, on_new_connection,
    )
    return JSONResponse(answer)


@app.patch("/offer")
async def offer_patch(request: Request):
    """Handle ICE candidate patches."""
    body = await request.json()
    patch_request = SmallWebRTCPatchRequest(
        pc_id=body["pc_id"],
        candidates=[IceCandidate(**c) for c in body.get("candidates", [])],
    )
    await request_handler.handle_patch_request(patch_request)
    return JSONResponse({"status": "ok"})


# Mount prebuilt call UI (AFTER API routes)
app.mount("/call", SmallWebRTCPrebuiltUI)


# ---------------------------------------------------------------------------
# Pipeline — called for each new WebRTC connection
# ---------------------------------------------------------------------------

async def on_new_connection(webrtc_connection):
    # Extract per-connection identifiers
    pc_id = getattr(webrtc_connection, "pc_id", None) or "unknown"
    request_data = getattr(webrtc_connection, "request_data", {}) or {}
    borrower_id = request_data.get("borrower_id")

    settings = get_settings()
    prompts = get_seed_prompts()

    # Load handoff data — get both summary and borrower_id
    handoff_data = _load_handoff_data(borrower_id)
    handoff_context = handoff_data.get("summary_text") or None

    # Resolve borrower_id: WebRTC request_data > handoff file > random fallback
    if not borrower_id:
        borrower_id = handoff_data.get("borrower_id")

    system_prompt = prompts["agent2_prompt"]

    # Load borrower context — by ID from handoff, fall back to random
    borrower_profile = None
    if borrower_id:
        borrower_profile = get_borrower(borrower_id)
    if borrower_profile is None:
        borrower_profile = get_random_borrower()
    b_context = borrower_context_block(borrower_profile, agent_type="agent2")

    # Per-connection transcript list — no global collision
    conversation_messages: list[dict[str, str]] = []
    _active_sessions[pc_id] = {
        "messages": conversation_messages,
        "borrower_id": borrower_id,
    }

    # --- Pipecat services ---
    stt = OpenAISTTService(
        api_key=settings.openai_api_key,
        model="whisper-1",
    )

    tts = OpenAITTSService(
        api_key=settings.openai_api_key,
        voice="onyx",
        model="tts-1",
    )

    # Production tracker with higher budget (evolution costs are in the same log)
    prod_settings = settings.model_copy(update={
        "cost": CostConfig(budget_limit=50.0, budget_reserve=1.0),
    })
    tracker = CostTracker(prod_settings)
    agent_processor = AgentRespondProcessor(
        system_prompt=system_prompt,
        handoff_context=handoff_context,
        tracker=tracker,
        conversation_log=conversation_messages,
        borrower_context=b_context,
    )

    # --- Transport ---
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # --- Pipeline: input -> STT -> agent_respond() -> TTS -> output ---
    pipeline = Pipeline([
        transport.input(),
        stt,
        agent_processor,
        tts,
        transport.output(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Speak opening when client connects
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        opening = (
            "Hello, this is the Resolution Agent from Meridian Financial Services. "
            "Based on our earlier assessment, I'd like to discuss your settlement options. "
            "How would you like to proceed?"
        )
        conversation_messages.append({"role": "assistant", "content": opening})
        await task.queue_frame(TTSSpeakFrame(text=opening))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        _save_transcript(conversation_messages, pc_id)
        _active_sessions.pop(pc_id, None)
        await task.queue_frame(EndFrame())

    # Run pipeline in background
    runner = PipelineRunner()
    asyncio.create_task(runner.run(task))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    port = int(os.environ.get("VOICE_PORT", "8001"))
    print(f"\nVoice call server starting on http://localhost:{port}")
    print(f"  Call UI: http://localhost:{port}/call")
    print(f"\n  Pipeline: Mic -> STT -> agent_respond() -> TTS -> Speaker")
    print(f"  Same code path as simulation/evaluation.")
    print()
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
