"""
Live state tracking for real-time dashboard updates.

Writes a JSON file that gets updated after every message in simulation.
Dashboard polls this file every 2 seconds.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LiveState:
    """Thread-safe live state writer. Single instance per process."""

    def __init__(self, logs_dir: Path) -> None:
        self._path = logs_dir / "live.json"
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "status": "idle",
            "generation": 0,
            "current_variant": "",
            "current_stage": "",
            "current_persona": "",
            "current_conversation_id": "",
            "messages": [],  # Live message feed
            "activity_log": [],  # High-level activity log
        }
        self._flush()

    def set_generation(self, gen: int) -> None:
        with self._lock:
            self._state["generation"] = gen
            self._state["status"] = "evolving"
            self._add_activity(f"=== Generation {gen} ===")
            self._flush()

    def set_evaluating_seed(self) -> None:
        with self._lock:
            self._state["status"] = "seeding"
            self._state["current_variant"] = "v0"
            self._add_activity("Evaluating seed v0...")
            self._flush()

    def set_mutating(self, parent_id: str, child_id: str) -> None:
        with self._lock:
            self._state["status"] = "mutating"
            self._state["current_variant"] = child_id
            self._add_activity(f"Mutating {parent_id} → {child_id}")
            self._flush()

    def set_simulating(self, variant_id: str, persona: str, conv_id: str, stage: str) -> None:
        with self._lock:
            self._state["status"] = "simulating"
            self._state["current_variant"] = variant_id
            self._state["current_persona"] = persona
            self._state["current_conversation_id"] = conv_id
            self._state["current_stage"] = stage
            self._state["messages"] = []  # Clear for new conversation stage
            self._add_activity(f"  {variant_id} / {persona} / {stage}")
            self._flush()

    def add_message(self, role: str, content: str, agent_stage: str = "") -> None:
        """Add a live message from the conversation."""
        with self._lock:
            self._state["messages"].append({
                "role": role,
                "content": content[:300],  # Truncate for live view
                "stage": agent_stage,
                "time": datetime.now(timezone.utc).isoformat(),
            })
            # Keep last 20 messages
            if len(self._state["messages"]) > 20:
                self._state["messages"] = self._state["messages"][-20:]
            self._flush()

    def set_scoring(self, variant_id: str, conv_id: str) -> None:
        with self._lock:
            self._state["status"] = "scoring"
            self._state["current_variant"] = variant_id
            self._state["current_conversation_id"] = conv_id
            self._add_activity(f"  Scoring {conv_id}")
            self._flush()

    def set_promoting(self, variant_id: str, result: str) -> None:
        with self._lock:
            self._add_activity(f"  {variant_id}: {result}")
            self._flush()

    def set_complete(self, best_id: str, best_score: float) -> None:
        with self._lock:
            self._state["status"] = "complete"
            self._add_activity(f"Evolution complete. Best: {best_id} ({best_score:.2f})")
            self._flush()

    def set_idle(self) -> None:
        with self._lock:
            self._state["status"] = "idle"
            self._flush()

    def _add_activity(self, msg: str) -> None:
        self._state["activity_log"].append({
            "msg": msg,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 50
        if len(self._state["activity_log"]) > 50:
            self._state["activity_log"] = self._state["activity_log"][-50:]

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._state, f)


# Module-level singleton
_instance: LiveState | None = None


def get_live_state(logs_dir: Path | None = None) -> LiveState:
    global _instance
    if _instance is None:
        if logs_dir is None:
            from config import get_settings
            logs_dir = get_settings().logs_dir
        _instance = LiveState(logs_dir)
    return _instance
