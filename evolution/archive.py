"""
DGM-style archive with batch/run tracking.

Each evolution run gets a unique batch_id. Multiple runs are stored
in separate directories under logs/runs/{batch_id}/.

Each batch stores:
  - archive.json: all variants with scores, lineage, configs
  - transcripts/{conv_id}.json: full conversation transcripts
  - costs.json: cost log for this run
  - meta.json: run metadata (start time, config, status)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from models import ArchiveEntry, Conversation, ConversationScores, VariantConfig


class Archive:
    """
    In-memory archive with disk persistence, scoped to a batch/run.

    Each run creates a new batch directory. Historical runs are preserved
    and can be loaded by the dashboard.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        batch_id: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._entries: dict[str, ArchiveEntry] = {}

        # Create or load batch
        if batch_id:
            self._batch_id = batch_id
        else:
            self._batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        self._batch_dir = self._settings.logs_dir / "runs" / self._batch_id
        self._batch_dir.mkdir(parents=True, exist_ok=True)

        # Also maintain a "current" symlink for dashboard
        current_link = self._settings.logs_dir / "current_run"
        # Remove old symlink if exists
        if current_link.is_symlink() or current_link.exists():
            current_link.unlink()
        current_link.symlink_to(self._batch_dir)

        self._save_meta()
        self._load_from_disk()

    @property
    def batch_id(self) -> str:
        return self._batch_id

    @property
    def batch_dir(self) -> Path:
        return self._batch_dir

    @property
    def entries(self) -> list[ArchiveEntry]:
        return list(self._entries.values())

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(self, entry: ArchiveEntry) -> None:
        if entry.version_id in self._entries:
            raise ValueError(f"Version {entry.version_id} already exists")
        self._entries[entry.version_id] = entry
        self._persist()

    def get(self, version_id: str) -> ArchiveEntry:
        if version_id not in self._entries:
            raise KeyError(f"Version {version_id} not in archive")
        return self._entries[version_id]

    def get_best(self) -> ArchiveEntry:
        promoted = [e for e in self._entries.values() if e.promoted and not e.discarded]
        if promoted:
            return max(promoted, key=lambda e: e.mean_score)
        active = [e for e in self._entries.values() if not e.discarded]
        if not active:
            raise ValueError("Archive is empty")
        return max(active, key=lambda e: e.mean_score)

    def get_lineage(self, version_id: str) -> list[ArchiveEntry]:
        lineage = []
        current = version_id
        while current:
            entry = self._entries.get(current)
            if entry is None:
                break
            lineage.append(entry)
            current = entry.parent_id
        return lineage

    def get_active(self) -> list[ArchiveEntry]:
        return [e for e in self._entries.values() if not e.discarded]

    def increment_children(self, version_id: str) -> None:
        if version_id in self._entries:
            self._entries[version_id].children_count += 1

    def update_scores(self, version_id: str, scores: list[ConversationScores]) -> None:
        if version_id not in self._entries:
            raise KeyError(f"Version {version_id} not in archive")
        self._entries[version_id].scores = scores
        self._persist()

    def rollback_to(self, version_id: str) -> VariantConfig:
        return self.get(version_id).variant_config

    # --- Transcript storage ---

    def store_conversation(self, conversation: Conversation) -> None:
        """Store full conversation transcript to disk."""
        transcripts_dir = self._batch_dir / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)

        data = {
            "conversation_id": conversation.conversation_id,
            "persona": conversation.persona.name,
            "persona_type": conversation.persona.persona_type.value,
            "seed": conversation.seed,
            "outcome": conversation.outcome.value,
            "stop_contact": conversation.stop_contact,
            "agent1": [{"role": m.role, "content": m.content} for m in conversation.agent1_transcript.messages],
            "agent2": [{"role": m.role, "content": m.content} for m in conversation.agent2_transcript.messages],
            "agent3": [{"role": m.role, "content": m.content} for m in conversation.agent3_transcript.messages],
            "handoff_1": {
                "text": conversation.handoff_1.text,
                "token_count": conversation.handoff_1.token_count,
            } if conversation.handoff_1 else None,
            "handoff_2": {
                "text": conversation.handoff_2.text,
                "token_count": conversation.handoff_2.token_count,
            } if conversation.handoff_2 else None,
        }

        path = transcripts_dir / f"{conversation.conversation_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_transcript(self, conversation_id: str) -> dict | None:
        """Load a stored transcript."""
        path = self._batch_dir / "transcripts" / f"{conversation_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_transcripts(self) -> list[str]:
        """List all stored conversation IDs."""
        transcripts_dir = self._batch_dir / "transcripts"
        if not transcripts_dir.exists():
            return []
        return [p.stem for p in transcripts_dir.glob("*.json")]

    # --- CSV export ---

    def export_raw_scores(self, output_path: Path | None = None) -> Path:
        path = output_path or (self._batch_dir / "raw_scores.csv")

        fieldnames = [
            "generation", "version_id", "parent_id", "conversation_id",
            "persona_type", "weighted_total", "resolution_rate",
            "agent1_goal_pass_rate", "agent1_quality_pass_rate", "agent1_compliance",
            "agent2_goal_pass_rate", "agent2_quality_pass_rate", "agent2_compliance",
            "agent3_goal_pass_rate", "agent3_quality_pass_rate", "agent3_compliance",
            "handoff_1_pass_rate", "handoff_2_pass_rate",
            "system_pass_rate", "promoted",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in sorted(self._entries.values(), key=lambda e: e.generation):
                for score in entry.scores:
                    row: dict[str, Any] = {
                        "generation": entry.generation,
                        "version_id": entry.version_id,
                        "parent_id": entry.parent_id or "",
                        "conversation_id": score.conversation_id,
                        "persona_type": score.persona_type.value,
                        "weighted_total": score.weighted_total,
                        "resolution_rate": score.resolution_rate,
                        "promoted": entry.promoted,
                    }
                    for agent_key in ["agent1", "agent2", "agent3"]:
                        if agent_key in score.agent_scores:
                            s = score.agent_scores[agent_key]
                            row[f"{agent_key}_goal_pass_rate"] = f"{s.goal.pass_rate:.2f}"
                            row[f"{agent_key}_quality_pass_rate"] = f"{s.quality.pass_rate:.2f}"
                            row[f"{agent_key}_compliance"] = s.compliance.all_passed

                    for hk in ["handoff_1", "handoff_2"]:
                        if hk in score.handoff_scores:
                            h = score.handoff_scores[hk]
                            row[f"{hk}_pass_rate"] = f"{h.score:.2f}" if hasattr(h, 'score') else ""

                    row["system_pass_rate"] = f"{score.system_checks.score:.2f}" if hasattr(score, 'system_checks') else ""
                    writer.writerow(row)

        return path

    # --- Persistence ---

    def _persist(self) -> None:
        path = self._batch_dir / "archive.json"
        data = {
            vid: entry.model_dump(mode="json")
            for vid, entry in self._entries.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Also write to legacy location for backward compat
        legacy = self._settings.logs_dir / "archive.json"
        with open(legacy, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_from_disk(self) -> None:
        path = self._batch_dir / "archive.json"
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for vid, entry_data in data.items():
            self._entries[vid] = ArchiveEntry.model_validate(entry_data)

    def _save_meta(self) -> None:
        meta = {
            "batch_id": self._batch_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        with open(self._batch_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def complete(self) -> None:
        """Mark batch as complete."""
        meta_path = self._batch_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        meta["status"] = "complete"
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        meta["total_variants"] = self.size
        meta["best_score"] = self.get_best().mean_score if self.size > 0 else 0
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


# --- Batch listing (for dashboard) ---

def list_batches(settings: Settings | None = None) -> list[dict]:
    """List all evolution runs with metadata."""
    s = settings or get_settings()
    runs_dir = s.logs_dir / "runs"
    if not runs_dir.exists():
        return []

    batches = []
    for batch_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not batch_dir.is_dir():
            continue
        meta_path = batch_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {"batch_id": batch_dir.name, "status": "unknown"}

        # Count variants
        archive_path = batch_dir / "archive.json"
        if archive_path.exists():
            with open(archive_path) as f:
                data = json.load(f)
            meta["variants"] = len(data)
        else:
            meta["variants"] = 0

        batches.append(meta)

    return batches
