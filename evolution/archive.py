"""
DGM-style open-ended archive.

Keeps ALL viable variants (not just the best). Each variant stores its
full config, scores, lineage, and mutation description. Flat list with
metadata — no tree structure needed.

Persists to disk as JSON for audit trail and reproducibility.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from models import ArchiveEntry, ConversationScores, VariantConfig


class Archive:
    """
    In-memory archive with disk persistence.

    Each entry is an ArchiveEntry. The archive is append-only during evolution.
    Re-scoring (after meta-eval rubric changes) updates scores in place.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._entries: dict[str, ArchiveEntry] = {}
        self._load_from_disk()

    @property
    def entries(self) -> list[ArchiveEntry]:
        return list(self._entries.values())

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(self, entry: ArchiveEntry) -> None:
        """Add a new variant to the archive. Persists immediately."""
        if entry.version_id in self._entries:
            raise ValueError(f"Version {entry.version_id} already exists in archive")
        self._entries[entry.version_id] = entry
        self._persist()

    def get(self, version_id: str) -> ArchiveEntry:
        """Get entry by version ID. Raises KeyError if not found."""
        if version_id not in self._entries:
            raise KeyError(f"Version {version_id} not in archive")
        return self._entries[version_id]

    def get_best(self) -> ArchiveEntry:
        """Get the highest-scoring promoted entry. Falls back to highest overall."""
        promoted = [e for e in self._entries.values() if e.promoted and not e.discarded]
        if promoted:
            return max(promoted, key=lambda e: e.mean_score)
        active = [e for e in self._entries.values() if not e.discarded]
        if not active:
            raise ValueError("Archive is empty")
        return max(active, key=lambda e: e.mean_score)

    def get_lineage(self, version_id: str) -> list[ArchiveEntry]:
        """Walk parent chain from this version back to root."""
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
        """Get all non-discarded entries."""
        return [e for e in self._entries.values() if not e.discarded]

    def increment_children(self, version_id: str) -> None:
        """Increment children count when this entry is selected as parent."""
        if version_id in self._entries:
            self._entries[version_id].children_count += 1

    def update_scores(self, version_id: str, scores: list[ConversationScores]) -> None:
        """Update scores for a variant (used after meta-eval re-scoring)."""
        if version_id not in self._entries:
            raise KeyError(f"Version {version_id} not in archive")
        self._entries[version_id].scores = scores
        self._persist()

    def rollback_to(self, version_id: str) -> VariantConfig:
        """Get the VariantConfig for a specific version (for rollback)."""
        entry = self.get(version_id)
        return entry.variant_config

    def export_raw_scores(self, output_path: Path | None = None) -> Path:
        """Export all per-conversation scores to CSV for reproducibility."""
        path = output_path or self._settings.raw_scores_file
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "generation", "version_id", "parent_id", "conversation_id",
            "agent1_goal", "agent1_quality", "agent1_compliance",
            "agent2_goal", "agent2_quality", "agent2_compliance",
            "agent3_goal", "agent3_quality", "agent3_compliance",
            "handoff_1", "handoff_2", "system_score",
            "weighted_total", "resolution_rate", "promoted",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in sorted(self._entries.values(), key=lambda e: e.generation):
                for score in entry.scores:
                    row = {
                        "generation": entry.generation,
                        "version_id": entry.version_id,
                        "parent_id": entry.parent_id or "",
                        "conversation_id": score.conversation_id,
                        "weighted_total": score.weighted_total,
                        "resolution_rate": score.resolution_rate,
                        "system_score": score.system_score,
                        "handoff_1": score.handoff_scores.get("handoff_1", ""),
                        "handoff_2": score.handoff_scores.get("handoff_2", ""),
                        "promoted": entry.promoted,
                    }
                    for agent_key in ["agent1", "agent2", "agent3"]:
                        if agent_key in score.agent_scores:
                            s = score.agent_scores[agent_key]
                            row[f"{agent_key}_goal"] = s.goal
                            row[f"{agent_key}_quality"] = s.quality
                            row[f"{agent_key}_compliance"] = s.compliance.all_passed
                    writer.writerow(row)

        return path

    # --- Persistence ---

    def _persist(self) -> None:
        """Save full archive to disk."""
        path = self._settings.archive_file
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            vid: entry.model_dump(mode="json")
            for vid, entry in self._entries.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_from_disk(self) -> None:
        """Load archive from disk if it exists."""
        path = self._settings.archive_file
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for vid, entry_data in data.items():
            self._entries[vid] = ArchiveEntry.model_validate(entry_data)
