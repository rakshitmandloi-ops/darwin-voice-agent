"""
DGM-style parent selection for the evolution loop.

Probabilistic selection based on:
  prob(v) = sigmoid(score) × 1/(1+children_count)

- High-scoring variants are preferred but not guaranteed
- Under-explored variants (fewer children) get a boost
- Even low-scoring variants can be selected → allows deceptive dips
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from models import ArchiveEntry


def select_parents(
    entries: Sequence[ArchiveEntry],
    k: int = 2,
    seed: int | None = None,
) -> list[ArchiveEntry]:
    """
    Select k parents from the archive using DGM-style probabilistic selection.

    Each entry's probability is proportional to:
        sigmoid(mean_score, k=2, mid=5) × 1/(1 + children_count)

    Returns k entries (with replacement possible for very small archives).
    """
    if not entries:
        raise ValueError("Cannot select parents from empty archive")

    active = [e for e in entries if not e.discarded]
    if not active:
        raise ValueError("No active entries in archive")

    # Compute selection probabilities
    raw_probs = []
    for entry in active:
        score_factor = _sigmoid(entry.mean_score, k=2.0, mid=5.0)
        exploration_factor = 1.0 / (1.0 + entry.children_count)
        raw_probs.append(score_factor * exploration_factor)

    # Normalize to sum to 1
    total = sum(raw_probs)
    if total == 0:
        # Uniform fallback
        probs = [1.0 / len(active)] * len(active)
    else:
        probs = [p / total for p in raw_probs]

    rng = random.Random(seed)
    selected = rng.choices(active, weights=probs, k=k)

    return selected


def _sigmoid(x: float, k: float = 2.0, mid: float = 5.0) -> float:
    """Sigmoid function centered at `mid` with steepness `k`."""
    return 1.0 / (1.0 + math.exp(-k * (x - mid)))
