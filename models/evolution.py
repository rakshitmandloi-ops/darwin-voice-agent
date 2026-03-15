"""Evolution and archive models."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from models.domain import VariantConfig
from models.scoring import ConversationScores, StrictGraderResult


# ---------------------------------------------------------------------------
# Evolution / Archive
# ---------------------------------------------------------------------------

class MutationResult(BaseModel, frozen=True):
    """Output of the rewriter — what was changed and why."""
    components_modified: list[str]
    changes: dict[str, str]         # component_name → new prompt text
    rationale: str
    failures_addressed: list[str]
    token_counts: dict[str, int]    # component_name → token count


class EvalComparison(BaseModel, frozen=True):
    """Statistical comparison between parent and child."""
    parent_version: str
    child_version: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    variance_too_high: bool = False
    compliance_preserved: bool = True
    per_persona_breakdown: dict[str, float] = Field(default_factory=dict)


class ArchiveEntry(BaseModel):
    """One version in the archive. Mutable only for score updates after meta-eval re-scoring."""
    version_id: str
    variant_config: VariantConfig
    scores: list[ConversationScores] = Field(default_factory=list)
    parent_id: str | None = None
    generation: int = 0
    mutation_description: str = ""
    components_modified: list[str] = Field(default_factory=list)
    rationale: str = ""
    strict_grader_result: StrictGraderResult | None = None
    promoted: bool = False
    discarded: bool = False
    discard_reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    children_count: int = 0

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.weighted_total for s in self.scores) / len(self.scores)

    @property
    def resolution_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.resolution_rate for s in self.scores) / len(self.scores)


# ---------------------------------------------------------------------------
# Trajectory Analysis
# ---------------------------------------------------------------------------

class TrajectoryAnalysis(BaseModel, frozen=True):
    """Aggregate performance analysis for the rewriter."""
    scores_by_persona: dict[str, float]
    scores_by_metric: dict[str, float]
    cross_gen_trends: dict[str, list[float]]
    systematic_failures: list[str]
    ceiling_floor_flags: list[str]
    win_loss_by_persona: dict[str, dict[str, int]]
