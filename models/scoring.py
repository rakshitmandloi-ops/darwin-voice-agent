"""Scoring models: compliance, checklists, agent scores, conversation scores."""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.enums import AgentType, PersonaType


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class ComplianceResult(BaseModel, frozen=True):
    """Per-rule pass/fail for one agent in one conversation."""
    rule_results: dict[str, bool]   # rule_name → pass/fail

    @property
    def all_passed(self) -> bool:
        return all(self.rule_results.values())

    @property
    def violations(self) -> list[str]:
        return [r for r, passed in self.rule_results.items() if not passed]


class ChecklistResult(BaseModel, frozen=True):
    """Binary pass/fail checklist — same pattern for goal, quality, compliance."""
    checks: dict[str, bool]   # criterion_name → pass/fail

    @property
    def all_passed(self) -> bool:
        return all(self.checks.values())

    @property
    def pass_rate(self) -> float:
        if not self.checks:
            return 0.0
        return sum(self.checks.values()) / len(self.checks)

    @property
    def failures(self) -> list[str]:
        return [c for c, passed in self.checks.items() if not passed]

    @property
    def score(self) -> float:
        """Convert to 1-10 scale for backward compat."""
        return 1.0 + self.pass_rate * 9.0


class AgentScores(BaseModel, frozen=True):
    """Scores for a single agent within a conversation."""
    agent: AgentType
    goal: ChecklistResult         # per-criterion pass/fail
    quality: ChecklistResult      # per-criterion pass/fail
    compliance: ComplianceResult  # per-rule pass/fail (kept separate for backward compat)

    @property
    def goal_score(self) -> float:
        return self.goal.score

    @property
    def quality_score(self) -> float:
        return self.quality.score


class HandoffChecklist(BaseModel, frozen=True):
    """Checklist for handoff quality."""
    checks: dict[str, bool]

    @property
    def score(self) -> float:
        if not self.checks:
            return 1.0
        return 1.0 + (sum(self.checks.values()) / len(self.checks)) * 9.0


class SystemChecklist(BaseModel, frozen=True):
    """Checklist for cross-agent system continuity."""
    checks: dict[str, bool]

    @property
    def score(self) -> float:
        if not self.checks:
            return 1.0
        return 1.0 + (sum(self.checks.values()) / len(self.checks)) * 9.0


class ConversationScores(BaseModel, frozen=True):
    """All scores for one full pipeline conversation."""
    conversation_id: str
    persona_type: PersonaType = PersonaType.COOPERATIVE
    agent_scores: dict[str, AgentScores]
    handoff_scores: dict[str, HandoffChecklist]
    system_checks: SystemChecklist = Field(default_factory=lambda: SystemChecklist(checks={}))
    weighted_total: float
    resolution_rate: float = Field(ge=0.0, le=1.0)

    @property
    def system_score(self) -> float:
        return self.system_checks.score

    @property
    def compliance_passed(self) -> bool:
        return all(
            s.compliance.all_passed for s in self.agent_scores.values()
        )


# ---------------------------------------------------------------------------
# Deal Quality
# ---------------------------------------------------------------------------

class DealQualityResult(BaseModel, frozen=True):
    """How good the deal was for the company (not the borrower)."""
    offered_percentage: float | None = None
    expected_max_percentage: float
    deal_quality_score: float = Field(ge=1.0, le=10.0)
    reasoning: str


# ---------------------------------------------------------------------------
# Strict Grader
# ---------------------------------------------------------------------------

class StrictGraderResult(BaseModel, frozen=True):
    validated: bool
    flags: list[str] = Field(default_factory=list)
    adjusted_scores: dict[str, float] = Field(default_factory=dict)
