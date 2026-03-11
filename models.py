"""
Domain models for the Darwin Voice Agent system.

Design principles:
- Immutable where possible (frozen=True) — prevents accidental mutation
- Enums for closed sets — no stringly-typed agent names or outcomes
- Validators at boundaries — catch bad data at creation, not downstream
- Separate config snapshots from runtime state
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentType(str, Enum):
    ASSESSMENT = "agent1"      # Chat — cold, clinical
    RESOLUTION = "agent2"      # Voice — transactional dealmaker
    FINAL_NOTICE = "agent3"    # Chat — consequence-driven closer


class Outcome(str, Enum):
    DEAL_AGREED = "deal_agreed"
    NO_DEAL = "no_deal"
    BORROWER_REFUSED = "borrower_refused"      # Explicit stop-contact
    HARDSHIP_REFERRAL = "hardship_referral"
    NO_RESPONSE = "no_response"
    IN_PROGRESS = "in_progress"


class CostCategory(str, Enum):
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    STRICT_GRADING = "strict_grading"
    REWRITING = "rewriting"
    SUMMARIZATION = "summarization"
    META_EVAL = "meta_eval"


class PersonaType(str, Enum):
    COOPERATIVE = "cooperative"
    COMBATIVE = "combative"
    EVASIVE = "evasive"
    CONFUSED = "confused"
    DISTRESSED = "distressed"


# ---------------------------------------------------------------------------
# Chat / Transcript
# ---------------------------------------------------------------------------

class Message(BaseModel, frozen=True):
    role: str           # "system", "assistant", "user"
    content: str


class Transcript(BaseModel, frozen=True):
    """Ordered list of messages for one agent stage."""
    messages: tuple[Message, ...] = ()

    @property
    def text(self) -> str:
        """Flat text representation for summarization input."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role in ("assistant", "user"))


# ---------------------------------------------------------------------------
# Agent Configuration (MUTABLE by evolution loop)
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel, frozen=True):
    """
    Snapshot of all agent prompts + summarizer. This is what the evolution
    loop mutates. Frozen so a given version is immutable once created.
    """
    version_id: str
    agent1_prompt: str
    agent2_prompt: str
    agent3_prompt: str
    summarizer_prompt: str

    def get_prompt(self, agent: AgentType) -> str:
        return {
            AgentType.ASSESSMENT: self.agent1_prompt,
            AgentType.RESOLUTION: self.agent2_prompt,
            AgentType.FINAL_NOTICE: self.agent3_prompt,
        }[agent]


# ---------------------------------------------------------------------------
# Evaluation Configuration (MUTABLE only by meta-eval cycle)
# ---------------------------------------------------------------------------

class EvalConfig(BaseModel, frozen=True):
    """
    Evaluation methodology snapshot. Rubrics + weights + compliance rules.
    NEVER changed by the evolution loop — only by the meta-eval cycle.
    Frozen so comparisons between parent/child use identical eval config.
    """
    version_id: str = "eval_v0"
    goal_rubric: dict[str, str] = Field(
        default_factory=dict,
        description="Per-agent goal completion rubric text",
    )
    quality_rubric: str = ""
    handoff_rubric: str = ""
    system_rubric: str = ""
    compliance_rules: list[str] = Field(
        default_factory=list,
        description="Append-only list of compliance rule definitions",
    )
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "goal": 0.30,
            "compliance": 0.20,
            "quality": 0.20,
            "handoff": 0.15,
            "system": 0.15,
        },
    )

    @field_validator("scoring_weights")
    @classmethod
    def _weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}")
        if v.get("compliance", 0) < 0.15:
            raise ValueError("Compliance weight floor is 0.15")
        return v


class VariantConfig(BaseModel, frozen=True):
    """
    Complete snapshot: agent behavior + eval methodology.
    Evolution changes agent_config. Meta-eval changes eval_config.
    """
    agent_config: AgentConfig
    eval_config: EvalConfig


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

class Persona(BaseModel, frozen=True):
    name: str
    persona_type: PersonaType
    system_prompt: str           # For chat-based simulation (Agent 1, 3)
    voice_system_prompt: str     # For voice-simulation (Agent 2)
    difficulty: float = Field(ge=0.0, le=1.0, description="0=easy, 1=hard")


# ---------------------------------------------------------------------------
# Conversation (simulation output)
# ---------------------------------------------------------------------------

class HandoffSummary(BaseModel, frozen=True):
    """Context passed between agents. Token-counted and budget-enforced."""
    text: str
    token_count: int
    source_agent: AgentType
    target_agent: AgentType


class Conversation(BaseModel, frozen=True):
    """Full pipeline output for one borrower simulation run."""
    conversation_id: str
    persona: Persona
    seed: int
    agent1_transcript: Transcript = Field(default_factory=Transcript)
    agent2_transcript: Transcript = Field(default_factory=Transcript)
    agent3_transcript: Transcript = Field(default_factory=Transcript)
    handoff_1: HandoffSummary | None = None    # Agent 1 → Agent 2
    handoff_2: HandoffSummary | None = None    # Agent 1+2 → Agent 3
    outcome: Outcome = Outcome.IN_PROGRESS
    stop_contact: bool = False


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


class AgentScores(BaseModel, frozen=True):
    """Scores for a single agent within a conversation."""
    agent: AgentType
    goal: float = Field(ge=1.0, le=10.0)
    quality: float = Field(ge=1.0, le=10.0)
    compliance: ComplianceResult


class ConversationScores(BaseModel, frozen=True):
    """All scores for one full pipeline conversation."""
    conversation_id: str
    agent_scores: dict[str, AgentScores]    # AgentType.value → scores
    handoff_scores: dict[str, float]         # "handoff_1", "handoff_2" → score
    system_score: float = Field(ge=1.0, le=10.0)
    weighted_total: float
    resolution_rate: float = Field(ge=0.0, le=1.0, description="1.0 if resolved, 0.0 if not")

    @property
    def compliance_passed(self) -> bool:
        return all(
            s.compliance.all_passed for s in self.agent_scores.values()
        )


# ---------------------------------------------------------------------------
# Strict Grader
# ---------------------------------------------------------------------------

class StrictGraderResult(BaseModel, frozen=True):
    validated: bool
    flags: list[str] = Field(default_factory=list)
    adjusted_scores: dict[str, float] = Field(default_factory=dict)


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
# Cost Tracking
# ---------------------------------------------------------------------------

class CostEntry(BaseModel, frozen=True):
    """Single LLM API call cost record."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str
    category: CostCategory
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class CostBreakdown(BaseModel, frozen=True):
    """Aggregated cost report."""
    total_usd: float
    by_category: dict[str, float]
    by_model: dict[str, float]
    call_counts: dict[str, int]     # conversations_simulated, evaluation_calls, etc.
    remaining_budget: float


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
