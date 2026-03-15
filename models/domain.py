"""Domain models: messages, transcripts, personas, conversations, configs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from models.enums import AgentType, Outcome, PersonaType


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
# Persona
# ---------------------------------------------------------------------------

class Persona(BaseModel, frozen=True):
    name: str
    persona_type: PersonaType
    system_prompt: str           # For chat-based simulation (Agent 1, 3)
    voice_system_prompt: str     # For voice-simulation (Agent 2)
    difficulty: float = Field(ge=0.0, le=1.0, description="0=easy, 1=hard")


# ---------------------------------------------------------------------------
# Agent Configuration (MUTABLE by evolution loop)
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel, frozen=True):
    """
    Snapshot of all agent prompts + summarizer + strategy.
    The strategy is the structured DNA. Prompts are generated from it.
    Both are stored for audit trail.
    """
    version_id: str
    agent1_prompt: str
    agent2_prompt: str
    agent3_prompt: str
    summarizer_prompt: str
    # Strategy JSON stored alongside prompts. Optional for backward compat.
    strategy_json: str = ""

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
    Evaluation methodology snapshot. Weights + rubric criteria text.
    NEVER changed by the evolution loop — only rubrics/weights by meta-eval.
    COMPLIANCE RULES ARE PERMANENTLY IMMUTABLE — no code path can modify them.
    """
    version_id: str = "eval_v0"
    compliance_rules: list[str] = Field(
        default_factory=list,
        description="PERMANENTLY IMMUTABLE — hardcoded in evaluation/compliance.py. This field exists for serialization only.",
    )
    # Evolvable rubric criteria text — meta-eval can rewrite these
    # Keys match the check names in rubrics.py. If a key exists here,
    # it OVERRIDES the hardcoded text. If absent, hardcoded is used.
    rubric_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Meta-eval can rewrite criteria text here. Key = check name, value = new criterion text.",
    )
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "goal": 0.27,
            "compliance": 0.23,
            "quality": 0.18,
            "handoff": 0.13,
            "system": 0.09,
            "deal_quality": 0.10,
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
