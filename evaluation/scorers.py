"""
Evaluation scorers — rate conversations across multiple dimensions.

Tier 1 (this file): Regular scorer using gpt-4o-mini. Runs on EVERY
conversation with FULL rubric. Cheap enough to run at scale.

Tier 2 (strict_grader.py): Independent gpt-4o grader for promotion validation.

Rubrics and weights are passed in (from EvalConfig), not hardcoded.
"""

from __future__ import annotations

import json
from typing import Any

from config import Settings, get_settings
from evaluation.compliance import check_compliance
from evaluation.cost_tracker import CostTracker
from models import (
    AgentScores,
    AgentType,
    Conversation,
    ConversationScores,
    CostCategory,
    EvalConfig,
    Outcome,
)


async def score_conversation(
    conversation: Conversation,
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings | None = None,
) -> ConversationScores:
    """
    Score a full pipeline conversation across all dimensions.

    Returns ConversationScores with per-agent scores, handoff scores,
    system score, and weighted total.
    """
    s = settings or get_settings()

    # --- Per-agent scoring ---
    agent_scores: dict[str, AgentScores] = {}

    agents_and_transcripts = [
        (AgentType.ASSESSMENT, conversation.agent1_transcript),
        (AgentType.RESOLUTION, conversation.agent2_transcript),
        (AgentType.FINAL_NOTICE, conversation.agent3_transcript),
    ]

    for agent_type, transcript in agents_and_transcripts:
        if not transcript.messages:
            # Agent didn't run (e.g., deal_agreed before Agent 3)
            agent_scores[agent_type.value] = AgentScores(
                agent=agent_type,
                goal=1.0,
                quality=1.0,
                compliance=check_compliance(transcript, agent_type),
            )
            continue

        # Goal completion
        goal_score = await _llm_score(
            prompt=_build_goal_prompt(
                transcript=transcript,
                agent_type=agent_type,
                rubric=eval_config.goal_rubric.get(agent_type.value, ""),
            ),
            tracker=tracker,
            settings=s,
        )

        # Quality
        quality_score = await _llm_score(
            prompt=_build_quality_prompt(
                transcript=transcript,
                agent_type=agent_type,
                rubric=eval_config.quality_rubric,
            ),
            tracker=tracker,
            settings=s,
        )

        # Compliance (rule-based, not LLM)
        compliance = check_compliance(transcript, agent_type)

        agent_scores[agent_type.value] = AgentScores(
            agent=agent_type,
            goal=goal_score,
            quality=quality_score,
            compliance=compliance,
        )

    # --- Handoff scoring ---
    handoff_scores: dict[str, float] = {}

    if conversation.handoff_1:
        handoff_scores["handoff_1"] = await _llm_score(
            prompt=_build_handoff_prompt(
                summary=conversation.handoff_1.text,
                source_transcript=conversation.agent1_transcript,
                target_transcript=conversation.agent2_transcript,
                rubric=eval_config.handoff_rubric,
            ),
            tracker=tracker,
            settings=s,
        )
    else:
        handoff_scores["handoff_1"] = 1.0

    if conversation.handoff_2:
        handoff_scores["handoff_2"] = await _llm_score(
            prompt=_build_handoff_prompt(
                summary=conversation.handoff_2.text,
                source_transcript=conversation.agent2_transcript,
                target_transcript=conversation.agent3_transcript,
                rubric=eval_config.handoff_rubric,
            ),
            tracker=tracker,
            settings=s,
        )
    else:
        handoff_scores["handoff_2"] = 1.0

    # --- System-level scoring (cross-agent continuity) ---
    system_score = await _llm_score(
        prompt=_build_system_prompt(conversation, eval_config.system_rubric),
        tracker=tracker,
        settings=s,
    )

    # --- Weighted total ---
    weights = eval_config.scoring_weights
    avg_goal = _mean([s.goal for s in agent_scores.values()])
    avg_quality = _mean([s.quality for s in agent_scores.values()])
    compliance_score = 10.0 if all(s.compliance.all_passed for s in agent_scores.values()) else 1.0
    avg_handoff = _mean(list(handoff_scores.values()))

    weighted_total = (
        weights.get("goal", 0.3) * avg_goal
        + weights.get("compliance", 0.2) * compliance_score
        + weights.get("quality", 0.2) * avg_quality
        + weights.get("handoff", 0.15) * avg_handoff
        + weights.get("system", 0.15) * system_score
    )

    # --- Resolution rate ---
    resolved = 1.0 if conversation.outcome in (Outcome.DEAL_AGREED, Outcome.HARDSHIP_REFERRAL) else 0.0

    return ConversationScores(
        conversation_id=conversation.conversation_id,
        agent_scores=agent_scores,
        handoff_scores=handoff_scores,
        system_score=system_score,
        weighted_total=round(weighted_total, 2),
        resolution_rate=resolved,
    )


# ---------------------------------------------------------------------------
# LLM scoring helper
# ---------------------------------------------------------------------------

async def _llm_score(
    *,
    prompt: str,
    tracker: CostTracker,
    settings: Settings,
) -> float:
    """Call LLM to get a 1-10 score. Returns the numeric score."""
    response = await tracker.tracked_completion(
        model=settings.models.eval,
        messages=[
            {"role": "system", "content": "You are an evaluation judge. Output ONLY a JSON object with a 'score' field (integer 1-10) and a brief 'reason' field. Example: {\"score\": 7, \"reason\": \"Good but missed X\"}"},
            {"role": "user", "content": prompt},
        ],
        category=CostCategory.EVALUATION,
        temperature=0.0,
    )

    text = response.choices[0].message.content or ""
    return _parse_score(text)


def _parse_score(text: str) -> float:
    """Extract numeric score from LLM response. Handles JSON and plain text."""
    # Try JSON first
    try:
        data = json.loads(text)
        score = float(data.get("score", 5))
        return max(1.0, min(10.0, score))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: find first number in text
    import re
    match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
    if match:
        score = float(match.group(1))
        return max(1.0, min(10.0, score))

    return 5.0  # Default if parsing fails


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_goal_prompt(
    transcript: Transcript,
    agent_type: AgentType,
    rubric: str,
) -> str:
    from models import Transcript
    return f"""{rubric}

AGENT TYPE: {agent_type.value}
CONVERSATION:
{transcript.text}

Rate this agent's goal completion (1-10):"""


def _build_quality_prompt(
    transcript: Transcript,
    agent_type: AgentType,
    rubric: str,
) -> str:
    from models import Transcript
    return f"""{rubric}

AGENT TYPE: {agent_type.value} ({"cold/clinical" if agent_type == AgentType.ASSESSMENT else "transactional" if agent_type == AgentType.RESOLUTION else "consequence-driven"})
CONVERSATION:
{transcript.text}

Rate this agent's conversation quality (1-10):"""


def _build_handoff_prompt(
    summary: str,
    source_transcript: Transcript,
    target_transcript: Transcript,
    rubric: str,
) -> str:
    from models import Transcript
    target_text = target_transcript.text if target_transcript.messages else "(agent did not run)"
    return f"""{rubric}

HANDOFF SUMMARY:
{summary}

SOURCE CONVERSATION (what the summary was built from):
{source_transcript.text[:1000]}

TARGET CONVERSATION (did the next agent use the summary?):
{target_text[:1000]}

Rate this handoff quality (1-10):"""


def _build_system_prompt(conversation: Conversation, rubric: str) -> str:
    parts = [rubric, "\nFULL CONVERSATION ACROSS ALL AGENTS:\n"]

    parts.append("--- AGENT 1 (Assessment/Chat) ---")
    parts.append(conversation.agent1_transcript.text[:800])

    if conversation.agent2_transcript.messages:
        parts.append("\n--- AGENT 2 (Resolution/Voice) ---")
        parts.append(conversation.agent2_transcript.text[:800])

    if conversation.agent3_transcript.messages:
        parts.append("\n--- AGENT 3 (Final Notice/Chat) ---")
        parts.append(conversation.agent3_transcript.text[:800])

    parts.append("\nRate the end-to-end continuity (1-10):")
    return "\n".join(parts)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
