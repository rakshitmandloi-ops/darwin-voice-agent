"""
Checklist-based evaluation scorers.

Every criterion is a binary yes/no check evaluated by LLM.
The LLM sees the transcript + one specific criterion and returns true/false.

This replaces vague 1-10 scores with clear, auditable pass/fail per criterion.
"""

from __future__ import annotations

import asyncio
import json

from config import Settings, get_settings
from evaluation.compliance import check_compliance
from evaluation.cost_tracker import CostTracker
from evaluation.rubrics import (
    DEFAULT_SCORING_WEIGHTS,
    GOAL_CHECKS,
    HANDOFF_CHECKS,
    QUALITY_CHECKS,
    SYSTEM_CHECKS,
)
from models import (
    AgentScores,
    AgentType,
    ChecklistResult,
    Conversation,
    ConversationScores,
    CostCategory,
    EvalConfig,
    HandoffChecklist,
    Outcome,
    SystemChecklist,
)


async def score_conversation(
    conversation: Conversation,
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings | None = None,
) -> ConversationScores:
    """Score a full pipeline conversation using binary checklists."""
    s = settings or get_settings()

    # --- Per-agent scoring (goal + quality + compliance) ---
    agent_scores: dict[str, AgentScores] = {}

    agents_and_transcripts = [
        (AgentType.ASSESSMENT, conversation.agent1_transcript),
        (AgentType.RESOLUTION, conversation.agent2_transcript),
        (AgentType.FINAL_NOTICE, conversation.agent3_transcript),
    ]

    agent_tasks = []
    for agent_type, transcript in agents_and_transcripts:
        agent_tasks.append(
            _score_agent(agent_type, transcript, tracker, s)
        )

    agent_results = await asyncio.gather(*agent_tasks)
    for agent_type, result in zip([a[0] for a in agents_and_transcripts], agent_results):
        agent_scores[agent_type.value] = result

    # --- Handoff scoring ---
    handoff_tasks = []
    handoff_keys = []

    if conversation.handoff_1 and conversation.agent2_transcript.messages:
        handoff_keys.append("handoff_1")
        handoff_tasks.append(
            _score_handoff(
                conversation.handoff_1.text,
                conversation.agent1_transcript.text,
                conversation.agent2_transcript.text,
                tracker, s,
            )
        )

    if conversation.handoff_2 and conversation.agent3_transcript.messages:
        handoff_keys.append("handoff_2")
        handoff_tasks.append(
            _score_handoff(
                conversation.handoff_2.text,
                conversation.agent2_transcript.text,
                conversation.agent3_transcript.text,
                tracker, s,
            )
        )

    handoff_results = await asyncio.gather(*handoff_tasks) if handoff_tasks else []
    handoff_scores: dict[str, HandoffChecklist] = {}
    for key, result in zip(handoff_keys, handoff_results):
        handoff_scores[key] = result

    # Fill missing handoffs
    empty_handoff = HandoffChecklist(checks={k: False for k in HANDOFF_CHECKS})
    if "handoff_1" not in handoff_scores:
        handoff_scores["handoff_1"] = empty_handoff
    if "handoff_2" not in handoff_scores:
        handoff_scores["handoff_2"] = empty_handoff

    # --- System continuity ---
    system_checks = await _score_system(conversation, tracker, s)

    # --- Weighted total ---
    weights = eval_config.scoring_weights
    avg_goal = _mean([a.goal_score for a in agent_scores.values()])
    avg_quality = _mean([a.quality_score for a in agent_scores.values()])
    compliance_score = 10.0 if all(a.compliance.all_passed for a in agent_scores.values()) else 1.0
    avg_handoff = _mean([h.score for h in handoff_scores.values()])
    sys_score = system_checks.score

    weighted_total = (
        weights.get("goal", 0.3) * avg_goal
        + weights.get("compliance", 0.2) * compliance_score
        + weights.get("quality", 0.2) * avg_quality
        + weights.get("handoff", 0.15) * avg_handoff
        + weights.get("system", 0.15) * sys_score
    )

    resolved = 1.0 if conversation.outcome in (Outcome.DEAL_AGREED, Outcome.HARDSHIP_REFERRAL) else 0.0

    return ConversationScores(
        conversation_id=conversation.conversation_id,
        persona_type=conversation.persona.persona_type,
        agent_scores=agent_scores,
        handoff_scores=handoff_scores,
        system_checks=system_checks,
        weighted_total=round(weighted_total, 2),
        resolution_rate=resolved,
    )


# ---------------------------------------------------------------------------
# Per-agent scoring
# ---------------------------------------------------------------------------

async def _score_agent(
    agent_type: AgentType,
    transcript: "Transcript",
    tracker: CostTracker,
    settings: Settings,
) -> AgentScores:
    """Score one agent: goal checklist + quality checklist + compliance."""
    from models import Transcript

    if not transcript.messages:
        empty_goal = ChecklistResult(checks={k: False for k in GOAL_CHECKS.get(agent_type.value, {})})
        empty_quality = ChecklistResult(checks={k: False for k in QUALITY_CHECKS})
        return AgentScores(
            agent=agent_type,
            goal=empty_goal,
            quality=empty_quality,
            compliance=check_compliance(transcript, agent_type),
        )

    text = transcript.text

    # Score all goal + quality checks in parallel
    goal_criteria = GOAL_CHECKS.get(agent_type.value, {})
    quality_criteria = QUALITY_CHECKS

    all_checks = []
    all_keys = []
    all_categories = []

    for key, description in goal_criteria.items():
        all_keys.append(key)
        all_categories.append("goal")
        all_checks.append(
            _check_criterion(text, description, agent_type.value, tracker, settings)
        )

    for key, description in quality_criteria.items():
        all_keys.append(key)
        all_categories.append("quality")
        all_checks.append(
            _check_criterion(text, description, agent_type.value, tracker, settings)
        )

    results = await asyncio.gather(*all_checks)

    goal_results = {}
    quality_results = {}
    for key, category, result in zip(all_keys, all_categories, results):
        if category == "goal":
            goal_results[key] = result
        else:
            quality_results[key] = result

    return AgentScores(
        agent=agent_type,
        goal=ChecklistResult(checks=goal_results),
        quality=ChecklistResult(checks=quality_results),
        compliance=check_compliance(transcript, agent_type),
    )


async def _score_handoff(
    summary: str,
    source_text: str,
    target_text: str,
    tracker: CostTracker,
    settings: Settings,
) -> HandoffChecklist:
    """Score handoff using checklist."""
    context = f"HANDOFF SUMMARY:\n{summary}\n\nSOURCE CONVERSATION:\n{source_text[:800]}\n\nTARGET CONVERSATION:\n{target_text[:800]}"

    tasks = []
    keys = []
    for key, description in HANDOFF_CHECKS.items():
        keys.append(key)
        tasks.append(
            _check_criterion(context, description, "handoff", tracker, settings)
        )

    results = await asyncio.gather(*tasks)
    return HandoffChecklist(checks=dict(zip(keys, results)))


async def _score_system(
    conversation: Conversation,
    tracker: CostTracker,
    settings: Settings,
) -> SystemChecklist:
    """Score system-level continuity using checklist."""
    parts = ["FULL CONVERSATION:\n"]
    parts.append("--- Agent 1 (Assessment/Chat) ---\n" + conversation.agent1_transcript.text[:600])
    if conversation.agent2_transcript.messages:
        parts.append("\n--- Agent 2 (Resolution/Voice) ---\n" + conversation.agent2_transcript.text[:600])
    if conversation.agent3_transcript.messages:
        parts.append("\n--- Agent 3 (Final Notice/Chat) ---\n" + conversation.agent3_transcript.text[:600])
    context = "\n".join(parts)

    tasks = []
    keys = []
    for key, description in SYSTEM_CHECKS.items():
        keys.append(key)
        tasks.append(
            _check_criterion(context, description, "system", tracker, settings)
        )

    results = await asyncio.gather(*tasks)
    return SystemChecklist(checks=dict(zip(keys, results)))


# ---------------------------------------------------------------------------
# Single criterion check
# ---------------------------------------------------------------------------

async def _check_criterion(
    text: str,
    criterion: str,
    context_label: str,
    tracker: CostTracker,
    settings: Settings,
) -> bool:
    """Ask LLM: does this text satisfy this criterion? Returns True/False."""
    try:
        response = await tracker.tracked_completion(
            model=settings.models.eval,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate AI debt collection conversations against specific criteria. "
                        "Answer with ONLY a JSON object: {\"pass\": true} or {\"pass\": false}. "
                        "Be fair and evidence-based. Pass if the criterion is reasonably satisfied in the conversation. "
                        "Fail only if there is clear evidence the criterion was violated or clearly not met."
                    ),
                },
                {
                    "role": "user",
                    "content": f"CRITERION: {criterion}\n\nCONVERSATION ({context_label}):\n{text[:2000]}",
                },
            ],
            category=CostCategory.EVALUATION,
            temperature=0.0,
            metadata={"criterion": criterion[:50], "context": context_label},
        )
        result_text = response.choices[0].message.content or ""
        return _parse_bool(result_text)
    except Exception:
        return False  # Default to fail on error — conservative


def _parse_bool(text: str) -> bool:
    """Parse LLM response to boolean."""
    try:
        data = json.loads(text)
        return bool(data.get("pass", False))
    except (json.JSONDecodeError, TypeError):
        lower = text.lower().strip()
        if "true" in lower or '"pass": true' in lower:
            return True
        return False


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
