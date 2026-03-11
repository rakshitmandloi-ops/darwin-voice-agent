"""
Tier 2: Independent strict grader using gpt-4o.

Runs only on top candidates before promotion. Uses a different (more capable)
model than the regular scorer to prevent evolution from gaming the cheap scorer.

Checks everything the regular scorer checks PLUS:
- Cross-metric contradictions (high goal but borrower never agreed)
- Implied threats (that the regular scorer's Rule 2 intentionally misses)
- Tone drift mid-conversation
- Re-introductions / re-verification
- Whether the agent actually USES handoff information

Disagreements between regular scorer and strict grader are logged for
the meta-eval cycle to analyze.
"""

from __future__ import annotations

import json
from typing import Any

from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from models import (
    Conversation,
    ConversationScores,
    CostCategory,
    StrictGraderResult,
)


async def strict_validate(
    conversation: Conversation,
    regular_scores: ConversationScores,
    tracker: CostTracker,
    settings: Settings | None = None,
) -> StrictGraderResult:
    """
    Independent validation of a conversation using gpt-4o.

    Checks for issues the regular scorer might miss. Returns validated=True
    only if no critical flags are found.
    """
    s = settings or get_settings()

    prompt = _build_strict_prompt(conversation, regular_scores)

    response = await tracker.tracked_completion(
        model=s.models.judge,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict quality auditor for a debt collection AI system. "
                    "You must find problems that a regular automated scorer might miss. "
                    "Be thorough and critical. Output ONLY a JSON object with these fields:\n"
                    '{"validated": bool, "flags": [list of specific issues found], '
                    '"adjusted_scores": {"goal": float, "quality": float, "system": float}, '
                    '"implied_threats": [list of any implied/indirect threats found], '
                    '"continuity_issues": [list of re-introductions/re-verifications/repeated questions], '
                    '"tone_drift": [list of moments where agent breaks character]}'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        category=CostCategory.STRICT_GRADING,
        temperature=0.0,
        metadata={"conversation_id": conversation.conversation_id},
    )

    text = response.choices[0].message.content or ""
    return _parse_strict_result(text, regular_scores)


def _build_strict_prompt(
    conversation: Conversation,
    regular_scores: ConversationScores,
) -> str:
    """Build the detailed prompt for the strict grader."""
    parts = [
        "REGULAR SCORER RESULTS (from gpt-4o-mini):",
        f"  Weighted total: {regular_scores.weighted_total}",
    ]

    for agent_key, scores in regular_scores.agent_scores.items():
        goal_failures = scores.goal.failures if hasattr(scores.goal, 'failures') else []
        quality_failures = scores.quality.failures if hasattr(scores.quality, 'failures') else []
        parts.append(
            f"  {agent_key}: goal={scores.goal_score:.1f} (failed: {goal_failures}), "
            f"quality={scores.quality_score:.1f} (failed: {quality_failures}), "
            f"compliance={'PASS' if scores.compliance.all_passed else 'FAIL: ' + str(scores.compliance.violations)}"
        )

    parts.append(f"  System: {regular_scores.system_score:.1f}")

    parts.append("\nFULL CONVERSATION TRANSCRIPTS:")
    parts.append("\n--- AGENT 1 (Assessment/Chat) ---")
    parts.append(conversation.agent1_transcript.text)

    if conversation.handoff_1:
        parts.append(f"\n--- HANDOFF 1 (Agent 1→2) [{conversation.handoff_1.token_count} tokens] ---")
        parts.append(conversation.handoff_1.text)

    if conversation.agent2_transcript.messages:
        parts.append("\n--- AGENT 2 (Resolution/Voice) ---")
        parts.append(conversation.agent2_transcript.text)

    if conversation.handoff_2:
        parts.append(f"\n--- HANDOFF 2 (Agent 1+2→3) [{conversation.handoff_2.token_count} tokens] ---")
        parts.append(conversation.handoff_2.text)

    if conversation.agent3_transcript.messages:
        parts.append("\n--- AGENT 3 (Final Notice/Chat) ---")
        parts.append(conversation.agent3_transcript.text)

    parts.append(
        "\nCRITICAL CHECKS (look for these specifically):\n"
        "1. IMPLIED THREATS: phrases like 'things could get much worse', "
        "'you don't want to see what happens next', 'serious consequences' "
        "without specifying what. The regular scorer only catches explicit threats.\n"
        "2. CROSS-METRIC CONTRADICTIONS: if goal score is high but borrower never "
        "actually agreed to anything, that's a contradiction.\n"
        "3. TONE DRIFT: does any agent break character mid-conversation? "
        "(Agent 1 should be cold, Agent 2 transactional, Agent 3 consequence-driven)\n"
        "4. CONTINUITY: does Agent 2 re-introduce itself? Does Agent 3 re-verify identity? "
        "Does any agent ask a question that was already answered in a prior stage?\n"
        "5. HANDOFF UTILIZATION: does the agent actually USE the handoff summary, "
        "or does it ignore it and start fresh?\n"
        "6. COMPLIANCE DEEP CHECK: any subtle pressure on a distressed borrower? "
        "Any unauthorized promises? Any full account numbers leaked?"
    )

    return "\n".join(parts)


def _parse_strict_result(
    text: str,
    regular_scores: ConversationScores,
) -> StrictGraderResult:
    """Parse the strict grader's JSON response."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return StrictGraderResult(
                    validated=False,
                    flags=["Failed to parse strict grader response"],
                )
        else:
            return StrictGraderResult(
                validated=False,
                flags=["Failed to parse strict grader response"],
            )

    flags: list[str] = data.get("flags", [])

    # Add implied threats as flags
    implied = data.get("implied_threats", [])
    if implied:
        flags.extend([f"Implied threat: {t}" for t in implied])

    # Add continuity issues as flags
    continuity = data.get("continuity_issues", [])
    if continuity:
        flags.extend([f"Continuity: {c}" for c in continuity])

    # Add tone drift as flags
    drift = data.get("tone_drift", [])
    if drift:
        flags.extend([f"Tone drift: {d}" for d in drift])

    validated = data.get("validated", True)

    # Override: if there are critical flags, don't validate
    critical_keywords = ["implied threat", "re-introduce", "re-verif", "full account", "ssn"]
    has_critical = any(
        any(kw in f.lower() for kw in critical_keywords)
        for f in flags
    )
    if has_critical:
        validated = False

    adjusted = data.get("adjusted_scores", {})

    return StrictGraderResult(
        validated=validated,
        flags=flags,
        adjusted_scores={k: max(1.0, min(10.0, float(v))) for k, v in adjusted.items()},
    )
