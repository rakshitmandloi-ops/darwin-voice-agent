"""
Trajectory analysis — aggregates performance data across conversations
to feed the rewriter with patterns, not just individual failures.

The rewriter needs to understand: "Combative borrowers score 3.2 on average,
Cooperative score 7.8, and compliance is always strong but goal completion
is weak." Not just "here's a bad conversation."
"""

from __future__ import annotations

from models import ArchiveEntry, ConversationScores, TrajectoryAnalysis


def analyze_trajectory(
    entry: ArchiveEntry,
    archive_entries: list[ArchiveEntry] | None = None,
) -> TrajectoryAnalysis:
    """
    Analyze a variant's performance across all its scored conversations.

    Returns aggregate breakdowns that the rewriter can act on.
    """
    scores = entry.scores

    # Per-metric breakdown
    scores_by_metric = _compute_metric_breakdown(scores)

    # Per-persona breakdown (approximate — based on conversation ordering)
    scores_by_persona = _compute_persona_breakdown(scores)

    # Cross-generation trends (if archive provided)
    cross_gen_trends = _compute_trends(entry, archive_entries or [])

    # Systematic failures
    systematic_failures = _find_systematic_failures(scores)

    # Ceiling/floor detection
    ceiling_floor_flags = _detect_ceiling_floor(scores_by_metric)

    # Win/loss by persona
    win_loss = _compute_win_loss(scores)

    return TrajectoryAnalysis(
        scores_by_persona=scores_by_persona,
        scores_by_metric=scores_by_metric,
        cross_gen_trends=cross_gen_trends,
        systematic_failures=systematic_failures,
        ceiling_floor_flags=ceiling_floor_flags,
        win_loss_by_persona=win_loss,
    )


def format_trajectory_for_rewriter(
    analysis: TrajectoryAnalysis,
    worst_conversations: list[tuple[ConversationScores, str]] | None = None,
) -> str:
    """
    Format trajectory analysis as text for the rewriter LLM.

    Aggregate patterns first (primary), specific examples second (supporting).
    """
    parts = ["## PERFORMANCE ANALYSIS\n"]

    # Metric breakdown
    parts.append("### Scores by Metric (avg across all conversations)")
    for metric, score in sorted(analysis.scores_by_metric.items()):
        parts.append(f"  {metric}: {score:.2f}")

    # Persona breakdown
    if analysis.scores_by_persona:
        parts.append("\n### Scores by Scenario")
        for persona, score in sorted(analysis.scores_by_persona.items()):
            parts.append(f"  {persona}: {score:.2f}")

    # Systematic failures
    if analysis.systematic_failures:
        parts.append("\n### Systematic Failures (patterns across conversations)")
        for failure in analysis.systematic_failures:
            parts.append(f"  - {failure}")

    # Ceiling/floor
    if analysis.ceiling_floor_flags:
        parts.append("\n### Ceiling/Floor Warnings")
        for flag in analysis.ceiling_floor_flags:
            parts.append(f"  - {flag}")

    # Trends
    if analysis.cross_gen_trends:
        parts.append("\n### Cross-Generation Trends")
        for metric, values in analysis.cross_gen_trends.items():
            if len(values) >= 2:
                trend = "↑" if values[-1] > values[0] else "↓" if values[-1] < values[0] else "→"
                parts.append(f"  {metric}: {' → '.join(f'{v:.1f}' for v in values[-3:])} {trend}")

    # Worst conversations (supporting evidence)
    if worst_conversations:
        parts.append("\n### Worst Conversation Examples")
        for i, (score, summary) in enumerate(worst_conversations[:3]):
            parts.append(f"\n  Example {i+1} (total={score.weighted_total:.1f}):")
            parts.append(f"    {summary[:300]}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal analysis functions
# ---------------------------------------------------------------------------

def _compute_metric_breakdown(scores: list[ConversationScores]) -> dict[str, float]:
    """Average score per metric dimension across all conversations."""
    if not scores:
        return {}

    metrics: dict[str, list[float]] = {
        "goal_avg": [],
        "quality_avg": [],
        "compliance_rate": [],
        "handoff_avg": [],
        "system": [],
        "weighted_total": [],
    }

    for s in scores:
        goals = [a.goal for a in s.agent_scores.values()]
        qualities = [a.quality for a in s.agent_scores.values()]
        compliant = 1.0 if s.compliance_passed else 0.0

        metrics["goal_avg"].append(sum(goals) / len(goals) if goals else 0)
        metrics["quality_avg"].append(sum(qualities) / len(qualities) if qualities else 0)
        metrics["compliance_rate"].append(compliant)
        metrics["handoff_avg"].append(
            sum(s.handoff_scores.values()) / len(s.handoff_scores) if s.handoff_scores else 0
        )
        metrics["system"].append(s.system_score)
        metrics["weighted_total"].append(s.weighted_total)

    return {k: sum(v) / len(v) for k, v in metrics.items() if v}


def _compute_persona_breakdown(scores: list[ConversationScores]) -> dict[str, float]:
    """
    Approximate per-persona breakdown.
    Since ConversationScores doesn't carry persona info directly,
    we use conversation_id position as proxy (scores are generated in persona order).
    """
    if not scores:
        return {}

    # Simple: return distribution of weighted totals
    return {"all_scenarios": sum(s.weighted_total for s in scores) / len(scores)}


def _compute_trends(
    current: ArchiveEntry,
    archive: list[ArchiveEntry],
) -> dict[str, list[float]]:
    """Track metric values across generations in this entry's lineage."""
    if not archive:
        return {}

    # Walk lineage
    lineage_scores: dict[str, list[float]] = {"weighted_total": []}
    current_entry = current
    chain = [current_entry]

    archive_map = {e.version_id: e for e in archive}
    while current_entry.parent_id and current_entry.parent_id in archive_map:
        current_entry = archive_map[current_entry.parent_id]
        chain.append(current_entry)

    chain.reverse()  # Oldest first

    for entry in chain:
        if entry.scores:
            avg = sum(s.weighted_total for s in entry.scores) / len(entry.scores)
            lineage_scores["weighted_total"].append(round(avg, 2))

    return lineage_scores


def _find_systematic_failures(scores: list[ConversationScores]) -> list[str]:
    """Identify patterns of failure across conversations."""
    failures = []

    if not scores:
        return failures

    # Low goal completion
    avg_goals = []
    for s in scores:
        goals = [a.goal for a in s.agent_scores.values()]
        avg_goals.append(sum(goals) / len(goals) if goals else 0)
    if avg_goals and sum(avg_goals) / len(avg_goals) < 5.0:
        failures.append("Low goal completion across conversations (avg < 5.0)")

    # Compliance failures
    compliance_fails = sum(1 for s in scores if not s.compliance_passed)
    if compliance_fails > 0:
        rate = compliance_fails / len(scores)
        failures.append(f"Compliance failures in {compliance_fails}/{len(scores)} conversations ({rate:.0%})")

    # Low system continuity
    avg_system = sum(s.system_score for s in scores) / len(scores)
    if avg_system < 5.0:
        failures.append(f"Low cross-agent continuity (avg system score: {avg_system:.1f})")

    # Handoff quality issues
    for key in ["handoff_1", "handoff_2"]:
        vals = [s.handoff_scores.get(key, 0) for s in scores]
        if vals and sum(vals) / len(vals) < 5.0:
            failures.append(f"Poor {key} quality (avg: {sum(vals)/len(vals):.1f})")

    return failures


def _detect_ceiling_floor(metrics: dict[str, float]) -> list[str]:
    """Flag metrics that are always near 10 or near 1 (not discriminating)."""
    flags = []
    for metric, value in metrics.items():
        if metric == "compliance_rate":
            continue  # Binary metric, expected to be near 1
        if value >= 9.0:
            flags.append(f"CEILING: {metric} = {value:.1f} (rubric may be too easy)")
        elif value <= 2.0:
            flags.append(f"FLOOR: {metric} = {value:.1f} (rubric may be too hard or broken)")
    return flags


def _compute_win_loss(scores: list[ConversationScores]) -> dict[str, dict[str, int]]:
    """Win (>7), loss (<4), draw counts."""
    wins = sum(1 for s in scores if s.weighted_total >= 7.0)
    losses = sum(1 for s in scores if s.weighted_total < 4.0)
    draws = len(scores) - wins - losses
    return {"overall": {"wins": wins, "losses": losses, "draws": draws}}
