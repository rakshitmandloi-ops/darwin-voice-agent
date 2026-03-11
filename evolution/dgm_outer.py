"""
DGM outer loop — the main evolution engine.

Each generation:
1. Select parents (probabilistic)
2. Mutate (trajectory-aware, prompts + summarizer only)
3. Staged simulation + full evaluation (2 → 10 → 25 convos, accumulative)
4. Per-child promotion against its own parent
5. Meta-eval hook (pluggable, every N generations)

Rubrics are FIXED during evolution. Only the meta-eval cycle changes them.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from agents.prompts import get_seed_prompts
from config import Settings, get_settings
from evaluation.cost_tracker import BudgetExhaustedError, CostTracker
from evaluation.rubrics import DEFAULT_SCORING_WEIGHTS
from evaluation.scorers import score_conversation
from evaluation.stats import check_persona_regression, paired_bootstrap
from evaluation.strict_grader import strict_validate
from evolution.archive import Archive
from evolution.rewriter import apply_mutation, rewrite
from evolution.selection import select_parents
from evolution.trajectory import analyze_trajectory
from models import (
    AgentConfig,
    ArchiveEntry,
    Conversation,
    ConversationScores,
    EvalConfig,
    Outcome,
    VariantConfig,
)
from simulation.personas import get_persona_with_variation
from simulation.simulator import simulate_pipeline

logger = logging.getLogger(__name__)


async def run_evolution(
    max_generations: int | None = None,
    budget_limit: float | None = None,
    meta_eval_fn: Callable | None = None,
    settings: Settings | None = None,
) -> Archive:
    """
    Run the full DGM evolution loop.

    Returns the archive with all variants and their scores.
    """
    s = settings or get_settings()
    max_gen = max_generations or s.evolution.max_generations
    tracker = CostTracker(s)
    archive = Archive(s)

    # Live state for dashboard
    from live_state import get_live_state
    ls = get_live_state(s.logs_dir)

    # Initialize eval config (fixed during evolution)
    # Rubric text fields are no longer used — scorers use GOAL_CHECKS etc. directly
    eval_config = EvalConfig(
        version_id="eval_v0",
        scoring_weights=DEFAULT_SCORING_WEIGHTS,
    )

    # Seed the archive if empty
    if archive.size == 0:
        ls.set_evaluating_seed()
        seed_entry = await _create_seed_entry(eval_config, tracker, s, archive=archive)
        archive.add(seed_entry)
        logger.info(f"Seed v0: mean_score={seed_entry.mean_score:.2f}")

    recent_mutations: list[str] = []
    best_score = archive.get_best().mean_score
    plateau_count = 0

    for gen in range(max_gen):
        logger.info(f"=== Generation {gen} ===")
        ls.set_generation(gen)

        # 0. Budget check
        if tracker.is_budget_exhausted():
            logger.info("Budget exhausted, stopping evolution")
            break

        try:
            # 1. Select parents
            parents = select_parents(
                archive.get_active(),
                k=s.evolution.children_per_generation,
                seed=gen * 1000,
            )

            children: list[ArchiveEntry] = []

            for parent_idx, parent in enumerate(parents):
                archive.increment_children(parent.version_id)

                # 2. Mutate — include worst conversation transcripts
                trajectory = analyze_trajectory(parent, archive.entries)

                # Get worst conversations from archive transcripts
                worst_convos = _get_worst_conversations(parent, archive)

                mutation = await rewrite(
                    parent, trajectory, tracker,
                    recent_mutations=recent_mutations,
                    worst_conversations=worst_convos,
                    settings=s,
                )

                if not mutation.components_modified:
                    logger.info(f"  Parent {parent.version_id}: rewriter returned no changes, skipping")
                    continue

                recent_mutations.append(mutation.rationale[:100])

                child_version = f"v{gen}_{parent_idx}_{uuid.uuid4().hex[:4]}"
                ls.set_mutating(parent.version_id, child_version)
                child_ac = apply_mutation(
                    parent.variant_config.agent_config,
                    mutation,
                    child_version,
                )

                # Hard check: eval_config unchanged
                child_vc = VariantConfig(
                    agent_config=child_ac,
                    eval_config=eval_config,
                )

                # 3. Staged simulation + full evaluation (accumulative)
                child_entry = await _staged_evaluate(
                    child_vc=child_vc,
                    parent=parent,
                    mutation=mutation,
                    generation=gen,
                    eval_config=eval_config,
                    tracker=tracker,
                    archive=archive,
                    settings=s,
                )

                if child_entry is not None:
                    children.append(child_entry)

            # 4. Per-child promotion
            gen_promoted = False
            for child in children:
                if child.discarded:
                    continue

                parent_entry = archive.get(child.parent_id) if child.parent_id else None
                if parent_entry is None:
                    continue

                promoted = await _try_promote(
                    child=child,
                    parent=parent_entry,
                    eval_config=eval_config,
                    tracker=tracker,
                    settings=s,
                )

                if promoted:
                    child.promoted = True
                    gen_promoted = True
                    logger.info(f"  PROMOTED: {child.version_id} (score={child.mean_score:.2f})")
                    ls.set_promoting(child.version_id, f"PROMOTED (score={child.mean_score:.2f})")
                else:
                    ls.set_promoting(child.version_id, f"not promoted (score={child.mean_score:.2f})")

            # 5. Meta-eval hook
            if meta_eval_fn and gen > 0 and gen % s.meta_eval.frequency == 0:
                logger.info(f"  Running meta-eval at generation {gen}")
                eval_config = await meta_eval_fn(archive, eval_config, tracker, s)

            # 6. Termination checks
            current_best = archive.get_best().mean_score
            if current_best >= s.evolution.success_threshold:
                logger.info(f"Success threshold reached: {current_best:.2f} >= {s.evolution.success_threshold}")
                break

            if current_best > best_score:
                best_score = current_best
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= s.evolution.plateau_generations:
                logger.info(f"Plateau detected ({plateau_count} generations), stopping")
                break

        except BudgetExhaustedError:
            logger.info("Budget exhausted mid-generation, stopping")
            break

    # Export results and mark batch complete
    archive.export_raw_scores()
    archive.complete()
    breakdown = tracker.get_breakdown()
    logger.info(f"Evolution complete. Archive size: {archive.size}, Best: {archive.get_best().mean_score:.2f}")
    logger.info(f"Total cost: ${breakdown.total_usd:.4f}")
    logger.info(f"Call counts: {breakdown.call_counts}")

    return archive


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_worst_conversations(
    parent: ArchiveEntry,
    archive: Archive,
) -> list[dict]:
    """
    Get the worst-scoring conversations for this parent variant,
    with full transcripts loaded from archive storage.

    Returns up to 3 worst conversations with transcripts + failing criteria.
    """
    if not parent.scores:
        return []

    # Sort by weighted_total, get worst 3
    sorted_scores = sorted(parent.scores, key=lambda s: s.weighted_total)
    worst = sorted_scores[:3]

    result = []
    for score in worst:
        # Load transcript from archive
        transcript = archive.get_transcript(score.conversation_id)
        if not transcript:
            continue

        # Collect all failing criteria across agents
        failing = []
        for agent_key, agent_score in score.agent_scores.items():
            for check_name, passed in agent_score.goal.checks.items():
                if not passed:
                    failing.append(f"{agent_key}/{check_name}")
            for check_name, passed in agent_score.quality.checks.items():
                if not passed:
                    failing.append(f"{agent_key}/quality:{check_name}")
            for rule, passed in agent_score.compliance.rule_results.items():
                if not passed:
                    failing.append(f"{agent_key}/compliance:{rule}")

        # Handoff failures
        for hk, hv in score.handoff_scores.items():
            for check_name, passed in hv.checks.items():
                if not passed:
                    failing.append(f"{hk}/{check_name}")

        # System failures
        for check_name, passed in score.system_checks.checks.items():
            if not passed:
                failing.append(f"system/{check_name}")

        conv_data = {
            "conversation_id": score.conversation_id,
            "persona_type": score.persona_type.value,
            "total": score.weighted_total,
            "failing_criteria": failing,
            "agent1": transcript.get("agent1", []),
            "agent2": transcript.get("agent2", []),
            "agent3": transcript.get("agent3", []),
            "handoff_1": transcript.get("handoff_1"),
            "handoff_2": transcript.get("handoff_2"),
        }
        result.append(conv_data)

    return result


async def _create_seed_entry(
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings,
    archive: Archive | None = None,
) -> ArchiveEntry:
    """Create the initial v0 entry with seed prompts, evaluated on 5 conversations."""
    prompts = get_seed_prompts()
    ac = AgentConfig(version_id="v0", **prompts)
    vc = VariantConfig(agent_config=ac, eval_config=eval_config)

    # Run 5 conversations (1 per persona)
    conversations = await _simulate_batch(vc, n_personas=5, runs_per=1, tracker=tracker, settings=settings, archive=archive, seed_offset=0)
    scores = await _evaluate_batch(conversations, eval_config, tracker, settings)

    return ArchiveEntry(
        version_id="v0",
        variant_config=vc,
        scores=scores,
        generation=0,
        mutation_description="Seed v0 prompts",
    )


async def _staged_evaluate(
    *,
    child_vc: VariantConfig,
    parent: ArchiveEntry,
    mutation: Any,
    generation: int,
    eval_config: EvalConfig,
    tracker: CostTracker,
    archive: Archive,
    settings: Settings,
) -> ArchiveEntry | None:
    """
    Staged simulation: 2 → +8 = 10 → +15 = 25 conversations.
    Conversations ACCUMULATE across stages.
    Returns ArchiveEntry if passes compliance, None if discarded.
    """
    child_version = child_vc.agent_config.version_id
    all_conversations: list[Conversation] = []
    all_scores: list[ConversationScores] = []

    # Stage 1: 2 conversations
    stage1_convos = await _simulate_batch(
        child_vc, n_personas=2, runs_per=1,
        tracker=tracker, settings=settings, archive=archive,
        seed_offset=generation * 100,
    )
    stage1_scores = await _evaluate_batch(stage1_convos, eval_config, tracker, settings)
    all_conversations.extend(stage1_convos)
    all_scores.extend(stage1_scores)

    # Compliance gate
    if not all(s.compliance_passed for s in stage1_scores):
        logger.info(f"  {child_version}: DISCARDED (compliance fail at Stage 1)")
        entry = ArchiveEntry(
            version_id=child_version,
            variant_config=child_vc,
            scores=all_scores,
            parent_id=parent.version_id,
            generation=generation,
            mutation_description=mutation.rationale[:200],
            components_modified=mutation.components_modified,
            discarded=True,
            discard_reason="compliance_fail_stage1",
        )
        archive.add(entry)
        return entry

    # Stage 2: +8 more conversations (total 10)
    stage2_convos = await _simulate_batch(
        child_vc, n_personas=5, runs_per=2,
        tracker=tracker, settings=settings, archive=archive,
        seed_offset=generation * 100 + 10,
        exclude_seeds={c.seed for c in all_conversations},
    )
    stage2_convos = stage2_convos[:8]
    stage2_scores = await _evaluate_batch(stage2_convos, eval_config, tracker, settings)
    all_conversations.extend(stage2_convos)
    all_scores.extend(stage2_scores)

    # Stage 3: +15 more (total 25) — only for promising candidates
    avg_score = sum(s.weighted_total for s in all_scores) / len(all_scores)
    best_in_archive = archive.get_best().mean_score if archive.size > 0 else 0

    if avg_score >= best_in_archive * 0.8:
        stage3_convos = await _simulate_batch(
            child_vc, n_personas=5, runs_per=5,
            tracker=tracker, settings=settings, archive=archive,
            seed_offset=generation * 100 + 50,
            exclude_seeds={c.seed for c in all_conversations},
        )
        stage3_convos = stage3_convos[:15]
        stage3_scores = await _evaluate_batch(stage3_convos, eval_config, tracker, settings)
        all_conversations.extend(stage3_convos)
        all_scores.extend(stage3_scores)

    entry = ArchiveEntry(
        version_id=child_version,
        variant_config=child_vc,
        scores=all_scores,
        parent_id=parent.version_id,
        generation=generation,
        mutation_description=mutation.rationale[:200],
        components_modified=mutation.components_modified,
        rationale=mutation.rationale,
    )
    archive.add(entry)
    logger.info(f"  {child_version}: scored {entry.mean_score:.2f} ({len(all_scores)} convos)")

    return entry


async def _try_promote(
    *,
    child: ArchiveEntry,
    parent: ArchiveEntry,
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings,
) -> bool:
    """
    Attempt to promote a child by comparing against its own parent.

    Returns True if promoted, False otherwise.
    """
    # Statistical comparison
    comparison = paired_bootstrap(parent.scores, child.scores)

    logger.info(
        f"  {child.version_id} vs {parent.version_id}: "
        f"diff={comparison.mean_diff:+.2f}, CI=[{comparison.ci_lower:.2f}, {comparison.ci_upper:.2f}], "
        f"p={comparison.p_value:.3f}, var_high={comparison.variance_too_high}"
    )

    # Log per-persona breakdown
    if comparison.per_persona_breakdown:
        for persona, diff in sorted(comparison.per_persona_breakdown.items()):
            logger.info(f"    {persona}: {diff:+.2f}")

    if not comparison.significant:
        logger.info(f"  {child.version_id}: NOT PROMOTED — not significant")
        return False

    if comparison.variance_too_high:
        logger.info(f"  {child.version_id}: NOT PROMOTED — variance too high")
        return False

    if not comparison.compliance_preserved:
        logger.info(f"  {child.version_id}: NOT PROMOTED — compliance regression")
        return False

    if check_persona_regression(comparison):
        regressed = [p for p, d in comparison.per_persona_breakdown.items() if d < -0.5]
        logger.info(f"  {child.version_id}: NOT PROMOTED — persona regression: {regressed}")
        return False

    child.strict_grader_result = None
    return True


async def _simulate_batch(
    vc: VariantConfig,
    n_personas: int,
    runs_per: int,
    tracker: CostTracker,
    settings: Settings,
    archive: Archive | None = None,
    seed_offset: int = 0,
    exclude_seeds: set[int] | None = None,
) -> list[Conversation]:
    """Simulate a batch of conversations across personas. Runs concurrently."""
    import asyncio
    from simulation.personas import get_persona_with_variation
    from models import PersonaType

    persona_types = list(PersonaType)[:n_personas]
    exclude = exclude_seeds or set()

    tasks = []
    for run in range(runs_per):
        for pt in persona_types:
            seed = seed_offset + hash((pt.value, run)) % 10000
            while seed in exclude:
                seed += 1
            exclude.add(seed)

            persona = get_persona_with_variation(pt, seed=seed)
            tasks.append(
                simulate_pipeline(
                    agent_config=vc.agent_config,
                    persona=persona,
                    seed=seed,
                    tracker=tracker,
                    settings=settings,
                )
            )

    conversations = await asyncio.gather(*tasks)
    result = list(conversations)

    # Store transcripts if archive provided
    if archive:
        for conv in result:
            archive.store_conversation(conv)

    return result


async def _evaluate_batch(
    conversations: list[Conversation],
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings,
) -> list[ConversationScores]:
    """Score a batch of conversations. Runs concurrently."""
    import asyncio
    tasks = [
        score_conversation(conv, eval_config, tracker, settings)
        for conv in conversations
    ]
    scores = await asyncio.gather(*tasks)
    return list(scores)
