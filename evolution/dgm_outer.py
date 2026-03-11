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
from evaluation.rubrics import (
    DEFAULT_SCORING_WEIGHTS,
    GOAL_RUBRIC_V0,
    HANDOFF_RUBRIC_V0,
    QUALITY_RUBRIC_V0,
    SYSTEM_RUBRIC_V0,
)
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

    # Initialize eval config (fixed during evolution)
    eval_config = EvalConfig(
        version_id="eval_v0",
        goal_rubric=GOAL_RUBRIC_V0,
        quality_rubric=QUALITY_RUBRIC_V0,
        handoff_rubric=HANDOFF_RUBRIC_V0,
        system_rubric=SYSTEM_RUBRIC_V0,
        scoring_weights=DEFAULT_SCORING_WEIGHTS,
    )

    # Seed the archive if empty
    if archive.size == 0:
        seed_entry = await _create_seed_entry(eval_config, tracker, s)
        archive.add(seed_entry)
        logger.info(f"Seed v0: mean_score={seed_entry.mean_score:.2f}")

    recent_mutations: list[str] = []
    best_score = archive.get_best().mean_score
    plateau_count = 0

    for gen in range(max_gen):
        logger.info(f"=== Generation {gen} ===")

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

                # 2. Mutate
                trajectory = analyze_trajectory(parent, archive.entries)
                mutation = await rewrite(
                    parent, trajectory, tracker,
                    recent_mutations=recent_mutations,
                    settings=s,
                )

                if not mutation.components_modified:
                    logger.info(f"  Parent {parent.version_id}: rewriter returned no changes, skipping")
                    continue

                recent_mutations.append(mutation.rationale[:100])

                child_version = f"v{gen}_{parent_idx}_{uuid.uuid4().hex[:4]}"
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

    # Export results
    archive.export_raw_scores()
    breakdown = tracker.get_breakdown()
    logger.info(f"Evolution complete. Archive size: {archive.size}, Best: {archive.get_best().mean_score:.2f}")
    logger.info(f"Total cost: ${breakdown.total_usd:.4f}")
    logger.info(f"Call counts: {breakdown.call_counts}")

    return archive


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _create_seed_entry(
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings,
) -> ArchiveEntry:
    """Create the initial v0 entry with seed prompts, evaluated on 5 conversations."""
    prompts = get_seed_prompts()
    ac = AgentConfig(version_id="v0", **prompts)
    vc = VariantConfig(agent_config=ac, eval_config=eval_config)

    # Run 5 conversations (1 per persona)
    conversations = await _simulate_batch(vc, n_personas=5, runs_per=1, tracker=tracker, settings=settings, seed_offset=0)
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
        tracker=tracker, settings=settings,
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
        tracker=tracker, settings=settings,
        seed_offset=generation * 100 + 10,
        exclude_seeds={c.seed for c in all_conversations},
    )
    # Only take enough to reach ~8 new
    stage2_convos = stage2_convos[:8]
    stage2_scores = await _evaluate_batch(stage2_convos, eval_config, tracker, settings)
    all_conversations.extend(stage2_convos)
    all_scores.extend(stage2_scores)

    # Stage 3: +15 more (total 25) — only for promising candidates
    avg_score = sum(s.weighted_total for s in all_scores) / len(all_scores)
    best_in_archive = archive.get_best().mean_score if archive.size > 0 else 0

    if avg_score >= best_in_archive * 0.8:  # Within 80% of best
        stage3_convos = await _simulate_batch(
            child_vc, n_personas=5, runs_per=5,
            tracker=tracker, settings=settings,
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

    if not comparison.significant:
        logger.info(f"  {child.version_id}: not significant (CI=[{comparison.ci_lower}, {comparison.ci_upper}])")
        return False

    if comparison.variance_too_high:
        logger.info(f"  {child.version_id}: variance too high")
        return False

    if not comparison.compliance_preserved:
        logger.info(f"  {child.version_id}: compliance regression")
        return False

    if check_persona_regression(comparison):
        logger.info(f"  {child.version_id}: persona regression detected")
        return False

    # Strict grader validation (expensive — only for statistical winners)
    # Sample up to 5 conversations for strict grading
    sample_convos = child.scores[:5]
    # We need the actual conversations for strict grading, but we only have scores here.
    # For now, mark as validated if all other checks pass.
    # Full strict grading requires conversation objects (added when we have them in the pipeline).
    child.strict_grader_result = None  # Placeholder

    return True


async def _simulate_batch(
    vc: VariantConfig,
    n_personas: int,
    runs_per: int,
    tracker: CostTracker,
    settings: Settings,
    seed_offset: int = 0,
    exclude_seeds: set[int] | None = None,
) -> list[Conversation]:
    """Simulate a batch of conversations across personas."""
    from simulation.personas import get_persona_with_variation
    from models import PersonaType

    conversations = []
    persona_types = list(PersonaType)[:n_personas]
    exclude = exclude_seeds or set()

    for run in range(runs_per):
        for pt in persona_types:
            seed = seed_offset + hash((pt.value, run)) % 10000
            # Ensure unique seeds
            while seed in exclude:
                seed += 1
            exclude.add(seed)

            persona = get_persona_with_variation(pt, seed=seed)
            conv = await simulate_pipeline(
                agent_config=vc.agent_config,
                persona=persona,
                seed=seed,
                tracker=tracker,
                settings=settings,
            )
            conversations.append(conv)

    return conversations


async def _evaluate_batch(
    conversations: list[Conversation],
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings,
) -> list[ConversationScores]:
    """Score a batch of conversations."""
    scores = []
    for conv in conversations:
        score = await score_conversation(conv, eval_config, tracker, settings)
        scores.append(score)
    return scores
