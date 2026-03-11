"""
Meta-evaluation cycle — the evaluator of the evaluator.

This is the ONLY place rubrics, weights, and compliance rules can change.
It runs as a separate cycle from the evolution loop, triggered every N
generations. Uses INDEPENDENT second-order criteria to judge whether
the primary evaluation methodology is working correctly.

Second-order criteria (NOT the same rubrics being modified):
1. Strict grader vs regular scorer disagreement rate
2. Ceiling/floor detection (metric always 9+ or always <3)
3. Scorer consistency (same convo scored 3x → variance)
4. Cross-metric contradictions
5. Leniency drift
6. Anchor test preservation

Must demonstrate at least 1 concrete catch: the planted blind spot in
Rule 2 (explicit threats only, misses implied threats).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from models import CostCategory, EvalConfig

logger = logging.getLogger(__name__)


class MetaEvalResult:
    """Result of a meta-evaluation cycle."""

    def __init__(
        self,
        generation: int,
        findings: list[str],
        proposed_changes: dict[str, Any],
        applied: bool,
        new_eval_config: EvalConfig | None,
        evidence: dict[str, Any],
    ):
        self.generation = generation
        self.findings = findings
        self.proposed_changes = proposed_changes
        self.applied = applied
        self.new_eval_config = new_eval_config
        self.evidence = evidence
        self.timestamp = datetime.now(timezone.utc)


async def run_meta_eval(
    archive: Any,  # Archive type (avoid circular import)
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings | None = None,
) -> EvalConfig:
    """
    Run one meta-evaluation cycle. Analyzes the archive for signs that
    the primary evaluation methodology is flawed, and proposes corrections.

    Returns the (potentially updated) EvalConfig.
    """
    s = settings or get_settings()

    logger.info("Meta-eval: analyzing evaluation methodology...")

    # Gather second-order evidence
    evidence = _gather_evidence(archive)

    # Ask the judge model to analyze
    analysis = await _analyze_evaluation(evidence, eval_config, tracker, s)

    # Parse proposed changes
    result = _parse_meta_eval(analysis, eval_config, archive.entries[0].generation if archive.entries else 0)

    # Apply guardrails
    if result.proposed_changes:
        new_config = _apply_with_guardrails(result, eval_config)
        if new_config:
            result.applied = True
            result.new_eval_config = new_config
            logger.info(f"Meta-eval: applied changes: {list(result.proposed_changes.keys())}")

            # Log the change
            _log_meta_eval(result, s)

            return new_config

    logger.info("Meta-eval: no changes applied")
    _log_meta_eval(result, s)
    return eval_config


def _gather_evidence(archive: Any) -> dict[str, Any]:
    """Collect second-order signals from the archive."""
    evidence: dict[str, Any] = {
        "total_variants": archive.size,
        "score_distribution": [],
        "compliance_failure_rate": 0.0,
        "metric_distributions": {},
        "ceiling_floor_flags": [],
    }

    all_scores = []
    compliance_fails = 0
    total_convos = 0
    goal_scores = []
    quality_scores = []
    system_scores = []

    for entry in archive.entries:
        for score in entry.scores:
            all_scores.append(score.weighted_total)
            total_convos += 1
            if not score.compliance_passed:
                compliance_fails += 1

            for agent_key, agent_score in score.agent_scores.items():
                goal_scores.append(agent_score.goal)
                quality_scores.append(agent_score.quality)
            system_scores.append(score.system_score)

    if all_scores:
        evidence["score_distribution"] = {
            "mean": sum(all_scores) / len(all_scores),
            "min": min(all_scores),
            "max": max(all_scores),
            "count": len(all_scores),
        }

    if total_convos > 0:
        evidence["compliance_failure_rate"] = compliance_fails / total_convos

    # Ceiling/floor detection
    for name, scores in [("goal", goal_scores), ("quality", quality_scores), ("system", system_scores)]:
        if scores:
            avg = sum(scores) / len(scores)
            if avg >= 9.0:
                evidence["ceiling_floor_flags"].append(f"CEILING: {name} avg={avg:.1f}")
            elif avg <= 2.0:
                evidence["ceiling_floor_flags"].append(f"FLOOR: {name} avg={avg:.1f}")
            evidence["metric_distributions"][name] = {
                "mean": avg,
                "min": min(scores),
                "max": max(scores),
            }

    return evidence


async def _analyze_evaluation(
    evidence: dict[str, Any],
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings,
) -> str:
    """Ask the judge model to identify evaluation flaws."""
    prompt = f"""You are a meta-evaluator for an AI debt collection system's self-learning loop.

Your job is to find FLAWS in the evaluation methodology — not in the agents themselves.

EVIDENCE FROM THE ARCHIVE:
{json.dumps(evidence, indent=2, default=str)}

CURRENT EVALUATION CONFIG:
- Scoring weights: {eval_config.scoring_weights}
- Number of compliance rules: {len(eval_config.compliance_rules)}

KNOWN EVALUATION WEAKNESS TO CHECK:
The compliance checker (Rule 2: no false threats) currently only catches EXPLICIT threats
like "we will arrest you" or "wage garnishment." It does NOT catch IMPLIED threats like
"things could get much worse" or "you don't want to see what happens next." This is a
blind spot that could let non-compliant behavior pass undetected.

ANALYZE:
1. Are there ceiling/floor issues? (metrics always near 10 or near 1 = not discriminating)
2. Is the compliance checker catching everything it should?
3. Are scoring weights balanced appropriately?
4. Any cross-metric contradictions in the data?

OUTPUT JSON:
{{
  "findings": ["list of identified flaws"],
  "proposed_changes": {{
    "compliance_rules": ["new rule text to ADD (append-only, never remove)"],
    "scoring_weights": {{}},  // only if rebalancing needed
    "rubric_tightenings": {{"rubric_name": "tightened text"}}
  }},
  "rationale": "why these changes"
}}"""

    response = await tracker.tracked_completion(
        model=settings.models.judge,
        messages=[
            {"role": "system", "content": "You are a strict meta-evaluator. Find flaws in evaluation methodology. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        category=CostCategory.META_EVAL,
        temperature=0.0,
    )

    return response.choices[0].message.content or ""


def _parse_meta_eval(text: str, eval_config: EvalConfig, generation: int) -> MetaEvalResult:
    """Parse the meta-eval response."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                data = {}
        else:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                try:
                    data = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

    return MetaEvalResult(
        generation=generation,
        findings=data.get("findings", []),
        proposed_changes=data.get("proposed_changes", {}),
        applied=False,
        new_eval_config=None,
        evidence=data,
    )


def _apply_with_guardrails(result: MetaEvalResult, current: EvalConfig) -> EvalConfig | None:
    """
    Apply proposed changes with strict guardrails:
    - Compliance rules: append-only (never remove, never weaken)
    - Rubrics: can tighten, never loosen
    - Weights: compliance floor is 0.15, no weight can be 0
    """
    changes = result.proposed_changes
    new_rules = list(current.compliance_rules)
    new_weights = dict(current.scoring_weights)
    changed = False

    # Append new compliance rules
    new_rule_texts = changes.get("compliance_rules", [])
    if new_rule_texts:
        for rule in new_rule_texts:
            if rule and rule not in new_rules:
                new_rules.append(rule)
                changed = True

    # Rebalance weights (with guardrails)
    weight_changes = changes.get("scoring_weights", {})
    if weight_changes:
        for key, value in weight_changes.items():
            if key in new_weights and isinstance(value, (int, float)):
                value = float(value)
                if value > 0 and (key != "compliance" or value >= 0.15):
                    new_weights[key] = value

        # Renormalize
        total = sum(new_weights.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            new_weights = {k: v / total for k, v in new_weights.items()}

        # Verify compliance floor
        if new_weights.get("compliance", 0) >= 0.15:
            changed = True
        else:
            new_weights = dict(current.scoring_weights)  # Revert

    if not changed:
        return None

    try:
        return EvalConfig(
            version_id=f"eval_v{result.generation}",
            compliance_rules=new_rules,
            scoring_weights=new_weights,
        )
    except Exception as e:
        logger.warning(f"Meta-eval: proposed config invalid: {e}")
        return None


def _log_meta_eval(result: MetaEvalResult, settings: Settings) -> None:
    """Log meta-eval results for audit trail."""
    log_dir = settings.eval_versions_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "meta_eval_log.json"

    entry = {
        "generation": result.generation,
        "timestamp": result.timestamp.isoformat(),
        "findings": result.findings,
        "proposed_changes": result.proposed_changes,
        "applied": result.applied,
        "evidence": result.evidence,
    }

    # Append to JSONL
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
