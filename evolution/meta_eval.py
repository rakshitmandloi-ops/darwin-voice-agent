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

    # Only apply if confidence is high
    confidence = result.evidence.get("confidence", "low")
    if confidence != "high":
        logger.info(f"Meta-eval: confidence={confidence}, skipping changes (only apply on high)")
        _log_meta_eval(result, eval_config, None, s)
        return eval_config

    # Apply guardrails
    if result.proposed_changes:
        new_config = _apply_with_guardrails(result, eval_config)
        if new_config:
            result.applied = True
            result.new_eval_config = new_config
            logger.info(f"Meta-eval: applied changes (confidence=high): {list(result.proposed_changes.keys())}")
            _log_meta_eval(result, eval_config, new_config, s)
            return new_config

    logger.info("Meta-eval: no changes applied")
    _log_meta_eval(result, eval_config, None, s)
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
                goal_scores.append(agent_score.goal_score)
                quality_scores.append(agent_score.quality_score)
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

    # Ceiling/floor detection — aggregate level
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

    # Per-check floor/ceiling detection — the real signal
    # A check that's ALWAYS 0% or ALWAYS 100% across all variants is miscalibrated
    from collections import defaultdict
    check_rates = defaultdict(lambda: {"pass": 0, "total": 0})

    def _get(obj, key, default=None):
        """Get attribute from Pydantic model or dict."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _get_checks(obj):
        """Extract checks dict from ChecklistResult, HandoffChecklist, SystemChecklist, or dict."""
        if isinstance(obj, dict):
            return obj.get("checks", obj.get("rule_results", {}))
        if hasattr(obj, "checks"):
            return obj.checks
        if hasattr(obj, "rule_results"):
            return obj.rule_results
        return {}

    for entry in archive.entries:
        for score in entry.scores:
            # Convert to dict to avoid Pydantic __getattr__ issues
            sd = score.model_dump() if hasattr(score, 'model_dump') else score

            for ak, asc in sd.get("agent_scores", {}).items():
                for section in ["goal", "quality"]:
                    section_data = asc.get(section, {})
                    checks = section_data.get("checks", {}) if isinstance(section_data, dict) else {}
                    for k, v in checks.items():
                        check_rates[f"{section}/{ak}/{k}"]["total"] += 1
                        if v:
                            check_rates[f"{section}/{ak}/{k}"]["pass"] += 1

            for hk, hv in sd.get("handoff_scores", {}).items():
                checks = hv.get("checks", {}) if isinstance(hv, dict) else {}
                for k, v in checks.items():
                    check_rates[f"handoff/{hk}/{k}"]["total"] += 1
                    if v:
                        check_rates[f"handoff/{hk}/{k}"]["pass"] += 1

            sys_data = sd.get("system_checks", {})
            sys_checks = sys_data.get("checks", {}) if isinstance(sys_data, dict) else {}
            for k, v in sys_checks.items():
                check_rates[f"system/{k}"]["total"] += 1
                if v:
                    check_rates[f"system/{k}"]["pass"] += 1

    evidence["per_check_issues"] = []
    for check, data in sorted(check_rates.items()):
        if data["total"] < 10:
            continue
        rate = data["pass"] / data["total"]
        if rate == 0:
            evidence["per_check_issues"].append(f"FLOOR (0% pass, {data['total']} samples): {check}")
        elif rate <= 0.05:
            evidence["per_check_issues"].append(f"NEAR-FLOOR ({rate:.0%} pass, {data['total']} samples): {check}")
        elif rate >= 0.98:
            evidence["per_check_issues"].append(f"CEILING ({rate:.0%} pass, {data['total']} samples): {check}")

    return evidence


async def _analyze_evaluation(
    evidence: dict[str, Any],
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings,
) -> str:
    """Ask the judge model to identify evaluation flaws."""
    per_check_issues = evidence.get("per_check_issues", [])
    per_check_str = "\n".join(f"  - {issue}" for issue in per_check_issues) if per_check_issues else "  None detected"

    prompt = f"""You are a CONSERVATIVE meta-evaluator for an AI debt collection system.

Your job: find OBVIOUS FLAWS in the evaluation methodology. Only propose changes when
there is clear evidence of a problem. Do NOT change things that are working.

RULE: Only fix things where the evidence is undeniable. A check at 0% across ALL variants
and ALL conversations is clearly broken. A check at 40% might just be hard — leave it alone.

EVIDENCE:
{json.dumps({k: v for k, v in evidence.items() if k != 'per_check_issues'}, indent=2, default=str)}

PER-CHECK FLOOR/CEILING ISSUES (most important signal):
{per_check_str}

CURRENT CONFIG:
- Weights: {eval_config.scoring_weights}
- Compliance rules: {len(eval_config.compliance_rules)}

WHAT TO LOOK FOR:
1. Checks at 0% across ALL variants = criterion text is WRONG. Rewrite it.
2. Checks at 100% across ALL variants = criterion is too easy. Not urgent.
3. Weight imbalance = only adjust if a category is structurally stuck.

WHAT YOU CAN CHANGE:
1. REWRITE criterion text for checks that are clearly miscalibrated (0% pass but agents are doing it right).
   Use "rubric_overrides" — key is the check name (e.g. "quality/agent1/concise"), value is the new criterion text.
2. Adjust scoring weights if needed.

WHAT NOT TO DO:
- NEVER change compliance rules. They are IMMUTABLE.
- Don't rewrite criteria that are working (30-70% pass range is healthy).
- Only rewrite criteria where 0% pass is clearly wrong based on evidence.

OUTPUT JSON:
{{
  "findings": ["list of clear, evidence-backed flaws"],
  "proposed_changes": {{
    "scoring_weights": {{}},
    "rubric_overrides": {{
      "quality/agent1/concise": "new criterion text that better captures what concise means",
      "system/coherent_continuation": "new criterion text"
    }}
  }},
  "confidence": "high/medium/low — only apply if high",
  "rationale": "specific evidence for each rewrite"
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
    - Compliance rules: NEVER CHANGED by meta-eval (hard rule)
    - Weights: compliance floor is 0.15, no weight can be 0
    """
    changes = result.proposed_changes
    new_rules = list(current.compliance_rules)  # Preserved as-is, never modified
    new_weights = dict(current.scoring_weights)
    changed = False

    # COMPLIANCE RULES ARE IMMUTABLE
    if changes.get("compliance_rules"):
        logger.info("Meta-eval: ignoring proposed compliance rule changes (immutable)")

    # Apply rubric overrides (the main power of meta-eval)
    new_overrides = dict(current.rubric_overrides) if hasattr(current, 'rubric_overrides') else {}
    rubric_changes = changes.get("rubric_overrides", {})
    if rubric_changes:
        for key, new_text in rubric_changes.items():
            if isinstance(new_text, str) and len(new_text) > 10:
                old_text = new_overrides.get(key, "(hardcoded default)")
                new_overrides[key] = new_text
                logger.info(f"Meta-eval: rubric override [{key}]: '{old_text[:50]}' → '{new_text[:50]}'")
                changed = True

    # Rebalance weights (with guardrails)
    weight_changes = changes.get("scoring_weights", {})
    if weight_changes:
        for key, value in weight_changes.items():
            if key in new_weights and isinstance(value, (int, float)):
                value = float(value)
                if value > 0 and (key != "compliance" or value >= 0.15):
                    new_weights[key] = value

        total = sum(new_weights.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            new_weights = {k: v / total for k, v in new_weights.items()}

        if new_weights.get("compliance", 0) >= 0.15:
            changed = True
        else:
            new_weights = dict(current.scoring_weights)

    if not changed:
        return None

    try:
        return EvalConfig(
            version_id=f"eval_v{result.generation}",
            compliance_rules=new_rules,
            rubric_overrides=new_overrides,
            scoring_weights=new_weights,
        )
    except Exception as e:
        logger.warning(f"Meta-eval: proposed config invalid: {e}")
        return None


def _log_meta_eval(
    result: MetaEvalResult,
    old_config: EvalConfig,
    new_config: EvalConfig | None,
    settings: Settings,
) -> None:
    """Log meta-eval results with exact before/after diff."""
    log_dir = settings.eval_versions_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "meta_eval_log.json"

    # Build before/after diff
    changes_applied = {}
    if new_config and result.applied:
        # Weight changes
        old_w = old_config.scoring_weights
        new_w = new_config.scoring_weights
        weight_diff = {}
        for k in set(list(old_w.keys()) + list(new_w.keys())):
            old_v = old_w.get(k, 0)
            new_v = new_w.get(k, 0)
            if abs(old_v - new_v) > 0.001:
                weight_diff[k] = {"before": round(old_v, 4), "after": round(new_v, 4), "diff": round(new_v - old_v, 4)}
        if weight_diff:
            changes_applied["scoring_weights"] = weight_diff

        # Rubric overrides
        old_overrides = old_config.rubric_overrides if hasattr(old_config, 'rubric_overrides') else {}
        new_overrides = new_config.rubric_overrides if hasattr(new_config, 'rubric_overrides') else {}
        rubric_diff = {}
        for k in set(list(old_overrides.keys()) + list(new_overrides.keys())):
            old_text = old_overrides.get(k, "(hardcoded default)")
            new_text = new_overrides.get(k, "(hardcoded default)")
            if old_text != new_text:
                rubric_diff[k] = {"before": old_text[:200], "after": new_text[:200]}
        if rubric_diff:
            changes_applied["rubric_overrides"] = rubric_diff

        # Version change
        if old_config.version_id != new_config.version_id:
            changes_applied["version"] = {"before": old_config.version_id, "after": new_config.version_id}

    entry = {
        "generation": result.generation,
        "timestamp": result.timestamp.isoformat(),
        "findings": result.findings,
        "proposed_changes": result.proposed_changes,
        "applied": result.applied,
        "confidence": result.evidence.get("confidence", "unknown"),
        "changes_applied": changes_applied,
        "per_check_issues": result.evidence.get("per_check_issues", []) if isinstance(result.evidence, dict) else [],
        "evidence": result.evidence,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
