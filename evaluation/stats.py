"""
Statistical testing for the evolution loop.

Paired bootstrap CI and Wilcoxon signed-rank test for comparing
parent vs child performance. Paired on (persona, seed) to control
for borrower behavior variance.

Key design: "8% improvement with 40% std dev is not an improvement."
We require BOTH statistical significance AND low variance.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from config import get_settings
from models import ConversationScores, EvalComparison, PersonaType


def paired_bootstrap(
    parent_scores: list[ConversationScores],
    child_scores: list[ConversationScores],
    n_bootstrap: int | None = None,
    confidence_level: float | None = None,
) -> EvalComparison:
    """
    Paired bootstrap confidence interval for score difference.

    Pairs are matched on conversation_id suffix or position.
    Returns EvalComparison with CI, significance, and per-persona breakdown.
    """
    s = get_settings()
    n_boot = n_bootstrap or s.evolution.bootstrap_n
    ci = confidence_level or s.evolution.confidence_level

    # Extract weighted totals
    parent_vals = np.array([s.weighted_total for s in parent_scores])
    child_vals = np.array([s.weighted_total for s in child_scores])

    # Handle mismatched lengths (use min)
    n = min(len(parent_vals), len(child_vals))
    parent_vals = parent_vals[:n]
    child_vals = child_vals[:n]

    if n == 0:
        return _empty_comparison("v_parent", "v_child")

    # Compute paired differences
    diffs = child_vals - parent_vals
    observed_mean = float(np.mean(diffs))
    observed_std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    # Bootstrap
    rng = np.random.default_rng(seed=42)  # Deterministic
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diffs[sample_idx])

    # Confidence interval
    alpha = 1 - ci
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # Significance: CI excludes zero
    significant = ci_lower > 0  # Only care about improvement (positive diff)

    # Variance check: std dev of differences > 40% of mean = unreliable
    variance_too_high = (observed_std > 0.4 * abs(observed_mean)) if observed_mean != 0 else (observed_std > 2.0)

    # Wilcoxon p-value as secondary check
    if n >= 5:
        try:
            _, p_value = scipy_stats.wilcoxon(diffs, alternative="greater")
            p_value = float(p_value)
        except ValueError:
            p_value = 1.0  # All differences are zero
    else:
        p_value = 1.0  # Not enough samples

    # Per-persona breakdown
    per_persona = _persona_breakdown(parent_scores, child_scores)

    # Check compliance preservation
    parent_compliance = all(s.compliance_passed for s in parent_scores)
    child_compliance = all(s.compliance_passed for s in child_scores)
    compliance_preserved = child_compliance or not parent_compliance  # Can't regress

    return EvalComparison(
        parent_version=parent_scores[0].conversation_id if parent_scores else "unknown",
        child_version=child_scores[0].conversation_id if child_scores else "unknown",
        mean_diff=round(observed_mean, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        p_value=round(p_value, 4),
        significant=significant,
        variance_too_high=variance_too_high,
        compliance_preserved=compliance_preserved,
        per_persona_breakdown=per_persona,
    )


def check_persona_regression(
    comparison: EvalComparison,
    threshold: float = -0.5,
) -> bool:
    """
    Check if any persona type regressed significantly.

    Returns True if regression detected (bad), False if OK.
    """
    for persona, diff in comparison.per_persona_breakdown.items():
        if diff < threshold:
            return True
    return False


def _persona_breakdown(
    parent_scores: list[ConversationScores],
    child_scores: list[ConversationScores],
) -> dict[str, float]:
    """Compute mean score difference per persona type."""
    # Group scores by persona type
    parent_by_persona: dict[str, list[float]] = {}
    child_by_persona: dict[str, list[float]] = {}

    for s in parent_scores:
        key = s.persona_type.value
        parent_by_persona.setdefault(key, []).append(s.weighted_total)

    for s in child_scores:
        key = s.persona_type.value
        child_by_persona.setdefault(key, []).append(s.weighted_total)

    # Compute mean diff per persona
    breakdown: dict[str, float] = {}
    all_personas = set(list(parent_by_persona.keys()) + list(child_by_persona.keys()))

    for persona in all_personas:
        parent_vals = parent_by_persona.get(persona, [])
        child_vals = child_by_persona.get(persona, [])
        if parent_vals and child_vals:
            parent_mean = sum(parent_vals) / len(parent_vals)
            child_mean = sum(child_vals) / len(child_vals)
            breakdown[persona] = round(child_mean - parent_mean, 4)

    return breakdown


def _empty_comparison(parent_version: str, child_version: str) -> EvalComparison:
    return EvalComparison(
        parent_version=parent_version,
        child_version=child_version,
        mean_diff=0.0,
        ci_lower=0.0,
        ci_upper=0.0,
        p_value=1.0,
        significant=False,
        variance_too_high=True,
    )
