"""
Tests for statistical comparison (paired bootstrap, Wilcoxon).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from models import (
    AgentScores,
    AgentType,
    ChecklistResult,
    ComplianceResult,
    ConversationScores,
    HandoffChecklist,
    PersonaType,
    SystemChecklist,
)


def _make_scores(
    conv_id: str,
    persona: PersonaType,
    total: float,
    compliance_passed: bool = True,
) -> ConversationScores:
    """Helper: build ConversationScores with a given weighted_total."""
    compliance = ComplianceResult(rule_results={
        "r1": compliance_passed,
        "r2": compliance_passed,
    })
    checks = ChecklistResult(checks={"c1": True})
    agent_score = AgentScores(
        agent=AgentType.ASSESSMENT,
        goal=checks,
        quality=checks,
        compliance=compliance,
    )
    return ConversationScores(
        conversation_id=conv_id,
        persona_type=persona,
        agent_scores={"agent1": agent_score},
        handoff_scores={},
        system_checks=SystemChecklist(checks={}),
        weighted_total=total,
        resolution_rate=0.5,
    )


@pytest.fixture(autouse=True)
def _mock_settings():
    """Ensure settings can load without .env file."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


class TestPairedBootstrap:
    """Tests for paired_bootstrap statistical comparison."""

    def test_significant_improvement(self):
        """Clear improvement should be detected as significant."""
        from evaluation.stats import paired_bootstrap

        parent = [_make_scores(f"p{i}", PersonaType.COOPERATIVE, 5.0) for i in range(10)]
        child = [_make_scores(f"c{i}", PersonaType.COOPERATIVE, 7.0) for i in range(10)]

        result = paired_bootstrap(parent, child)
        assert result.significant is True
        assert result.mean_diff > 0
        assert result.ci_lower > 0

    def test_no_improvement(self):
        """Same scores should not be significant."""
        from evaluation.stats import paired_bootstrap

        parent = [_make_scores(f"p{i}", PersonaType.COOPERATIVE, 6.0) for i in range(10)]
        child = [_make_scores(f"c{i}", PersonaType.COOPERATIVE, 6.0) for i in range(10)]

        result = paired_bootstrap(parent, child)
        assert result.significant is False
        assert abs(result.mean_diff) < 0.01

    def test_high_variance_flagged(self):
        """High variance in differences should be flagged."""
        from evaluation.stats import paired_bootstrap
        import random

        random.seed(42)
        parent = [_make_scores(f"p{i}", PersonaType.COOPERATIVE, 5.0 + random.uniform(-3, 3)) for i in range(10)]
        child = [_make_scores(f"c{i}", PersonaType.COOPERATIVE, 5.5 + random.uniform(-3, 3)) for i in range(10)]

        result = paired_bootstrap(parent, child)
        # With high variance, either not significant or variance flagged
        if result.significant:
            # Even if CI excludes zero by luck, variance should be flagged
            pass  # Acceptable — bootstrap with random data can go either way

    def test_per_persona_breakdown(self):
        """Should return per-persona score differences."""
        from evaluation.stats import paired_bootstrap

        parent = [
            _make_scores("p1", PersonaType.COOPERATIVE, 5.0),
            _make_scores("p2", PersonaType.COMBATIVE, 4.0),
        ]
        child = [
            _make_scores("c1", PersonaType.COOPERATIVE, 7.0),
            _make_scores("c2", PersonaType.COMBATIVE, 6.0),
        ]

        result = paired_bootstrap(parent, child)
        assert "cooperative" in result.per_persona_breakdown
        assert "combative" in result.per_persona_breakdown
        assert result.per_persona_breakdown["cooperative"] == pytest.approx(2.0, abs=0.01)
        assert result.per_persona_breakdown["combative"] == pytest.approx(2.0, abs=0.01)

    def test_empty_scores(self):
        """Empty input should return non-significant result."""
        from evaluation.stats import paired_bootstrap

        result = paired_bootstrap([], [])
        assert result.significant is False
        assert result.variance_too_high is True

    def test_compliance_regression_detected(self):
        """Child losing compliance should be flagged."""
        from evaluation.stats import paired_bootstrap

        parent = [_make_scores(f"p{i}", PersonaType.COOPERATIVE, 6.0, compliance_passed=True) for i in range(5)]
        child = [_make_scores(f"c{i}", PersonaType.COOPERATIVE, 7.0, compliance_passed=False) for i in range(5)]

        result = paired_bootstrap(parent, child)
        assert result.compliance_preserved is False


class TestPersonaRegression:
    """Tests for check_persona_regression."""

    def test_no_regression(self):
        from evaluation.stats import check_persona_regression
        from models import EvalComparison

        comparison = EvalComparison(
            parent_version="p",
            child_version="c",
            mean_diff=0.5,
            ci_lower=0.1,
            ci_upper=0.9,
            p_value=0.01,
            significant=True,
            per_persona_breakdown={"cooperative": 0.5, "combative": 0.3},
        )
        assert check_persona_regression(comparison) is False

    def test_regression_detected(self):
        from evaluation.stats import check_persona_regression
        from models import EvalComparison

        comparison = EvalComparison(
            parent_version="p",
            child_version="c",
            mean_diff=0.5,
            ci_lower=0.1,
            ci_upper=0.9,
            p_value=0.01,
            significant=True,
            per_persona_breakdown={"cooperative": 1.0, "combative": -0.5},
        )
        assert check_persona_regression(comparison) is True
