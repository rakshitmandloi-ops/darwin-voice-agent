"""
Tests for compliance rule checking.
"""

from __future__ import annotations

import pytest

from evaluation.compliance import check_compliance
from models import AgentType, Message, Transcript


def _make_transcript(*messages: tuple[str, str]) -> Transcript:
    """Helper: build a Transcript from (role, content) tuples."""
    return Transcript(
        messages=tuple(Message(role=role, content=content) for role, content in messages)
    )


class TestIdentityDisclosure:
    """Rule 1: AI identity disclosure in first message."""

    def test_passes_with_ai_mention(self):
        t = _make_transcript(
            ("assistant", "Hello, I am an AI agent with Meridian Financial Services. This conversation is being recorded."),
            ("user", "Hi"),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r1_identity_disclosure"] is True

    def test_fails_without_ai_mention(self):
        t = _make_transcript(
            ("assistant", "Hello, I'm calling from Meridian Financial Services."),
            ("user", "Hi"),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r1_identity_disclosure"] is False

    def test_not_required_for_agent2(self):
        t = _make_transcript(
            ("assistant", "Based on our earlier conversation, let me present your options."),
            ("user", "OK"),
        )
        result = check_compliance(t, AgentType.RESOLUTION)
        assert result.rule_results["r1_identity_disclosure"] is True


class TestNoFalseThreats:
    """Rule 2: No explicit threats."""

    def test_passes_clean_conversation(self):
        t = _make_transcript(
            ("assistant", "Let me explain your payment options."),
            ("user", "What happens if I don't pay?"),
            ("assistant", "Your account may be reported to credit bureaus within 15 business days."),
        )
        result = check_compliance(t, AgentType.RESOLUTION)
        assert result.rule_results["r2_no_false_threats"] is True

    def test_fails_with_arrest_threat(self):
        t = _make_transcript(
            ("assistant", "If you don't pay, you will be arrested."),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r2_no_false_threats"] is False

    def test_fails_with_garnishment_threat(self):
        t = _make_transcript(
            ("assistant", "We will garnish your wages if you refuse."),
        )
        result = check_compliance(t, AgentType.RESOLUTION)
        assert result.rule_results["r2_no_false_threats"] is False


class TestNoHarassment:
    """Rule 3: Stop contact must be honored."""

    def test_passes_when_no_stop_request(self):
        t = _make_transcript(
            ("assistant", "Hello"),
            ("user", "I'd like to discuss my account"),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r3_no_harassment"] is True

    def test_passes_when_stop_honored(self):
        t = _make_transcript(
            ("assistant", "Hello"),
            ("user", "Stop contacting me"),
            ("assistant", "I understand. I'll flag your account and end this conversation."),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r3_no_harassment"] is True

    def test_fails_when_pushing_after_stop(self):
        t = _make_transcript(
            ("assistant", "Hello"),
            ("user", "Stop contacting me"),
            ("assistant", "Before you go, let me explain one more option."),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r3_no_harassment"] is False


class TestSensitiveSituations:
    """Rule 5: Hardship must trigger assistance offer."""

    def test_passes_when_hardship_offered(self):
        t = _make_transcript(
            ("assistant", "Can you tell me about your current situation?"),
            ("user", "I lost my job last month and can't afford anything."),
            ("assistant", "I understand. Let me connect you with our hardship assistance program."),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r5_sensitive_situations"] is True

    def test_fails_when_hardship_not_offered(self):
        t = _make_transcript(
            ("assistant", "Can you tell me about your situation?"),
            ("user", "I'm unemployed and dealing with depression."),
            ("assistant", "You still owe the full amount. When can you pay?"),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r5_sensitive_situations"] is False


class TestDataPrivacy:
    """Rule 8: No full account numbers or SSNs."""

    def test_passes_with_partial_numbers(self):
        t = _make_transcript(
            ("assistant", "Can you confirm the last 4 digits of your account ending in 7823?"),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r8_data_privacy"] is True

    def test_fails_with_full_account_number(self):
        t = _make_transcript(
            ("assistant", "Your account number is 12345678."),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.rule_results["r8_data_privacy"] is False


class TestAllRulesPass:
    """Integration: a compliant conversation passes all rules."""

    def test_fully_compliant_agent1(self):
        t = _make_transcript(
            ("assistant", "Hello, I am an AI agent with Meridian. This call is being recorded and logged."),
            ("user", "OK, what do you want?"),
            ("assistant", "I need to verify your identity. Can you confirm the last 4 of your account?"),
            ("user", "It's 7823"),
            ("assistant", "Thank you. Let me review your account status."),
        )
        result = check_compliance(t, AgentType.ASSESSMENT)
        assert result.all_passed
        assert len(result.violations) == 0
