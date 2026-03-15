"""Enum types for the Darwin Voice Agent system."""

from __future__ import annotations

from enum import Enum


class AgentType(str, Enum):
    ASSESSMENT = "agent1"      # Chat — cold, clinical
    RESOLUTION = "agent2"      # Voice — transactional dealmaker
    FINAL_NOTICE = "agent3"    # Chat — consequence-driven closer


class Outcome(str, Enum):
    DEAL_AGREED = "deal_agreed"
    NO_DEAL = "no_deal"
    BORROWER_REFUSED = "borrower_refused"      # Explicit stop-contact
    HARDSHIP_REFERRAL = "hardship_referral"
    NO_RESPONSE = "no_response"
    IN_PROGRESS = "in_progress"


class CostCategory(str, Enum):
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    STRICT_GRADING = "strict_grading"
    REWRITING = "rewriting"
    SUMMARIZATION = "summarization"
    META_EVAL = "meta_eval"


class PersonaType(str, Enum):
    COOPERATIVE = "cooperative"
    COMBATIVE = "combative"
    EVASIVE = "evasive"
    CONFUSED = "confused"
    DISTRESSED = "distressed"
    MANIPULATIVE = "manipulative"
    LITIGIOUS = "litigious"
    PROMPT_INJECTION = "prompt_injection"
