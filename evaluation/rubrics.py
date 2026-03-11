"""
Evaluation rubrics as clear binary checklists.

Every criterion is a yes/no question. No vague 1-10 scales.
The score is simply: pass_count / total_count × 9 + 1.

These are FIXED during evolution. Only meta-eval can change them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Goal Completion Checklists (per agent)
# ---------------------------------------------------------------------------

GOAL_CHECKS: dict[str, dict[str, str]] = {
    "agent1": {
        "ai_disclosure": "Agent identifies itself as AI in the first message",
        "recording_disclosure": "Agent discloses the conversation is being recorded/logged in the first message",
        "identity_verification": "Agent attempts to verify borrower identity using partial account info (last 4 digits only)",
        "debt_established": "Agent establishes the debt: mentions amount owed or account status or days past due",
        "financial_situation": "Agent asks about borrower's financial situation (employment, income, or obligations)",
        "resolution_path": "Agent determines or suggests a viable resolution path (lump-sum, payment plan, or hardship)",
        "no_settlement_offers": "Agent does NOT make specific settlement offers (that's Agent 2's role)",
        "cold_clinical_tone": "Agent maintains a cold, clinical, all-business tone throughout",
    },
    "agent2": {
        "references_prior": "Agent references information from the prior assessment naturally (not re-asking)",
        "no_re_verification": "Agent does NOT re-verify borrower identity or re-ask questions already answered",
        "presents_lump_sum": "Agent presents a lump-sum settlement option",
        "presents_payment_plan": "Agent presents a payment plan option",
        "offers_in_policy": "All settlement offers are within policy ranges (60-80% lump-sum, 3-12 month plans)",
        "handles_objections": "Agent handles objections by restating terms/policy, not by comforting",
        "states_deadline": "Agent mentions a deadline or expiration for the offer",
        "pushes_commitment": "Agent pushes for a verbal commitment or clear next step",
        "transactional_tone": "Agent maintains a transactional, dealmaker tone throughout",
    },
    "agent3": {
        "references_call": "Agent references the prior phone call outcome naturally",
        "no_re_introduction": "Agent does NOT re-introduce itself or start from scratch",
        "states_consequences": "Agent states specific consequences (credit reporting, legal review, collection activity)",
        "consequences_factual": "Stated consequences are factual/documented next steps, not fabricated threats",
        "final_offer": "Agent makes one final offer with a clear deadline",
        "offer_matches_prior": "The final offer matches or improves what was discussed on the call",
        "zero_ambiguity": "Agent leaves zero ambiguity about what happens next if borrower doesn't respond",
        "consequence_driven_tone": "Agent maintains a consequence-driven, no-negotiation tone throughout",
    },
}

# ---------------------------------------------------------------------------
# Quality Checklists (same for all agents, evaluated per-agent)
# ---------------------------------------------------------------------------

QUALITY_CHECKS: dict[str, str] = {
    "tone_consistent": "Agent maintains its designated personality throughout (no breaks in character)",
    "no_hallucination": "Agent only references real information from the conversation or handoff (no invented facts)",
    "no_repetition": "Agent does not repeat the same point or question more than once",
    "natural_flow": "Conversation flows naturally without awkward transitions or robotic phrasing",
    "concise": "Agent is concise — no unnecessary filler or overly long messages",
    "professional": "Agent stays professional regardless of borrower behavior",
    "appropriate_response": "Agent responds appropriately to borrower's emotional state",
}

# ---------------------------------------------------------------------------
# Handoff Checklists
# ---------------------------------------------------------------------------

HANDOFF_CHECKS: dict[str, str] = {
    "identity_preserved": "Summary preserves identity verification status (verified yes/no, method used)",
    "financial_preserved": "Summary captures borrower's financial situation accurately",
    "emotional_preserved": "Summary notes the borrower's emotional state",
    "offers_preserved": "Summary records any offers made and borrower's response",
    "objections_preserved": "Summary captures objections raised by the borrower",
    "routing_included": "Summary includes a routing recommendation",
    "stop_contact_noted": "Summary notes stop-contact request if borrower made one",
    "next_agent_uses_it": "The next agent actually USES the handoff info (doesn't re-ask known facts)",
}

# ---------------------------------------------------------------------------
# System Continuity Checklists
# ---------------------------------------------------------------------------

SYSTEM_CHECKS: dict[str, str] = {
    "no_re_introductions": "No agent re-introduces itself after Agent 1's initial introduction",
    "no_re_verification": "Agent 2 and 3 do not re-verify borrower identity",
    "no_repeated_questions": "No agent asks a question that was already answered in a prior stage",
    "coherent_continuation": "The conversation reads as one continuous interaction, not three separate ones",
    "tone_progression": "Tone progresses naturally: cold assessment → transactional negotiation → final notice",
    "info_flows_forward": "Details from early stages appear naturally in later ones",
    "borrower_no_seam": "A borrower would not know they talked to different agents",
}

# ---------------------------------------------------------------------------
# Default Scoring Weights
# ---------------------------------------------------------------------------

DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "goal": 0.30,
    "compliance": 0.20,
    "quality": 0.20,
    "handoff": 0.15,
    "system": 0.15,
}

RUBRIC_VERSION = "v0"
