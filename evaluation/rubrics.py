"""
Evaluation rubrics — the measuring stick for agent quality.

These are FIXED during the evolution loop. Only the meta-eval cycle
(Phase 15) can modify them. This separation prevents the optimizer
from gaming its own evaluation.

Each rubric is versioned for audit trail.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Goal Completion Rubrics (per agent)
# ---------------------------------------------------------------------------

GOAL_RUBRIC_V0: dict[str, str] = {
    "agent1": """\
Rate 1-10 how well the Assessment Agent accomplished its goals:
- Did it identify itself as AI and disclose recording? (required)
- Did it verify borrower identity using partial account info only?
- Did it establish the debt (amount, status, days past due)?
- Did it gather financial situation (employment, income, obligations)?
- Did it determine a viable resolution path?
- Did it avoid making settlement offers (not its role)?
- Did it stay cold/clinical without being rude?
1=failed completely, 5=partial, 10=all goals achieved perfectly.""",

    "agent2": """\
Rate 1-10 how well the Resolution Agent accomplished its goals:
- Did it reference the prior assessment naturally (no re-introductions)?
- Did it present settlement options clearly (lump-sum, payment plan, hardship)?
- Were offers within policy ranges (60-80% lump-sum, 3-12 month plans)?
- Did it handle objections by restating terms, not comforting?
- Did it anchor on deadlines and push for commitment?
- Did it maintain a transactional, dealmaker tone?
- Did it avoid re-verifying identity or re-asking known information?
1=failed completely, 5=partial, 10=all goals achieved perfectly.""",

    "agent3": """\
Rate 1-10 how well the Final Notice Agent accomplished its goals:
- Did it reference the prior phone call outcome naturally?
- Did it state consequences clearly and factually?
- Are stated consequences documented next steps (not fabricated)?
- Did it make one final offer with a hard deadline?
- Did the offer match or improve what was discussed on the call?
- Did it maintain a consequence-driven, no-negotiation tone?
- Did it leave zero ambiguity about what happens next?
1=failed completely, 5=partial, 10=all goals achieved perfectly.""",
}

# ---------------------------------------------------------------------------
# Quality Rubric
# ---------------------------------------------------------------------------

QUALITY_RUBRIC_V0 = """\
Rate 1-10 the quality of this agent's conversation:
- Tone consistency: does the agent maintain its personality throughout?
  (Agent 1: cold/clinical, Agent 2: transactional, Agent 3: consequence-driven)
- No hallucination: does the agent only reference real information?
- No repetition: does the agent avoid repeating itself?
- Natural flow: does the conversation feel like a real interaction?
- Conciseness: does the agent avoid unnecessary filler?
- Professionalism: does the agent stay professional regardless of borrower behavior?
- Appropriate response to borrower's emotional state?
1=poor quality, 5=adequate, 10=excellent professional conversation."""

# ---------------------------------------------------------------------------
# Handoff Rubric
# ---------------------------------------------------------------------------

HANDOFF_RUBRIC_V0 = """\
Rate 1-10 the quality of this handoff summary:
- Does it preserve identity verification status?
- Does it capture the borrower's financial situation accurately?
- Does it note the borrower's emotional state?
- Does it record any offers made and borrower's response?
- Does it capture objections raised?
- Does it include a routing recommendation?
- Does it note stop-contact requests?
- Is it concise without dropping critical information?
- Does the NEXT agent actually USE this information (no re-asking)?
1=critical info lost, 5=adequate, 10=perfect information preservation."""

# ---------------------------------------------------------------------------
# System Rubric (cross-agent continuity)
# ---------------------------------------------------------------------------

SYSTEM_RUBRIC_V0 = """\
Rate 1-10 the end-to-end continuity of this 3-agent conversation:
- No re-introductions: does Agent 2/3 avoid re-introducing themselves?
- No re-verification: does Agent 2/3 avoid re-asking identity questions?
- No repeated questions: does any agent ask something already answered?
- Coherent continuation: does the conversation read as one continuous interaction?
- Tone progression: Assessment→Resolution→Final Notice feels like escalation?
- Information flows forward: details from early stages appear naturally in later ones?
- Borrower never feels the seam: would the borrower know they talked to different agents?
- Resolution references assessment findings, Final Notice references call outcomes?
1=feels like 3 separate conversations, 5=some continuity, 10=seamless single experience."""

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

# ---------------------------------------------------------------------------
# Version tracking
# ---------------------------------------------------------------------------

RUBRIC_VERSION = "v0"
