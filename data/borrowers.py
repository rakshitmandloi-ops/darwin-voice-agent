"""
Mock borrower database and context injection for agent prompts.

Provides realistic borrower profiles so agents reference real account data
instead of hallucinating details. The `borrower_context_block()` function
generates a text block that gets prepended to the agent's system prompt.
"""

from __future__ import annotations

import random
from typing import Optional

from pydantic import BaseModel, Field


class BorrowerProfile(BaseModel, frozen=True):
    """A single borrower's account profile."""

    borrower_id: str
    name: str
    account_number: str = Field(min_length=8, max_length=8)
    balance_owed: float
    days_past_due: int
    account_status: str
    loan_type: str
    last_payment_date: str
    phone: str
    email: str

    @property
    def masked_account(self) -> str:
        """Return account number with only last 4 digits visible."""
        return f"****{self.account_number[-4:]}"


# ---------------------------------------------------------------------------
# Mock database
# ---------------------------------------------------------------------------

BORROWER_DB: dict[str, BorrowerProfile] = {
    "B001": BorrowerProfile(
        borrower_id="B001",
        name="John Smith",
        account_number="48217823",
        balance_owed=2500.00,
        days_past_due=45,
        account_status="delinquent",
        loan_type="personal loan",
        last_payment_date="2025-11-15",
        phone="(555) 234-8901",
        email="john.smith@email.com",
    ),
    "B002": BorrowerProfile(
        borrower_id="B002",
        name="Maria Garcia",
        account_number="73956104",
        balance_owed=8750.00,
        days_past_due=90,
        account_status="default",
        loan_type="auto loan",
        last_payment_date="2025-09-03",
        phone="(555) 678-2345",
        email="maria.garcia@email.com",
    ),
    "B003": BorrowerProfile(
        borrower_id="B003",
        name="David Chen",
        account_number="61048392",
        balance_owed=1200.00,
        days_past_due=30,
        account_status="delinquent",
        loan_type="credit card",
        last_payment_date="2025-12-20",
        phone="(555) 912-4567",
        email="david.chen@email.com",
    ),
    "B004": BorrowerProfile(
        borrower_id="B004",
        name="Sarah Johnson",
        account_number="29573816",
        balance_owed=15300.00,
        days_past_due=120,
        account_status="default",
        loan_type="personal loan",
        last_payment_date="2025-08-01",
        phone="(555) 345-6789",
        email="sarah.johnson@email.com",
    ),
    "B005": BorrowerProfile(
        borrower_id="B005",
        name="Michael Brown",
        account_number="84629175",
        balance_owed=4200.00,
        days_past_due=60,
        account_status="delinquent",
        loan_type="medical debt",
        last_payment_date="2025-10-10",
        phone="(555) 456-7890",
        email="michael.brown@email.com",
    ),
    "B006": BorrowerProfile(
        borrower_id="B006",
        name="Emily Davis",
        account_number="50318746",
        balance_owed=22000.00,
        days_past_due=150,
        account_status="default",
        loan_type="auto loan",
        last_payment_date="2025-07-22",
        phone="(555) 567-8901",
        email="emily.davis@email.com",
    ),
    "B007": BorrowerProfile(
        borrower_id="B007",
        name="Robert Wilson",
        account_number="17940263",
        balance_owed=650.00,
        days_past_due=15,
        account_status="delinquent",
        loan_type="credit card",
        last_payment_date="2026-01-05",
        phone="(555) 789-0123",
        email="robert.wilson@email.com",
    ),
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_borrower(borrower_id: str) -> Optional[BorrowerProfile]:
    """Look up a borrower by ID. Returns None if not found."""
    return BORROWER_DB.get(borrower_id)


def get_borrower_by_name(name: str) -> Optional[BorrowerProfile]:
    """
    Find a borrower by name (case-insensitive partial match).
    Returns the first match, or None.
    """
    name_lower = name.lower()
    for profile in BORROWER_DB.values():
        if name_lower in profile.name.lower():
            return profile
    return None


def list_borrowers() -> list[BorrowerProfile]:
    """Return all borrowers sorted by borrower_id."""
    return sorted(BORROWER_DB.values(), key=lambda b: b.borrower_id)


def get_random_borrower(seed: int | None = None) -> BorrowerProfile:
    """Pick a random borrower, optionally using a seed for reproducibility."""
    rng = random.Random(seed)
    borrower_id = rng.choice(list(BORROWER_DB.keys()))
    return BORROWER_DB[borrower_id]


# ---------------------------------------------------------------------------
# Context block for agent prompts
# ---------------------------------------------------------------------------


def borrower_context_block(profile: BorrowerProfile, agent_type: str = "agent1") -> str:
    """
    Generate a text block to prepend to the agent's system prompt.

    Agent-aware: Agent 1 gets verification instructions.
    Agent 2/3 get "identity already verified — do NOT re-ask."
    """
    base = (
        f"BORROWER FILE (internal — do NOT read this verbatim to the borrower):\n"
        f"- Name: {profile.name}\n"
        f"- Account: {profile.masked_account} (last 4 only — never reveal full number)\n"
        f"- Balance: ${profile.balance_owed:,.2f}\n"
        f"- Days Past Due: {profile.days_past_due}\n"
        f"- Status: {profile.account_status.title()}\n"
        f"- Loan Type: {profile.loan_type.title()}\n"
        f"- Last Payment: {profile.last_payment_date}\n"
    )

    if agent_type == "agent1":
        # Agent 1: verification instructions
        base += (
            f"\n"
            f"IDENTITY VERIFICATION INSTRUCTIONS:\n"
            f"- Say: \"I have an account on file ending in {profile.account_number[-4:]}. Can you confirm those last 4 digits?\"\n"
            f"- The correct answer is: {profile.account_number[-4:]}\n"
            f"- ALWAYS give the hint by stating the last 4 digits — the borrower needs to confirm, not guess.\n"
            f"- If they say the wrong digits, repeat: \"The account I have ends in {profile.account_number[-4:]}, does that match your records?\"\n"
            f"- Never reveal the full account number. Only reference the last 4 digits.\n"
        )
    else:
        # Agent 2/3: identity already verified — DO NOT re-ask
        base += (
            f"\n"
            f"IDENTITY STATUS: ALREADY VERIFIED by prior agent.\n"
            f"- Do NOT re-verify identity. Do NOT ask for account digits.\n"
            f"- The borrower's identity has been confirmed. Proceed directly with your role.\n"
            f"- Reference the borrower by name ({profile.name}) naturally.\n"
        )

    base += (
        f"\n"
        f"Use this data naturally throughout the conversation. Reference specific amounts and dates."
    )
    return base


def borrower_persona_context(profile: BorrowerProfile) -> str:
    """
    Generate a context block for the SIMULATED BORROWER so it knows
    its own account details and can respond correctly during verification.

    This is injected into the borrower persona's system prompt during simulation.
    """
    return (
        f"\nYOUR ACCOUNT DETAILS (use these when the agent asks):\n"
        f"- Your name is {profile.name}\n"
        f"- Your account number ends in {profile.account_number[-4:]}\n"
        f"- You owe ${profile.balance_owed:,.2f} on a {profile.loan_type}\n"
        f"- Your last payment was on {profile.last_payment_date}\n"
        f"- When asked to verify your identity, confirm the last 4 digits: {profile.account_number[-4:]}\n"
        f"- You know these details. Respond naturally when asked about them."
    )
