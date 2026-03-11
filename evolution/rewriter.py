"""
Trajectory-aware prompt rewriter.

Diagnoses systematic failures across ALL conversations, then proposes
targeted mutations to agent prompts and/or summarizer.

CRITICAL CONSTRAINT: Can ONLY modify prompts + summarizer.
CANNOT modify rubrics, weights, or compliance rules (those are
changed only by the meta-eval cycle).
"""

from __future__ import annotations

import json

from agents.prompts import count_tokens
from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from evolution.trajectory import TrajectoryAnalysis, format_trajectory_for_rewriter
from models import (
    AgentConfig,
    ArchiveEntry,
    CostCategory,
    MutationResult,
)

# Components the rewriter is ALLOWED to modify
MUTABLE_COMPONENTS = {"agent1_prompt", "agent2_prompt", "agent3_prompt", "summarizer_prompt"}


async def rewrite(
    parent: ArchiveEntry,
    trajectory: TrajectoryAnalysis,
    tracker: CostTracker,
    recent_mutations: list[str] | None = None,
    settings: Settings | None = None,
) -> MutationResult:
    """
    Propose a mutation to the parent's agent prompts/summarizer.

    The rewriter sees:
    1. Aggregate trajectory analysis (primary signal)
    2. Current prompts
    3. Token budget constraints
    4. Recent mutations (to avoid repeating)

    Returns a MutationResult with the proposed changes.
    """
    s = settings or get_settings()

    prompt = _build_rewriter_prompt(
        parent=parent,
        trajectory=trajectory,
        recent_mutations=recent_mutations or [],
        settings=s,
    )

    response = await tracker.tracked_completion(
        model=s.models.rewrite,
        messages=[
            {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        category=CostCategory.REWRITING,
        temperature=0.3,
        metadata={"parent_version": parent.version_id},
    )

    text = response.choices[0].message.content or ""
    result = _parse_mutation(text, parent.variant_config.agent_config, s)

    return result


REWRITER_SYSTEM_PROMPT = """\
You are an expert prompt engineer improving AI debt collection agents.

You will receive performance data showing how the current prompts perform
across different borrower scenarios. Your job is to diagnose systematic
failures and propose targeted prompt changes.

OUTPUT FORMAT: JSON object with these fields:
{
  "components_modified": ["agent1_prompt", "agent2_prompt", ...],
  "changes": {
    "agent1_prompt": "full new prompt text here",
    ...
  },
  "rationale": "Why these changes, which patterns they address",
  "failures_addressed": ["specific failure 1", "specific failure 2"]
}

RULES:
- You may modify: agent1_prompt, agent2_prompt, agent3_prompt, summarizer_prompt
- You may modify MULTIPLE components if the diagnosis warrants it
- You CANNOT modify: rubrics, scoring weights, compliance rules
- Token budgets: agent1 ≤ 2000 tokens, agent2/agent3 ≤ 1500 tokens
- Preserve ALL compliance behaviors (AI disclosure, recording disclosure, etc.)
- Small targeted changes are better than complete rewrites
- Address the SYSTEMATIC failures, not individual edge cases"""


def _build_rewriter_prompt(
    parent: ArchiveEntry,
    trajectory: TrajectoryAnalysis,
    recent_mutations: list[str],
    settings: Settings,
) -> str:
    """Build the full prompt for the rewriter."""
    ac = parent.variant_config.agent_config

    parts = [
        format_trajectory_for_rewriter(trajectory),
        "\n## CURRENT PROMPTS\n",
        f"### Agent 1 (Assessment) [{count_tokens(ac.agent1_prompt)} tokens / {settings.tokens.agent1_prompt} limit]",
        ac.agent1_prompt,
        f"\n### Agent 2 (Resolution) [{count_tokens(ac.agent2_prompt)} tokens / {settings.tokens.agent2_prompt} limit]",
        ac.agent2_prompt,
        f"\n### Agent 3 (Final Notice) [{count_tokens(ac.agent3_prompt)} tokens / {settings.tokens.agent3_prompt} limit]",
        ac.agent3_prompt,
        f"\n### Summarizer [{count_tokens(ac.summarizer_prompt)} tokens]",
        ac.summarizer_prompt,
    ]

    if recent_mutations:
        parts.append("\n## RECENT MUTATIONS (avoid repeating these)")
        for m in recent_mutations[-3:]:
            parts.append(f"  - {m}")

    parts.append(
        "\n## TASK"
        "\nAnalyze the performance data above. Identify the most impactful"
        " improvement(s) and output your proposed changes as JSON."
    )

    return "\n".join(parts)


def _parse_mutation(
    text: str,
    current_config: AgentConfig,
    settings: Settings,
) -> MutationResult:
    """
    Parse the rewriter's response into a validated MutationResult.

    Hard enforcement:
    - Reject any non-prompt components
    - Reject prompts that exceed token budgets
    """
    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            # Try to find JSON object in text
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(text[start:end+1])
            else:
                return _fallback_mutation(current_config)

    components = data.get("components_modified", [])
    changes = data.get("changes", {})
    rationale = data.get("rationale", "No rationale provided")
    failures = data.get("failures_addressed", [])

    # --- Hard enforcement: reject non-prompt components ---
    valid_components = []
    valid_changes = {}
    for comp in components:
        if comp in MUTABLE_COMPONENTS and comp in changes:
            valid_components.append(comp)
            valid_changes[comp] = changes[comp]

    # --- Token budget enforcement ---
    budget_map = {
        "agent1_prompt": settings.tokens.agent1_prompt,
        "agent2_prompt": settings.tokens.agent2_prompt,
        "agent3_prompt": settings.tokens.agent3_prompt,
        "summarizer_prompt": 500,  # Summarizer should be concise
    }

    token_counts = {}
    final_components = []
    final_changes = {}

    for comp, new_text in valid_changes.items():
        tokens = count_tokens(new_text)
        budget = budget_map.get(comp, 2000)
        token_counts[comp] = tokens

        if tokens <= budget:
            final_components.append(comp)
            final_changes[comp] = new_text
        # Silently drop over-budget components

    if not final_components:
        return _fallback_mutation(current_config)

    return MutationResult(
        components_modified=final_components,
        changes=final_changes,
        rationale=rationale,
        failures_addressed=failures if isinstance(failures, list) else [str(failures)],
        token_counts=token_counts,
    )


def _fallback_mutation(current_config: AgentConfig) -> MutationResult:
    """Return a no-op mutation if parsing fails."""
    return MutationResult(
        components_modified=[],
        changes={},
        rationale="Failed to parse rewriter output — no changes applied",
        failures_addressed=[],
        token_counts={},
    )


def apply_mutation(
    parent_config: AgentConfig,
    mutation: MutationResult,
    new_version_id: str,
) -> AgentConfig:
    """
    Apply a mutation to produce a new AgentConfig.

    Only changes components listed in the mutation. All other prompts
    carry forward from parent unchanged.
    """
    changes = mutation.changes
    return AgentConfig(
        version_id=new_version_id,
        agent1_prompt=changes.get("agent1_prompt", parent_config.agent1_prompt),
        agent2_prompt=changes.get("agent2_prompt", parent_config.agent2_prompt),
        agent3_prompt=changes.get("agent3_prompt", parent_config.agent3_prompt),
        summarizer_prompt=changes.get("summarizer_prompt", parent_config.summarizer_prompt),
    )
