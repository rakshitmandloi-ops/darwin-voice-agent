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
    worst_conversations: list[dict] | None = None,
    settings: Settings | None = None,
) -> MutationResult:
    """
    Propose a mutation to the parent's strategy.

    The rewriter sees:
    1. Aggregate trajectory analysis
    2. Current strategy (structured parameters)
    3. Worst conversation transcripts (actual messages that caused failures)
    4. Recent mutations (to avoid repeating)

    Returns a MutationResult with the proposed changes.
    """
    s = settings or get_settings()

    prompt = _build_rewriter_prompt(
        parent=parent,
        trajectory=trajectory,
        recent_mutations=recent_mutations or [],
        worst_conversations=worst_conversations or [],
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
You are an expert AI agent strategist improving debt collection agents.

You will receive:
1. Performance data (which criteria pass/fail per persona per agent)
2. The current STRATEGY (structured parameters controlling agent behavior)

Your job: diagnose failures and propose TARGETED strategy mutations.

WHAT YOU CAN CHANGE:
- Goal priorities (reorder what the agent tackles first)
- Turn allocation per goal (give more turns to struggling goals)
- Goal instructions (tighten wording for failing criteria)
- Persona tactics (change approach for specific borrower types)
- Summarizer field priorities (reorder what gets kept vs dropped)
- Opening lines
- Rules (add specificity, never remove)

OUTPUT FORMAT: JSON with these fields:
{
  "strategy_changes": {
    "agent1": {
      "goals": [{"name": "goal_name", "priority": 2, "max_turns": 2, "instruction": "new text"}],
      "persona_tactics": [{"persona_type": "combative", "approach": "new", "special_instructions": "new"}],
      "rules": ["new rule to ADD"],
      "opening_line": "new opening"
    },
    "agent2": { ... },
    "agent3": { ... },
    "summarizer": {
      "fields": [{"name": "field_name", "priority": 1, "max_tokens": 40}]
    }
  },
  "rationale": "Why these changes",
  "failures_addressed": ["specific failure 1", ...]
}

RULES:
- Only include agents/fields you want to CHANGE (omit unchanged ones)
- For goals: include the full goal object with the change, matched by name
- COMPLIANCE VIOLATIONS ARE TOP PRIORITY — fix those FIRST
- Token budgets: agent1 ≤ 2000 total, agent2/agent3 ≤ 1500 total
- Each agent has 4-6 turns MAX. Prioritize goals that matter most.
- If a persona type consistently fails: adjust that persona's tactic
- If summarizer drops critical info: increase that field's priority"""


def _build_rewriter_prompt(
    parent: ArchiveEntry,
    trajectory: TrajectoryAnalysis,
    recent_mutations: list[str],
    worst_conversations: list[dict],
    settings: Settings,
) -> str:
    """Build the full prompt for the rewriter."""
    ac = parent.variant_config.agent_config
    import json as _json

    parts = [format_trajectory_for_rewriter(trajectory)]

    # Show current strategy if available
    if ac.strategy_json:
        try:
            from agents.strategy import PipelineStrategy
            strategy = PipelineStrategy.model_validate_json(ac.strategy_json)
            parts.append("\n## CURRENT STRATEGY\n")
            for name, agent in [("agent1", strategy.agent1), ("agent2", strategy.agent2), ("agent3", strategy.agent3)]:
                parts.append(f"### {name} ({agent.agent_name})")
                parts.append(f"  Tone: {agent.tone}")
                parts.append(f"  Goals (priority order):")
                for g in sorted(agent.goals, key=lambda g: g.priority):
                    parts.append(f"    {g.priority}. {g.name}: max_turns={g.max_turns}, instruction=\"{g.instruction[:80]}\"")
                parts.append(f"  Persona tactics:")
                for t in agent.persona_tactics:
                    parts.append(f"    {t.persona_type}: {t.approach} — {t.special_instructions[:60]}")
                parts.append("")
            parts.append("### Summarizer fields (priority order):")
            for f in sorted(strategy.summarizer.fields, key=lambda f: f.priority):
                parts.append(f"    {f.priority}. {f.name}: max_tokens={f.max_tokens}")
        except Exception:
            parts.append("\n## CURRENT PROMPTS (no structured strategy available)\n")
            parts.append(ac.agent1_prompt[:500])
    else:
        parts.append("\n## CURRENT PROMPTS\n")
        parts.append(f"Agent 1 [{count_tokens(ac.agent1_prompt)} tok]: {ac.agent1_prompt[:300]}")
        parts.append(f"Agent 2 [{count_tokens(ac.agent2_prompt)} tok]: {ac.agent2_prompt[:300]}")
        parts.append(f"Agent 3 [{count_tokens(ac.agent3_prompt)} tok]: {ac.agent3_prompt[:300]}")

    # Include worst conversation transcripts so rewriter can see WHAT went wrong
    if worst_conversations:
        parts.append("\n## WORST CONVERSATIONS (actual transcripts showing failures)")
        for i, conv in enumerate(worst_conversations[:3]):  # Max 3 to stay within context
            persona = conv.get("persona_type", "?")
            score = conv.get("total", 0)
            parts.append(f"\n### Conversation {i+1}: {persona} (score: {score:.2f})")

            # Show failing criteria for this conversation
            failing = conv.get("failing_criteria", [])
            if failing:
                parts.append(f"  FAILING CRITERIA: {', '.join(failing)}")

            # Show each agent's messages (truncated)
            for stage, label in [("agent1", "Agent 1 (Assessment)"), ("agent2", "Agent 2 (Resolution)"), ("agent3", "Agent 3 (Final Notice)")]:
                msgs = conv.get(stage, [])
                if not msgs:
                    continue
                parts.append(f"\n  --- {label} ---")
                for m in msgs[:8]:  # Max 8 messages per stage
                    role = "AGENT" if m.get("role") == "assistant" else "BORROWER"
                    content = m.get("content", "")[:200]  # Truncate long messages
                    parts.append(f"  [{role}] {content}")

            # Show handoff
            h1 = conv.get("handoff_1")
            if h1:
                parts.append(f"\n  --- Handoff 1 ({h1.get('token_count', '?')} tok) ---")
                parts.append(f"  {h1.get('text', '')[:200]}")

            h2 = conv.get("handoff_2")
            if h2:
                parts.append(f"\n  --- Handoff 2 ({h2.get('token_count', '?')} tok) ---")
                parts.append(f"  {h2.get('text', '')[:200]}")

    if recent_mutations:
        parts.append("\n## RECENT MUTATIONS (avoid repeating)")
        for m in recent_mutations[-3:]:
            parts.append(f"  - {m}")

    parts.append(
        "\n## TASK"
        "\nAnalyze the performance data AND the actual conversation transcripts above."
        "\nIdentify exactly what went wrong and propose targeted strategy mutations."
        "\nFocus on the WORST-PERFORMING criteria and personas."
    )

    return "\n".join(parts)


def _parse_mutation(
    text: str,
    current_config: AgentConfig,
    settings: Settings,
) -> MutationResult:
    """
    Parse the rewriter's strategy changes and apply them to produce new prompts.
    """
    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return _fallback_mutation()
        else:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                try:
                    data = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    return _fallback_mutation()
            else:
                return _fallback_mutation()

    rationale = data.get("rationale", "No rationale provided")
    failures = data.get("failures_addressed", [])
    strategy_changes = data.get("strategy_changes", {})

    if not strategy_changes:
        # Fallback: check for old-format prompt changes
        changes = data.get("changes", {})
        if changes:
            components = [k for k in changes if k in MUTABLE_COMPONENTS]
            return MutationResult(
                components_modified=components,
                changes=changes,
                rationale=rationale,
                failures_addressed=failures if isinstance(failures, list) else [str(failures)],
                token_counts={k: count_tokens(v) for k, v in changes.items() if k in MUTABLE_COMPONENTS},
            )
        return _fallback_mutation()

    # Apply strategy changes to parent strategy → generate new prompts
    from agents.strategy import PipelineStrategy, strategy_to_prompt, summarizer_strategy_to_prompt

    try:
        if current_config.strategy_json:
            strategy = PipelineStrategy.model_validate_json(current_config.strategy_json)
        else:
            from agents.strategy import get_seed_strategy
            strategy = get_seed_strategy()
    except Exception:
        from agents.strategy import get_seed_strategy
        strategy = get_seed_strategy()

    # Apply mutations to strategy
    modified_components = []
    strategy_dict = strategy.model_dump()

    for agent_key in ["agent1", "agent2", "agent3"]:
        if agent_key in strategy_changes:
            changes_for_agent = strategy_changes[agent_key]
            _apply_agent_changes(strategy_dict[agent_key], changes_for_agent)
            modified_components.append(f"{agent_key}_prompt")

    if "summarizer" in strategy_changes:
        _apply_summarizer_changes(strategy_dict["summarizer"], strategy_changes["summarizer"])
        modified_components.append("summarizer_prompt")

    if not modified_components:
        return _fallback_mutation()

    # Rebuild strategy and generate prompts
    new_strategy = PipelineStrategy.model_validate(strategy_dict)
    new_prompts = {
        "agent1_prompt": strategy_to_prompt(new_strategy.agent1, is_first_agent=True),
        "agent2_prompt": strategy_to_prompt(new_strategy.agent2, is_first_agent=False),
        "agent3_prompt": strategy_to_prompt(new_strategy.agent3, is_first_agent=False),
        "summarizer_prompt": summarizer_strategy_to_prompt(new_strategy.summarizer),
        "strategy_json": new_strategy.model_dump_json(),
    }

    # Token budget check
    budget_map = {"agent1_prompt": 2000, "agent2_prompt": 1500, "agent3_prompt": 1500, "summarizer_prompt": 500}
    token_counts = {}
    for comp in modified_components:
        if comp in new_prompts:
            tokens = count_tokens(new_prompts[comp])
            token_counts[comp] = tokens
            if comp in budget_map and tokens > budget_map[comp]:
                return _fallback_mutation()

    return MutationResult(
        components_modified=modified_components,
        changes=new_prompts,
        rationale=rationale,
        failures_addressed=failures if isinstance(failures, list) else [str(failures)],
        token_counts=token_counts,
    )


def _apply_agent_changes(agent_dict: dict, changes: dict) -> None:
    """Apply changes to an agent strategy dict in place."""
    # Update goals by name
    if "goals" in changes:
        existing_goals = {g["name"]: g for g in agent_dict.get("goals", [])}
        for goal_change in changes["goals"]:
            name = goal_change.get("name", "")
            if name in existing_goals:
                existing_goals[name].update(goal_change)
            else:
                agent_dict.setdefault("goals", []).append(goal_change)
        agent_dict["goals"] = list(existing_goals.values())

    # Update persona tactics by type
    if "persona_tactics" in changes:
        existing_tactics = {t["persona_type"]: t for t in agent_dict.get("persona_tactics", [])}
        for tactic_change in changes["persona_tactics"]:
            pt = tactic_change.get("persona_type", "")
            if pt in existing_tactics:
                existing_tactics[pt].update(tactic_change)
            else:
                agent_dict.setdefault("persona_tactics", []).append(tactic_change)
        agent_dict["persona_tactics"] = list(existing_tactics.values())

    # Add rules (append only, never remove)
    if "rules" in changes:
        existing_rules = set(agent_dict.get("rules", []))
        for rule in changes["rules"]:
            if rule not in existing_rules:
                agent_dict.setdefault("rules", []).append(rule)

    # Update simple fields
    for key in ["opening_line", "tone", "role_description"]:
        if key in changes:
            agent_dict[key] = changes[key]


def _apply_summarizer_changes(summarizer_dict: dict, changes: dict) -> None:
    """Apply changes to summarizer strategy dict in place."""
    if "fields" in changes:
        existing_fields = {f["name"]: f for f in summarizer_dict.get("fields", [])}
        for field_change in changes["fields"]:
            name = field_change.get("name", "")
            if name in existing_fields:
                existing_fields[name].update(field_change)
        summarizer_dict["fields"] = list(existing_fields.values())

    if "compression_instruction" in changes:
        summarizer_dict["compression_instruction"] = changes["compression_instruction"]


def _fallback_mutation() -> MutationResult:
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
    """Apply mutation to produce new AgentConfig. Prompts from mutation.changes."""
    changes = mutation.changes
    return AgentConfig(
        version_id=new_version_id,
        agent1_prompt=changes.get("agent1_prompt", parent_config.agent1_prompt),
        agent2_prompt=changes.get("agent2_prompt", parent_config.agent2_prompt),
        agent3_prompt=changes.get("agent3_prompt", parent_config.agent3_prompt),
        summarizer_prompt=changes.get("summarizer_prompt", parent_config.summarizer_prompt),
        strategy_json=changes.get("strategy_json", parent_config.strategy_json),
    )
