"""
Simulate meta-eval rubric rewrites on existing data.

1. Run meta-eval with full transcripts → get rubric rewrites
2. Re-score a sample of existing conversations with new rubrics
3. Compare old scores vs new scores
4. Generate report showing: what changed, why, score impact
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from agents.prompts import count_tokens
from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from models import CostCategory, EvalConfig

logger = logging.getLogger(__name__)


async def run_rubric_simulation(
    archive_path: Path,
    transcripts_dir: Path,
    tracker: CostTracker,
    settings: Settings | None = None,
    sample_size: int = 10,
) -> dict:
    """
    Full simulation:
    1. Gather evidence from archive
    2. Ask meta-eval to propose rubric rewrites
    3. Re-score sample conversations with old AND new rubrics
    4. Compare scores
    """
    s = settings or get_settings()

    # Load data
    with open(archive_path) as f:
        archive_data = json.load(f)

    logger.info(f"Loaded {len(archive_data)} variants")

    # Gather per-check issues + sample transcripts for meta-eval
    evidence = _gather_full_evidence(archive_data, transcripts_dir)

    # Step 1: Ask meta-eval to propose rubric rewrites
    logger.info("Asking meta-eval for rubric rewrites...")
    rewrite_response = await _ask_for_rewrites(evidence, tracker, s)

    proposed_overrides = rewrite_response.get("rubric_overrides", {})
    findings = rewrite_response.get("findings", [])
    rationale = rewrite_response.get("rationale", "")
    confidence = rewrite_response.get("confidence", "low")

    logger.info(f"Proposed {len(proposed_overrides)} rubric rewrites (confidence: {confidence})")
    for k, v in proposed_overrides.items():
        logger.info(f"  {k}: {v[:80]}")

    if not proposed_overrides:
        return {
            "status": "no_changes_proposed",
            "findings": findings,
            "rationale": rationale,
            "confidence": confidence,
        }

    # Step 2: Pick sample conversations to re-score
    sample = _pick_sample(archive_data, transcripts_dir, sample_size)
    logger.info(f"Re-scoring {len(sample)} conversations with old and new rubrics...")

    # Step 3: Score with OLD rubrics
    old_eval = EvalConfig(
        version_id="eval_old",
        scoring_weights={"goal": 0.30, "compliance": 0.25, "quality": 0.20, "handoff": 0.15, "system": 0.10},
    )

    old_scores = await _score_sample(sample, old_eval, tracker, s)

    # Step 4: Score with NEW rubrics (overrides applied)
    new_eval = EvalConfig(
        version_id="eval_new",
        rubric_overrides=proposed_overrides,
        scoring_weights={"goal": 0.30, "compliance": 0.25, "quality": 0.20, "handoff": 0.15, "system": 0.10},
    )

    new_scores = await _score_sample(sample, new_eval, tracker, s)

    # Step 5: Compare
    comparisons = []
    for i, (conv_data, old_s, new_s) in enumerate(zip(sample, old_scores, new_scores)):
        comp = {
            "conversation_id": conv_data["conversation_id"],
            "persona": conv_data["persona_type"],
            "old_total": old_s["weighted_total"],
            "new_total": new_s["weighted_total"],
            "diff": round(new_s["weighted_total"] - old_s["weighted_total"], 3),
            "old_checks": old_s,
            "new_checks": new_s,
            "check_diffs": [],
        }

        # Find checks that flipped
        for category in ["goal", "quality", "handoff", "system"]:
            old_checks = old_s.get(category, {})
            new_checks = new_s.get(category, {})
            for check_key in set(list(old_checks.keys()) + list(new_checks.keys())):
                old_val = old_checks.get(check_key)
                new_val = new_checks.get(check_key)
                if old_val != new_val:
                    comp["check_diffs"].append({
                        "check": f"{category}/{check_key}",
                        "old": old_val,
                        "new": new_val,
                        "flipped": "FAIL→PASS" if new_val and not old_val else "PASS→FAIL",
                    })

        comparisons.append(comp)

    # Build report
    old_mean = sum(c["old_total"] for c in comparisons) / len(comparisons)
    new_mean = sum(c["new_total"] for c in comparisons) / len(comparisons)
    total_flips = sum(len(c["check_diffs"]) for c in comparisons)
    fail_to_pass = sum(1 for c in comparisons for d in c["check_diffs"] if d["flipped"] == "FAIL→PASS")
    pass_to_fail = sum(1 for c in comparisons for d in c["check_diffs"] if d["flipped"] == "PASS→FAIL")

    report = {
        "status": "complete",
        "confidence": confidence,
        "findings": findings,
        "rationale": rationale,
        "rubric_rewrites": {
            k: {
                "old": _get_original_criterion(k),
                "new": v,
            }
            for k, v in proposed_overrides.items()
        },
        "score_impact": {
            "sample_size": len(comparisons),
            "old_mean": round(old_mean, 3),
            "new_mean": round(new_mean, 3),
            "diff": round(new_mean - old_mean, 3),
            "pct_change": round((new_mean - old_mean) / old_mean * 100, 1) if old_mean > 0 else 0,
        },
        "check_flips": {
            "total": total_flips,
            "fail_to_pass": fail_to_pass,
            "pass_to_fail": pass_to_fail,
        },
        "per_conversation": comparisons,
        "assessment": _assess_impact(old_mean, new_mean, fail_to_pass, pass_to_fail, proposed_overrides),
    }

    return report


def _gather_full_evidence(archive_data: dict, transcripts_dir: Path) -> dict:
    """Gather per-check rates + sample transcripts for meta-eval."""
    from collections import defaultdict
    check_rates = defaultdict(lambda: {"pass": 0, "total": 0})

    all_scores = []
    for vid, entry in archive_data.items():
        for s in entry.get("scores", []):
            all_scores.append(s)
            for ak, asc in s.get("agent_scores", {}).items():
                for section in ["goal", "quality"]:
                    sd = asc.get(section, {})
                    checks = sd.get("checks", {}) if isinstance(sd, dict) else {}
                    for k, v in checks.items():
                        check_rates[f"{section}/{ak}/{k}"]["total"] += 1
                        if v:
                            check_rates[f"{section}/{ak}/{k}"]["pass"] += 1
            for hk, hv in s.get("handoff_scores", {}).items():
                checks = hv.get("checks", {}) if isinstance(hv, dict) else {}
                for k, v in checks.items():
                    check_rates[f"handoff/{hk}/{k}"]["total"] += 1
                    if v:
                        check_rates[f"handoff/{hk}/{k}"]["pass"] += 1
            sys_data = s.get("system_checks", {})
            checks = sys_data.get("checks", {}) if isinstance(sys_data, dict) else {}
            for k, v in checks.items():
                check_rates[f"system/{k}"]["total"] += 1
                if v:
                    check_rates[f"system/{k}"]["pass"] += 1

    # Floor/ceiling issues
    issues = []
    for check, data in sorted(check_rates.items()):
        if data["total"] < 10:
            continue
        rate = data["pass"] / data["total"]
        if rate <= 0.05:
            issues.append(f"{'FLOOR' if rate == 0 else 'NEAR-FLOOR'} ({rate:.0%} pass, {data['total']} samples): {check}")

    # Sample transcripts — pick worst conversations
    sorted_scores = sorted(all_scores, key=lambda s: s.get("weighted_total", 0))
    worst = sorted_scores[:5]

    sample_transcripts = []
    for s in worst:
        conv_id = s.get("conversation_id", "")
        t_path = transcripts_dir / f"{conv_id}.json"
        if t_path.exists():
            with open(t_path) as f:
                t = json.load(f)
            sample_transcripts.append({
                "score": s,
                "transcript": t,
            })

    return {
        "per_check_issues": issues,
        "total_conversations": len(all_scores),
        "total_variants": len(archive_data),
        "sample_transcripts": sample_transcripts,
    }


async def _ask_for_rewrites(evidence: dict, tracker: CostTracker, settings: Settings) -> dict:
    """Ask GPT-4o to propose rubric rewrites based on evidence + transcripts."""
    issues_str = "\n".join(f"  - {i}" for i in evidence["per_check_issues"])

    # Include sample transcripts
    transcript_str = ""
    for i, st in enumerate(evidence.get("sample_transcripts", [])[:3]):
        s = st["score"]
        t = st["transcript"]
        transcript_str += f"\n### Sample {i+1}: {s.get('persona_type', '?')} (score: {s.get('weighted_total', 0):.2f})\n"

        # Show failing checks
        failing = []
        for ak, asc in s.get("agent_scores", {}).items():
            for section in ["goal", "quality"]:
                sd = asc.get(section, {})
                for k, v in (sd.get("checks", {}) if isinstance(sd, dict) else {}).items():
                    if not v:
                        failing.append(f"{section}/{ak}/{k}")
        transcript_str += f"FAILING: {', '.join(failing[:8])}\n"

        for stage in ["agent1", "agent2", "agent3"]:
            msgs = t.get(stage, [])
            if msgs:
                transcript_str += f"\n--- {stage} ---\n"
                for m in msgs[:4]:
                    role = "AGENT" if m.get("role") == "assistant" else "BORROWER"
                    transcript_str += f"[{role}] {m.get('content', '')[:250]}\n"

    # Get current rubric texts for reference
    from evaluation.rubrics import GOAL_CHECKS, QUALITY_CHECKS, SYSTEM_CHECKS, HANDOFF_CHECKS
    current_rubrics = {}
    for agent_key, checks in GOAL_CHECKS.items():
        for k, v in checks.items():
            current_rubrics[f"goal/{agent_key}/{k}"] = v
    for k, v in QUALITY_CHECKS.items():
        for agent_key in ["agent1", "agent2", "agent3"]:
            current_rubrics[f"quality/{agent_key}/{k}"] = v
    for k, v in SYSTEM_CHECKS.items():
        current_rubrics[f"system/{k}"] = v

    # Only include rubrics that are failing
    failing_rubrics = {}
    for issue in evidence["per_check_issues"]:
        # Extract check name from issue string
        parts = issue.split(": ", 1)
        if len(parts) == 2:
            check_name = parts[1].strip()
            if check_name in current_rubrics:
                failing_rubrics[check_name] = current_rubrics[check_name]

    response = await tracker.tracked_completion(
        model=settings.models.judge,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a meta-evaluator who fixes broken evaluation criteria. "
                    "You will see criteria that ALWAYS FAIL (0% pass rate) even though "
                    "the actual conversation transcripts show agents doing the right thing. "
                    "Your job: REWRITE the criterion text so the evaluator can correctly "
                    "assess it. Make criteria specific, observable, and fair.\n\n"
                    "Output JSON with rubric_overrides mapping check names to new text."
                ),
            },
            {
                "role": "user",
                "content": f"""FLOOR ISSUES (criteria that always fail):
{issues_str}

CURRENT CRITERION TEXT (the ones that need rewriting):
{json.dumps(failing_rubrics, indent=2)}

SAMPLE CONVERSATIONS (read these to understand what agents actually do):
{transcript_str}

TASK: For each FLOOR criterion, read the transcripts and decide:
- Is the agent ACTUALLY failing this? → Leave the criterion alone.
- Is the agent doing it right but the criterion is badly worded? → REWRITE it.

For rewrites, make the new text:
- Specific and observable (not vague)
- Fair to the agent given the 2000 token budget constraint (4-5 turns)
- Something the LLM evaluator can objectively assess

OUTPUT JSON:
{{
  "findings": ["what you found in the transcripts for each criterion"],
  "rubric_overrides": {{
    "quality/agent1/concise": "new criterion text",
    "system/coherent_continuation": "new criterion text"
  }},
  "confidence": "high/medium/low",
  "rationale": "evidence-based explanation"
}}""",
            },
        ],
        category=CostCategory.META_EVAL,
        temperature=0.0,
    )

    text = response.choices[0].message.content or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return {"findings": [], "rubric_overrides": {}, "confidence": "low"}


def _pick_sample(archive_data: dict, transcripts_dir: Path, n: int) -> list[dict]:
    """Pick diverse sample conversations for re-scoring."""
    all_convos = []
    for vid, entry in archive_data.items():
        for s in entry.get("scores", []):
            conv_id = s.get("conversation_id", "")
            t_path = transcripts_dir / f"{conv_id}.json"
            if t_path.exists():
                with open(t_path) as f:
                    t = json.load(f)
                all_convos.append({
                    "conversation_id": conv_id,
                    "variant_id": vid,
                    "persona_type": s.get("persona_type", "?"),
                    "old_score": s,
                    "transcript": t,
                })

    # Pick diverse: worst, best, middle, and one per persona
    all_convos.sort(key=lambda c: c["old_score"].get("weighted_total", 0))
    sample = []
    sample.append(all_convos[0])  # worst
    sample.append(all_convos[-1])  # best
    sample.append(all_convos[len(all_convos)//2])  # middle

    # One per persona
    seen_personas = set()
    for c in all_convos:
        pt = c["persona_type"]
        if pt not in seen_personas and len(sample) < n:
            seen_personas.add(pt)
            if c not in sample:
                sample.append(c)

    # Fill remaining
    for c in all_convos[::len(all_convos)//max(n, 1)]:
        if len(sample) >= n:
            break
        if c not in sample:
            sample.append(c)

    return sample[:n]


async def _score_sample(
    sample: list[dict],
    eval_config: EvalConfig,
    tracker: CostTracker,
    settings: Settings,
) -> list[dict]:
    """Score sample conversations and return per-check results."""
    from evaluation.scorers import score_conversation, _resolve_criteria
    from evaluation.rubrics import GOAL_CHECKS, QUALITY_CHECKS, HANDOFF_CHECKS, SYSTEM_CHECKS
    from simulation.simulator import simulate_pipeline
    from models import (
        AgentConfig, AgentType, Conversation, Persona, PersonaType,
        Transcript, Message, HandoffSummary, Outcome,
    )

    overrides = eval_config.rubric_overrides if hasattr(eval_config, 'rubric_overrides') else {}
    results = []

    for conv_data in sample:
        t = conv_data["transcript"]

        # Rebuild Conversation object from transcript
        def _make_transcript(msgs):
            return Transcript(messages=tuple(
                Message(role=m["role"], content=m["content"]) for m in msgs
            ))

        persona = Persona(
            name=t.get("persona", "unknown"),
            persona_type=PersonaType(t.get("persona_type", "cooperative")),
            system_prompt="", voice_system_prompt="", difficulty=0.5,
        )

        h1 = None
        if t.get("handoff_1"):
            h1 = HandoffSummary(
                text=t["handoff_1"]["text"],
                token_count=t["handoff_1"]["token_count"],
                source_agent=AgentType.ASSESSMENT,
                target_agent=AgentType.RESOLUTION,
            )
        h2 = None
        if t.get("handoff_2"):
            h2 = HandoffSummary(
                text=t["handoff_2"]["text"],
                token_count=t["handoff_2"]["token_count"],
                source_agent=AgentType.RESOLUTION,
                target_agent=AgentType.FINAL_NOTICE,
            )

        conv = Conversation(
            conversation_id=conv_data["conversation_id"],
            persona=persona,
            seed=t.get("seed", 0),
            agent1_transcript=_make_transcript(t.get("agent1", [])),
            agent2_transcript=_make_transcript(t.get("agent2", [])),
            agent3_transcript=_make_transcript(t.get("agent3", [])),
            handoff_1=h1,
            handoff_2=h2,
            outcome=Outcome(t.get("outcome", "no_deal")),
        )

        scores = await score_conversation(conv, eval_config, tracker, settings)

        # Extract all checks into flat dict
        result = {"weighted_total": scores.weighted_total}
        for ak, asc in scores.agent_scores.items():
            for k, v in asc.goal.checks.items():
                result.setdefault("goal", {})[f"{ak}/{k}"] = v
            for k, v in asc.quality.checks.items():
                result.setdefault("quality", {})[f"{ak}/{k}"] = v
        for hk, hv in scores.handoff_scores.items():
            for k, v in hv.checks.items():
                result.setdefault("handoff", {})[f"{hk}/{k}"] = v
        for k, v in scores.system_checks.checks.items():
            result.setdefault("system", {})[k] = v

        results.append(result)

    return results


def _get_original_criterion(check_name: str) -> str:
    """Get the hardcoded criterion text for a check."""
    from evaluation.rubrics import GOAL_CHECKS, QUALITY_CHECKS, SYSTEM_CHECKS, HANDOFF_CHECKS

    parts = check_name.split("/")
    if len(parts) == 3:
        cat, agent, key = parts
        if cat == "goal":
            return GOAL_CHECKS.get(agent, {}).get(key, check_name)
        elif cat == "quality":
            return QUALITY_CHECKS.get(key, check_name)
        elif cat == "handoff":
            return HANDOFF_CHECKS.get(key, check_name)
    elif len(parts) == 2:
        cat, key = parts
        if cat == "system":
            return SYSTEM_CHECKS.get(key, check_name)
        elif cat == "handoff":
            return HANDOFF_CHECKS.get(key, check_name)
    return check_name


def _assess_impact(old_mean, new_mean, fail_to_pass, pass_to_fail, overrides) -> str:
    """Generate human-readable assessment."""
    diff = new_mean - old_mean
    parts = []

    if diff > 0.3:
        parts.append(f"SIGNIFICANT IMPROVEMENT: Scores increased by {diff:.2f} ({diff/old_mean*100:.1f}%).")
    elif diff > 0.1:
        parts.append(f"Moderate improvement: Scores increased by {diff:.2f} ({diff/old_mean*100:.1f}%).")
    elif diff > -0.1:
        parts.append(f"Minimal change: Scores shifted by {diff:+.2f}.")
    else:
        parts.append(f"Score decreased by {abs(diff):.2f} — rubric rewrites may be too lenient.")

    if fail_to_pass > pass_to_fail:
        parts.append(f"{fail_to_pass} checks flipped FAIL→PASS vs {pass_to_fail} PASS→FAIL. Net: criteria became more accurate.")
    elif pass_to_fail > fail_to_pass:
        parts.append(f"WARNING: {pass_to_fail} checks flipped PASS→FAIL. Criteria may have become stricter.")

    parts.append(f"{len(overrides)} criteria were rewritten.")

    return " ".join(parts)
