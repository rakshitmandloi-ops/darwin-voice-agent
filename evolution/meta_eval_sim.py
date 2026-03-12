"""
Sliding window meta-eval simulation.

Runs meta-eval across existing data in a sliding window:
- Size each window by the ACTUAL prompt token count
- Include EVERY conversation in that window in the prompt
- Slide: 50% old data + 50% new data each iteration
- At each window: meta-eval proposes rubric/weight changes
- Recalculate scores with new weights to show impact

This is a simulation — no new conversations are generated.
Only meta-eval gpt-4o calls are made (~$0.10 per window).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.prompts import count_tokens
from config import Settings, get_settings
from evaluation.cost_tracker import CostTracker
from models import CostCategory, EvalConfig

logger = logging.getLogger(__name__)

# 128K context, 70% = ~89K tokens for data
MAX_CONTEXT_TOKENS = 89_000
# Slide when 50% new data available
SLIDE_RATIO = 0.5


def _load_conversations_with_transcripts(
    archive_path: Path,
    transcripts_dir: Path,
) -> list[dict]:
    """Load all conversations with scores + transcripts, sorted by generation."""
    with open(archive_path) as f:
        archive = json.load(f)

    conversations = []
    for vid, entry in sorted(archive.items(), key=lambda x: x[1].get("generation", 0)):
        for score in entry.get("scores", []):
            conv_id = score.get("conversation_id", "")
            transcript_path = transcripts_dir / f"{conv_id}.json"

            transcript = None
            if transcript_path.exists():
                with open(transcript_path) as f:
                    transcript = json.load(f)

            conversations.append({
                "variant_id": vid,
                "generation": entry.get("generation", 0),
                "score": score,
                "transcript": transcript,
            })

    return conversations


def _build_window(
    conversations: list[dict],
    start_idx: int,
    current_weights: dict[str, float],
    window_number: int,
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> tuple[list[dict], int]:
    """Build the largest window whose fully formatted prompt fits in context."""
    window = []
    end_idx = start_idx

    for i in range(start_idx, len(conversations)):
        candidate = window + [conversations[i]]
        prompt = _format_window_for_meta_eval(candidate, current_weights, window_number)
        prompt_tokens = count_tokens(prompt)

        if prompt_tokens > max_tokens:
            if not window:
                logger.warning(
                    "Conversation %s alone exceeds window budget (%s tokens > %s). "
                    "Including it anyway so sliding window can make progress.",
                    conversations[i].get("score", {}).get("conversation_id", "?"),
                    prompt_tokens,
                    max_tokens,
                )
                return candidate, i + 1
            break

        window = candidate
        end_idx = i + 1

    return window, end_idx


def _collect_failing_checks(score: dict[str, Any]) -> list[str]:
    """Flatten failed checks for one conversation score."""
    failing = []

    for ak, asc in score.get("agent_scores", {}).items():
        for section in ["goal", "quality"]:
            sd = asc.get(section, {})
            checks = sd.get("checks", {}) if isinstance(sd, dict) else {}
            for k, v in checks.items():
                if not v:
                    failing.append(f"{ak}/{section}/{k}")

        comp = asc.get("compliance", {})
        comp_checks = comp.get("rule_results", {}) if isinstance(comp, dict) else {}
        for k, v in comp_checks.items():
            if not v:
                failing.append(f"{ak}/compliance/{k}")

    for hk, hv in score.get("handoff_scores", {}).items():
        checks = hv.get("checks", {}) if isinstance(hv, dict) else {}
        for k, v in checks.items():
            if not v:
                failing.append(f"{hk}/{k}")

    sys_data = score.get("system_checks", {})
    sys_checks = sys_data.get("checks", {}) if isinstance(sys_data, dict) else {}
    for k, v in sys_checks.items():
        if not v:
            failing.append(f"system/{k}")

    return failing


def _format_full_conversation(conv: dict[str, Any], index: int) -> str:
    """Format one full conversation for the meta-eval prompt."""
    score = conv["score"]
    parts = [
        f"### Conversation {index}",
        f"Variant: {conv['variant_id']}",
        f"Generation: {conv['generation']}",
        f"Conversation ID: {score.get('conversation_id', '')}",
        f"Persona: {score.get('persona_type', '?')}",
        f"Weighted total: {score.get('weighted_total', 0):.2f}",
        f"Resolution rate: {score.get('resolution_rate', 0)}",
        "Score JSON:",
        json.dumps(score, indent=2, default=str),
    ]

    failing = _collect_failing_checks(score)
    if failing:
        parts.append("Failing checks:")
        parts.append(", ".join(failing))

    transcript = conv.get("transcript")
    if transcript:
        parts.append("Full transcript:")
        for stage, label in [("agent1", "Agent 1"), ("agent2", "Agent 2"), ("agent3", "Agent 3")]:
            msgs = transcript.get(stage, [])
            if msgs:
                parts.append(f"--- {label} ---")
                for msg in msgs:
                    role = "AGENT" if msg.get("role") == "assistant" else "BORROWER"
                    parts.append(f"[{role}] {msg.get('content', '')}")

        for hk in ["handoff_1", "handoff_2"]:
            handoff = transcript.get(hk)
            if handoff:
                parts.append(f"--- {hk} ({handoff.get('token_count', '?')} tok) ---")
                parts.append(handoff.get("text", ""))
    else:
        parts.append("Transcript: MISSING")

    return "\n".join(parts)


def _format_window_for_meta_eval(
    window: list[dict],
    current_weights: dict[str, float],
    window_number: int,
) -> str:
    """Format a window of conversations as a prompt for meta-eval."""
    parts = []
    parts.append(f"SLIDING WINDOW META-EVAL — Window {window_number}")
    parts.append(f"Conversations in this window: {len(window)}")
    parts.append(f"Current scoring weights: {current_weights}")
    parts.append("Every conversation in this window is included below in full.")
    parts.append("")

    # Aggregate pass rates for this window
    from collections import defaultdict
    check_rates = defaultdict(lambda: {"pass": 0, "total": 0})

    for conv in window:
        s = conv["score"]
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

    parts.append("## PER-CHECK PASS RATES (this window)")
    for check, data in sorted(check_rates.items(), key=lambda x: x[1]["pass"] / max(x[1]["total"], 1)):
        rate = data["pass"] / data["total"] * 100 if data["total"] else 0
        flag = " ⚠️ FLOOR" if rate == 0 else " ⚠️ NEAR-FLOOR" if rate < 5 else " ⚠️ CEILING" if rate > 95 else ""
        if rate < 30 or rate > 95:
            parts.append(f"  {rate:5.1f}% {check} ({data['pass']}/{data['total']}){flag}")

    # Full window contents
    parts.append("")
    parts.append("## FULL CONVERSATIONS IN THIS WINDOW")
    for i, conv in enumerate(window, start=1):
        parts.append("")
        parts.append(_format_full_conversation(conv, i))

    return "\n".join(parts)


def _recalculate_scores(
    conversations: list[dict],
    new_weights: dict[str, float],
) -> list[dict]:
    """Recalculate weighted_total using new weights without re-running LLM scorers."""
    results = []
    for conv in conversations:
        s = conv["score"]
        agent_scores = s.get("agent_scores", {})

        # Goal: average goal pass_rate across agents
        goal_rates = []
        quality_rates = []
        for ak, asc in agent_scores.items():
            for section, rates_list in [("goal", goal_rates), ("quality", quality_rates)]:
                sd = asc.get(section, {})
                checks = sd.get("checks", {}) if isinstance(sd, dict) else {}
                if checks:
                    rates_list.append(sum(checks.values()) / len(checks))

        goal_score = 1 + (sum(goal_rates) / len(goal_rates) * 9 if goal_rates else 0)
        quality_score = 1 + (sum(quality_rates) / len(quality_rates) * 9 if quality_rates else 0)

        # Compliance
        compliance_passed = all(
            all(asc.get("compliance", {}).get("rule_results", {}).values())
            for asc in agent_scores.values()
        )
        compliance_score = 10.0 if compliance_passed else 1.0

        # Handoff
        handoff_rates = []
        for hk, hv in s.get("handoff_scores", {}).items():
            checks = hv.get("checks", {}) if isinstance(hv, dict) else {}
            if checks:
                handoff_rates.append(sum(checks.values()) / len(checks))
        handoff_score = 1 + (sum(handoff_rates) / len(handoff_rates) * 9 if handoff_rates else 0)

        # System
        sys_data = s.get("system_checks", {})
        sys_checks = sys_data.get("checks", {}) if isinstance(sys_data, dict) else {}
        sys_rate = sum(sys_checks.values()) / len(sys_checks) if sys_checks else 0
        system_score = 1 + sys_rate * 9

        new_total = (
            new_weights.get("goal", 0.3) * goal_score
            + new_weights.get("compliance", 0.2) * compliance_score
            + new_weights.get("quality", 0.2) * quality_score
            + new_weights.get("handoff", 0.15) * handoff_score
            + new_weights.get("system", 0.1) * system_score
        )

        results.append({
            "conversation_id": s.get("conversation_id", ""),
            "variant_id": conv["variant_id"],
            "persona_type": s.get("persona_type", "?"),
            "old_total": s.get("weighted_total", 0),
            "new_total": round(new_total, 2),
            "diff": round(new_total - s.get("weighted_total", 0), 2),
        })

    return results


async def run_sliding_meta_eval(
    archive_path: Path,
    transcripts_dir: Path,
    tracker: CostTracker,
    settings: Settings | None = None,
) -> list[dict]:
    """
    Run sliding window meta-eval simulation on existing data.

    Returns list of window results with proposed changes and score impacts.
    """
    s = settings or get_settings()

    logger.info("Loading conversations with transcripts...")
    conversations = _load_conversations_with_transcripts(archive_path, transcripts_dir)
    logger.info(f"Loaded {len(conversations)} conversations")

    current_weights = dict(EvalConfig().scoring_weights)
    results = []
    window_start = 0
    window_num = 0

    while window_start < len(conversations):
        window_num += 1
        window, window_end = _build_window(
            conversations,
            window_start,
            current_weights,
            window_num,
        )
        if not window:
            break

        logger.info(f"Window {window_num}: conversations {window_start}-{window_end} ({len(window)} convos)")

        # Build prompt with all conversations in the window
        prompt = _format_window_for_meta_eval(window, current_weights, window_num)
        prompt_tokens = count_tokens(prompt)
        logger.info(f"  Prompt: {prompt_tokens} tokens")

        # Call gpt-4o
        response = await tracker.tracked_completion(
            model=s.models.judge,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a CONSERVATIVE meta-evaluator. You see actual conversation transcripts "
                        "and their scores. Find OBVIOUS evaluation flaws — criteria that are clearly "
                        "miscalibrated based on what you see in the actual conversations.\n\n"
                        "For example: if 'concise' fails at 0% but the conversations ARE concise "
                        "(short messages, no filler), then the criterion is broken.\n\n"
                        "NEVER change compliance rules. Only adjust scoring weights.\n\n"
                        "Output JSON: {\"findings\": [...], \"proposed_changes\": {\"scoring_weights\": {}}, "
                        "\"confidence\": \"high/medium/low\", \"rationale\": \"...\"}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            category=CostCategory.META_EVAL,
            temperature=0.0,
        )

        response_text = response.choices[0].message.content or ""

        # Parse response
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', response_text, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}

        findings = data.get("findings", [])
        proposed_weights = data.get("proposed_changes", {}).get("scoring_weights", {})
        confidence = data.get("confidence", "low")
        rationale = data.get("rationale", "")

        # Calculate score impact if weights changed
        score_impact = None
        if proposed_weights and confidence == "high":
            # Normalize weights
            new_w = dict(current_weights)
            for k, v in proposed_weights.items():
                if k in new_w and isinstance(v, (int, float)) and v > 0:
                    new_w[k] = float(v)
            total = sum(new_w.values())
            if total > 0:
                new_w = {k: v/total for k, v in new_w.items()}

            if new_w.get("compliance", 0) >= 0.15:
                recalced = _recalculate_scores(conversations, new_w)
                old_mean = sum(r["old_total"] for r in recalced) / len(recalced)
                new_mean = sum(r["new_total"] for r in recalced) / len(recalced)

                score_impact = {
                    "old_weights": dict(current_weights),
                    "new_weights": {k: round(v, 4) for k, v in new_w.items()},
                    "old_mean_score": round(old_mean, 3),
                    "new_mean_score": round(new_mean, 3),
                    "score_diff": round(new_mean - old_mean, 3),
                    "per_variant": {},
                }

                # Per-variant impact
                from collections import defaultdict
                by_variant = defaultdict(lambda: {"old": [], "new": []})
                for r in recalced:
                    by_variant[r["variant_id"]]["old"].append(r["old_total"])
                    by_variant[r["variant_id"]]["new"].append(r["new_total"])

                for vid, scores in sorted(by_variant.items()):
                    old_avg = sum(scores["old"]) / len(scores["old"])
                    new_avg = sum(scores["new"]) / len(scores["new"])
                    if abs(new_avg - old_avg) > 0.05:
                        score_impact["per_variant"][vid] = {
                            "old": round(old_avg, 2),
                            "new": round(new_avg, 2),
                            "diff": round(new_avg - old_avg, 2),
                        }

                # Apply new weights for next window
                current_weights = new_w

        window_result = {
            "window": window_num,
            "convos_range": f"{window_start}-{window_end}",
            "num_convos": len(window),
            "prompt_tokens": prompt_tokens,
            "findings": findings,
            "proposed_weights": proposed_weights,
            "confidence": confidence,
            "rationale": rationale,
            "score_impact": score_impact,
            "applied": confidence == "high" and score_impact is not None,
        }
        results.append(window_result)

        logger.info(f"  Findings: {len(findings)}, Confidence: {confidence}, Applied: {window_result['applied']}")
        if score_impact:
            logger.info(f"  Score impact: {score_impact['old_mean_score']:.3f} → {score_impact['new_mean_score']:.3f} ({score_impact['score_diff']:+.3f})")

        # Slide window: 50% overlap
        slide_amount = max(1, len(window) // 2)
        window_start += slide_amount

    return results
