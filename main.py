"""
CLI entry point for the Darwin Voice Agent system.

Commands:
  python main.py evolve [--max-gen N] [--budget B]
  python main.py status
  python main.py costs
  python main.py simulate --persona TYPE [--seed S]
  python main.py rerun-eval --config PATH
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def cmd_evolve(args: argparse.Namespace) -> None:
    """Run the DGM evolution loop."""
    from config import get_settings
    from evolution.dgm_outer import run_evolution

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    s = get_settings()
    archive = asyncio.run(
        run_evolution(
            max_generations=args.max_gen,
            resume_batch=args.resume,
            settings=s,
        )
    )

    best = archive.get_best()
    print(f"\nBest: {best.version_id} (score={best.mean_score:.2f})")
    print(f"Archive size: {archive.size}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show archive summary and cost."""
    import json
    from config import get_settings
    from evaluation.cost_tracker import CostTracker
    from evolution.archive import Archive, list_batches

    s = get_settings()

    # Read current archive directly without creating new batch
    from models import ArchiveEntry
    archive_file = s.logs_dir / "archive.json"
    entries: dict[str, ArchiveEntry] = {}
    if archive_file.exists():
        with open(archive_file) as f:
            data = json.load(f)
        for vid, entry_data in data.items():
            entries[vid] = ArchiveEntry.model_validate(entry_data)
    tracker = CostTracker(s)

    if not entries:
        print("Archive is empty. Run 'python main.py evolve' first.")
        return

    active = [e for e in entries.values() if not e.discarded]
    best = max(active, key=lambda e: e.mean_score) if active else None

    print(f"Archive: {len(entries)} variants")
    if best:
        print(f"Best: {best.version_id} (score={best.mean_score:.2f})")
    print()

    for entry in sorted(entries.values(), key=lambda e: e.generation):
        status = "PROMOTED" if entry.promoted else ("DISCARDED" if entry.discarded else "active")
        print(
            f"  {entry.version_id}: gen={entry.generation}, "
            f"score={entry.mean_score:.2f}, "
            f"convos={len(entry.scores)}, "
            f"parent={entry.parent_id or 'none'}, "
            f"{status}"
        )

    breakdown = tracker.get_breakdown()
    print(f"\nCost: ${breakdown.total_usd:.4f} / ${s.cost.budget_limit:.2f}")
    print(f"Remaining: ${breakdown.remaining_budget:.4f}")


def cmd_costs(args: argparse.Namespace) -> None:
    """Show cost breakdown by category."""
    from config import get_settings
    from evaluation.cost_tracker import CostTracker

    s = get_settings()
    tracker = CostTracker(s)
    breakdown = tracker.get_breakdown()

    print(f"Total: ${breakdown.total_usd:.4f}")
    print(f"Budget: ${s.cost.budget_limit:.2f}")
    print(f"Remaining: ${breakdown.remaining_budget:.4f}")
    print()
    print("By category:")
    for cat, cost in sorted(breakdown.by_category.items()):
        count = breakdown.call_counts.get(cat, 0)
        print(f"  {cat}: ${cost:.4f} ({count} calls)")
    print()
    print("By model:")
    for model, cost in sorted(breakdown.by_model.items()):
        print(f"  {model}: ${cost:.4f}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run a single full pipeline simulation."""
    from agents.prompts import get_seed_prompts
    from config import get_settings
    from evaluation.cost_tracker import CostTracker
    from evolution.archive import Archive
    from models import AgentConfig, PersonaType
    from simulation.personas import get_persona_with_variation
    from simulation.simulator import simulate_pipeline

    s = get_settings()
    tracker = CostTracker(s)

    # Get agent config from archive or seed
    archive = Archive(s)
    if archive.size > 0 and not args.seed_prompts:
        ac = archive.get_best().variant_config.agent_config
        print(f"Using best agent config: {ac.version_id}")
    else:
        prompts = get_seed_prompts()
        ac = AgentConfig(version_id="v0", **prompts)
        print("Using seed v0 prompts")

    persona_type = PersonaType(args.persona)
    persona = get_persona_with_variation(persona_type, seed=args.seed)

    print(f"Persona: {persona_type.value}, seed={args.seed}")
    print()

    conv = asyncio.run(
        simulate_pipeline(
            agent_config=ac,
            persona=persona,
            seed=args.seed,
            tracker=tracker,
            settings=s,
        )
    )

    print(f"Outcome: {conv.outcome.value}")
    print(f"Stop contact: {conv.stop_contact}")
    print()

    for label, transcript in [
        ("Agent 1 (Assessment)", conv.agent1_transcript),
        ("Agent 2 (Resolution)", conv.agent2_transcript),
        ("Agent 3 (Final Notice)", conv.agent3_transcript),
    ]:
        if transcript.messages:
            print(f"--- {label} ({transcript.turn_count} turns) ---")
            for msg in transcript.messages:
                prefix = "AGENT" if msg.role == "assistant" else "BORROWER"
                print(f"  [{prefix}] {msg.content[:200]}")
            print()

    if conv.handoff_1:
        print(f"--- Handoff 1 ({conv.handoff_1.token_count} tokens) ---")
        print(f"  {conv.handoff_1.text[:300]}")
        print()
    if conv.handoff_2:
        print(f"--- Handoff 2 ({conv.handoff_2.token_count} tokens) ---")
        print(f"  {conv.handoff_2.text[:300]}")


def cmd_rerun_eval(args: argparse.Namespace) -> None:
    """Rerun evaluation on stored transcripts to verify scores are reproducible."""
    from config import get_settings
    from evaluation.cost_tracker import CostTracker
    from evaluation.scorers import score_conversation
    from models import (
        AgentType, Conversation, EvalConfig, HandoffSummary,
        Message, Outcome, Persona, PersonaType, Transcript,
    )

    s = get_settings()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        run_config = json.load(f)

    print(f"Rerunning evaluation from {config_path}")
    print(f"Models: {run_config.get('models', {})}")
    print(f"Weights: {run_config.get('scoring_weights', {})}")
    print()

    # Find transcripts directory
    batch_id = run_config.get("batch_id", "")
    transcripts_dir = s.logs_dir / "runs" / batch_id / "transcripts"
    archive_path = s.logs_dir / "runs" / batch_id / "archive.json"

    if not transcripts_dir.exists():
        print(f"Transcripts not found at {transcripts_dir}")
        sys.exit(1)

    # Load archive for original scores
    with open(archive_path) as f:
        archive_data = json.load(f)

    # Build eval config from run_config
    eval_config = EvalConfig(
        scoring_weights=run_config.get("scoring_weights", {}),
        rubric_overrides=run_config.get("rubric_overrides", {}),
    )

    tracker = CostTracker(s)

    # Pick sample to re-score
    sample_convos = run_config.get("conversations", [])[:args.sample]
    print(f"Re-scoring {len(sample_convos)} conversations...")

    async def _rerun():
        diffs = []
        for conv_info in sample_convos:
            conv_id = conv_info.get("conversation_id", "")
            t_path = transcripts_dir / f"{conv_id}.json"
            if not t_path.exists():
                continue

            with open(t_path) as f:
                t = json.load(f)

            # Rebuild Conversation from transcript
            def _mk_t(msgs):
                return Transcript(messages=tuple(Message(role=m["role"], content=m["content"]) for m in msgs))

            persona = Persona(
                name=t.get("persona", "unknown"),
                persona_type=PersonaType(t.get("persona_type", "cooperative")),
                system_prompt="", voice_system_prompt="", difficulty=0.5,
            )
            h1 = HandoffSummary(text=t["handoff_1"]["text"], token_count=t["handoff_1"]["token_count"],
                                source_agent=AgentType.ASSESSMENT, target_agent=AgentType.RESOLUTION) if t.get("handoff_1") else None
            h2 = HandoffSummary(text=t["handoff_2"]["text"], token_count=t["handoff_2"]["token_count"],
                                source_agent=AgentType.RESOLUTION, target_agent=AgentType.FINAL_NOTICE) if t.get("handoff_2") else None

            conv = Conversation(
                conversation_id=conv_id, persona=persona, seed=t.get("seed", 0),
                agent1_transcript=_mk_t(t.get("agent1", [])),
                agent2_transcript=_mk_t(t.get("agent2", [])),
                agent3_transcript=_mk_t(t.get("agent3", [])),
                handoff_1=h1, handoff_2=h2,
                outcome=Outcome(t.get("outcome", "no_deal")),
            )

            new_scores = await score_conversation(conv, eval_config, tracker, s)

            # Find original score
            original_total = None
            for vid, entry in archive_data.items():
                for sc in entry.get("scores", []):
                    if sc.get("conversation_id") == conv_id:
                        original_total = sc["weighted_total"]
                        break

            diff = abs(new_scores.weighted_total - original_total) if original_total else None
            diffs.append({
                "conv_id": conv_id,
                "persona": conv_info.get("persona", "?"),
                "original": original_total,
                "rerun": new_scores.weighted_total,
                "diff": diff,
            })

            status = "✅" if diff and diff < 0.5 else "⚠️" if diff else "?"
            print(f"  {status} {conv_id}: original={original_total:.2f}, rerun={new_scores.weighted_total:.2f}, diff={diff:.3f}" if diff else f"  ? {conv_id}: no original score found")

        # Summary
        valid_diffs = [d["diff"] for d in diffs if d["diff"] is not None]
        if valid_diffs:
            avg_diff = sum(valid_diffs) / len(valid_diffs)
            max_diff = max(valid_diffs)
            within_tolerance = sum(1 for d in valid_diffs if d < 0.5)
            print(f"\nReproducibility: {within_tolerance}/{len(valid_diffs)} within ±0.5 tolerance")
            print(f"Average diff: {avg_diff:.3f}, Max diff: {max_diff:.3f}")
            if avg_diff < 0.5:
                print("✅ REPRODUCIBLE — scores match within tolerance")
            else:
                print("⚠️ SCORES DIVERGED — check model versions and temperatures")

    asyncio.run(_rerun())


def main() -> None:
    parser = argparse.ArgumentParser(description="Darwin Voice Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # evolve
    evolve_parser = subparsers.add_parser("evolve", help="Run DGM evolution loop")
    evolve_parser.add_argument("--max-gen", type=int, default=None, help="Max generations")
    evolve_parser.add_argument("--budget", type=float, default=None, help="Budget limit")
    evolve_parser.add_argument("--resume", type=str, default=None, help="Resume a previous batch by ID (e.g. 20260311_172220)")

    # status
    subparsers.add_parser("status", help="Show archive summary")

    # costs
    subparsers.add_parser("costs", help="Show cost breakdown")

    # simulate
    sim_parser = subparsers.add_parser("simulate", help="Run single simulation")
    sim_parser.add_argument("--persona", type=str, required=True, choices=[p.value for p in __import__("models").PersonaType])
    sim_parser.add_argument("--seed", type=int, default=42)
    sim_parser.add_argument("--seed-prompts", action="store_true", help="Use seed v0 prompts instead of best from archive")

    # rerun-eval
    rerun_parser = subparsers.add_parser("rerun-eval", help="Rerun evaluation for reproducibility")
    rerun_parser.add_argument("--config", type=str, required=True, help="Path to run_config.json")
    rerun_parser.add_argument("--sample", type=int, default=10, help="Number of conversations to re-score")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "evolve": cmd_evolve,
        "status": cmd_status,
        "costs": cmd_costs,
        "simulate": cmd_simulate,
        "rerun-eval": cmd_rerun_eval,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
