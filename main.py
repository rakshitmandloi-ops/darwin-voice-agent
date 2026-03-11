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
    """Rerun evaluation for reproducibility verification."""
    from config import get_settings
    from evaluation.cost_tracker import CostTracker
    from evolution.archive import Archive

    s = get_settings()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        run_config = json.load(f)

    print(f"Rerunning evaluation from {config_path}")
    print(f"Config: {json.dumps(run_config, indent=2)[:200]}...")
    print()
    print("(Full rerun implementation requires storing conversation seeds — coming in next step)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Darwin Voice Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # evolve
    evolve_parser = subparsers.add_parser("evolve", help="Run DGM evolution loop")
    evolve_parser.add_argument("--max-gen", type=int, default=None, help="Max generations")
    evolve_parser.add_argument("--budget", type=float, default=None, help="Budget limit")

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
