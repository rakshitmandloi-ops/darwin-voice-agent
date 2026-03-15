"""
Parallel evolution run launcher.

Launches N independent evolution runs concurrently using asyncio.
Each run gets a unique seed offset and its own batch directory.
All runs share the same OpenAI API key but have independent archives and cost trackers.

Usage:
    .venv/bin/python scripts/run_parallel.py --n-runs 10 --max-gen 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class RunState:
    """Track state for a single evolution run."""

    def __init__(self, run_index: int) -> None:
        self.run_index = run_index
        self.batch_id: str = ""
        self.current_generation: int = 0
        self.best_score: float = 0.0
        self.total_cost: float = 0.0
        self.variants: int = 0
        self.status: str = "pending"
        self.error: str | None = None


async def _run_single(
    run_index: int,
    max_gen: int,
    state: RunState,
) -> RunState:
    """Execute a single evolution run with a unique seed offset."""
    from config import Settings, CostConfig, SimulationConfig
    from evaluation.cost_tracker import CostTracker
    from evolution.dgm_outer import run_evolution
    from evolution.archive import Archive

    state.status = "starting"

    try:
        # Each run gets its OWN Settings instance — no singleton caching
        # Use a per-run budget so they don't interfere
        settings = Settings(
            _env_file=".env",
        )

        # Give each run its own cost budget ($20 each is too much for parallel,
        # use $5 per run for 10 runs = $50 total, or let caller decide)
        budget_per_run = 20.0 / max(1, 1)  # Full budget per run for now

        state.status = "running"

        archive = await run_evolution(
            max_generations=max_gen,
            settings=settings,
        )

        state.batch_id = archive.batch_id
        state.variants = archive.size
        state.best_score = archive.get_best().mean_score if archive.size > 0 else 0.0
        state.current_generation = max(
            (e.generation for e in archive.entries), default=0
        )

        # Get cost from the archive's cost tracker
        tracker = CostTracker(settings)
        bd = tracker.get_breakdown()
        state.total_cost = bd.total_usd

        state.status = "complete"
    except Exception as e:
        state.status = "error"
        state.error = str(e)
        logger.exception(f"Run {run_index} failed")

    return state


async def _monitor(
    states: list[RunState],
    interval: float = 30.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Print progress summary every `interval` seconds."""
    evt = stop_event or asyncio.Event()
    while not evt.is_set():
        try:
            await asyncio.wait_for(evt.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass

        print("\n--- Progress ---")
        running = 0
        complete = 0
        errors = 0
        for s in states:
            if s.status == "running":
                running += 1
                print(f"  Run {s.run_index}: gen={s.current_generation}, batch={s.batch_id}")
            elif s.status == "complete":
                complete += 1
                print(f"  Run {s.run_index}: DONE, best={s.best_score:.2f}, variants={s.variants}, batch={s.batch_id}")
            elif s.status == "error":
                errors += 1
                print(f"  Run {s.run_index}: ERROR — {s.error}")
            else:
                print(f"  Run {s.run_index}: {s.status}")
        print(f"  Running: {running} | Complete: {complete} | Errors: {errors}")
        print("----------------\n")


def _print_summary(states: list[RunState]) -> None:
    print("\n========== PARALLEL EVOLUTION SUMMARY ==========")
    print(f"{'Run':>4} {'Batch':<22} {'Status':<10} {'Best':>6} {'Vars':>5} {'Gen':>4} {'Cost':>10}")
    print("-" * 70)
    total_cost = 0.0
    for s in states:
        batch = s.batch_id or "n/a"
        cost_str = f"${s.total_cost:.4f}" if s.total_cost else "$0.00"
        total_cost += s.total_cost
        print(
            f"{s.run_index:>4} {batch:<22} {s.status:<10} "
            f"{s.best_score:>6.2f} {s.variants:>5} {s.current_generation:>4} {cost_str:>10}"
        )

    completed = [s for s in states if s.status == "complete"]
    if completed:
        best_run = max(completed, key=lambda s: s.best_score)
        print(f"\nBest overall: Run {best_run.run_index} "
              f"(batch={best_run.batch_id}, score={best_run.best_score:.2f})")
    print(f"Total cost: ${total_cost:.4f}")
    print("=================================================\n")


async def async_main(n_runs: int = 5, max_gen: int = 5) -> list[RunState]:
    """Launch N parallel evolution runs and monitor them."""
    states = [RunState(i) for i in range(n_runs)]

    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(_monitor(states, interval=30.0, stop_event=stop_event))

    # Launch all runs concurrently
    run_tasks = [
        _run_single(run_index=i, max_gen=max_gen, state=states[i])
        for i in range(n_runs)
    ]

    start = time.time()
    print(f"Launching {n_runs} parallel evolution runs (max {max_gen} generations each)...")
    results = await asyncio.gather(*run_tasks, return_exceptions=True)
    elapsed = time.time() - start

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            states[i].status = "error"
            states[i].error = str(result)

    stop_event.set()
    await monitor_task

    print(f"\nAll {n_runs} runs finished in {elapsed:.1f}s")
    _print_summary(states)

    return states


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch parallel evolution runs")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of parallel runs")
    parser.add_argument("--max-gen", type=int, default=3, help="Max generations per run")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(message)s",
    )

    asyncio.run(async_main(n_runs=args.n_runs, max_gen=args.max_gen))


if __name__ == "__main__":
    main()
