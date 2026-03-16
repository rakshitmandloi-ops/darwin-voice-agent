"""
Temporal worker: registers workflow + activities and runs the event loop.

Usage:
    python -m workflow.worker

Requires a running Temporal server (e.g. `temporal server start-dev`).
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

from config import get_settings
from evaluation.cost_tracker import CostTracker
from workflow.activities import init_activity_context, run_agent, run_agent_conversation, run_handoff
from workflow.pipeline import CollectionPipelineWorkflow

TASK_QUEUE = "collection-pipeline"


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    settings = get_settings()
    tracker = CostTracker(settings)

    # Inject shared dependencies into activities
    init_activity_context(tracker, settings)

    # Connect to Temporal server
    temporal_address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    client = await Client.connect(temporal_address)

    # Run worker
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CollectionPipelineWorkflow],
        activities=[run_agent, run_agent_conversation, run_handoff],
    )

    logging.info("Worker started on task queue: %s", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
