"""xCRG ARA entry worker"""

import asyncio
import json
import logging
import time
import uuid

from shepherd_utils.otel import setup_tracer
from shepherd_utils.shared import get_tasks, handle_task_failure, wrap_up_task

STREAM = "xcrg"
GROUP = "consumer"
CONSUMER = str(uuid.uuid4())[:8]
TASK_LIMIT = 100
tracer = setup_tracer(STREAM)


async def xcrg(task, logger: logging.Logger):
    """Set the minimal Shepherd-native xCRG workflow."""
    workflow = json.loads(task[1]["workflow"])

    if workflow is None:
        workflow = [
            {"id": "xcrg.lookup"},
        ]

    task[1]["workflow"] = json.dumps(workflow)


async def process_task(task, parent_ctx, logger: logging.Logger, limiter):
    """Process a given task and ACK in redis."""
    start = time.time()
    span = tracer.start_span(STREAM, context=parent_ctx)
    try:
        await xcrg(task, logger)
        try:
            await wrap_up_task(STREAM, GROUP, task, logger)
        except Exception as e:
            logger.error(f"Task {task[0]}: Failed to wrap up task: {e}")
    except asyncio.CancelledError:
        logger.warning(f"Task {task[0]} was cancelled")
    except Exception as e:
        logger.error(f"Task {task[0]} failed with unhandled error: {e}", exc_info=True)
        await handle_task_failure(STREAM, GROUP, task, logger)
    finally:
        span.end()
        limiter.release()
        logger.info(f"Finished task {task[0]} in {time.time() - start}")


async def poll_for_tasks():
    """On initialization, poll indefinitely for available tasks."""
    while True:
        try:
            async for task, parent_ctx, logger, limiter in get_tasks(
                STREAM, GROUP, CONSUMER, TASK_LIMIT
            ):
                asyncio.create_task(process_task(task, parent_ctx, logger, limiter))
        except asyncio.CancelledError:
            logging.info("Poll loop cancelled, shutting down.")
        except Exception as e:
            logging.error(f"Error in task polling loop: {e}", exc_info=True)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(poll_for_tasks())
