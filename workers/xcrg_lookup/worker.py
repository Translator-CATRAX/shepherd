"""xCRG direct lookup worker."""

import asyncio
import logging
import time
import uuid

import httpx

from shepherd_utils.config import settings
from shepherd_utils.db import (
    add_callback_id,
    cleanup_callbacks,
    get_message,
    get_running_callbacks,
)
from shepherd_utils.otel import setup_tracer
from shepherd_utils.shared import get_tasks, handle_task_failure, wrap_up_task

STREAM = "xcrg.lookup"
GROUP = "consumer"
CONSUMER = str(uuid.uuid4())[:8]
TASK_LIMIT = 100
tracer = setup_tracer(STREAM)


def validate_lookup_query(message: dict) -> None:
    """Validate the first xCRG direct-lookup query shape."""
    qgraph = message.get("message", {}).get("query_graph", {})
    qnodes = qgraph.get("nodes", {})
    qedges = qgraph.get("edges", {})

    if len(qedges) != 1:
        raise ValueError("xCRG lookup MVP supports exactly one query edge.")

    edge = next(iter(qedges.values()))
    if edge.get("knowledge_type", "lookup") == "inferred":
        raise ValueError("xCRG lookup MVP does not support inferred edges.")

    subject = edge.get("subject")
    obj = edge.get("object")
    if subject not in qnodes or obj not in qnodes:
        raise ValueError("Query edge references missing query nodes.")

    pinned_nodes = [qid for qid, qnode in qnodes.items() if qnode.get("ids")]
    if len(pinned_nodes) != 1:
        raise ValueError("xCRG lookup MVP supports exactly one pinned query node.")


async def xcrg_lookup(task, logger: logging.Logger):
    """Dispatch a direct lookup query to the configured graph backend."""
    query_id = task[1]["query_id"]
    message = await get_message(query_id, logger)
    parameters = message.get("parameters") or {}
    parameters["timeout"] = parameters.get("timeout", settings.lookup_timeout)
    message["parameters"] = parameters

    validate_lookup_query(message)

    callback_id = str(uuid.uuid4())[:8]
    await add_callback_id(query_id, callback_id, logger)
    message["callback"] = f"{settings.callback_host}/xcrg/callback/{callback_id}"

    logger.info(f"Sending xCRG lookup query to {settings.xcrg_lookup_url}")
    async with httpx.AsyncClient(timeout=100) as client:
        response = await client.post(settings.xcrg_lookup_url, json=message)
        response.raise_for_status()

    max_query_time = message["parameters"]["timeout"]
    start_time = time.time()
    running_callback_ids = [callback_id]
    while time.time() - start_time < max_query_time:
        try:
            running_callback_ids = await get_running_callbacks(query_id, logger)
        except Exception:
            await asyncio.sleep(5)
            continue

        if len(running_callback_ids) == 0:
            logger.debug("xCRG lookup callbacks completed.")
            break

        await asyncio.sleep(1)

    if time.time() - start_time > max_query_time:
        logger.warning(
            "Timed out getting xCRG lookup callbacks. "
            f"{len(running_callback_ids)} queries were still running..."
        )
        await cleanup_callbacks(query_id, logger)


async def process_task(task, parent_ctx, logger: logging.Logger, limiter):
    """Process a given task and ACK in redis."""
    start = time.time()
    span = tracer.start_span(STREAM, context=parent_ctx)
    try:
        await xcrg_lookup(task, logger)
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
