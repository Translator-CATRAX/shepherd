"""xCRG direct lookup worker."""

import asyncio
import logging
import time
import uuid

import httpx

from shepherd_utils.config import settings
from shepherd_utils.db import (
    get_message,
    save_message,
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

    predicates = edge.get("predicates") or []
    if "biolink:affects" not in predicates:
        raise ValueError("xCRG lookup MVP requires predicate biolink:affects.")

    subject = edge.get("subject")
    obj = edge.get("object")
    if subject not in qnodes or obj not in qnodes:
        raise ValueError("Query edge references missing query nodes.")

    pinned_nodes = [qid for qid, qnode in qnodes.items() if qnode.get("ids")]
    if len(pinned_nodes) != 1:
        raise ValueError("xCRG lookup MVP supports exactly one pinned query node.")

    unbound_nodes = [qid for qid, qnode in qnodes.items() if not qnode.get("ids")]
    if len(unbound_nodes) != 1:
        raise ValueError("xCRG lookup MVP supports exactly one unbound query node.")

    pinned_node = qnodes[pinned_nodes[0]]
    unbound_node = qnodes[unbound_nodes[0]]

    pinned_categories = pinned_node.get("categories") or []
    if "biolink:Gene" not in pinned_categories:
        raise ValueError("xCRG lookup MVP requires the pinned node to be a Gene.")

    unbound_categories = unbound_node.get("categories") or []
    if "biolink:ChemicalEntity" not in unbound_categories:
        raise ValueError(
            "xCRG lookup MVP requires the unbound node to be a ChemicalEntity."
        )


async def xcrg_lookup(task, logger: logging.Logger):
    """Dispatch a direct lookup query to Retriever and save the sync response."""
    query_id = task[1]["query_id"]
    response_id = task[1]["response_id"]
    message = await get_message(query_id, logger)
    parameters = message.get("parameters") or {}
    parameters["timeout"] = parameters.get("timeout", settings.lookup_timeout)
    parameters["tiers"] = parameters.get("tiers") or [settings.default_data_tier]
    message["parameters"] = parameters

    validate_lookup_query(message)

    if "submitter" not in message:
        message["submitter"] = (
            "infores:shepherd-xcrg:{maturity}@{location}@{url}".format(
                maturity=settings.server_maturity,
                location=settings.server_location,
                url=settings.server_url,
            )
        )

    logger.info(f"Sending xCRG lookup query to {settings.sync_kg_retrieval_url}")
    async with httpx.AsyncClient(timeout=message["parameters"]["timeout"]) as client:
        response = await client.post(settings.sync_kg_retrieval_url, json=message)
        response.raise_for_status()
        result = response.json()

    if "message" not in result:
        raise ValueError("Retriever response did not contain a TRAPI message.")
    result["message"].setdefault("knowledge_graph", {"nodes": {}, "edges": {}})
    result["message"].setdefault("results", [])
    result["message"].setdefault("auxiliary_graphs", {})

    await save_message(response_id, result, logger)


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
