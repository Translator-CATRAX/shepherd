"""xCRG direct and inferred lookup worker."""

import asyncio
import json
import logging
import time
import uuid
from copy import deepcopy
from pathlib import Path

import httpx

from shepherd_utils.config import settings
from shepherd_utils.db import get_message, save_message
from shepherd_utils.otel import setup_tracer
from shepherd_utils.shared import get_tasks, handle_task_failure, wrap_up_task

STREAM = "xcrg.lookup"
GROUP = "consumer"
CONSUMER = str(uuid.uuid4())[:8]
TASK_LIMIT = 100
tracer = setup_tracer(STREAM)

TF_QNODE_ID = "tf"
TP53_CURIE = "NCBIGene:7157"
TF_PATH = Path(__file__).resolve().parent / "transcription_factors.json"
DEBUG_DIR = Path("logs") / "xcrg_debug"


def get_single_query_edge(message: dict) -> tuple[str, dict]:
    """Return the single query edge for xCRG MVP queries."""
    qedges = message.get("message", {}).get("query_graph", {}).get("edges", {})
    if len(qedges) != 1:
        raise ValueError("xCRG MVP supports exactly one query edge.")
    edge_id = next(iter(qedges))
    return edge_id, qedges[edge_id]


def get_qualifier_value(edge: dict, qualifier_type_id: str) -> str | None:
    """Return a qualifier value from the first qualifier set, if present."""
    qualifier_constraints = edge.get("qualifier_constraints") or []
    if not qualifier_constraints:
        return None
    qualifier_set = qualifier_constraints[0].get("qualifier_set") or []
    for qualifier in qualifier_set:
        if qualifier.get("qualifier_type_id") == qualifier_type_id:
            return qualifier.get("qualifier_value")
    return None


def get_endpoint_type(categories: list[str]) -> str | None:
    """Return the supported xCRG endpoint type for a qnode."""
    if "biolink:ChemicalEntity" in categories:
        return "chemical"
    if "biolink:Gene" in categories:
        return "gene"
    return None


def validate_direct_lookup_query(message: dict) -> None:
    """Validate the direct one-hop xCRG query shape."""
    qgraph = message.get("message", {}).get("query_graph", {})
    qnodes = qgraph.get("nodes", {})
    _, edge = get_single_query_edge(message)

    if edge.get("knowledge_type", "lookup") == "inferred":
        raise ValueError("xCRG direct lookup does not support inferred edges.")

    predicates = edge.get("predicates") or []
    if "biolink:affects" not in predicates:
        raise ValueError("xCRG direct lookup requires predicate biolink:affects.")

    subject = edge.get("subject")
    obj = edge.get("object")
    if subject not in qnodes or obj not in qnodes:
        raise ValueError("Query edge references missing query nodes.")

    pinned_nodes = [qid for qid, qnode in qnodes.items() if qnode.get("ids")]
    if len(pinned_nodes) != 1:
        raise ValueError("xCRG direct lookup supports exactly one pinned query node.")

    unbound_nodes = [qid for qid, qnode in qnodes.items() if not qnode.get("ids")]
    if len(unbound_nodes) != 1:
        raise ValueError("xCRG direct lookup supports exactly one unbound query node.")

    pinned_node = qnodes[pinned_nodes[0]]
    unbound_node = qnodes[unbound_nodes[0]]

    if "biolink:Gene" not in (pinned_node.get("categories") or []):
        raise ValueError("xCRG direct lookup requires the pinned node to be a Gene.")
    if "biolink:ChemicalEntity" not in (unbound_node.get("categories") or []):
        raise ValueError(
            "xCRG direct lookup requires the unbound node to be a ChemicalEntity."
        )


def validate_inferred_query(message: dict) -> tuple[str, str, dict]:
    """Validate a phase-one inferred xCRG query while preserving user direction."""
    qgraph = message.get("message", {}).get("query_graph", {})
    qnodes = qgraph.get("nodes", {})
    _, edge = get_single_query_edge(message)

    if edge.get("knowledge_type") != "inferred":
        raise ValueError("Expected an inferred query edge.")

    if "biolink:affects" not in (edge.get("predicates") or []):
        raise ValueError("xCRG inferred lookup requires predicate biolink:affects.")

    source_qnode = edge.get("subject")
    target_qnode = edge.get("object")
    if source_qnode not in qnodes or target_qnode not in qnodes:
        raise ValueError("Query edge references missing query nodes.")

    source_node = qnodes[source_qnode]
    target_node = qnodes[target_qnode]

    endpoint_nodes = [source_node, target_node]
    pinned_count = sum(1 for node in endpoint_nodes if node.get("ids"))
    if pinned_count != 1:
        raise ValueError(
            "Phase-one inferred xCRG requires exactly one pinned endpoint node."
        )

    source_type = get_endpoint_type(source_node.get("categories") or [])
    target_type = get_endpoint_type(target_node.get("categories") or [])
    if {source_type, target_type} != {"chemical", "gene"}:
        raise ValueError(
            "Phase-one inferred xCRG currently requires one ChemicalEntity endpoint "
            "and one Gene endpoint."
        )

    direction = get_qualifier_value(edge, "biolink:object_direction_qualifier")
    aspect = get_qualifier_value(edge, "biolink:object_aspect_qualifier")
    if direction not in {"increased", "decreased"}:
        raise ValueError(
            "Phase-one inferred xCRG requires increased/decreased directionality."
        )
    if aspect != "activity_or_abundance":
        raise ValueError(
            "Phase-one inferred xCRG requires activity_or_abundance qualifiers."
        )

    return source_qnode, target_qnode, edge


def load_tf_list() -> list[str]:
    """Load the transcription factor list from the local Shepherd scripts folder."""
    with open(TF_PATH, "r", encoding="utf-8") as tf_file:
        tf_data = json.load(tf_file)
    tf_list = tf_data.get("tf") or []
    if not tf_list:
        raise ValueError("No transcription factors were found in transcription_factors.json.")
    return tf_list


def get_sign_templates(final_direction: str) -> list[tuple[str, str]]:
    """Return sign-compatible two-hop templates for the desired final direction."""
    if final_direction == "increased":
        return [("increased", "increased"), ("decreased", "decreased")]
    if final_direction == "decreased":
        return [("increased", "decreased"), ("decreased", "increased")]
    raise ValueError(f"Unsupported final direction: {final_direction}")


def chunk_values(values: list[str], chunk_size: int) -> list[list[str]]:
    """Split values into non-empty batches."""
    if chunk_size <= 0:
        raise ValueError("xCRG TF batch size must be positive.")
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def build_two_hop_query(
    original_message: dict,
    source_qnode: str,
    target_qnode: str,
    tf_list: list[str],
    first_direction: str,
    second_direction: str,
) -> dict:
    """Build a TF-mediated two-hop TRAPI query from the original inferred query."""
    original_qgraph = original_message["message"]["query_graph"]
    source_node = deepcopy(original_qgraph["nodes"][source_qnode])
    target_node = deepcopy(original_qgraph["nodes"][target_qnode])

    return {
        "message": {
            "query_graph": {
                "nodes": {
                    source_qnode: source_node,
                    TF_QNODE_ID: {
                        "categories": ["biolink:Gene"],
                        "ids": tf_list,
                    },
                    target_qnode: target_node,
                },
                "edges": {
                    "e0": {
                        "subject": source_qnode,
                        "object": TF_QNODE_ID,
                        "predicates": ["biolink:affects"],
                        "qualifier_constraints": [
                            {
                                "qualifier_set": [
                                    {
                                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                                        "qualifier_value": "activity_or_abundance",
                                    },
                                    {
                                        "qualifier_type_id": "biolink:object_direction_qualifier",
                                        "qualifier_value": first_direction,
                                    },
                                ]
                            }
                        ],
                    },
                    "e1": {
                        "subject": TF_QNODE_ID,
                        "object": target_qnode,
                        "predicates": ["biolink:affects"],
                        "qualifier_constraints": [
                            {
                                "qualifier_set": [
                                    {
                                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                                        "qualifier_value": "activity_or_abundance",
                                    },
                                    {
                                        "qualifier_type_id": "biolink:object_direction_qualifier",
                                        "qualifier_value": second_direction,
                                    },
                                ]
                            }
                        ],
                    },
                },
            },
            "knowledge_graph": {"nodes": {}, "edges": {}},
            "results": [],
            "auxiliary_graphs": {},
        },
        "parameters": deepcopy(original_message.get("parameters") or {}),
        "submitter": original_message.get("submitter"),
    }


def result_has_bad_edge_predicate(result: dict, kg_edges: dict, predicate: str) -> bool:
    """Return True when any bound knowledge graph edge has the given predicate."""
    for analysis in result.get("analyses") or []:
        for bindings in (analysis.get("edge_bindings") or {}).values():
            for binding in bindings or []:
                edge_id = binding.get("id")
                if kg_edges.get(edge_id, {}).get("predicate") == predicate:
                    return True
    return False


def get_bound_node_id(result: dict, qnode_id: str) -> str | None:
    """Return the first node binding id for the given qnode."""
    bindings = (result.get("node_bindings") or {}).get(qnode_id) or []
    if not bindings:
        return None
    return bindings[0].get("id")


def result_preserves_direction(
    result: dict,
    kg_edges: dict,
    source_qnode: str,
    target_qnode: str,
) -> bool:
    """Check that the result preserves source->tf and tf->target edge directions."""
    source_id = get_bound_node_id(result, source_qnode)
    tf_id = get_bound_node_id(result, TF_QNODE_ID)
    target_id = get_bound_node_id(result, target_qnode)
    if not source_id or not tf_id or not target_id:
        return False

    for analysis in result.get("analyses") or []:
        edge_bindings = analysis.get("edge_bindings") or {}

        e0_bindings = edge_bindings.get("e0") or []
        e1_bindings = edge_bindings.get("e1") or []
        if not e0_bindings or not e1_bindings:
            return False

        for binding in e0_bindings:
            kg_edge = kg_edges.get(binding.get("id")) or {}
            if kg_edge.get("subject") != source_id or kg_edge.get("object") != tf_id:
                return False

        for binding in e1_bindings:
            kg_edge = kg_edges.get(binding.get("id")) or {}
            if kg_edge.get("subject") != tf_id or kg_edge.get("object") != target_id:
                return False

    return True


def merge_filtered_responses(
    filtered_responses: list[dict],
    query_graph: dict,
) -> dict:
    """Merge filtered Retriever responses into a single TRAPI response."""
    merged = {
        "message": {
            "query_graph": query_graph,
            "knowledge_graph": {"nodes": {}, "edges": {}},
            "results": [],
            "auxiliary_graphs": {},
        }
    }
    seen_results = set()

    for response in filtered_responses:
        message = response.get("message") or {}
        merged["message"]["knowledge_graph"]["nodes"].update(
            (message.get("knowledge_graph") or {}).get("nodes") or {}
        )
        merged["message"]["knowledge_graph"]["edges"].update(
            (message.get("knowledge_graph") or {}).get("edges") or {}
        )
        merged["message"]["auxiliary_graphs"].update(
            (message.get("auxiliary_graphs") or {}) or {}
        )

        for result in message.get("results") or []:
            key = json.dumps(
                {
                    "node_bindings": result.get("node_bindings"),
                    "analyses": result.get("analyses"),
                },
                sort_keys=True,
            )
            if key not in seen_results:
                seen_results.add(key)
                merged["message"]["results"].append(result)

    return merged


def debug_dump_json(query_id: str, label: str, payload: dict, logger: logging.Logger) -> None:
    """Best-effort debug JSON dump for inferred xCRG runs."""
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = DEBUG_DIR / f"{query_id}_{label}.json"
        with open(dump_path, "w", encoding="utf-8") as debug_file:
            json.dump(payload, debug_file, indent=2, sort_keys=True)
    except Exception as exc:
        logger.warning(f"Failed to write debug JSON {label}: {exc}")


def filter_inferred_response(
    response: dict,
    source_qnode: str,
    target_qnode: str,
) -> dict:
    """Filter subclass and wrong-direction results from a two-hop Retriever response."""
    message = response.get("message") or {}
    kg = message.get("knowledge_graph") or {}
    kg_edges = kg.get("edges") or {}

    filtered_results = []
    for result in message.get("results") or []:
        tf_id = get_bound_node_id(result, TF_QNODE_ID)
        if tf_id == TP53_CURIE:
            continue
        if result_has_bad_edge_predicate(result, kg_edges, "biolink:subclass_of"):
            continue
        if not result_preserves_direction(result, kg_edges, source_qnode, target_qnode):
            continue
        filtered_results.append(result)

    filtered_message = deepcopy(message)
    filtered_message["results"] = filtered_results
    filtered_message.setdefault("knowledge_graph", {"nodes": {}, "edges": {}})
    filtered_message.setdefault("auxiliary_graphs", {})
    return {"message": filtered_message}


def summarize_response_counts(response: dict) -> dict:
    """Return compact counts for a TRAPI response."""
    message = response.get("message") or {}
    knowledge_graph = message.get("knowledge_graph") or {}
    return {
        "result_count": len(message.get("results") or []),
        "node_count": len(knowledge_graph.get("nodes") or {}),
        "edge_count": len(knowledge_graph.get("edges") or {}),
    }


async def run_sync_retriever_lookup(message: dict, logger: logging.Logger) -> dict:
    """Run a sync Retriever lookup and return its TRAPI response."""
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
    return result


async def run_direct_lookup(message: dict, logger: logging.Logger) -> dict:
    """Run the original one-hop direct xCRG lookup."""
    validate_direct_lookup_query(message)
    return await run_sync_retriever_lookup(message, logger)


async def run_inferred_lookup(query_id: str, message: dict, logger: logging.Logger) -> dict:
    """Run phase-one TF-mediated inferred xCRG lookup."""
    debug_dump_json(query_id, "original_inferred_query", message, logger)
    source_qnode, target_qnode, edge = validate_inferred_query(message)
    source_ids = message["message"]["query_graph"]["nodes"][source_qnode].get("ids") or []
    target_ids = message["message"]["query_graph"]["nodes"][target_qnode].get("ids") or []
    endpoint_ids = set(source_ids) | set(target_ids)
    tf_list = [
        tf_id
        for tf_id in load_tf_list()
        if tf_id != TP53_CURIE and tf_id not in endpoint_ids
    ]
    if not tf_list:
        raise ValueError("No transcription factors remain after TP53/target filtering.")

    final_direction = get_qualifier_value(edge, "biolink:object_direction_qualifier")
    sign_templates = get_sign_templates(final_direction)
    tf_batches = chunk_values(tf_list, settings.xcrg_tf_batch_size)
    logger.info(
        "Running inferred xCRG lookup with %s TFs across %s batches of up to %s IDs.",
        len(tf_list),
        len(tf_batches),
        settings.xcrg_tf_batch_size,
    )

    filtered_responses = []
    debug_summary = {
        "query_id": query_id,
        "final_direction": final_direction,
        "tf_count": len(tf_list),
        "batch_size": settings.xcrg_tf_batch_size,
        "batch_count": len(tf_batches),
        "templates": [],
    }
    for template_idx, (first_direction, second_direction) in enumerate(
        sign_templates, start=1
    ):
        template_summary = {
            "template_index": template_idx,
            "first_direction": first_direction,
            "second_direction": second_direction,
            "batches": [],
        }
        for batch_idx, tf_batch in enumerate(tf_batches, start=1):
            inferred_message = build_two_hop_query(
                message,
                source_qnode,
                target_qnode,
                tf_batch,
                first_direction,
                second_direction,
            )
            inferred_message["parameters"]["timeout"] = (
                inferred_message["parameters"].get("timeout") or settings.lookup_timeout
            )
            inferred_message["parameters"]["tiers"] = (
                inferred_message["parameters"].get("tiers")
                or [settings.default_data_tier]
            )
            if (
                "submitter" not in inferred_message
                or inferred_message["submitter"] is None
            ):
                inferred_message["submitter"] = (
                    "infores:shepherd-xcrg:{maturity}@{location}@{url}".format(
                        maturity=settings.server_maturity,
                        location=settings.server_location,
                        url=settings.server_url,
                    )
                )
            debug_dump_json(
                query_id,
                f"template_{template_idx}_batch_{batch_idx}_query",
                inferred_message,
                logger,
            )
            response = await run_sync_retriever_lookup(inferred_message, logger)
            debug_dump_json(
                query_id,
                f"template_{template_idx}_batch_{batch_idx}_raw_response",
                response,
                logger,
            )
            filtered_response = filter_inferred_response(
                response, source_qnode, target_qnode
            )
            debug_dump_json(
                query_id,
                f"template_{template_idx}_batch_{batch_idx}_filtered_response",
                filtered_response,
                logger,
            )
            filtered_responses.append(filtered_response)
            template_summary["batches"].append(
                {
                    "batch_index": batch_idx,
                    "tf_ids": tf_batch,
                    "tf_count": len(tf_batch),
                    "raw_response": summarize_response_counts(response),
                    "filtered_response": summarize_response_counts(filtered_response),
                }
            )
        debug_summary["templates"].append(template_summary)

    merged_query_graph = build_two_hop_query(
        message,
        source_qnode,
        target_qnode,
        tf_list,
        sign_templates[0][0],
        sign_templates[0][1],
    )["message"]["query_graph"]

    merged = merge_filtered_responses(
        filtered_responses,
        merged_query_graph,
    )
    debug_summary["merged_response"] = summarize_response_counts(merged)
    debug_dump_json(query_id, "inferred_debug_summary", debug_summary, logger)
    debug_dump_json(query_id, "merged_inferred_response", merged, logger)
    return merged


async def xcrg_lookup(task, logger: logging.Logger):
    """Dispatch direct or inferred xCRG lookups and save the result."""
    query_id = task[1]["query_id"]
    response_id = task[1]["response_id"]
    message = await get_message(query_id, logger)
    parameters = message.get("parameters") or {}
    parameters["timeout"] = parameters.get("timeout", settings.lookup_timeout)
    parameters["tiers"] = parameters.get("tiers") or [settings.default_data_tier]
    message["parameters"] = parameters

    if "submitter" not in message:
        message["submitter"] = (
            "infores:shepherd-xcrg:{maturity}@{location}@{url}".format(
                maturity=settings.server_maturity,
                location=settings.server_location,
                url=settings.server_url,
            )
        )

    _, edge = get_single_query_edge(message)
    if edge.get("knowledge_type") == "inferred":
        result = await run_inferred_lookup(query_id, message, logger)
    else:
        result = await run_direct_lookup(message, logger)

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
