"""xCRG direct and inferred lookup worker."""

import asyncio
import json
import logging
import math
import sqlite3
import time
import uuid
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
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
DIRECT_QEDGE_ID = "direct"
SCORING_METHOD = "xcrg-result-filtering-v2"
XCRG_RESOURCE_ID = "infores:arax"
MISSING_SORT_VALUE = float("inf")
NGD_CACHE_MAX_ROWS = 256
NGD_VALUE_URL = "https://arax.ncats.io/api/rtx/v1/ui/#/PubmedMeshNgd"
NGD_DESCRIPTION = (
    "Normalized google distance is a metric based on edge subject/object node "
    "co-occurrence in abstracts of PubMed articles."
)
COMPUTED_EDGE_CONTAINER_DESCRIPTION = (
    "This edge is a container for a computed value between two nodes that is not "
    "directly attachable to other edges."
)

try:
    from bmt import Toolkit
except ImportError:  # pragma: no cover - local unit env may not install worker deps.
    Toolkit = None

_BMT_TOOLKIT = None
_BMT_WARNING_EMITTED = False
_NGD_CONNECTION = None
_NGD_WARNING_EMITTED = False
_NGD_NEIGHBOR_CACHE = OrderedDict()
FALLBACK_CATEGORY_DEPTH = {
    "biolink:ChemicalEntity": 1,
    "biolink:ChemicalMixture": 2,
    "biolink:EnvironmentalFoodContaminant": 2,
    "biolink:FoodAdditive": 2,
    "biolink:MolecularEntity": 2,
    "biolink:ComplexMolecularMixture": 3,
    "biolink:Food": 3,
    "biolink:MolecularMixture": 3,
    "biolink:NucleicAcidEntity": 3,
    "biolink:ProcessedMaterial": 3,
    "biolink:SmallMolecule": 3,
    "biolink:Drug": 4,
}


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


def safe_debug_token(value: str | None) -> str:
    """Return a filesystem-friendly token for debug run names."""
    if not value:
        return "unbound"
    token = "".join(char if char.isalnum() else "_" for char in value)
    token = "_".join(part for part in token.split("_") if part)
    return token[:80] or "unbound"


def describe_qnode_for_debug(qnode: dict) -> str:
    """Return a compact qnode label for human-readable debug paths."""
    ids = qnode.get("ids") or []
    if ids:
        return safe_debug_token(ids[0])
    categories = qnode.get("categories") or []
    if categories:
        return safe_debug_token(categories[0].removeprefix("biolink:"))
    return "unbound"


def make_debug_run_context(query_id: str, message: dict) -> dict:
    """Create human-readable debug path metadata for one xCRG query."""
    created_at = datetime.now(timezone.utc)
    qgraph = message.get("message", {}).get("query_graph", {})
    qnodes = qgraph.get("nodes") or {}
    edge_id, edge = get_single_query_edge(message)
    source_qnode = edge.get("subject")
    target_qnode = edge.get("object")
    direction = get_qualifier_value(edge, "biolink:object_direction_qualifier")
    source_label = describe_qnode_for_debug(qnodes.get(source_qnode) or {})
    target_label = describe_qnode_for_debug(qnodes.get(target_qnode) or {})
    direction_label = safe_debug_token(direction)
    run_name = (
        f"{created_at.strftime('%Y%m%d_%H%M%S')}_{query_id}_"
        f"{source_label}_to_{target_label}_{direction_label}"
    )
    return {
        "query_id": query_id,
        "created_at": created_at.isoformat(),
        "run_name": run_name,
        "run_dir": DEBUG_DIR / run_name,
        "query_edge_id": edge_id,
        "source_qnode": source_qnode,
        "target_qnode": target_qnode,
        "source_label": source_label,
        "target_label": target_label,
        "direction": direction,
        "artifacts": [],
    }


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


def get_bmt_toolkit(logger: logging.Logger):
    """Return a cached Biolink Toolkit instance when the dependency is available."""
    global _BMT_TOOLKIT, _BMT_WARNING_EMITTED
    if Toolkit is None:
        if not _BMT_WARNING_EMITTED:
            logger.warning("BMT is unavailable; using fallback specificity scores.")
            _BMT_WARNING_EMITTED = True
        return None
    if _BMT_TOOLKIT is None:
        try:
            _BMT_TOOLKIT = Toolkit()
        except Exception as exc:
            if not _BMT_WARNING_EMITTED:
                logger.warning(
                    f"Failed to initialize BMT; using fallback specificity scores: {exc}"
                )
                _BMT_WARNING_EMITTED = True
            return None
    return _BMT_TOOLKIT


def get_category_specificity(category: str, logger: logging.Logger) -> int:
    """Return a Biolink specificity heuristic based on non-mixin ancestor count."""
    bmt_toolkit = get_bmt_toolkit(logger)
    if bmt_toolkit:
        try:
            if not bmt_toolkit.get_element(category):
                return FALLBACK_CATEGORY_DEPTH.get(category, 0)
            ancestors = (
                bmt_toolkit.get_ancestors(
                    category,
                    reflexive=False,
                    formatted=True,
                    mixin=False,
                )
                or []
            )
            return max(len(ancestors), FALLBACK_CATEGORY_DEPTH.get(category, 0))
        except Exception as exc:
            logger.warning(
                f"Could not calculate BMT specificity for {category}: {exc}"
            )
    return FALLBACK_CATEGORY_DEPTH.get(category, 0)


def is_chemical_category(category: str, logger: logging.Logger) -> bool:
    """Return True when a category is ChemicalEntity or a chemical descendant."""
    if category == "biolink:ChemicalEntity" or category in FALLBACK_CATEGORY_DEPTH:
        return True
    bmt_toolkit = get_bmt_toolkit(logger)
    if not bmt_toolkit:
        return False
    try:
        ancestors = (
            bmt_toolkit.get_ancestors(
                category,
                reflexive=False,
                formatted=True,
                mixin=False,
            )
            or []
        )
        return "biolink:ChemicalEntity" in ancestors
    except Exception as exc:
        logger.warning(f"Could not inspect category ancestry for {category}: {exc}")
        return False


def get_node_category_specificity(node: dict, logger: logging.Logger) -> int:
    """Return the most specific chemical category score attached to a KG node."""
    chemical_categories = [
        category
        for category in (node.get("categories") or [])
        if is_chemical_category(category, logger)
    ]
    if not chemical_categories:
        return 0

    bmt_toolkit = get_bmt_toolkit(logger)
    if bmt_toolkit:
        try:
            most_specific = bmt_toolkit.get_most_specific_category(
                chemical_categories,
                formatted=True,
            )
            if is_chemical_category(most_specific, logger):
                return get_category_specificity(most_specific, logger)
        except Exception as exc:
            logger.warning(f"Could not select most specific category with BMT: {exc}")

    return max(
        FALLBACK_CATEGORY_DEPTH.get(category, 0)
        for category in chemical_categories
    )


def get_node_information_content(node: dict) -> float | None:
    """Return a node's Biolink information content attribute, when present."""
    values = []
    for attribute in node.get("attributes") or []:
        if attribute.get("attribute_type_id") != "biolink:information_content":
            continue
        raw_value = attribute.get("value")
        raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in raw_values:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return max(values) if values else None


def get_ngd_connection(logger: logging.Logger) -> sqlite3.Connection | None:
    """Return a cached read-only NGD SQLite connection when the local DB exists."""
    global _NGD_CONNECTION, _NGD_WARNING_EMITTED

    if _NGD_CONNECTION is not None:
        return _NGD_CONNECTION

    db_path = Path(settings.xcrg_ngd_db_path)
    if not db_path.exists():
        if not _NGD_WARNING_EMITTED:
            logger.warning(
                "xCRG NGD DB not found at %s; NGD tie-breaker is disabled.",
                db_path,
            )
            _NGD_WARNING_EMITTED = True
        return None

    try:
        _NGD_CONNECTION = sqlite3.connect(
            f"file:{db_path.as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        return _NGD_CONNECTION
    except sqlite3.Error as exc:
        if not _NGD_WARNING_EMITTED:
            logger.warning(
                "Failed to open xCRG NGD DB at %s; NGD tie-breaker is disabled: %s",
                db_path,
                exc,
            )
            _NGD_WARNING_EMITTED = True
        return None


def get_ngd_neighbors(curie: str, logger: logging.Logger) -> dict[str, float] | None:
    """Return cached NGD neighbors for one CURIE from the adjacency-list DB."""
    if not curie:
        return None

    if curie in _NGD_NEIGHBOR_CACHE:
        _NGD_NEIGHBOR_CACHE.move_to_end(curie)
        return _NGD_NEIGHBOR_CACHE[curie]

    connection = get_ngd_connection(logger)
    if connection is None:
        return None

    try:
        row = connection.execute(
            "SELECT ngd FROM curie_ngd WHERE curie = ?",
            (curie,),
        ).fetchone()
    except sqlite3.Error:
        return None

    if row is None:
        neighbors = {}
    else:
        try:
            neighbors = {}
            for neighbor, score in json.loads(row[0]):
                try:
                    ngd_score = float(score)
                except (TypeError, ValueError):
                    continue
                # Only keep meaningful NGD values. Missing/invalid values should
                # behave like "no tie-breaker" instead of helping a result.
                if not math.isfinite(ngd_score) or ngd_score <= 0.0:
                    continue
                neighbors[str(neighbor)] = ngd_score
        except (TypeError, ValueError, json.JSONDecodeError):
            neighbors = {}

    _NGD_NEIGHBOR_CACHE[curie] = neighbors
    _NGD_NEIGHBOR_CACHE.move_to_end(curie)
    while len(_NGD_NEIGHBOR_CACHE) > NGD_CACHE_MAX_ROWS:
        _NGD_NEIGHBOR_CACHE.popitem(last=False)
    return neighbors


def get_ngd_score(
    curie_a: str | None,
    curie_b: str | None,
    logger: logging.Logger,
) -> float | None:
    """Return lower-is-better NGD for a CURIE pair, if present in the local DB."""
    if not curie_a or not curie_b:
        return None
    if curie_a == curie_b:
        return None

    neighbors = get_ngd_neighbors(curie_a, logger)
    if neighbors:
        score = neighbors.get(curie_b)
        if score is not None:
            return score

    # The DB is expected to be symmetric, but this fallback is cheap insurance
    # for partial rows or future DB variants.
    reverse_neighbors = get_ngd_neighbors(curie_b, logger)
    if reverse_neighbors:
        score = reverse_neighbors.get(curie_a)
        if score is not None:
            return score
    return None


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


def build_direct_query_for_inferred(
    original_message: dict,
    source_qnode: str,
    target_qnode: str,
) -> dict:
    """Build the direct one-hop query that accompanies inferred xCRG mode."""
    original_qgraph = original_message["message"]["query_graph"]
    _, original_edge = get_single_query_edge(original_message)
    direct_edge = deepcopy(original_edge)
    direct_edge.pop("knowledge_type", None)

    return {
        "message": {
            "query_graph": {
                "nodes": {
                    source_qnode: deepcopy(original_qgraph["nodes"][source_qnode]),
                    target_qnode: deepcopy(original_qgraph["nodes"][target_qnode]),
                },
                "edges": {
                    DIRECT_QEDGE_ID: direct_edge,
                },
            },
            "knowledge_graph": {"nodes": {}, "edges": {}},
            "results": [],
            "auxiliary_graphs": {},
        },
        "parameters": deepcopy(original_message.get("parameters") or {}),
        "submitter": original_message.get("submitter"),
    }


def build_combined_query_graph(
    original_message: dict,
    source_qnode: str,
    target_qnode: str,
    tf_list: list[str],
) -> dict:
    """Build a response query graph that can bind direct and TF-mediated results."""
    direct_query = build_direct_query_for_inferred(
        original_message,
        source_qnode,
        target_qnode,
    )
    query_graph = direct_query["message"]["query_graph"]
    query_graph["nodes"][TF_QNODE_ID] = {
        "categories": ["biolink:Gene"],
        "ids": tf_list,
    }
    query_graph["edges"]["e0"] = {
        "subject": source_qnode,
        "object": TF_QNODE_ID,
        "predicates": ["biolink:affects"],
    }
    query_graph["edges"]["e1"] = {
        "subject": TF_QNODE_ID,
        "object": target_qnode,
        "predicates": ["biolink:affects"],
    }
    return query_graph


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


def result_preserves_direct_direction(
    result: dict,
    kg_edges: dict,
    source_qnode: str,
    target_qnode: str,
) -> bool:
    """Check that a direct result preserves the original source->target direction."""
    source_id = get_bound_node_id(result, source_qnode)
    target_id = get_bound_node_id(result, target_qnode)
    if not source_id or not target_id:
        return False

    for analysis in result.get("analyses") or []:
        edge_bindings = analysis.get("edge_bindings") or {}
        direct_bindings = edge_bindings.get(DIRECT_QEDGE_ID) or []
        if not direct_bindings:
            return False

        for binding in direct_bindings:
            kg_edge = kg_edges.get(binding.get("id")) or {}
            if kg_edge.get("subject") != source_id or kg_edge.get("object") != target_id:
                return False

    return True


def filter_direct_response(
    response: dict,
    source_qnode: str,
    target_qnode: str,
) -> dict:
    """Filter subclass and wrong-direction results from a direct Retriever response."""
    message = response.get("message") or {}
    kg = message.get("knowledge_graph") or {}
    kg_edges = kg.get("edges") or {}

    filtered_results = []
    for result in message.get("results") or []:
        if result_has_bad_edge_predicate(result, kg_edges, "biolink:subclass_of"):
            continue
        if not result_preserves_direct_direction(
            result,
            kg_edges,
            source_qnode,
            target_qnode,
        ):
            continue
        filtered_results.append(result)

    filtered_message = deepcopy(message)
    filtered_message["results"] = filtered_results
    filtered_message.setdefault("knowledge_graph", {"nodes": {}, "edges": {}})
    filtered_message.setdefault("auxiliary_graphs", {})
    return ensure_response_versions({"message": filtered_message}, response)


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
    ensure_response_versions(merged, *filtered_responses)
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


def get_answer_qnode_id(
    query_graph: dict,
    source_qnode: str,
    target_qnode: str,
) -> str:
    """Return the unpinned endpoint qnode whose bindings are the answer list."""
    qnodes = query_graph.get("nodes") or {}
    for qnode_id in (source_qnode, target_qnode):
        if not (qnodes.get(qnode_id) or {}).get("ids"):
            return qnode_id
    return target_qnode


def result_edge_binding_keys(result: dict) -> set[str]:
    """Return qedge ids bound by any analysis in the result."""
    keys = set()
    for analysis in result.get("analyses") or []:
        keys.update((analysis.get("edge_bindings") or {}).keys())
    return keys


def is_two_hop_result(result: dict) -> bool:
    """Return True for TF-mediated inferred results."""
    keys = result_edge_binding_keys(result)
    return "e0" in keys and "e1" in keys


def answer_qnode_uses_category_specificity(
    query_graph: dict,
    answer_qnode_id: str,
    logger: logging.Logger,
) -> bool:
    """Chemical answers use Biolink specificity before information content."""
    qnode = (query_graph.get("nodes") or {}).get(answer_qnode_id) or {}
    categories = qnode.get("categories") or []
    return any(is_chemical_category(category, logger) for category in categories)


def get_result_answer_metrics(
    result: dict,
    kg_nodes: dict,
    query_graph: dict,
    answer_qnode_id: str,
    use_category_specificity: bool,
    logger: logging.Logger,
) -> tuple[int, float | None, str]:
    """Return sort metrics for the answer node bound by a result."""
    answer_id = get_bound_node_id(result, answer_qnode_id) or ""
    answer_node = kg_nodes.get(answer_id) or {}
    specificity = (
        get_node_category_specificity(answer_node, logger)
        if use_category_specificity
        else 0
    )
    information_content = get_node_information_content(answer_node)
    return specificity, information_content, answer_id


def descending_optional(value: float | int | None) -> float:
    """Convert optional descending values into ascending sort components."""
    return -float(value) if value is not None else MISSING_SORT_VALUE


def ascending_optional(value: float | int | None) -> float:
    """Convert optional ascending values into sort components."""
    return float(value) if value is not None else MISSING_SORT_VALUE


def get_result_endpoint_ngd(
    result: dict,
    source_qnode: str,
    target_qnode: str,
    logger: logging.Logger,
) -> float | None:
    """Return direct source-answer NGD for the final source/target pair."""
    source_id = get_bound_node_id(result, source_qnode)
    target_id = get_bound_node_id(result, target_qnode)
    return get_ngd_score(source_id, target_id, logger)


def get_result_answer_tf_ngd(
    result: dict,
    answer_qnode_id: str,
    logger: logging.Logger,
) -> float | None:
    """Return answer-to-TF NGD for ordering results within a TF bucket."""
    answer_id = get_bound_node_id(result, answer_qnode_id)
    tf_id = get_bound_node_id(result, TF_QNODE_ID)
    return get_ngd_score(answer_id, tf_id, logger)


def get_original_query_edge_id(message: dict) -> str:
    """Return the original single qedge id from an xCRG response/query."""
    qedges = message.get("message", {}).get("query_graph", {}).get("edges", {})
    if len(qedges) != 1:
        raise ValueError("xCRG final response expects the original single query edge.")
    return next(iter(qedges))


def get_edge_bindings(result: dict, qedge_id: str) -> list[dict]:
    """Return copied edge bindings for a qedge across all analyses."""
    edge_bindings = []
    seen = set()
    for analysis in result.get("analyses") or []:
        for binding in (analysis.get("edge_bindings") or {}).get(qedge_id) or []:
            edge_id = binding.get("id")
            if edge_id in seen:
                continue
            seen.add(edge_id)
            copied_binding = deepcopy(binding)
            copied_binding.setdefault("attributes", [])
            edge_bindings.append(copied_binding)
    return edge_bindings


def get_result_score(result: dict) -> float | None:
    """Return the first score attached to a result analysis."""
    for analysis in result.get("analyses") or []:
        if "score" not in analysis:
            continue
        try:
            return float(analysis["score"])
        except (TypeError, ValueError):
            continue
    return None


def stamp_rank_scores(results: list[dict]) -> None:
    """Assign rank-derived TRAPI Analysis.score values after sorting."""
    total = len(results)
    if total == 0:
        return
    for index, result in enumerate(results):
        score = float(total - index) / total
        for analysis in result.get("analyses") or []:
            analysis["score"] = score
            analysis["scoring_method"] = SCORING_METHOD


def stamp_xcrg_rank_scores(results: list[dict]) -> None:
    """Assign rank-derived scores only to xCRG analyses in final output."""
    total = len(results)
    if total == 0:
        return
    for index, result in enumerate(results):
        score = float(total - index) / total
        for analysis in result.get("analyses") or []:
            if analysis.get("resource_id") != XCRG_RESOURCE_ID:
                continue
            analysis["score"] = score
            analysis["scoring_method"] = SCORING_METHOD


def sort_xcrg_combined_results(
    message: dict,
    source_qnode: str,
    target_qnode: str,
    logger: logging.Logger,
) -> None:
    """Sort direct results first, then TF-mediated results by the xCRG policy."""
    result_message = message.get("message") or {}
    query_graph = result_message.get("query_graph") or {}
    kg_nodes = (result_message.get("knowledge_graph") or {}).get("nodes") or {}
    answer_qnode_id = get_answer_qnode_id(query_graph, source_qnode, target_qnode)
    use_category_specificity = answer_qnode_uses_category_specificity(
        query_graph,
        answer_qnode_id,
        logger,
    )

    direct_results = []
    inferred_results = []
    for index, result in enumerate(result_message.get("results") or []):
        result["_xcrg_original_index"] = index
        if is_two_hop_result(result):
            inferred_results.append(result)
        else:
            direct_results.append(result)

    tf_degrees = Counter(
        get_bound_node_id(result, TF_QNODE_ID)
        for result in inferred_results
        if get_bound_node_id(result, TF_QNODE_ID)
    )

    def direct_key(result: dict) -> tuple:
        specificity, information_content, answer_id = get_result_answer_metrics(
            result,
            kg_nodes,
            query_graph,
            answer_qnode_id,
            use_category_specificity,
            logger,
        )
        ngd_score = get_result_endpoint_ngd(
            result,
            source_qnode,
            target_qnode,
            logger,
        )
        return (
            descending_optional(specificity),
            descending_optional(information_content),
            ascending_optional(ngd_score),
            answer_id,
            result.get("_xcrg_original_index", 0),
        )

    def inferred_key(result: dict) -> tuple:
        tf_id = get_bound_node_id(result, TF_QNODE_ID) or ""
        specificity, information_content, answer_id = get_result_answer_metrics(
            result,
            kg_nodes,
            query_graph,
            answer_qnode_id,
            use_category_specificity,
            logger,
        )
        ngd_score = get_result_answer_tf_ngd(
            result,
            answer_qnode_id,
            logger,
        )
        return (
            tf_degrees.get(tf_id, 0),
            tf_id,
            descending_optional(specificity),
            descending_optional(information_content),
            ascending_optional(ngd_score),
            answer_id,
            result.get("_xcrg_original_index", 0),
        )

    sorted_results = sorted(direct_results, key=direct_key) + sorted(
        inferred_results,
        key=inferred_key,
    )
    for result in sorted_results:
        result.pop("_xcrg_original_index", None)
    stamp_rank_scores(sorted_results)
    result_message["results"] = sorted_results


def query_qualifiers_to_edge_qualifiers(query_edge: dict) -> list[dict]:
    """Convert the first qedge qualifier set into KG edge qualifiers."""
    qualifier_constraints = query_edge.get("qualifier_constraints") or []
    if not qualifier_constraints:
        return []
    qualifier_set = qualifier_constraints[0].get("qualifier_set") or []
    return [
        {
            "qualifier_type_id": qualifier.get("qualifier_type_id"),
            "qualifier_value": qualifier.get("qualifier_value"),
        }
        for qualifier in qualifier_set
        if qualifier.get("qualifier_type_id") and qualifier.get("qualifier_value")
    ]


def make_stable_id(prefix: str, payload: dict) -> str:
    """Return a deterministic compact id for generated KG/support entries."""
    key = json.dumps(payload, sort_keys=True)
    suffix = uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:16]
    return f"{prefix}_{suffix}"


def ensure_response_versions(response: dict, *sources: dict) -> dict:
    """Add response-level TRAPI/Biolink versions when upstream omitted them."""
    schema_version = response.get("schema_version") or next(
        (source.get("schema_version") for source in sources if source.get("schema_version")),
        settings.trapi_schema_version,
    )
    biolink_version = response.get("biolink_version") or next(
        (source.get("biolink_version") for source in sources if source.get("biolink_version")),
        settings.biolink_version,
    )
    response["schema_version"] = schema_version
    response["biolink_version"] = biolink_version
    return response


def make_xcrg_inferred_edge(
    source_id: str,
    target_id: str,
    original_query_edge: dict,
    support_graph_ids: list[str],
) -> tuple[str, dict]:
    """Create the final source->target inferred edge supported by TF paths."""
    predicate = (original_query_edge.get("predicates") or ["biolink:affects"])[0]
    edge_id = make_stable_id(
        "xcrg_inferred_edge",
        {
            "source": source_id,
            "target": target_id,
            "predicate": predicate,
            "support_graphs": support_graph_ids,
        },
    )
    edge = {
        "subject": source_id,
        "predicate": predicate,
        "object": target_id,
        "qualifiers": query_qualifiers_to_edge_qualifiers(original_query_edge),
        "attributes": [
            {
                "attribute_type_id": "biolink:knowledge_level",
                "value": "prediction",
                "attribute_source": XCRG_RESOURCE_ID,
            },
            {
                "attribute_type_id": "biolink:agent_type",
                "value": "computational_model",
                "attribute_source": XCRG_RESOURCE_ID,
            },
            {
                "attribute_type_id": "biolink:support_graphs",
                "value": support_graph_ids,
                "attribute_source": XCRG_RESOURCE_ID,
            },
        ],
        "sources": [
            {
                "resource_id": XCRG_RESOURCE_ID,
                "resource_role": "primary_knowledge_source",
            }
        ],
    }
    if not edge["qualifiers"]:
        edge.pop("qualifiers")
    return edge_id, edge


def make_xcrg_ngd_edge(
    source_id: str,
    target_id: str,
    ngd_score: float | str,
) -> tuple[str, dict]:
    """Create an ARAX-style virtual NGD edge for analysis support graphs."""
    edge_id = make_stable_id(
        "xcrg_ngd_edge",
        {
            "source": source_id,
            "target": target_id,
            "ngd": ngd_score,
        },
    )
    edge = {
        "subject": source_id,
        "predicate": "biolink:occurs_together_in_literature_with",
        "object": target_id,
        "attributes": [
            {
                "attribute_source": XCRG_RESOURCE_ID,
                "attribute_type_id": "EDAM-DATA:2526",
                "description": NGD_DESCRIPTION,
                "original_attribute_name": "normalized_google_distance",
                "value_url": NGD_VALUE_URL,
                "value": ngd_score,
            },
            {
                "attribute_source": XCRG_RESOURCE_ID,
                "attribute_type_id": "EDAM-OPERATION:0226",
                "original_attribute_name": "virtual_relation_label",
                "value": "N1",
            },
            {
                "attribute_source": XCRG_RESOURCE_ID,
                "attribute_type_id": "biolink:creation_date",
                "original_attribute_name": "defined_datetime",
                "value": datetime.now(timezone.utc).isoformat(),
            },
            {
                "attribute_source": XCRG_RESOURCE_ID,
                "attribute_type_id": "EDAM-DATA:1772",
                "description": COMPUTED_EDGE_CONTAINER_DESCRIPTION,
                "value_type_id": "metatype:Boolean",
                "value": True,
            },
            {
                "attribute_source": XCRG_RESOURCE_ID,
                "attribute_type_id": "biolink:knowledge_level",
                "value": "statistical_association",
            },
            {
                "attribute_source": XCRG_RESOURCE_ID,
                "attribute_type_id": "biolink:agent_type",
                "value": "automated_agent",
            },
        ],
        "sources": [
            {
                "resource_id": XCRG_RESOURCE_ID,
                "resource_role": "primary_knowledge_source",
            }
        ],
    }
    return edge_id, edge


def add_ngd_analysis_support_graph(
    analysis: dict,
    kg_edges: dict,
    auxiliary_graphs: dict,
    source_id: str,
    target_id: str,
    logger: logging.Logger,
) -> None:
    """Attach a virtual NGD edge as analysis-level support.

    ARAX/xDTD keeps the analysis support graph even when no NGD is available,
    displaying the NGD value as "inf". Keep that as a string to avoid emitting
    non-standard JSON numeric Infinity.
    """
    ngd_score = get_ngd_score(source_id, target_id, logger)
    ngd_value: float | str = ngd_score if ngd_score is not None else "inf"

    ngd_edge_id, ngd_edge = make_xcrg_ngd_edge(
        source_id,
        target_id,
        ngd_value,
    )
    kg_edges[ngd_edge_id] = ngd_edge
    support_graph_id = make_stable_id(
        "xcrg_ngd_support",
        {
            "source": source_id,
            "target": target_id,
            "edge": ngd_edge_id,
        },
    )
    auxiliary_graphs[support_graph_id] = {
        "edges": [ngd_edge_id],
    }
    support_graphs = analysis.setdefault("support_graphs", [])
    if support_graph_id not in support_graphs:
        support_graphs.append(support_graph_id)


def get_or_create_final_result(
    final_results_by_pair: dict,
    final_results: list[dict],
    source_qnode: str,
    target_qnode: str,
    source_id: str,
    target_id: str,
) -> dict:
    """Return the final one-hop result for a source/target answer pair."""
    pair_key = (source_id, target_id)
    if pair_key not in final_results_by_pair:
        final_results_by_pair[pair_key] = {
            "node_bindings": {
                source_qnode: [{"id": source_id, "attributes": []}],
                target_qnode: [{"id": target_id, "attributes": []}],
            },
            "analyses": [],
            "_xcrg_direct_bindings": [],
            "_xcrg_direct_binding_ids": set(),
            "_xcrg_support_edges": [],
            "_xcrg_support_edge_ids": set(),
            "_xcrg_first_score": None,
            "_xcrg_first_index": len(final_results),
        }
        final_results.append(final_results_by_pair[pair_key])
    return final_results_by_pair[pair_key]


def add_direct_evidence(final_result: dict, bindings: list[dict]) -> None:
    """Attach direct one-hop KG edge bindings to a final result."""
    for binding in bindings:
        edge_id = binding.get("id")
        if edge_id in final_result["_xcrg_direct_binding_ids"]:
            continue
        final_result["_xcrg_direct_binding_ids"].add(edge_id)
        final_result["_xcrg_direct_bindings"].append(binding)


def add_support_path_edges(
    final_result: dict,
    path_edge_ids: list[str],
) -> None:
    """Collect unique TF-mediated path edges for the final predicted edge."""
    for edge_id in path_edge_ids:
        if edge_id in final_result["_xcrg_support_edge_ids"]:
            continue
        final_result["_xcrg_support_edge_ids"].add(edge_id)
        final_result["_xcrg_support_edges"].append(edge_id)


def finalize_clean_result_analyses(
    final_result: dict,
    kg_edges: dict,
    auxiliary_graphs: dict,
    original_qedge_id: str,
    original_query_edge: dict,
    logger: logging.Logger,
) -> None:
    """Build final Retriever/xCRG analyses after evidence grouping."""
    direct_bindings = final_result.pop("_xcrg_direct_bindings")
    final_result.pop("_xcrg_direct_binding_ids")
    support_edges = final_result.pop("_xcrg_support_edges")
    final_result.pop("_xcrg_support_edge_ids")
    source_id = final_result["node_bindings"][original_query_edge["subject"]][0]["id"]
    target_id = final_result["node_bindings"][original_query_edge["object"]][0]["id"]

    xcrg_bindings = deepcopy(direct_bindings)
    if support_edges:
        support_graph_id = make_stable_id(
            "xcrg_support",
            {
                "source": source_id,
                "target": target_id,
                "edges": support_edges,
            },
        )
        auxiliary_graphs[support_graph_id] = {
            "edges": support_edges,
        }
        inferred_edge_id, inferred_edge = make_xcrg_inferred_edge(
            source_id,
            target_id,
            original_query_edge,
            [support_graph_id],
        )
        kg_edges[inferred_edge_id] = inferred_edge
        xcrg_bindings.append({"id": inferred_edge_id, "attributes": []})

    if xcrg_bindings:
        analysis = {
            "resource_id": XCRG_RESOURCE_ID,
            "edge_bindings": {
                original_qedge_id: xcrg_bindings,
            },
        }
        add_ngd_analysis_support_graph(
            analysis,
            kg_edges,
            auxiliary_graphs,
            source_id,
            target_id,
            logger,
        )
        final_result["analyses"] = [analysis]
    else:
        final_result["analyses"] = []


def build_trapi_clean_response(
    original_message: dict,
    combined_message: dict,
    source_qnode: str,
    target_qnode: str,
    logger: logging.Logger | None = None,
) -> dict:
    """Convert debug-shaped direct+2-hop results into one-hop TRAPI results."""
    logger = logger or logging.getLogger(__name__)
    original_qedge_id, original_query_edge = get_single_query_edge(original_message)
    combined = combined_message.get("message") or {}
    combined_kg = combined.get("knowledge_graph") or {}

    final_message = {
        "message": {
            "query_graph": deepcopy(original_message["message"]["query_graph"]),
            "knowledge_graph": {
                "nodes": deepcopy(combined_kg.get("nodes") or {}),
                "edges": deepcopy(combined_kg.get("edges") or {}),
            },
            "results": [],
        }
    }
    auxiliary_graphs = {}
    final_results_by_pair = {}
    final_results = final_message["message"]["results"]

    for result_index, result in enumerate(combined.get("results") or []):
        source_id = get_bound_node_id(result, source_qnode)
        target_id = get_bound_node_id(result, target_qnode)
        if not source_id or not target_id:
            continue

        final_result = get_or_create_final_result(
            final_results_by_pair,
            final_results,
            source_qnode,
            target_qnode,
            source_id,
            target_id,
        )
        if final_result["_xcrg_first_score"] is None:
            final_result["_xcrg_first_score"] = get_result_score(result)
            final_result["_xcrg_first_index"] = result_index

        if is_two_hop_result(result):
            path_edge_ids = [
                binding.get("id")
                for qedge_id in ("e0", "e1")
                for binding in get_edge_bindings(result, qedge_id)
                if binding.get("id")
            ]
            if path_edge_ids:
                add_support_path_edges(
                    final_result,
                    path_edge_ids,
                )
        else:
            add_direct_evidence(
                final_result,
                get_edge_bindings(result, DIRECT_QEDGE_ID),
            )

    for final_result in final_results:
        finalize_clean_result_analyses(
            final_result,
            final_message["message"]["knowledge_graph"]["edges"],
            auxiliary_graphs,
            original_qedge_id,
            original_query_edge,
            logger,
        )

    final_results.sort(
        key=lambda result: (
            descending_optional(result.get("_xcrg_first_score")),
            result.get("_xcrg_first_index", 0),
        )
    )
    for final_result in final_results:
        final_result.pop("_xcrg_first_score", None)
        final_result.pop("_xcrg_first_index", None)
    stamp_xcrg_rank_scores(final_results)

    if auxiliary_graphs:
        final_message["message"]["auxiliary_graphs"] = auxiliary_graphs
    return ensure_response_versions(final_message, original_message, combined_message)


def write_debug_manifest(debug_context: dict, logger: logging.Logger) -> None:
    """Write or refresh the human-readable debug manifest for one query."""
    try:
        manifest = {
            key: value
            for key, value in debug_context.items()
            if key not in {"run_dir"}
        }
        manifest["run_dir"] = str(debug_context["run_dir"])
        manifest_path = debug_context["run_dir"] / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
    except Exception as exc:
        logger.warning(f"Failed to write xCRG debug manifest: {exc}")


def debug_dump_json(
    query_id: str,
    label: str,
    payload: dict,
    logger: logging.Logger,
    debug_context: dict | None = None,
) -> None:
    """Best-effort debug JSON dump for inferred xCRG runs."""
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        legacy_path = DEBUG_DIR / f"{query_id}_{label}.json"
        with open(legacy_path, "w", encoding="utf-8") as debug_file:
            json.dump(payload, debug_file, indent=2, sort_keys=True)

        if debug_context:
            debug_context["run_dir"].mkdir(parents=True, exist_ok=True)
            readable_path = debug_context["run_dir"] / f"{label}.json"
            with open(readable_path, "w", encoding="utf-8") as debug_file:
                json.dump(payload, debug_file, indent=2, sort_keys=True)
            debug_context["artifacts"].append(
                {
                    "label": label,
                    "legacy_path": str(legacy_path),
                    "path": str(readable_path),
                    "written_at": datetime.now(timezone.utc).isoformat(),
                    "summary": summarize_response_counts(payload),
                }
            )
            write_debug_manifest(debug_context, logger)
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
    return ensure_response_versions({"message": filtered_message}, response)


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
    debug_context = make_debug_run_context(query_id, message)
    debug_dump_json(
        query_id,
        "original_inferred_query",
        message,
        logger,
        debug_context,
    )
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

    direct_message = build_direct_query_for_inferred(
        message,
        source_qnode,
        target_qnode,
    )
    debug_dump_json(
        query_id,
        "direct_lookup_query",
        direct_message,
        logger,
        debug_context,
    )
    direct_response = await run_sync_retriever_lookup(direct_message, logger)
    debug_dump_json(
        query_id,
        "direct_raw_response",
        direct_response,
        logger,
        debug_context,
    )
    filtered_direct_response = filter_direct_response(
        direct_response,
        source_qnode,
        target_qnode,
    )
    debug_dump_json(
        query_id,
        "direct_filtered_response",
        filtered_direct_response,
        logger,
        debug_context,
    )

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
        "direct_response": summarize_response_counts(filtered_direct_response),
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
                inferred_message["submitter"] = XCRG_RESOURCE_ID
            debug_dump_json(
                query_id,
                f"template_{template_idx}_batch_{batch_idx}_query",
                inferred_message,
                logger,
                debug_context,
            )
            response = await run_sync_retriever_lookup(inferred_message, logger)
            debug_dump_json(
                query_id,
                f"template_{template_idx}_batch_{batch_idx}_raw_response",
                response,
                logger,
                debug_context,
            )
            filtered_response = filter_inferred_response(
                response, source_qnode, target_qnode
            )
            debug_dump_json(
                query_id,
                f"template_{template_idx}_batch_{batch_idx}_filtered_response",
                filtered_response,
                logger,
                debug_context,
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

    merged_query_graph = build_combined_query_graph(
        message,
        source_qnode,
        target_qnode,
        tf_list,
    )

    merged_inferred = merge_filtered_responses(
        filtered_responses,
        build_two_hop_query(
            message,
            source_qnode,
            target_qnode,
            tf_list,
            sign_templates[0][0],
            sign_templates[0][1],
        )["message"]["query_graph"],
    )
    merged = merge_filtered_responses(
        [filtered_direct_response, merged_inferred],
        merged_query_graph,
    )
    sort_xcrg_combined_results(merged, source_qnode, target_qnode, logger)
    debug_dump_json(
        query_id,
        "merged_debug_response",
        merged,
        logger,
        debug_context,
    )
    final_response = build_trapi_clean_response(
        message,
        merged,
        source_qnode,
        target_qnode,
        logger,
    )
    debug_summary["merged_response"] = summarize_response_counts(final_response)
    debug_summary["debug_run_dir"] = str(debug_context["run_dir"])
    debug_dump_json(
        query_id,
        "inferred_debug_summary",
        debug_summary,
        logger,
        debug_context,
    )
    debug_dump_json(
        query_id,
        "merged_inferred_response",
        final_response,
        logger,
        debug_context,
    )
    return final_response


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
        message["submitter"] = XCRG_RESOURCE_ID

    _, edge = get_single_query_edge(message)
    if edge.get("knowledge_type") == "inferred":
        result = await run_inferred_lookup(query_id, message, logger)
    else:
        result = await run_direct_lookup(message, logger)

    ensure_response_versions(result, message)
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
