import logging
from copy import deepcopy

import pytest

from workers.xcrg.worker import xcrg
from workers.xcrg_lookup import worker as xcrg_lookup_worker
from workers.xcrg_lookup.worker import xcrg_lookup


@pytest.mark.asyncio
async def test_xcrg_sets_lookup_workflow(redis_mock):
    logger = logging.getLogger(__name__)
    task = [
        "test",
        {
            "query_id": "test",
            "response_id": "test_response",
            "log_level": "20",
            "otel": "{}",
            "workflow": "null",
        },
    ]

    await xcrg(task, logger)

    assert task[1]["workflow"] == '[{"id": "xcrg.lookup"}]'


@pytest.mark.asyncio
async def test_xcrg_lookup_dispatches_callback(mocker, redis_mock):
    mocker.patch(
        "workers.xcrg_lookup.worker.get_message",
        return_value={
            "message": {
                "query_graph": {
                    "nodes": {
                        "sn": {"categories": ["biolink:ChemicalEntity"]},
                        "on": {
                            "categories": ["biolink:Gene"],
                            "ids": ["NCBIGene:51341"],
                        },
                    },
                    "edges": {
                        "t_edge": {
                            "subject": "sn",
                            "object": "on",
                            "predicates": ["biolink:affects"],
                        }
                    },
                }
            },
            "parameters": {},
        },
    )
    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "message": {
            "knowledge_graph": {"nodes": {}, "edges": {}},
            "results": [],
        }
    }
    mock_post = mocker.patch("httpx.AsyncClient.post", return_value=mock_response)
    mock_save = mocker.patch("workers.xcrg_lookup.worker.save_message")
    logger = logging.getLogger(__name__)

    await xcrg_lookup(
        [
            "test",
            {
                "query_id": "test",
                "response_id": "test_response",
                "workflow": '[{"id": "xcrg.lookup"}]',
                "log_level": "20",
                "otel": "{}",
            },
        ],
        logger,
    )

    assert mock_post.called
    assert mock_save.called


@pytest.mark.asyncio
async def test_xcrg_inferred_lookup_expands_to_tf_queries(
    mocker, monkeypatch, redis_mock
):
    mocker.patch.object(xcrg_lookup_worker, "get_ngd_score", return_value=None)
    mocker.patch(
        "workers.xcrg_lookup.worker.get_message",
        return_value={
            "message": {
                "query_graph": {
                    "nodes": {
                        "sn": {"categories": ["biolink:ChemicalEntity"]},
                        "on": {
                            "categories": ["biolink:Gene"],
                            "ids": ["NCBIGene:51341"],
                        },
                    },
                    "edges": {
                        "t_edge": {
                            "subject": "sn",
                            "object": "on",
                            "predicates": ["biolink:affects"],
                            "knowledge_type": "inferred",
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {
                                            "qualifier_type_id": "biolink:object_aspect_qualifier",
                                            "qualifier_value": "activity_or_abundance",
                                        },
                                        {
                                            "qualifier_type_id": "biolink:object_direction_qualifier",
                                            "qualifier_value": "increased",
                                        },
                                    ]
                                }
                            ],
                        }
                    },
                }
            },
            "parameters": {},
        },
    )
    mocker.patch.object(
        xcrg_lookup_worker,
        "load_tf_list",
        return_value=[
            "NCBIGene:4066",
            "NCBIGene:1105",
            "NCBIGene:9324",
            "NCBIGene:7157",
            "NCBIGene:51341",
        ],
    )
    mocker.patch.object(xcrg_lookup_worker, "debug_dump_json")
    monkeypatch.setattr(xcrg_lookup_worker.settings, "xcrg_tf_batch_size", 2)

    direct_payload = {
        "message": {
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:2": {
                        "categories": ["biolink:Drug"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 10,
                            }
                        ],
                    },
                    "NCBIGene:51341": {"categories": ["biolink:Gene"]},
                },
                "edges": {
                    "direct_edge": {
                        "subject": "CHEBI:2",
                        "object": "NCBIGene:51341",
                        "predicate": "biolink:affects",
                    },
                },
            },
            "results": [
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:2"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "direct": [{"id": "direct_edge"}],
                            },
                        }
                    ],
                }
            ],
        }
    }
    inferred_payload = {
        "message": {
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:1": {
                        "categories": ["biolink:ChemicalEntity"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 95,
                            }
                        ],
                    },
                    "NCBIGene:4066": {"categories": ["biolink:Gene"]},
                    "NCBIGene:51341": {"categories": ["biolink:Gene"]},
                },
                "edges": {
                    "edge0": {
                        "subject": "CHEBI:1",
                        "object": "NCBIGene:4066",
                        "predicate": "biolink:affects",
                    },
                    "edge1": {
                        "subject": "NCBIGene:4066",
                        "object": "NCBIGene:51341",
                        "predicate": "biolink:affects",
                    },
                },
            },
            "results": [
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:1"}],
                        "tf": [{"id": "NCBIGene:4066"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "edge_bindings": {
                                "e0": [{"id": "edge0"}],
                                "e1": [{"id": "edge1"}],
                            }
                        }
                    ],
                }
            ],
        }
    }

    def make_response(payload):
        response = mocker.Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = lambda: deepcopy(payload)
        return response

    def post_side_effect(*args, **kwargs):
        qgraph = kwargs["json"]["message"]["query_graph"]
        if "direct" in qgraph["edges"]:
            return make_response(direct_payload)
        return make_response(inferred_payload)

    mock_post = mocker.patch("httpx.AsyncClient.post", side_effect=post_side_effect)
    mock_save = mocker.patch("workers.xcrg_lookup.worker.save_message")
    logger = logging.getLogger(__name__)

    await xcrg_lookup(
        [
            "test",
            {
                "query_id": "test",
                "response_id": "test_response",
                "workflow": '[{"id": "xcrg.lookup"}]',
                "log_level": "20",
                "otel": "{}",
            },
        ],
        logger,
    )

    assert mock_post.call_count == 5
    direct_payload_call = mock_post.call_args_list[0].kwargs["json"]
    assert list(direct_payload_call["message"]["query_graph"]["edges"]) == ["direct"]
    assert (
        "knowledge_type"
        not in direct_payload_call["message"]["query_graph"]["edges"]["direct"]
    )

    seen_tf_ids = set()
    inferred_calls = mock_post.call_args_list[1:]
    assert len(inferred_calls) == 4
    for call in inferred_calls:
        payload = call.kwargs["json"]
        tf_ids = payload["message"]["query_graph"]["nodes"]["tf"]["ids"]
        assert len(tf_ids) <= 2
        assert "NCBIGene:7157" not in tf_ids
        assert "NCBIGene:51341" not in tf_ids
        seen_tf_ids.update(tf_ids)

    saved_response = mock_save.call_args.args[1]
    result_edges = [
        set(result["analyses"][0]["edge_bindings"])
        for result in saved_response["message"]["results"]
    ]
    assert result_edges[0] == {"t_edge"}
    assert result_edges[1] == {"t_edge"}
    assert (
        saved_response["message"]["results"][0]["analyses"][0]["score"]
        > saved_response["message"]["results"][1]["analyses"][0]["score"]
    )
    assert list(saved_response["message"]["query_graph"]["edges"]) == ["t_edge"]
    assert "tf" not in saved_response["message"]["query_graph"]["nodes"]
    inferred_analysis = saved_response["message"]["results"][1]["analyses"][0]
    assert inferred_analysis["resource_id"] == "infores:arax"
    assert inferred_analysis["support_graphs"][0].startswith("xcrg_ngd_support_")
    assert saved_response["message"]["auxiliary_graphs"]
    inferred_edge_ids = [
        binding["id"]
        for binding in inferred_analysis["edge_bindings"]["t_edge"]
        if binding["id"].startswith("xcrg_inferred_edge_")
    ]
    assert inferred_edge_ids
    assert seen_tf_ids == {"NCBIGene:4066", "NCBIGene:1105", "NCBIGene:9324"}
    debug_labels = [call.args[1] for call in xcrg_lookup_worker.debug_dump_json.call_args_list]
    assert "merged_debug_response" in debug_labels
    assert "inferred_debug_summary" in debug_labels
    assert mock_save.called


@pytest.mark.asyncio
async def test_xcrg_inferred_lookup_preserves_user_direction(
    mocker, monkeypatch, redis_mock
):
    mocker.patch(
        "workers.xcrg_lookup.worker.get_message",
        return_value={
            "message": {
                "query_graph": {
                    "nodes": {
                        "gene_q": {"categories": ["biolink:Gene"]},
                        "chem_q": {
                            "categories": ["biolink:ChemicalEntity"],
                            "ids": ["CHEBI:123"],
                        },
                    },
                    "edges": {
                        "t_edge": {
                            "subject": "gene_q",
                            "object": "chem_q",
                            "predicates": ["biolink:affects"],
                            "knowledge_type": "inferred",
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {
                                            "qualifier_type_id": "biolink:object_aspect_qualifier",
                                            "qualifier_value": "activity_or_abundance",
                                        },
                                        {
                                            "qualifier_type_id": "biolink:object_direction_qualifier",
                                            "qualifier_value": "decreased",
                                        },
                                    ]
                                }
                            ],
                        }
                    },
                }
            },
            "parameters": {},
        },
    )
    mocker.patch.object(
        xcrg_lookup_worker,
        "load_tf_list",
        return_value=["NCBIGene:4066", "NCBIGene:7157", "CHEBI:123"],
    )
    mocker.patch.object(xcrg_lookup_worker, "debug_dump_json")
    monkeypatch.setattr(xcrg_lookup_worker.settings, "xcrg_tf_batch_size", 10)

    direct_payload = {
        "message": {
            "knowledge_graph": {
                "nodes": {
                    "NCBIGene:999": {"categories": ["biolink:Gene"]},
                    "CHEBI:123": {"categories": ["biolink:ChemicalEntity"]},
                },
                "edges": {
                    "direct_edge": {
                        "subject": "NCBIGene:999",
                        "object": "CHEBI:123",
                        "predicate": "biolink:affects",
                    },
                },
            },
            "results": [
                {
                    "node_bindings": {
                        "gene_q": [{"id": "NCBIGene:999"}],
                        "chem_q": [{"id": "CHEBI:123"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "direct": [{"id": "direct_edge"}],
                            },
                        }
                    ],
                }
            ],
        }
    }
    inferred_payload = {
        "message": {
            "knowledge_graph": {
                "nodes": {
                    "NCBIGene:4066": {"categories": ["biolink:Gene"]},
                    "NCBIGene:999": {"categories": ["biolink:Gene"]},
                    "CHEBI:123": {"categories": ["biolink:ChemicalEntity"]},
                },
                "edges": {
                    "edge0": {
                        "subject": "NCBIGene:999",
                        "object": "NCBIGene:4066",
                        "predicate": "biolink:affects",
                    },
                    "edge1": {
                        "subject": "NCBIGene:4066",
                        "object": "CHEBI:123",
                        "predicate": "biolink:affects",
                    },
                },
            },
            "results": [
                {
                    "node_bindings": {
                        "gene_q": [{"id": "NCBIGene:999"}],
                        "tf": [{"id": "NCBIGene:4066"}],
                        "chem_q": [{"id": "CHEBI:123"}],
                    },
                    "analyses": [
                        {
                            "edge_bindings": {
                                "e0": [{"id": "edge0"}],
                                "e1": [{"id": "edge1"}],
                            }
                        }
                    ],
                }
            ],
        }
    }

    def make_response(payload):
        response = mocker.Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = lambda: deepcopy(payload)
        return response

    def post_side_effect(*args, **kwargs):
        qgraph = kwargs["json"]["message"]["query_graph"]
        if "direct" in qgraph["edges"]:
            return make_response(direct_payload)
        return make_response(inferred_payload)

    mock_post = mocker.patch("httpx.AsyncClient.post", side_effect=post_side_effect)
    mock_save = mocker.patch("workers.xcrg_lookup.worker.save_message")
    logger = logging.getLogger(__name__)

    await xcrg_lookup(
        [
            "test",
            {
                "query_id": "test",
                "response_id": "test_response",
                "workflow": '[{"id": "xcrg.lookup"}]',
                "log_level": "20",
                "otel": "{}",
            },
        ],
        logger,
    )

    assert mock_post.call_count == 3
    direct_qgraph = mock_post.call_args_list[0].kwargs["json"]["message"]["query_graph"]
    assert direct_qgraph["edges"]["direct"]["subject"] == "gene_q"
    assert direct_qgraph["edges"]["direct"]["object"] == "chem_q"

    for call in mock_post.call_args_list[1:]:
        payload = call.kwargs["json"]
        qgraph = payload["message"]["query_graph"]
        assert qgraph["edges"]["e0"]["subject"] == "gene_q"
        assert qgraph["edges"]["e0"]["object"] == "tf"
        assert qgraph["edges"]["e1"]["subject"] == "tf"
        assert qgraph["edges"]["e1"]["object"] == "chem_q"
        assert qgraph["nodes"]["tf"]["ids"] == ["NCBIGene:4066"]

    saved_response = mock_save.call_args.args[1]
    assert list(saved_response["message"]["query_graph"]["edges"]) == ["t_edge"]
    assert saved_response["message"]["query_graph"]["edges"]["t_edge"]["subject"] == "gene_q"
    assert saved_response["message"]["query_graph"]["edges"]["t_edge"]["object"] == "chem_q"
    assert "tf" not in saved_response["message"]["query_graph"]["nodes"]


def test_xcrg_combined_sort_orders_direct_then_inferred_by_policy():
    logger = logging.getLogger(__name__)
    message = {
        "message": {
            "query_graph": {
                "nodes": {
                    "sn": {"categories": ["biolink:ChemicalEntity"]},
                    "tf": {"categories": ["biolink:Gene"]},
                    "on": {
                        "categories": ["biolink:Gene"],
                        "ids": ["NCBIGene:51341"],
                    },
                },
                "edges": {
                    "direct": {
                        "subject": "sn",
                        "object": "on",
                        "predicates": ["biolink:affects"],
                    },
                    "e0": {
                        "subject": "sn",
                        "object": "tf",
                        "predicates": ["biolink:affects"],
                    },
                    "e1": {
                        "subject": "tf",
                        "object": "on",
                        "predicates": ["biolink:affects"],
                    },
                },
            },
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:drug_direct": {
                        "categories": ["biolink:Drug"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 10,
                            }
                        ],
                    },
                    "CHEBI:chemical_direct": {
                        "categories": ["biolink:ChemicalEntity"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 99,
                            }
                        ],
                    },
                    "CHEBI:rare_tf": {
                        "categories": ["biolink:ChemicalEntity"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 1,
                            }
                        ],
                    },
                    "CHEBI:hub_drug": {
                        "categories": ["biolink:Drug"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 5,
                            }
                        ],
                    },
                    "CHEBI:hub_chemical": {
                        "categories": ["biolink:ChemicalEntity"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 100,
                            }
                        ],
                    },
                    "NCBIGene:rare_tf": {"categories": ["biolink:Gene"]},
                    "NCBIGene:hub_tf": {"categories": ["biolink:Gene"]},
                    "NCBIGene:51341": {"categories": ["biolink:Gene"]},
                },
                "edges": {},
            },
            "results": [
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:hub_chemical"}],
                        "tf": [{"id": "NCBIGene:hub_tf"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "e0": [{"id": "hub_e0a"}],
                                "e1": [{"id": "hub_e1"}],
                            },
                        }
                    ],
                },
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:chemical_direct"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "direct": [{"id": "direct_chemical"}],
                            },
                        }
                    ],
                },
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:rare_tf"}],
                        "tf": [{"id": "NCBIGene:rare_tf"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "e0": [{"id": "rare_e0"}],
                                "e1": [{"id": "rare_e1"}],
                            },
                        }
                    ],
                },
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:drug_direct"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "direct": [{"id": "direct_drug"}],
                            },
                        }
                    ],
                },
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:hub_drug"}],
                        "tf": [{"id": "NCBIGene:hub_tf"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "e0": [{"id": "hub_e0b"}],
                                "e1": [{"id": "hub_e1"}],
                            },
                        }
                    ],
                },
            ],
        }
    }

    xcrg_lookup_worker.sort_xcrg_combined_results(message, "sn", "on", logger)

    answer_order = [
        result["node_bindings"]["sn"][0]["id"]
        for result in message["message"]["results"]
    ]
    assert answer_order == [
        "CHEBI:drug_direct",
        "CHEBI:chemical_direct",
        "CHEBI:rare_tf",
        "CHEBI:hub_drug",
        "CHEBI:hub_chemical",
    ]
    scores = [
        result["analyses"][0]["score"]
        for result in message["message"]["results"]
    ]
    assert scores == sorted(scores, reverse=True)


def test_xcrg_combined_sort_uses_ngd_as_final_tie_breaker(mocker):
    logger = logging.getLogger(__name__)
    ngd_scores = {
        frozenset(("CHEBI:direct_near", "NCBIGene:pinned")): 0.2,
        frozenset(("CHEBI:direct_far", "NCBIGene:pinned")): 0.8,
        frozenset(("CHEBI:tf_near", "NCBIGene:tf")): 0.1,
        frozenset(("CHEBI:tf_far", "NCBIGene:tf")): 0.9,
    }
    mocker.patch.object(
        xcrg_lookup_worker,
        "get_ngd_score",
        side_effect=lambda curie_a, curie_b, logger: ngd_scores.get(
            frozenset((curie_a, curie_b))
        ),
    )
    message = {
        "message": {
            "query_graph": {
                "nodes": {
                    "sn": {"categories": ["biolink:ChemicalEntity"]},
                    "tf": {"categories": ["biolink:Gene"]},
                    "on": {
                        "categories": ["biolink:Gene"],
                        "ids": ["NCBIGene:pinned"],
                    },
                },
                "edges": {
                    "direct": {
                        "subject": "sn",
                        "object": "on",
                        "predicates": ["biolink:affects"],
                    },
                    "e0": {
                        "subject": "sn",
                        "object": "tf",
                        "predicates": ["biolink:affects"],
                    },
                    "e1": {
                        "subject": "tf",
                        "object": "on",
                        "predicates": ["biolink:affects"],
                    },
                },
            },
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:direct_far": {
                        "categories": ["biolink:ChemicalEntity"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 50,
                            }
                        ],
                    },
                    "CHEBI:direct_near": {
                        "categories": ["biolink:ChemicalEntity"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 50,
                            }
                        ],
                    },
                    "CHEBI:tf_far": {
                        "categories": ["biolink:ChemicalEntity"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 50,
                            }
                        ],
                    },
                    "CHEBI:tf_near": {
                        "categories": ["biolink:ChemicalEntity"],
                        "attributes": [
                            {
                                "attribute_type_id": "biolink:information_content",
                                "value": 50,
                            }
                        ],
                    },
                    "NCBIGene:pinned": {"categories": ["biolink:Gene"]},
                    "NCBIGene:tf": {"categories": ["biolink:Gene"]},
                },
                "edges": {},
            },
            "results": [
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:direct_far"}],
                        "on": [{"id": "NCBIGene:pinned"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {"direct": [{"id": "direct_far"}]},
                        }
                    ],
                },
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:direct_near"}],
                        "on": [{"id": "NCBIGene:pinned"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {"direct": [{"id": "direct_near"}]},
                        }
                    ],
                },
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:tf_far"}],
                        "tf": [{"id": "NCBIGene:tf"}],
                        "on": [{"id": "NCBIGene:pinned"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "e0": [{"id": "tf_far_e0"}],
                                "e1": [{"id": "tf_far_e1"}],
                            },
                        }
                    ],
                },
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:tf_near"}],
                        "tf": [{"id": "NCBIGene:tf"}],
                        "on": [{"id": "NCBIGene:pinned"}],
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:retriever",
                            "edge_bindings": {
                                "e0": [{"id": "tf_near_e0"}],
                                "e1": [{"id": "tf_near_e1"}],
                            },
                        }
                    ],
                },
            ],
        }
    }

    xcrg_lookup_worker.sort_xcrg_combined_results(message, "sn", "on", logger)

    answer_order = [
        result["node_bindings"]["sn"][0]["id"]
        for result in message["message"]["results"]
    ]
    assert answer_order == [
        "CHEBI:direct_near",
        "CHEBI:direct_far",
        "CHEBI:tf_near",
        "CHEBI:tf_far",
    ]


def test_xcrg_ngd_ignores_invalid_scores(mocker):
    logger = logging.getLogger(__name__)
    xcrg_lookup_worker._NGD_NEIGHBOR_CACHE.clear()

    mock_connection = mocker.Mock()
    mock_connection.execute.return_value.fetchone.return_value = (
        '[["CURIE:good", 0.25], ["CURIE:zero", 0], ["CURIE:neg", -1], '
        '["CURIE:inf", 1e999], ["CURIE:text", "bad"]]',
    )
    mocker.patch.object(
        xcrg_lookup_worker,
        "get_ngd_connection",
        return_value=mock_connection,
    )

    neighbors = xcrg_lookup_worker.get_ngd_neighbors("CURIE:test", logger)
    assert neighbors == {"CURIE:good": 0.25}
    assert xcrg_lookup_worker.get_ngd_score("CURIE:test", "CURIE:good", logger) == 0.25
    assert xcrg_lookup_worker.get_ngd_score("CURIE:test", "CURIE:zero", logger) is None
    assert xcrg_lookup_worker.get_ngd_score("CURIE:test", "CURIE:inf", logger) is None
    assert xcrg_lookup_worker.get_ngd_score("CURIE:test", "CURIE:test", logger) is None


def test_xcrg_clean_response_groups_multiple_tf_paths_as_support_graphs(mocker):
    mocker.patch.object(xcrg_lookup_worker, "get_ngd_score", return_value=0.42)
    original_message = {
        "message": {
            "query_graph": {
                "nodes": {
                    "sn": {"categories": ["biolink:ChemicalEntity"]},
                    "on": {
                        "categories": ["biolink:Gene"],
                        "ids": ["NCBIGene:51341"],
                    },
                },
                "edges": {
                    "t_edge": {
                        "subject": "sn",
                        "object": "on",
                        "predicates": ["biolink:affects"],
                        "knowledge_type": "inferred",
                    }
                },
            }
        }
    }
    combined_message = {
        "message": {
            "query_graph": {
                "nodes": {
                    "sn": {"categories": ["biolink:ChemicalEntity"]},
                    "tf": {"categories": ["biolink:Gene"]},
                    "on": {
                        "categories": ["biolink:Gene"],
                        "ids": ["NCBIGene:51341"],
                    },
                },
                "edges": {
                    "direct": {},
                    "e0": {},
                    "e1": {},
                },
            },
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:1": {"categories": ["biolink:ChemicalEntity"]},
                    "NCBIGene:TF1": {"categories": ["biolink:Gene"]},
                    "NCBIGene:TF2": {"categories": ["biolink:Gene"]},
                    "NCBIGene:51341": {"categories": ["biolink:Gene"]},
                },
                "edges": {
                    "edge0a": {
                        "subject": "CHEBI:1",
                        "predicate": "biolink:affects",
                        "object": "NCBIGene:TF1",
                    },
                    "edge1a": {
                        "subject": "NCBIGene:TF1",
                        "predicate": "biolink:affects",
                        "object": "NCBIGene:51341",
                    },
                    "edge0b": {
                        "subject": "CHEBI:1",
                        "predicate": "biolink:affects",
                        "object": "NCBIGene:TF2",
                    },
                    "edge1b": {
                        "subject": "NCBIGene:TF2",
                        "predicate": "biolink:affects",
                        "object": "NCBIGene:51341",
                    },
                },
            },
            "results": [
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:1"}],
                        "tf": [{"id": "NCBIGene:TF1"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "score": 1.0,
                            "edge_bindings": {
                                "e0": [{"id": "edge0a"}],
                                "e1": [{"id": "edge1a"}],
                            },
                        }
                    ],
                },
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:1"}],
                        "tf": [{"id": "NCBIGene:TF2"}],
                        "on": [{"id": "NCBIGene:51341"}],
                    },
                    "analyses": [
                        {
                            "score": 0.5,
                            "edge_bindings": {
                                "e0": [{"id": "edge0b"}],
                                "e1": [{"id": "edge1b"}],
                            },
                        }
                    ],
                },
            ],
        }
    }

    clean_response = xcrg_lookup_worker.build_trapi_clean_response(
        original_message,
        combined_message,
        "sn",
        "on",
    )

    message = clean_response["message"]
    assert list(message["query_graph"]["edges"]) == ["t_edge"]
    assert "tf" not in message["query_graph"]["nodes"]
    assert len(message["results"]) == 1
    analysis = message["results"][0]["analyses"][0]
    assert analysis["resource_id"] == "infores:arax"
    assert set(analysis["edge_bindings"]) == {"t_edge"}
    assert len(analysis["support_graphs"]) == 1
    assert analysis["support_graphs"][0].startswith("xcrg_ngd_support_")
    assert len(message["auxiliary_graphs"]) == 2

    ngd_edges = message["auxiliary_graphs"][analysis["support_graphs"][0]]["edges"]
    assert len(ngd_edges) == 1
    assert ngd_edges[0].startswith("xcrg_ngd_edge_")
    ngd_edge = message["knowledge_graph"]["edges"][ngd_edges[0]]
    assert ngd_edge["subject"] == "CHEBI:1"
    assert ngd_edge["predicate"] == "biolink:occurs_together_in_literature_with"
    assert ngd_edge["object"] == "NCBIGene:51341"
    assert ngd_edge["sources"] == [
        {
            "resource_id": "infores:arax",
            "resource_role": "primary_knowledge_source",
        }
    ]
    ngd_attr = next(
        attr
        for attr in ngd_edge["attributes"]
        if attr["attribute_type_id"] == "EDAM-DATA:2526"
    )
    assert ngd_attr["original_attribute_name"] == "normalized_google_distance"
    assert ngd_attr["value"] == 0.42
    assert ngd_attr["attribute_source"] == "infores:arax"

    inferred_edge_id = analysis["edge_bindings"]["t_edge"][0]["id"]
    inferred_edge = message["knowledge_graph"]["edges"][inferred_edge_id]
    support_attr = next(
        attr
        for attr in inferred_edge["attributes"]
        if attr["attribute_type_id"] == "biolink:support_graphs"
    )
    assert support_attr["attribute_source"] == "infores:arax"
    assert len(support_attr["value"]) == 1
    assert support_attr["value"][0].startswith("xcrg_support_")
    support_edges = message["auxiliary_graphs"][support_attr["value"][0]]["edges"]
    assert support_edges == ["edge0a", "edge1a", "edge0b", "edge1b"]


def test_xcrg_clean_response_uses_inf_ngd_support_when_ngd_is_missing(mocker):
    mocker.patch.object(xcrg_lookup_worker, "get_ngd_score", return_value=None)
    original_message = {
        "message": {
            "query_graph": {
                "nodes": {
                    "sn": {
                        "ids": ["CHEBI:1"],
                        "categories": ["biolink:ChemicalEntity"],
                    },
                    "on": {"categories": ["biolink:Gene"]},
                },
                "edges": {
                    "t_edge": {
                        "subject": "sn",
                        "object": "on",
                        "predicates": ["biolink:affects"],
                        "knowledge_type": "inferred",
                    }
                },
            }
        }
    }
    combined_message = {
        "message": {
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:1": {"categories": ["biolink:ChemicalEntity"]},
                    "NCBIGene:1": {"categories": ["biolink:Gene"]},
                },
                "edges": {
                    "direct_edge": {
                        "subject": "CHEBI:1",
                        "predicate": "biolink:affects",
                        "object": "NCBIGene:1",
                    }
                },
            },
            "results": [
                {
                    "node_bindings": {
                        "sn": [{"id": "CHEBI:1"}],
                        "on": [{"id": "NCBIGene:1"}],
                    },
                    "analyses": [
                        {
                            "score": 1.0,
                            "edge_bindings": {
                                "direct": [{"id": "direct_edge"}],
                            },
                        }
                    ],
                }
            ],
        }
    }

    clean_response = xcrg_lookup_worker.build_trapi_clean_response(
        original_message,
        combined_message,
        "sn",
        "on",
    )

    message = clean_response["message"]
    analysis = message["results"][0]["analyses"][0]
    ngd_support_id = analysis["support_graphs"][0]
    ngd_edge_id = message["auxiliary_graphs"][ngd_support_id]["edges"][0]
    ngd_edge = message["knowledge_graph"]["edges"][ngd_edge_id]
    ngd_attr = next(
        attr
        for attr in ngd_edge["attributes"]
        if attr["attribute_type_id"] == "EDAM-DATA:2526"
    )

    assert ngd_support_id.startswith("xcrg_ngd_support_")
    assert ngd_edge_id.startswith("xcrg_ngd_edge_")
    assert ngd_attr["original_attribute_name"] == "normalized_google_distance"
    assert ngd_attr["value"] == "inf"
