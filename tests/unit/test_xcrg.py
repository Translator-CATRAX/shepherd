import logging

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

    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "message": {
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:1": {"categories": ["biolink:ChemicalEntity"]},
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

    assert mock_post.call_count == 4
    seen_tf_ids = set()
    for call in mock_post.call_args_list:
        payload = call.kwargs["json"]
        tf_ids = payload["message"]["query_graph"]["nodes"]["tf"]["ids"]
        assert len(tf_ids) <= 2
        assert "NCBIGene:7157" not in tf_ids
        assert "NCBIGene:51341" not in tf_ids
        seen_tf_ids.update(tf_ids)

    saved_response = mock_save.call_args.args[1]
    assert saved_response["message"]["query_graph"]["nodes"]["tf"]["ids"] == [
        "NCBIGene:4066",
        "NCBIGene:1105",
        "NCBIGene:9324",
    ]
    assert seen_tf_ids == {"NCBIGene:4066", "NCBIGene:1105", "NCBIGene:9324"}
    debug_labels = [call.args[1] for call in xcrg_lookup_worker.debug_dump_json.call_args_list]
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

    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
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

    assert mock_post.call_count == 2
    for call in mock_post.call_args_list:
        payload = call.kwargs["json"]
        qgraph = payload["message"]["query_graph"]
        assert qgraph["edges"]["e0"]["subject"] == "gene_q"
        assert qgraph["edges"]["e0"]["object"] == "tf"
        assert qgraph["edges"]["e1"]["subject"] == "tf"
        assert qgraph["edges"]["e1"]["object"] == "chem_q"
        assert qgraph["nodes"]["tf"]["ids"] == ["NCBIGene:4066"]

    saved_response = mock_save.call_args.args[1]
    assert saved_response["message"]["query_graph"]["edges"]["e0"]["subject"] == "gene_q"
    assert saved_response["message"]["query_graph"]["edges"]["e1"]["object"] == "chem_q"
