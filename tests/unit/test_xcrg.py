import logging

import pytest

from workers.xcrg.worker import xcrg
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
