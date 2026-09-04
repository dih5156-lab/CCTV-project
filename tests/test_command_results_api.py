import asyncio

from src.api.v1.command_results import get_command_result, list_command_results
from src.edgex.command_result_collector import CommandResultStore


def test_command_results_api_reads_saved_result(tmp_path, monkeypatch):
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("EDGEX_COMMAND_RESULT_DB", str(db_path))
    CommandResultStore(db_path).record(
        "edgex/results/cctv/jetson-01/speaker",
        {
            "request_id": "req-api",
            "event_id": "event-api",
            "device_id": "cctv-speaker-01",
            "status": "simulated",
        },
    )

    response = asyncio.run(get_command_result("req-api", None))

    assert response.success is True
    assert response.data["status"] == "simulated"


def test_command_results_api_filters_list(tmp_path, monkeypatch):
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("EDGEX_COMMAND_RESULT_DB", str(db_path))
    store = CommandResultStore(db_path)
    store.record(
        "edgex/results/cctv/jetson-01/siren",
        {
            "request_id": "req-filter",
            "event_id": "event-filter",
            "device_id": "cctv-siren-01",
            "status": "failed",
        },
    )

    response = asyncio.run(
        list_command_results(
            device_id="cctv-siren-01",
            status="failed",
            limit=10,
            _=None,
        )
    )

    assert response.data[0]["request_id"] == "req-filter"
