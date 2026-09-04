from src.edgex.command_result_collector import CommandResultStore


def test_result_store_saves_and_reads_device_result(tmp_path):
    store = CommandResultStore(tmp_path / "results.db")

    store.record(
        "edgex/results/cctv/jetson-01/speaker",
        {
            "request_id": "req-1",
            "event_id": "event-1",
            "device_id": "cctv-speaker-01",
            "status": "simulated",
            "error_code": None,
        },
    )

    result = store.get("req-1")
    assert result["device_id"] == "cctv-speaker-01"
    assert result["status"] == "simulated"
    assert result["topic"].endswith("/speaker")


def test_result_store_updates_same_request_id_without_duplicate_rows(tmp_path):
    store = CommandResultStore(tmp_path / "results.db")
    base = {
        "request_id": "req-2",
        "event_id": "event-2",
        "device_id": "cctv-siren-01",
        "status": "running",
    }
    store.record("edgex/results/cctv/jetson-01/siren", base)
    store.record(
        "edgex/results/cctv/jetson-01/siren",
        {**base, "status": "acknowledged"},
    )

    assert store.get("req-2")["status"] == "acknowledged"
    assert len(store.list_recent()) == 1


def test_invalid_result_is_ignored(tmp_path):
    store = CommandResultStore(tmp_path / "results.db")

    assert store.record("edgex/results/cctv/jetson-01/speaker", {}) is False
    assert store.list_recent() == []


def test_result_store_filters_by_device_and_status(tmp_path):
    store = CommandResultStore(tmp_path / "results.db")
    for request_id, device_id, status in (
        ("req-speaker", "cctv-speaker-01", "acknowledged"),
        ("req-siren", "cctv-siren-01", "failed"),
    ):
        store.record(
            "edgex/results/cctv/jetson-01/device",
            {
                "request_id": request_id,
                "event_id": request_id.replace("req", "event"),
                "device_id": device_id,
                "status": status,
            },
        )

    results = store.list_recent(device_id="cctv-speaker-01", status="acknowledged")
    assert [item["request_id"] for item in results] == ["req-speaker"]
