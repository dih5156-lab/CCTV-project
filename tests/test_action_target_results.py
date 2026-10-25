"""여러 물리 장치의 결과·응답 대기를 도착 순서와 무관하게 보존한다."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from src.services._action_bridge_repo import _EventRepo
from src.services.command_result_collector import CommandResultCollector
from src.services.device_command_transport import DeviceCommand, EdgeXCommandTransport
from src.services.edgex_result_handler import EdgeXResultHandler


def target(device, status):
    return {"request_id": f"event-1:signboard:{device}", "event_id": "event-1",
            "device_id": device, "status": status, "error_code": None}


@pytest.mark.parametrize("reverse", [False, True])
def test_failed_device_is_not_overwritten_by_another_success(tmp_path, reverse):
    repo = _EventRepo(tmp_path / "events.db")
    repo.init()
    results = [target("board-1", "failed"), target("board-2", "acknowledged")]
    repo.record_command("event-1:signboard", "device/signboard", "pending", {
        "event_id": "event-1", "target_request_ids": [row["request_id"] for row in results],
        "device_results": [target(row["device_id"], "pending") for row in results],
    })
    handler = EdgeXResultHandler(repository=repo, publish_status=lambda *args: None)
    for row in results[::-1] if reverse else results:
        handler.handle("edgex/results/cctv/jetson-01/signboard", row)
    row = repo.list_commands()[0]
    assert row["status"] == "partial_failure"
    assert {v["device_id"]: v["status"] for v in row["payload"]["device_results"]} == {
        "board-1": "failed", "board-2": "acknowledged",
    }


def test_missing_reply_remains_pending_and_late_reply_survives_publish_record(tmp_path):
    repo = _EventRepo(tmp_path / "events.db")
    repo.init()
    pending = {"event_id": "event-1",
               "target_request_ids": [target(d, "pending")["request_id"] for d in ["board-1", "board-2"]],
               "device_results": [target(d, "accepted") for d in ["board-1", "board-2"]]}
    repo.record_command("event-1:signboard", "device/signboard", "accepted", pending)
    handler = EdgeXResultHandler(repository=repo, publish_status=lambda *args: None)
    handler.handle("edgex/results/cctv/jetson-01/signboard", target("board-1", "acknowledged"))
    assert repo.list_commands()[0]["status"] == "pending"
    handler.handle("edgex/results/cctv/jetson-01/signboard", target("board-2", "acknowledged"))
    # 장치 응답이 발행 함수 반환보다 빨라도 전송 상태가 완료 결과를 덮지 않는다.
    repo.record_command("event-1:signboard", "device/signboard", "accepted", pending)
    assert repo.list_commands()[0]["status"] == "acknowledged"


def test_concurrent_device_results_preserve_both_targets(tmp_path):
    repo = _EventRepo(tmp_path / "events.db")
    repo.init()
    handler = EdgeXResultHandler(repository=repo, publish_status=lambda *args: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(handler.handle, "edgex/results/cctv/jetson-01/signboard", target(d, s))
                   for d, s in [("board-1", "failed"), ("board-2", "acknowledged")]]
        for future in futures:
            future.result(timeout=3)
    assert repo.list_commands()[0]["status"] == "partial_failure"


@pytest.mark.parametrize("outcome", ["lost_reply", "partial_failure", "publish_error", "early_replies"])
def test_transport_and_result_pipeline_keep_expected_targets(tmp_path, outcome):
    repo = _EventRepo(tmp_path / "events.db")
    repo.init()
    collector = CommandResultCollector(repository=repo, device_commands_metric=MagicMock(),
                                       device_results_metric=MagicMock())
    handler = EdgeXResultHandler(repository=repo, publish_status=lambda *args: None)
    calls = []

    def publish(topic, request):
        calls.append(request["device_id"])
        if request["device_id"] == "board-1":
            if outcome == "publish_error":
                raise TimeoutError("수신 여부 미확인")
            if outcome == "partial_failure":
                handler.handle("edgex/results/cctv/jetson-01/signboard", target("board-1", "failed"))
                return True
        if outcome != "lost_reply":
            handler.handle("edgex/results/cctv/jetson-01/signboard", target(request["device_id"], "acknowledged"))
        return True

    transport = EdgeXCommandTransport(publish=publish, resolve_device_ids=lambda *_: ["board-1", "board-2"],
        jetson_id="jetson-01", topic_prefix="edgex/commands/cctv", register_targets=collector.register_targets)
    command = DeviceCommand("signboard", "display", {"text": "경고"}, "event-1:signboard", "event-1", "camera_1")
    response = collector.collect(command, transport.send(command))
    expected = {"lost_reply": "pending", "partial_failure": "partial_failure",
                "publish_error": "uncertain", "early_replies": "acknowledged"}[outcome]
    assert calls == ["board-1", "board-2"]
    assert response["status"] == repo.list_commands()[0]["status"] == expected
    assert len(response["device_results"]) == 2
    # 저장소 객체를 새로 만들어도 대상/미응답/부분 실패는 그대로 조회된다.
    assert _EventRepo(repo.db_path).list_commands()[0]["status"] == expected


def test_target_registration_failure_prevents_publish(tmp_path):
    repo = _EventRepo(tmp_path / "not_initialized.db")
    collector = CommandResultCollector(repository=repo, device_commands_metric=MagicMock(),
                                       device_results_metric=MagicMock())
    publish = MagicMock(return_value=True)
    transport = EdgeXCommandTransport(publish=publish, resolve_device_ids=lambda *_: ["board-1"],
        jetson_id="jetson-01", topic_prefix="edgex/commands/cctv", register_targets=collector.register_targets)
    result = transport.send(DeviceCommand("signboard", "display", {}, "event-1:signboard", "event-1", "camera_1"))
    assert result.status == "failed"
    publish.assert_not_called()


@pytest.mark.parametrize("change", ["event_id", "device_id", "unknown_target", "status"])
def test_mismatched_reply_does_not_overwrite_pending_command(tmp_path, change):
    repo = _EventRepo(tmp_path / "events.db")
    repo.init()
    original = target("board-1", "pending")
    repo.record_command("event-1:signboard", "device/signboard", "pending", {
        "event_id": "event-1", "target_request_ids": [original["request_id"]], "device_results": [original],
    })
    reply = target("board-1", "acknowledged")
    if change == "unknown_target":
        reply = target("board-unknown", "acknowledged")
    else:
        reply[change] = "invalid"
    handler = EdgeXResultHandler(repository=repo, publish_status=lambda *args: None)
    handler.handle("edgex/results/cctv/jetson-01/signboard", reply)
    assert repo.list_commands()[0]["status"] == "pending"


def test_reply_cannot_inject_other_targets_or_erase_waiting_device(tmp_path):
    repo = _EventRepo(tmp_path / "events.db")
    repo.init()
    expected = [target(d, "pending") for d in ["board-1", "board-2"]]
    repo.record_command("event-1:signboard", "device/signboard", "pending", {
        "event_id": "event-1", "target_request_ids": [row["request_id"] for row in expected],
        "device_results": expected,
    })
    reply = {**target("board-1", "acknowledged"),
             "device_results": [target("board-2", "acknowledged")],
             "target_request_ids": [expected[0]["request_id"]]}
    EdgeXResultHandler(repository=repo, publish_status=lambda *args: None).handle(
        "edgex/results/cctv/jetson-01/signboard", reply)
    result = repo.list_commands()[0]
    assert result["status"] == "pending"
    assert [row["status"] for row in result["payload"]["device_results"]] == ["acknowledged", "pending"]
