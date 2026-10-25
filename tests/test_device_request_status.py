"""장치 상태 조회가 실행/선점 없이 알려진 결과와 불확실성을 구분하는지 확인한다."""

import hashlib
import importlib
import sqlite3
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from src.devices.signboard import SignboardConfig
from src.devices.siren import SensorConfig
from src.devices.speaker import SpeakerConfig
from src.edgex.command_http import handle_request_status
from src.edgex.dabit_device_service import DabitDeviceService
from src.edgex.siren_device_service import SirenDeviceService
from src.edgex.speaker_device_service import SpeakerDeviceService


@pytest.mark.parametrize("kind,service_class,config,action,method", [
    ("signboard", DabitDeviceService, SignboardConfig, "display", "display"),
    ("speaker", SpeakerDeviceService, SpeakerConfig, "play", "play"),
    ("siren", SirenDeviceService, SensorConfig, "trigger", "trigger"),
])
def test_runner_get_reads_restored_result_without_device_execution(tmp_path, kind, service_class, config, action, method):
    device = MagicMock()
    path = tmp_path / "requests.db"
    options = dict(device_id="device-1", config=config(), device=device, request_db_path=path)
    request_id = f"event-1:{kind}:device-1"
    request = {"request_id": request_id, "event_id": "event-1", "device": kind,
               "device_id": "device-1", "action": action, "payload": {"text": "경고"}}
    assert service_class(**options).execute_request(request).status == "acknowledged"
    restored = service_class(**options)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    module = importlib.import_module(f"runners.run_{kind}_device_service")
    handler = object.__new__(module._CommandHandler)
    handler.service = restored
    handler.path = "/requests/" + quote(request_id, safe="")
    handler._write_json = MagicMock()
    handler.do_GET()
    status, body = handler._write_json.call_args.args
    assert status == 200
    assert body["request_id"] == request_id
    assert body["status"] == "acknowledged"
    assert body["durable"] is True
    assert body["retry_allowed"] is False
    assert body["retry_reason"] == "already_completed"
    assert "payload" not in body
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    getattr(device, method).assert_called_once()


@pytest.mark.parametrize("outcome", ["unknown", "unconfirmed", "failed", "missing_db", "corrupt_result"])
def test_unknown_or_unconfirmed_lookup_never_allows_retry(tmp_path, outcome):
    device = MagicMock()
    device.play.return_value = False
    path = tmp_path / "requests.db"
    service = SpeakerDeviceService(device_id="device-1", config=SpeakerConfig(), device=device,
                                   request_db_path=path)
    request_id = "event-1:speaker:device-1"
    if outcome == "unconfirmed":
        with sqlite3.connect(path) as conn:
            conn.execute("INSERT INTO device_requests(request_id, request_hash) VALUES (?, ?)",
                         (request_id, "hash"))
    elif outcome in {"failed", "corrupt_result"}:
        service.execute_request({"request_id": request_id, "event_id": "event-1", "device": "speaker",
                                 "device_id": "device-1", "action": "play", "payload": {}})
        if outcome == "corrupt_result":
            with sqlite3.connect(path) as conn:
                conn.execute("UPDATE device_requests SET result_json='[]'")
    elif outcome == "missing_db":
        path.rename(tmp_path / "preserved.db")
    count = device.play.call_count
    status, result = handle_request_status(service, "/requests/" + quote(request_id, safe=""))
    assert status == (404 if outcome == "unknown" else 503 if outcome in {"missing_db", "corrupt_result"} else 200)
    assert result["status"] == ("unknown" if outcome == "unknown" else "uncertain")
    assert result["retry_allowed"] is False
    assert device.play.call_count == count
    if outcome == "missing_db":
        assert not path.exists()


@pytest.mark.parametrize("path", ["/requests/", "/other", "/requests"])
def test_invalid_lookup_path_does_not_call_service(path):
    service = MagicMock()
    assert handle_request_status(service, path)[0] == 404
    service.get_request_status.assert_not_called()
