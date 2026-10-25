"""
test_integration_action_pipeline.py — ActionBridge 파이프라인 통합 테스트

MQTT → ActionBridge._on_message → _handle_event → _execute_action → DB 저장
까지의 전체 흐름을 인 프로세스에서 검증한다. 외부 브로커·장치 없이 동작한다.
"""

import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.devices.signboard import SignboardConfig
from src.edgex.dabit_device_service import DabitDeviceService
from src.services.action_bridge import (
    ActionBridge,
    ControlMode,
)

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

_INTRUSION_TOPIC = "cctv/rules/intrusion/filtered"
_SENSOR_TILT_TOPIC = "aiot/rules/sensor/tilt"


def _make_mqtt_message(topic: str, payload: dict) -> SimpleNamespace:
    """paho MQTTMessage 를 흉내낸 더미 객체를 반환한다."""
    msg = SimpleNamespace()
    msg.topic = topic
    msg.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return msg


def _count_db_rows(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0]


def _last_db_row(db_path: Path) -> dict:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM action_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else {}


def test_edgex_device_result_updates_action_command_history(bridge: ActionBridge):
    """EdgeX 장치 결과가 Action Layer 명령 상태를 갱신한다."""
    bridge._on_message(
        None,
        None,
        _make_mqtt_message(
            "edgex/results/cctv/jetson-01/speaker",
            {
                "request_id": "event-1:speaker",
                "event_id": "event-1",
                "device_id": "cctv-speaker-01",
                "status": "simulated",
                "error_code": None,
            },
        ),
    )

    command = bridge._repo.list_commands(limit=1)[0]
    assert command["command_id"] == "event-1:speaker"
    assert command["status"] == "simulated"


def test_camera_alarm_reaches_only_registered_signboard_and_records_result(tmp_path):
    """일반 탐지·저신뢰·미등록 입력은 차단하고 낙상 명령의 중복·실패 회신을 확인한다."""
    registry = tmp_path / "devices.json"
    registry.write_text(json.dumps({"devices": [{
        "device_id": "test-signboard", "device_type": "signboard",
        "camera_ids": ["camera_1"], "enabled": True,
    }]}))
    bridge = ActionBridge(
        db_path=str(tmp_path / "events.db"), edgex_command_mode="edgex",
        edgex_allowed_devices={"signboard"}, edgex_device_registry_path=str(registry),
        signboard_config=SignboardConfig(host="192.0.2.1"),
    )
    bridge._repo.init()
    bridge._sites.set_default_action_settings(confidence_threshold=0.7)
    published = []
    bridge._mqtt_client = SimpleNamespace(
        publish=lambda topic, payload, **kwargs: published.append((topic, json.loads(payload)))
        or SimpleNamespace(rc=0)
    )

    def deliver(event_id, event_type, confidence=0.95, camera_id="camera_1"):
        bridge._on_message(None, None, _make_mqtt_message(
            f"cctv/ai/events/{camera_id}/{event_type}",
            {"event_id": event_id, "camera_id": camera_id, "type": event_type,
             "confidence": confidence, "severity": "critical"},
        ))

    deliver("normal", "person")
    deliver("low", "fall_detected", confidence=0.6)
    deliver("fall", "fall_detected")
    deliver("fall", "fall_detected")
    deliver("unmapped", "fall_detected", camera_id="camera_other")
    commands = [(topic, payload) for topic, payload in published if topic.startswith("edgex/commands/")]
    assert len(commands) == 1
    topic, request = commands[0]
    assert topic.endswith("/signboard/test-signboard")
    assert request["event_id"] == "fall"
    assert request["request_id"] == "fall:signboard:test-signboard"
    device = MagicMock()
    device.display.return_value = True
    service = DabitDeviceService(device_id="test-signboard", config=SignboardConfig(), device=device)
    result = service.execute_request(request)
    assert service.execute_request(request) == result
    device.display.assert_called_once()
    result_topic = "edgex/results/cctv/jetson-01/signboard"
    bridge._on_message(None, None, _make_mqtt_message(result_topic, result.to_dict()))
    command = next(row for row in bridge.list_commands() if row["command_id"] == "fall:signboard")
    assert command["status"] == "acknowledged"

    # 장치 실패는 성공 캐시에 남기지 않아 같은 요청의 복구 재시도를 허용한다.
    failed_request = {**request, "request_id": "recovery:signboard:test-signboard", "event_id": "recovery"}
    device.display.return_value = False
    failed = service.execute_request(failed_request)
    assert failed.status == "failed" and failed.error_code == "device_unreachable"
    bridge._on_message(None, None, _make_mqtt_message(result_topic, failed.to_dict()))
    assert bridge.list_commands()[0]["status"] == "failed"
    device.display.return_value = True
    recovered = service.execute_request(failed_request)
    bridge._on_message(None, None, _make_mqtt_message(result_topic, recovered.to_dict()))
    assert bridge.list_commands()[0]["status"] == "acknowledged"
    assert device.display.call_count == 3


@pytest.mark.parametrize("rule_first", [False, True])
def test_raw_and_rule_routes_share_command_identity_after_cooldown(tmp_path, rule_first):
    """서로 다른 토픽의 동일 이벤트가 쿨다운 이후에도 장치를 재실행하지 않는다."""
    bridge = ActionBridge(
        db_path=str(tmp_path / "events.db"), edgex_command_mode="edgex",
        edgex_allowed_devices={"signboard"},
        edgex_device_registry_path="config/output_devices.json",
        signboard_config=SignboardConfig(host="192.0.2.1"),
    )
    bridge._repo.init()
    published = []
    bridge._mqtt_client = SimpleNamespace(
        publish=lambda topic, payload, **kwargs: published.append((topic, json.loads(payload)))
        or SimpleNamespace(rc=0)
    )
    event = {"event_id": "original-fall", "camera_id": "camera_1",
             "type": "fall_detected", "confidence": 0.95,
             "timestamp": "2026-10-12T15:00:00+09:00", "object_id": "12123278610165399591"}
    routes = [
        ("cctv/ai/events/camera_1/fall_detected", event),
        ("cctv/rules/intrusion/critical", [{**event, "severity": "critical"}]),
    ]
    if rule_first:
        routes.reverse()
    for index, (topic, payload) in enumerate(routes):
        # 시간 제한에 가려진 중복이 아니라 request_id에 의한 억제를 확인한다.
        with patch("src.services._action_bridge_alarm.time.time", return_value=1700000000 + index * 600):
            bridge._on_message(None, None, _make_mqtt_message(topic, payload))

    requests = [payload for topic, payload in published if topic.startswith("edgex/commands/")]
    assert len(requests) == 2
    assert {request["event_id"] for request in requests} == {"original-fall"}
    assert {request["request_id"] for request in requests} == {
        "original-fall:signboard:cctv-signboard-01"
    }
    device = MagicMock()
    device.display.return_value = True
    service = DabitDeviceService(device_id="cctv-signboard-01", config=SignboardConfig(), device=device)
    for request in requests:
        result = service.execute_request(request)
        bridge._on_message(None, None, _make_mqtt_message(
            "edgex/results/cctv/jetson-01/signboard", result.to_dict(),
        ))
    device.display.assert_called_once()
    assert len(bridge.list_recent_events()) == 1
    assert {row["event_id"] for row in bridge.list_recent_events()} == {"original-fall"}
    commands = bridge.list_commands()
    assert len(commands) == 1
    assert commands[0]["command_id"] == "original-fall:signboard"
    assert commands[0]["status"] == "acknowledged"


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture()
def bridge(tmp_path: Path) -> ActionBridge:
    """디바이스 없는 ActionBridge 인스턴스 (DB 는 tmp_path 에 생성)."""
    db = tmp_path / "test_action.db"
    ab = ActionBridge(
        mqtt_broker="localhost",
        mqtt_port=1883,
        db_path=str(db),
        rest_enabled=False,
    )
    ab._repo.init()
    # MQTT 클라이언트 없이도 _publish_status 가 조용히 실패하도록
    ab._mqtt_client = None
    return ab


# ---------------------------------------------------------------------------
# 통합 시나리오 : AUTO 모드 — MQTT 수신 → DB 저장
# ---------------------------------------------------------------------------


class TestAutoModeFullPipeline:
    """AUTO 모드에서 이벤트가 처리되어 SQLite 에 저장되는지 검증."""

    def test_intrusion_event_saved_to_db(self, bridge: ActionBridge, tmp_path: Path):
        """침입 감지 이벤트가 _on_message → DB까지 저장된다."""
        payload = {
            "camera_id": "cam-01",
            "type": "intrusion",
            "confidence": 0.85,
            "severity": "high",
            "timestamp": time.time(),
        }
        msg = _make_mqtt_message(_INTRUSION_TOPIC, payload)

        bridge._on_message(None, None, msg)

        db_path = bridge._repo.db_path
        assert _count_db_rows(db_path) == 1
        row = _last_db_row(db_path)
        assert row["event_id"].startswith("evt_")
        assert row["camera_id"] == "cam-01"
        assert row["event_type"] == "intrusion"
        assert row["topic"] == _INTRUSION_TOPIC

    def test_canonical_payload_saved_to_db(self, bridge: ActionBridge):
        """표준 event/device 구조만 있어도 ActionBridge가 저장 필드를 채운다."""
        payload = {
            "schema_version": "1.0",
            "message_type": "ai_detection_event",
            "occurred_at": "2026-05-04T01:00:00+00:00",
            "device": {"camera_id": "cam-canonical"},
            "event": {
                "event_type": "fall_detected",
                "severity": "critical",
                "confidence": 0.97,
            },
            "raw": {},
        }
        msg = _make_mqtt_message("cctv/ai/events/cam-canonical/fall_detected", payload)

        bridge._on_message(None, None, msg)

        row = _last_db_row(bridge._repo.db_path)
        assert row["camera_id"] == "cam-canonical"
        assert row["event_type"] == "fall_detected"
        assert row["confidence"] == 0.97
        assert row["severity"] == "critical"

    def test_multiple_events_accumulated(self, bridge: ActionBridge):
        """연속 이벤트가 각각 DB에 저장된다."""
        for cam in ("cam-01", "cam-02", "cam-03"):
            msg = _make_mqtt_message(
                _INTRUSION_TOPIC,
                {"camera_id": cam, "type": "intrusion", "confidence": 0.9},
            )
            bridge._on_message(None, None, msg)

        assert _count_db_rows(bridge._repo.db_path) == 3

    def test_kuiper_array_payload_handled(self, bridge: ActionBridge):
        """Kuiper 싱크는 배열 형태로 발행할 수 있다 — 개별 처리 확인."""
        payload = [
            {"camera_id": "cam-A", "type": "intrusion", "confidence": 0.8},
            {"camera_id": "cam-B", "type": "intrusion", "confidence": 0.75},
        ]
        msg = _make_mqtt_message(_INTRUSION_TOPIC, payload)

        bridge._on_message(None, None, msg)

        assert _count_db_rows(bridge._repo.db_path) == 2

    def test_malformed_json_does_not_crash(self, bridge: ActionBridge):
        """JSON 파싱 실패 시 예외 없이 무시된다."""
        msg = SimpleNamespace(
            topic=_INTRUSION_TOPIC,
            payload=b"{not valid json",
        )
        bridge._on_message(None, None, msg)  # 예외 불가

        assert _count_db_rows(bridge._repo.db_path) == 0


# ---------------------------------------------------------------------------
# 통합 시나리오 : MANUAL 모드 — 대기 큐 → 승인 → 실행 → DB
# ---------------------------------------------------------------------------


class TestManualModeApprovalPipeline:
    """MANUAL 모드에서 대기→승인→DB 저장 경로를 검증."""

    @pytest.fixture()
    def manual_bridge(self, tmp_path: Path) -> ActionBridge:
        db = tmp_path / "manual.db"
        ab = ActionBridge(
            mqtt_broker="localhost",
            mqtt_port=1883,
            db_path=str(db),
            rest_enabled=False,
            default_mode=ControlMode.MANUAL,
        )
        ab._repo.init()
        ab._mqtt_client = None
        return ab

    def test_event_enters_pending_queue(self, manual_bridge: ActionBridge):
        """MANUAL 모드 수신 이벤트는 대기 큐에 들어간다."""
        msg = _make_mqtt_message(
            _INTRUSION_TOPIC,
            {"camera_id": "cam-01", "type": "intrusion", "confidence": 0.9},
        )
        manual_bridge._on_message(None, None, msg)

        pending = manual_bridge.get_pending_events()
        assert len(pending) == 1
        assert pending[0]["camera_id"] == "cam-01"

    def test_approve_event_executes_and_saves(self, manual_bridge: ActionBridge):
        """승인된 이벤트는 즉시 실행되어 DB에 저장된다."""
        msg = _make_mqtt_message(
            _INTRUSION_TOPIC,
            {"camera_id": "cam-01", "type": "intrusion", "confidence": 0.9},
        )
        manual_bridge._on_message(None, None, msg)

        event_id = manual_bridge.get_pending_events()[0]["event_id"]
        ok, message = manual_bridge.approve_event(event_id)

        assert ok is True
        assert _count_db_rows(manual_bridge._repo.db_path) >= 1

    def test_reject_event_removes_from_queue(self, manual_bridge: ActionBridge):
        """거부된 이벤트는 대기 큐에서 제거된다."""
        msg = _make_mqtt_message(
            _INTRUSION_TOPIC,
            {"camera_id": "cam-01", "type": "intrusion", "confidence": 0.9},
        )
        manual_bridge._on_message(None, None, msg)

        event_id = manual_bridge.get_pending_events()[0]["event_id"]
        ok, _ = manual_bridge.reject_event(event_id)

        assert ok is True
        assert manual_bridge.get_pending_events() == []


# ---------------------------------------------------------------------------
# 통합 시나리오 : 센서 이벤트 정규화
# ---------------------------------------------------------------------------


class TestSensorEventNormalization:
    """aiot sensor 토픽의 device_id → camera_id 정규화 검증."""

    def test_tilt_sensor_payload_normalized(self, bridge: ActionBridge):
        """aiot/rules/sensor/tilt 수신 시 device_id 가 camera_id 로 매핑된다."""
        payload = {
            "device_id": "sensor-001",
            "angle_x_deg": 15.3,
            "angle_y_deg": 2.1,
            "timestamp": time.time(),
        }
        msg = _make_mqtt_message(_SENSOR_TILT_TOPIC, payload)

        bridge._on_message(None, None, msg)

        row = _last_db_row(bridge._repo.db_path)
        assert row["camera_id"] == "sensor-001"
        assert row["event_type"] == "tilt_alert"


# ---------------------------------------------------------------------------
# 통합 시나리오 : HTTP 포워더 + 디바이스 호출 검증
# ---------------------------------------------------------------------------


class TestDeviceCommandIntegration:
    """이벤트 처리 시 장치 명령(speaker.play)이 호출되는지 확인."""

    def test_speaker_called_on_alarm_topic(self, tmp_path: Path):
        """알람 토픽 이벤트가 수신되면 speaker.play 가 1회 호출된다."""
        db = tmp_path / "device.db"
        ab = ActionBridge(
            mqtt_broker="localhost",
            mqtt_port=1883,
            db_path=str(db),
            rest_enabled=False,
            alarm_cooldown_seconds=0,
        )
        ab._repo.init()
        ab._mqtt_client = None
        ab._speaker.config.host = "127.0.0.1"
        ab._speaker.config.username = "admin"
        ab._speaker.config.password = "pw"
        ab._device_reachability_cache["speaker"] = (time.time(), True)

        with patch.object(ab._speaker, "play", return_value=True) as mock_play:
            msg = _make_mqtt_message(
                "cctv/ai/events/cam-01/fall_detected",
                {
                    "camera_id": "cam-01",
                    "type": "fall_detected",
                    "confidence": 0.95,
                    "event_id": "fall-e2e-1",
                    "request_id": "fall-request-1",
                },
            )
            ab._on_message(None, None, msg)

        mock_play.assert_called_once()
        command = ab._repo.list_commands(limit=1)[0]
        assert command["command_id"] == "fall-e2e-1:speaker"
        assert command["status"] == "acknowledged"

    def test_http_forwarder_called_when_target_set(self, tmp_path: Path):
        """HTTP 포워더 타겟이 있으면 forward() 가 호출된다."""
        db = tmp_path / "fwd.db"
        ab = ActionBridge(
            mqtt_broker="localhost",
            mqtt_port=1883,
            db_path=str(db),
            external_api_url="http://example.invalid/api/alerts",
            rest_enabled=False,
            alarm_cooldown_seconds=0,
        )
        ab._repo.init()
        ab._mqtt_client = None

        with patch.object(ab._forwarder, "forward") as mock_fwd:
            msg = _make_mqtt_message(
                _INTRUSION_TOPIC,
                {"camera_id": "cam-01", "type": "intrusion", "confidence": 0.9},
            )
            ab._on_message(None, None, msg)

        mock_fwd.assert_called_once()
