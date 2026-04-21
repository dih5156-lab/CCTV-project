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

from src.services.action_bridge import ActionBridge, ControlMode, AlarmDevice, SiteConfig

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

        with patch.object(ab._speaker, "play", return_value=True) as mock_play:
            msg = _make_mqtt_message(
                "cctv/rules/intrusion/critical",
                {"camera_id": "cam-01", "type": "intrusion", "confidence": 0.95},
            )
            ab._on_message(None, None, msg)

        mock_play.assert_called_once()

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
