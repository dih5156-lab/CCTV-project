"""
test_device_service.py — CCTVDeviceService 순수 유틸리티 메서드 단위 테스트

외부 연결(Redis / MQTT / HTTP)이 필요한 메서드는
mock 을 통해 격리하거나 빠른 타임아웃으로 실패 경로를 테스트합니다.
"""
import asyncio
import importlib
import time
import pytest
from unittest.mock import MagicMock, patch
from src.edgex.device_service import CCTVDeviceService

# redis 설치 여부 확인 — 없으면 관련 테스트를 skip
_redis_available = importlib.util.find_spec("redis") is not None
redis_required = pytest.mark.skipif(
    not _redis_available, reason="redis 패키지가 설치되지 않아 건너뜀"
)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def svc(tmp_path) -> CCTVDeviceService:
    """연결 없이 CCTVDeviceService 인스턴스 생성."""
    return CCTVDeviceService({
        "coreMetadataUrl": "http://localhost:59881",
        "coreDataUrl":     "http://localhost:59880",
        "deviceServiceName": "test-service",
        "mqttBroker": "localhost",
        "mqttPort": "1883",
        "redisHost": "localhost",
        "redisPort": "6379",
        "enableStoreAndForward": True,
        "outboxDbPath": str(tmp_path / "event_outbox.db"),
    })


# ---------------------------------------------------------------------------
# _to_bool
# ---------------------------------------------------------------------------


class TestToBool:
    def test_true_strings(self, svc):
        for v in ("true", "True", "TRUE", "1", "yes", "on", "y"):
            assert svc._to_bool(v) is True, f"실패: {v!r}"

    def test_false_strings(self, svc):
        for v in ("false", "False", "FALSE", "0", "no", "off", "n"):
            assert svc._to_bool(v) is False, f"실패: {v!r}"

    def test_bool_passthrough(self, svc):
        assert svc._to_bool(True) is True
        assert svc._to_bool(False) is False

    def test_int(self, svc):
        assert svc._to_bool(1) is True
        assert svc._to_bool(0) is False

    def test_unknown_string_returns_truthy(self, svc):
        # 알 수 없는 문자열은 bool("문자열") == True 로 수렴
        assert svc._to_bool("maybe") is True


# ---------------------------------------------------------------------------
# _parse_rtsp_source
# ---------------------------------------------------------------------------


class TestParseRtspSource:
    def test_full_url(self, svc):
        result = svc._parse_rtsp_address_port("rtsp://192.168.1.10:8554/stream")
        assert result["Address"] == "192.168.1.10"
        assert result["Port"] == "8554"

    def test_url_without_port(self, svc):
        result = svc._parse_rtsp_address_port("rtsp://192.168.1.10/stream")
        assert result["Address"] == "192.168.1.10"
        assert result["Port"] == "554"         # 기본 포트

    def test_host_only(self, svc):
        result = svc._parse_rtsp_address_port("192.168.1.10:9000")
        assert result["Port"] == "9000"

    def test_empty_string_returns_default(self, svc):
        result = svc._parse_rtsp_address_port("")
        assert result == {"Address": "localhost", "Port": "554"}

    def test_none_returns_default(self, svc):
        result = svc._parse_rtsp_address_port(None)
        assert result == {"Address": "localhost", "Port": "554"}

    def test_non_string_returns_default(self, svc):
        result = svc._parse_rtsp_address_port(12345)
        assert result == {"Address": "localhost", "Port": "554"}


# ---------------------------------------------------------------------------
# _to_origin_nanos
# ---------------------------------------------------------------------------


class TestToOriginNanos:
    def test_numeric_float(self, svc):
        ts = 1_700_000_000.0
        ns = svc._to_origin_nanos(ts)
        assert ns == int(ts * 1_000_000_000)

    def test_numeric_int(self, svc):
        ns = svc._to_origin_nanos(1_700_000_000)
        assert ns > 0

    def test_string_float(self, svc):
        ns = svc._to_origin_nanos("1700000000.5")
        assert ns > 0

    def test_iso_string(self, svc):
        iso = "2024-01-01T00:00:00Z"
        ns = svc._to_origin_nanos(iso)
        assert ns > 0

    def test_none_returns_approx_now(self, svc):
        before = time.time() * 1_000_000_000
        ns = svc._to_origin_nanos(None)
        after = time.time() * 1_000_000_000
        assert before <= ns <= after + 1_000_000  # 1ms 여유

    def test_garbage_string_returns_approx_now(self, svc):
        before = time.time() * 1_000_000_000
        ns = svc._to_origin_nanos("not_a_timestamp")
        after = time.time() * 1_000_000_000
        assert before <= ns <= after + 1_000_000


# ---------------------------------------------------------------------------
# _versioned_endpoints
# ---------------------------------------------------------------------------


class TestVersionedEndpoints:
    def test_returns_v3_v2_v1(self, svc):
        endpoints = svc._versioned_endpoints("http://host:1234", "event")
        assert any("/v3/" in e for e in endpoints)
        assert any("/v2/" in e for e in endpoints)
        assert any("/v1/" in e for e in endpoints)

    def test_no_legacy_by_default(self, svc):
        endpoints = svc._versioned_endpoints("http://host:1234", "event")
        # 레거시 경로(api/ 없는 것)가 없어야 함
        legacy = [e for e in endpoints if "/api/" not in e and "localhost" not in e]
        assert len(legacy) == 0

    def test_include_legacy(self, svc):
        endpoints = svc._versioned_endpoints("http://host", "ping", include_legacy=True)
        assert len(endpoints) == 4

    def test_trailing_slash_stripped(self, svc):
        endpoints = svc._versioned_endpoints("http://host:1234/", "event")
        for e in endpoints:
            assert "//api" not in e   # 이중 슬래시 없음


# ---------------------------------------------------------------------------
# _payload_for_endpoint
# ---------------------------------------------------------------------------


class TestPayloadForEndpoint:
    def test_v3_wraps_in_list(self, svc):
        payload = {"foo": "bar"}
        result = svc._payload_for_endpoint("http://host/api/v3/event", payload)
        assert isinstance(result, list)
        assert result[0] == payload

    def test_v2_returns_object(self, svc):
        payload = {"foo": "bar"}
        result = svc._payload_for_endpoint("http://host/api/v2/event", payload)
        assert result == payload

    def test_v1_returns_object(self, svc):
        payload = {"foo": "bar"}
        result = svc._payload_for_endpoint("http://host/api/v1/event", payload)
        assert result == payload


# ---------------------------------------------------------------------------
# _map_event_type_to_resource
# ---------------------------------------------------------------------------


class TestMapEventTypeToResource:
    def test_helmet_types(self, svc):
        for t in ("helmet", "head", "unsafe_behavior", "wearing_helmet"):
            assert svc._map_event_type_to_resource(t) == "helmet_detection"

    def test_fall_types(self, svc):
        for t in ("fall_detected", "not_fall"):
            assert svc._map_event_type_to_resource(t) == "fall_detection"

    def test_person_type(self, svc):
        assert svc._map_event_type_to_resource("person") == "person_detection"

    def test_unknown_type(self, svc):
        result = svc._map_event_type_to_resource("some_unknown")
        # 알 수 없는 타입은 person_detection 또는 unknown 계열 반환 (예외 없이)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _ensure_redis_client / _ensure_mqtt_client — 실패 경로 (지수 백오프)
# ---------------------------------------------------------------------------


class TestEnsureConnectionBackoff:
    """Redis / MQTT 연결 시도는 실제 소켓 없이 mock 으로 격리."""

    def _reset_redis(self, svc):
        svc._redis_fail_count = 0
        svc._redis_last_fail_time = 0
        svc._redis_client = None

    def _reset_mqtt(self, svc):
        svc._mqtt_fail_count = 0
        svc._mqtt_last_fail_time = 0
        svc._mqtt_client = None

    def test_connection_state_is_isolated_per_instance(self, svc, tmp_path):
        """한 서비스 인스턴스의 백오프 상태가 다른 인스턴스에 전파되지 않아야 한다."""
        other = CCTVDeviceService({
            "coreMetadataUrl": "http://localhost:59881",
            "coreDataUrl": "http://localhost:59880",
            "deviceServiceName": "other-service",
            "enableStoreAndForward": True,
            "outboxDbPath": str(tmp_path / "other_outbox.db"),
        })

        svc._redis_fail_count = 3
        svc._redis_last_fail_time = time.time()
        svc._mqtt_fail_count = 2
        svc._mqtt_last_fail_time = time.time()

        assert other._redis_fail_count == 0
        assert other._redis_last_fail_time == 0.0
        assert other._mqtt_fail_count == 0
        assert other._mqtt_last_fail_time == 0.0

    def test_connection_locks_are_isolated_per_instance(self, svc, tmp_path):
        other = CCTVDeviceService({
            "coreMetadataUrl": "http://localhost:59881",
            "coreDataUrl": "http://localhost:59880",
            "deviceServiceName": "other-service",
            "enableStoreAndForward": True,
            "outboxDbPath": str(tmp_path / "other_outbox.db"),
        })

        assert svc._redis_state.lock is not other._redis_state.lock
        assert svc._mqtt_state.lock is not other._mqtt_state.lock

    @redis_required
    def test_redis_fails_and_increments_fail_count(self, svc):
        """Redis 연결 실패 → fail_count 증가."""
        self._reset_redis(svc)
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionRefusedError("연결 거부")
        with patch("src.edgex._publisher_mixin.redis.Redis", return_value=mock_client):
            result = svc._ensure_redis_client()
        assert result is False
        assert svc._redis_fail_count >= 1

    def test_redis_cooldown_blocks_retry(self, svc):
        """쿨다운 중에는 즉시 False 반환 (소켓 시도 없음)."""
        svc._redis_fail_count = 1
        svc._redis_last_fail_time = time.time()
        svc._redis_client = None
        result = svc._ensure_redis_client()
        assert result is False

    def test_mqtt_fails_and_increments_fail_count(self, svc):
        """MQTT 연결 실패 → fail_count 증가."""
        self._reset_mqtt(svc)
        mock_instance = MagicMock()
        mock_instance.connect.side_effect = ConnectionRefusedError("MQTT 연결 거부")
        with patch("src.edgex._publisher_mixin.create_mqtt_client", return_value=mock_instance):
            result = svc._ensure_mqtt_client()
        assert result is False
        assert svc._mqtt_fail_count >= 1

    def test_mqtt_cooldown_blocks_retry(self, svc):
        """MQTT 쿨다운 중에는 즉시 False 반환 (소켓 시도 없음)."""
        svc._mqtt_fail_count = 1
        svc._mqtt_last_fail_time = time.time()
        svc._mqtt_client = None
        result = svc._ensure_mqtt_client()
        assert result is False

    @redis_required
    def test_redis_success_resets_fail_count(self, svc):
        """Redis 연결 성공 시 fail_count 가 0으로 초기화."""
        svc._redis_fail_count = 5
        svc._redis_last_fail_time = 0
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        with patch("src.edgex._publisher_mixin.redis.Redis", return_value=mock_client):
            svc._redis_client = None
            result = svc._ensure_redis_client()
        assert result is True
        assert svc._redis_fail_count == 0

    def test_mqtt_success_resets_fail_count(self, svc):
        """MQTT 연결 성공 시 fail_count 가 0으로 초기화."""
        svc._mqtt_fail_count = 3
        svc._mqtt_last_fail_time = 0
        mock_instance = MagicMock()
        mock_instance.connect.return_value = None
        with patch("src.edgex._publisher_mixin.create_mqtt_client", return_value=mock_instance):
            svc._mqtt_client = None
            result = svc._ensure_mqtt_client()
        assert result is True
        assert svc._mqtt_fail_count == 0


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_clears_clients(self, svc):
        mock_mqtt = MagicMock()
        mock_redis = MagicMock()
        svc._mqtt_client = mock_mqtt
        svc._redis_client = mock_redis

        svc.close()

        assert svc._mqtt_client is None
        assert svc._redis_client is None
        mock_mqtt.loop_stop.assert_called_once()
        mock_mqtt.disconnect.assert_called_once()
        mock_redis.close.assert_called_once()

    def test_close_handles_mqtt_exception(self, svc):
        """disconnect 오류 발생 시 예외 없이 처리되어야 함."""
        mock_mqtt = MagicMock()
        mock_mqtt.disconnect.side_effect = RuntimeError("연결 끊김")
        svc._mqtt_client = mock_mqtt
        svc._redis_client = None

        svc.close()  # 예외 없어야 함
        assert svc._mqtt_client is None

    def test_close_handles_redis_exception(self, svc):
        mock_redis = MagicMock()
        mock_redis.close.side_effect = RuntimeError("Redis 오류")
        svc._mqtt_client = None
        svc._redis_client = mock_redis

        svc.close()  # 예외 없어야 함
        assert svc._redis_client is None

    def test_close_idempotent(self, svc):
        """이미 None 상태에서 close() 를 두 번 호출해도 안전."""
        svc._mqtt_client = None
        svc._redis_client = None
        svc.close()
        svc.close()


class TestEdgeXOutbox:
    def test_failed_send_is_persisted_to_outbox(self, svc):
        svc._init_outbox()
        svc.devices["cam1"] = "device-1"

        sample_event = {
            "camera_id": "cam1",
            "type": "fall_detected",
            "confidence": 0.98,
            "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
            "object_id": 7,
            "timestamp": "2026-03-26T10:00:00Z",
        }

        with patch.object(svc, "_publish_event_redis", return_value=False), \
             patch.object(svc, "_publish_event_mqtt", return_value=False):
            result = asyncio.run(svc.send_detection_event("cam1", [sample_event]))

        assert result is False
        pending = svc.get_pending_detection_events()
        assert len(pending) == 1
        assert pending[0]["camera_id"] == "cam1"
        assert pending[0]["event_data"]["type"] == "fall_detected"
        # fall_detected → person 카테고리
        assert pending[0]["data_category"] == "person"

    def test_replay_marks_outbox_row_sent(self, svc):
        svc._init_outbox()
        svc.devices["cam1"] = "device-1"
        sample_event = {
            "camera_id": "cam1",
            "type": "person",
            "confidence": 0.91,
            "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
            "timestamp": "2026-03-26T10:00:00Z",
        }
        svc._store_failed_detection_event("cam1", sample_event, "network down")
        pending = svc.get_pending_detection_events()

        with patch.object(svc, "_publish_event_redis", return_value=False), \
             patch.object(svc, "_publish_event_mqtt", return_value=True):
            outbox_ref = (pending[0]["_table"], pending[0]["id"])
            result = asyncio.run(
                svc.replay_detection_event(outbox_ref, "cam1", sample_event)
            )

        assert result is True
        assert svc.get_pending_detection_events() == []

    def test_expire_pending_detection_events_marks_ttl_rows(self, svc):
        svc._init_outbox()
        row_ref = svc._store_pending_event("cam1", {"type": "person"})
        assert row_ref is not None

        table, row_id = row_ref

        import sqlite3
        with sqlite3.connect(str(svc.outbox_db_path)) as conn:
            conn.execute(
                f"UPDATE {table} SET expire_at = datetime('now', '-1 day') WHERE id = ?",
                (row_id,),
            )
            conn.commit()

        expired = svc.expire_pending_detection_events()

        assert expired == 1
        assert svc.get_pending_detection_events() == []

        with sqlite3.connect(str(svc.outbox_db_path)) as conn:
            status = conn.execute(
                f"SELECT status FROM {table} WHERE id = ?",
                (row_id,),
            ).fetchone()[0]
        assert status == "expired"


# ---------------------------------------------------------------------------
# data_category 분류 테스트
# ---------------------------------------------------------------------------

class TestEventCategoryClassification:
    """_classify_event_category 와 data_category 저장 검증."""

    def test_person_events_classified_as_person(self, svc):
        for event_type in ("helmet", "head", "fall_detected", "not_fall",
                           "face_recognized", "face_unknown", "person",
                           "unsafe_behavior", "wearing_helmet"):
            assert svc._classify_event_category(event_type) == "person", \
                f"{event_type!r} should be 'person'"

    def test_camera_events_classified_as_camera(self, svc):
        for event_type in ("other", "unknown_event"):
            assert svc._classify_event_category(event_type) == "camera", \
                f"{event_type!r} should be 'camera'"

    def test_zone_events_classified_as_zone(self, svc):
        for event_type in ("danger_zone", "intrusion", "zone_entered",
                           "zone_dwelling", "zone_object_detected",
                           "crowd_warning"):
            assert svc._classify_event_category(event_type) == "zone", \
                f"{event_type!r} should be 'zone'"

    def test_sensor_events_classified_as_sensor(self, svc):
        for event_type in ("tilt_alert", "temperature_alert",
                           "vibration_alert", "sensor_data"):
            assert svc._classify_event_category(event_type) == "sensor", \
                f"{event_type!r} should be 'sensor'"

    def test_empty_event_type_returns_camera(self, svc):
        assert svc._classify_event_category("") == "camera"
        assert svc._classify_event_category(None) == "camera"  # type: ignore[arg-type]

    def test_stored_person_event_has_category(self, svc):
        svc._init_outbox()
        svc._store_failed_detection_event(
            "cam1", {"type": "helmet", "confidence": 0.9}, "net error"
        )
        pending = svc.get_pending_detection_events()
        assert pending[0]["data_category"] == "person"

    def test_stored_camera_event_has_category(self, svc):
        svc._init_outbox()
        svc._store_failed_detection_event(
            "cam2", {"type": "intrusion", "confidence": 0.75}, "net error"
        )
        pending = svc.get_pending_detection_events()
        assert pending[0]["data_category"] == "zone"

    def test_filter_by_person_category(self, svc):
        svc._init_outbox()
        svc._store_failed_detection_event("cam1", {"type": "helmet"}, "err")
        svc._store_failed_detection_event("cam1", {"type": "intrusion"}, "err")
        svc._store_failed_detection_event("cam1", {"type": "fall_detected"}, "err")

        person_rows = svc.get_pending_detection_events(data_category="person")
        zone_rows   = svc.get_pending_detection_events(data_category="zone")
        all_rows    = svc.get_pending_detection_events()

        assert len(person_rows) == 2   # helmet + fall_detected
        assert len(zone_rows) == 1     # intrusion
        assert len(all_rows) == 3

    def test_filter_by_zone_category(self, svc):
        svc._init_outbox()
        svc._store_failed_detection_event("cam1", {"type": "danger_zone"}, "err")
        svc._store_failed_detection_event("cam1", {"type": "person"}, "err")

        zone_only = svc.get_pending_detection_events(data_category="zone")
        assert len(zone_only) == 1
        assert zone_only[0]["event_data"]["type"] == "danger_zone"

    def test_sensor_stored_in_event_outbox(self, svc):
        """sensor 이벤트는 공통 event_outbox 테이블에 category로 저장된다."""
        svc._init_outbox()
        svc._store_failed_detection_event("s1", {"type": "tilt_alert"}, "err")
        svc._store_failed_detection_event("s2", {"type": "vibration_alert"}, "err")

        import sqlite3
        with sqlite3.connect(str(svc.outbox_db_path)) as conn:
            sensor_count = conn.execute(
                "SELECT COUNT(*) FROM event_outbox WHERE data_category = 'sensor'"
            ).fetchone()[0]
            total_count = conn.execute(
                "SELECT COUNT(*) FROM event_outbox"
            ).fetchone()[0]

        assert sensor_count == 2
        assert total_count == 2

    def test_zone_stored_in_event_outbox(self, svc):
        """zone 이벤트는 공통 event_outbox 테이블에 category로 저장된다."""
        svc._init_outbox()
        svc._store_failed_detection_event("cam1", {"type": "intrusion"}, "err")
        svc._store_failed_detection_event("cam2", {"type": "crowd_warning"}, "err")

        import sqlite3
        with sqlite3.connect(str(svc.outbox_db_path)) as conn:
            zone_count = conn.execute(
                "SELECT COUNT(*) FROM event_outbox WHERE data_category = 'zone'"
            ).fetchone()[0]
            total_count = conn.execute("SELECT COUNT(*) FROM event_outbox").fetchone()[0]

        assert zone_count == 2
        assert total_count == 2

    def test_pending_returns_table_key(self, svc):
        """get_pending_detection_events 반환값에 _table 키가 포함된다."""
        svc._init_outbox()
        svc._store_failed_detection_event("cam1", {"type": "tilt_alert"}, "err")
        pending = svc.get_pending_detection_events(data_category="sensor")
        assert len(pending) == 1
        assert pending[0]["_table"] == "event_outbox"


class TestEdgeXAdapterOutbox:
    @redis_required
    def test_replay_outbox_once_runs_expire_cleanup_first(self, tmp_path):
        from src.edgex.adapter_service import EdgeXDeviceAdapterService

        service = EdgeXDeviceAdapterService(
            ai_mqtt_broker="localhost",
            ai_mqtt_port=1883,
            metadata_url="http://localhost:59881",
            data_url="http://localhost:59880",
            outbox_db_path=str(tmp_path / "adapter_outbox.db"),
        )

        service.edgex_service._init_outbox()
        row_ref = service.edgex_service._store_pending_event(
            "cam1", {"type": "person", "camera_id": "cam1"}
        )
        assert row_ref is not None

        table, row_id = row_ref

        import sqlite3
        with sqlite3.connect(str(service.edgex_service.outbox_db_path)) as conn:
            conn.execute(
                f"UPDATE {table} SET expire_at = datetime('now', '-1 day') WHERE id = ?",
                (row_id,),
            )
            conn.commit()

        with patch.object(service, "_run_coro") as run_coro:
            service._replay_outbox_once()

        run_coro.assert_not_called()
        assert service.edgex_service.get_pending_detection_events() == []
