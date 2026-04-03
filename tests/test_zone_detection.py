"""
test_zone_detection.py — Zone / ZoneManager / ZoneEvent 단위 테스트

커버리지 대상:
  - Zone.contains_point, Zone.intersects_bbox
  - ZoneEvent.to_dict
  - ZoneManager.load_zones, check_zones, save_zones
"""

import json
import time

import numpy as np
import pytest

from src.core.events import DetectionEvent, EventType
from src.utils.zone_detection import PolygonZone as Zone, ZoneEvent, ZoneEventType, ZoneManager

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

# 100×100 정사각형 폴리곤 (좌상단 원점)
SQUARE_100 = [[0, 0], [100, 0], [100, 100], [0, 100]]


def _det(x: int, y: int, w: int = 20, h: int = 20, oid: int = 1, conf: float = 0.9) -> DetectionEvent:
    """테스트용 DetectionEvent 팩토리."""
    return DetectionEvent(
        event_type=EventType.PERSON,
        x=x, y=y, width=w, height=h,
        confidence=conf,
        timestamp=time.time(),
        object_id=oid,
    )


# ---------------------------------------------------------------------------
# Zone
# ---------------------------------------------------------------------------


class TestZone:
    def _square(self) -> Zone:
        return Zone("z1", SQUARE_100)

    # contains_point ---------------------------------------------------------
    def test_contains_point_inside(self):
        assert self._square().contains_point((50, 50)) is True

    def test_contains_point_on_vertex(self):
        # cv2.pointPolygonTest >= 0 → 경계 포함
        assert self._square().contains_point((0, 0)) is True

    def test_contains_point_outside(self):
        assert self._square().contains_point((150, 150)) is False

    def test_contains_point_negative_coords(self):
        assert self._square().contains_point((-1, 50)) is False

    # intersects_bbox --------------------------------------------------------
    def test_intersects_bbox_fully_inside(self):
        z = self._square()
        assert z.intersects_bbox({"x": 10, "y": 10, "width": 20, "height": 20}) is True

    def test_intersects_bbox_outside(self):
        z = self._square()
        assert z.intersects_bbox({"x": 200, "y": 200, "width": 20, "height": 20}) is False

    def test_intersects_bbox_center_inside(self):
        """코너는 밖이지만 중심점이 폴리곤 안에 있는 경우."""
        z = self._square()
        # 박스 (-50, -50, 200×200) → 중심 (50, 50) 은 안에 있음
        assert z.intersects_bbox({"x": -50, "y": -50, "width": 200, "height": 200}) is True

    def test_intersects_bbox_corner_overlap(self):
        """박스 한쪽 코너만 폴리곤 안에 있는 경우."""
        z = self._square()
        # 박스 (80, 80, 50×50) → 좌상단 (80,80) 이 안에 있음
        assert z.intersects_bbox({"x": 80, "y": 80, "width": 50, "height": 50}) is True

    # name -------------------------------------------------------------------
    def test_name_defaults_to_zone_id(self):
        z = Zone("myzone", SQUARE_100)
        assert z.name == "myzone"

    def test_name_custom(self):
        z = Zone("z1", SQUARE_100, "위험구역")
        assert z.name == "위험구역"

    # draw (smoke test) -------------------------------------------------------
    def test_draw_does_not_raise(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        self._square().draw(frame)


# ---------------------------------------------------------------------------
# ZoneEvent
# ---------------------------------------------------------------------------


class TestZoneEvent:
    def _make(self, etype: ZoneEventType = ZoneEventType.ENTERED) -> ZoneEvent:
        return ZoneEvent(
            event_type=etype,
            zone_id="zone1",
            object_id=42,
            camera_id="cam1",
            bbox={"x": 10, "y": 10, "width": 30, "height": 30},
            confidence=0.85,
            timestamp=1000.0,
        )

    def test_to_dict_entered(self):
        d = self._make(ZoneEventType.ENTERED).to_dict()
        assert d["event_type"] == "zone_entered"

    def test_to_dict_exited(self):
        d = self._make(ZoneEventType.EXITED).to_dict()
        assert d["event_type"] == "zone_exited"

    def test_to_dict_dwelling_with_seconds(self):
        ev = ZoneEvent(
            event_type=ZoneEventType.DWELLING,
            zone_id="z1", object_id=1, camera_id="cam1",
            bbox={}, confidence=0.5, dwelling_seconds=7.3,
        )
        d = ev.to_dict()
        assert d["event_type"] == "zone_dwelling"
        assert d["dwelling_seconds"] == pytest.approx(7.3)

    def test_to_dict_contains_required_keys(self):
        d = self._make().to_dict()
        for key in ("zone_id", "object_id", "camera_id", "bbox", "confidence"):
            assert key in d, f"필수 키 누락: {key}"

    def test_to_dict_zone_id_value(self):
        d = self._make().to_dict()
        assert d["zone_id"] == "zone1"
        assert d["object_id"] == 42


# ---------------------------------------------------------------------------
# ZoneManager – 로드
# ---------------------------------------------------------------------------


class TestZoneManagerLoad:
    @pytest.fixture
    def zm(self, tmp_path) -> ZoneManager:
        cfg_file = tmp_path / "zones_config.json"
        cfg_file.write_text(
            json.dumps({"dwelling_threshold_seconds": 2.0, "cameras": {}})
        )
        return ZoneManager(zones_config=str(cfg_file))

    def test_load_zones_from_data(self, zm):
        zm.load_zones("cam1", [{"id": "z1", "name": "위험", "polygon": SQUARE_100}])
        assert "cam1" in zm.zones
        assert "z1" in zm.zones["cam1"]

    def test_load_multiple_zones(self, zm):
        zm.load_zones("cam1", [
            {"id": "z1", "polygon": SQUARE_100},
            {"id": "z2", "polygon": SQUARE_100},
        ])
        assert len(zm.zones["cam1"]) == 2

    def test_load_empty_data_clears_zones(self, zm):
        zm.load_zones("cam1", [{"id": "z1", "polygon": SQUARE_100}])
        zm.load_zones("cam1", [])
        assert zm.zones["cam1"] == {}

    def test_reload_clears_object_states(self, zm):
        zm.load_zones("cam1", [{"id": "z1", "polygon": SQUARE_100}])
        zm.object_states["cam1"][99] = True  # 임의 상태 주입
        zm.load_zones("cam1", [{"id": "z1", "polygon": SQUARE_100}])
        assert 99 not in zm.object_states["cam1"]

    def test_dwelling_threshold_from_config(self, tmp_path):
        cfg = {"dwelling_threshold_seconds": 7.0, "cameras": {}}
        f = tmp_path / "z.json"
        f.write_text(json.dumps(cfg))
        zm = ZoneManager(zones_config=str(f))
        assert zm.dwelling_threshold == pytest.approx(7.0)

    def test_missing_config_file_uses_default_threshold(self):
        zm = ZoneManager(zones_config="/no/such/file.json")
        assert zm.dwelling_threshold == pytest.approx(3.0)

    def test_load_distinct_cameras_independent(self, zm):
        zm.load_zones("cam1", [{"id": "z1", "polygon": SQUARE_100}])
        zm.load_zones("cam2", [{"id": "z2", "polygon": SQUARE_100}])
        assert "z1" in zm.zones["cam1"]
        assert "z2" in zm.zones["cam2"]


# ---------------------------------------------------------------------------
# ZoneManager – check_zones
# ---------------------------------------------------------------------------


class TestZoneManagerCheckZones:
    @pytest.fixture
    def zm(self, tmp_path) -> ZoneManager:
        f = tmp_path / "zones_config.json"
        f.write_text(json.dumps({"dwelling_threshold_seconds": 9999.0, "cameras": {}}))
        mgr = ZoneManager(zones_config=str(f))
        mgr.load_zones("cam1", [{"id": "z1", "name": "Test", "polygon": SQUARE_100}])
        return mgr

    def test_unknown_camera_returns_empty(self, zm):
        events = zm.check_zones("unknown_cam", [_det(50, 50)])
        assert events == []

    def test_no_zones_returns_empty(self, zm):
        zm.load_zones("cam_no_zone", [])
        events = zm.check_zones("cam_no_zone", [_det(50, 50)])
        assert events == []

    def test_entered_event_on_first_detection_inside(self, zm):
        d = _det(50, 50)
        events = zm.check_zones("cam1", [d])
        entered = [e for e in events if e.event_type == ZoneEventType.ENTERED]
        assert len(entered) == 1
        assert entered[0].zone_id == "z1"
        assert entered[0].object_id == 1

    def test_no_repeated_entered_event(self, zm):
        zm.check_zones("cam1", [_det(50, 50)])  # 1차 진입
        events = zm.check_zones("cam1", [_det(50, 50)])  # 이미 안에 있음
        entered = [e for e in events if e.event_type == ZoneEventType.ENTERED]
        assert len(entered) == 0

    def test_exited_event_when_leaving_zone(self, zm):
        zm.check_zones("cam1", [_det(50, 50)])              # 진입
        events = zm.check_zones("cam1", [_det(200, 200)])   # 퇴장 (밖)
        exited = [e for e in events if e.event_type == ZoneEventType.EXITED]
        assert len(exited) == 1

    def test_outside_detection_produces_no_events(self, zm):
        events = zm.check_zones("cam1", [_det(200, 200)])
        assert events == []

    def test_disappeared_object_cleaned_from_state(self, zm):
        zm.check_zones("cam1", [_det(50, 50, oid=5)])
        # object_states keys are (zone_id, object_id) tuples
        assert any(oid == 5 for _, oid in zm.object_states["cam1"])
        zm.check_zones("cam1", [])  # 객체 사라짐
        assert not any(oid == 5 for _, oid in zm.object_states["cam1"])

    def test_dwelling_event_after_threshold(self, tmp_path):
        f = tmp_path / "zones_config.json"
        f.write_text(json.dumps({"dwelling_threshold_seconds": 0.05, "cameras": {}}))
        mgr = ZoneManager(zones_config=str(f))
        mgr.load_zones("cam1", [{"id": "z1", "polygon": SQUARE_100}])

        mgr.check_zones("cam1", [_det(50, 50)])  # 진입
        time.sleep(0.1)                           # 50ms 초과
        events = mgr.check_zones("cam1", [_det(50, 50)])  # 체류 확인
        dwelling = [e for e in events if e.event_type == ZoneEventType.DWELLING]
        assert len(dwelling) == 1
        assert dwelling[0].dwelling_seconds > 0

    def test_no_dwelling_before_threshold(self, zm):
        zm.dwelling_threshold = 9999.0
        zm.check_zones("cam1", [_det(50, 50)])  # 진입
        events = zm.check_zones("cam1", [_det(50, 50)])  # 즉시 체크
        dwelling = [e for e in events if e.event_type == ZoneEventType.DWELLING]
        assert len(dwelling) == 0

    def test_multiple_objects_independent(self, zm):
        d1 = _det(50, 50, oid=1)
        d2 = _det(60, 60, oid=2)
        events = zm.check_zones("cam1", [d1, d2])
        entered = [e for e in events if e.event_type == ZoneEventType.ENTERED]
        assert len(entered) == 2


# ---------------------------------------------------------------------------
# ZoneManager – save_zones
# ---------------------------------------------------------------------------


class TestZoneManagerSave:
    ZONES = [{"id": "z1", "name": "위험구역", "polygon": SQUARE_100}]

    def test_save_to_zones_config(self, tmp_path):
        cfg = {"dwelling_threshold_seconds": 3.0, "cameras": {}}
        f = tmp_path / "zones_config.json"
        f.write_text(json.dumps(cfg), encoding="utf-8")
        zm = ZoneManager(zones_config=str(f))
        zm.save_zones("cam1", self.ZONES)

        saved = json.loads(f.read_text(encoding="utf-8"))
        assert "cam1" in saved["cameras"]
        assert saved["cameras"]["cam1"]["zones"] == self.ZONES

    def test_save_to_cameras_json(self, tmp_path):
        cameras_data = [{"id": "cam1", "source": "rtsp://example.com/stream"}]
        cam_file = tmp_path / "cameras.json"
        cam_file.write_text(json.dumps(cameras_data), encoding="utf-8")
        zm = ZoneManager(zones_config="/no/file.json")
        zm.save_zones("cam1", self.ZONES, cameras_config_path=str(cam_file))

        saved = json.loads(cam_file.read_text(encoding="utf-8"))
        cam = next(c for c in saved if c["id"] == "cam1")
        assert cam["zones"] == self.ZONES

    def test_save_atomic_no_tmp_left(self, tmp_path):
        """tmp → replace 원자적 저장 후 .tmp 파일이 남지 않아야 함."""
        f = tmp_path / "zones_config.json"
        f.write_text(json.dumps({"dwelling_threshold_seconds": 3.0, "cameras": {}}))
        zm = ZoneManager(zones_config=str(f))
        zm.save_zones("cam1", self.ZONES)
        assert not (tmp_path / "zones_config.tmp").exists()

    def test_save_updates_in_memory_zones(self, tmp_path):
        f = tmp_path / "zones_config.json"
        f.write_text(json.dumps({"dwelling_threshold_seconds": 3.0, "cameras": {}}))
        zm = ZoneManager(zones_config=str(f))
        zm.save_zones("cam1", self.ZONES)
        assert "z1" in zm.zones.get("cam1", {})

    def test_save_cameras_json_unknown_camera_logs_warning(self, tmp_path, caplog):
        cameras_data = [{"id": "cam999", "source": "rtsp://x"}]
        cam_file = tmp_path / "cameras.json"
        cam_file.write_text(json.dumps(cameras_data))
        zm = ZoneManager(zones_config="/no/file.json")
        import logging
        with caplog.at_level(logging.WARNING):
            zm.save_zones("cam_missing", self.ZONES, cameras_config_path=str(cam_file))
        assert any("찾을 수 없습니다" in r.message for r in caplog.records)
