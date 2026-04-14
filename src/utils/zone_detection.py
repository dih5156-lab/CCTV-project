"""위험구역 진입/체류 탐지 모듈.

카메라별 폴리곤/라인 기반 위험 구역을 정의하고, 객체 바운딩박스와의
교차 여부를 구해 진입/퇴장/체류 이벤트를 생성한다.
다중 주시::

    zone_mgr = ZoneManager(zones_config='zones_config.json')
    zone_mgr.load_zones('cam1')
    events = zone_mgr.check_zones('cam1', detections)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ZoneMode(str, Enum):
    """구역 동작 모드"""
    DANGER      = "danger"        # 위험구역: 객체 진입 즉시 경고
    OBJECT_WATCH = "object_watch" # 객체감시: 특정 클래스 진입 시 경고
    CROWD_COUNT  = "crowd_count"  # 인파분석: 구역 내 객체 수 임계값 초과 시 경고


class ZoneEventType(str, Enum):
    """구역 이벤트 타입"""
    ENTERED       = "zone_entered"
    EXITED        = "zone_exited"
    DWELLING      = "zone_dwelling"
    CROWD_WARNING = "crowd_warning"       # 인파 임계값 초과
    OBJECT_DETECTED = "zone_object_detected"  # 감시 객체 탐지


@dataclass
class ZoneEvent:
    """구역 이벤트"""
    event_type: ZoneEventType  # ZoneEventType enum 사용
    zone_id: str
    object_id: int
    camera_id: str
    bbox: Dict  # {'x', 'y', 'width', 'height'}
    confidence: float
    timestamp: float = field(default_factory=time.time)
    dwelling_seconds: float = 0.0  # 체류 시간 (dwelling 이벤트일 때만)
    metadata: Dict = field(default_factory=dict)  # 추가 정보 (count 등)

    def to_dict(self) -> Dict:
        data = asdict(self)
        # event_type을 문자열로 변환
        data['event_type'] = self.event_type.value
        return data


class Zone:
    """위험 구역 정의 베이스 클래스."""

    zone_type = "polygon"

    def __init__(self, zone_id: str, name: str = ""):
        self.zone_id = zone_id
        self.name = name or zone_id

    def to_dict(self) -> Dict:
        raise NotImplementedError

    def intersects_bbox(self, bbox: Dict) -> bool:
        raise NotImplementedError

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ):
        raise NotImplementedError


# 구역 모드별 시각화 색상 (BGR)
_ZONE_MODE_COLORS: Dict[str, Tuple[int, int, int]] = {
    ZoneMode.DANGER:       (0, 0, 255),    # 빨강 - 위험구역
    ZoneMode.OBJECT_WATCH: (0, 165, 255),  # 주황 - 객체감시
    ZoneMode.CROWD_COUNT:  (255, 0, 255),  # 자주 - 인파분석
}


class PolygonZone(Zone):
    """폴리곤 기반 위험 구역."""

    zone_type = "polygon"

    def __init__(
        self,
        zone_id: str,
        polygon: List[Tuple[int, int]],
        name: str = "",
        mode: str = ZoneMode.DANGER,
        watch_classes: Optional[List[str]] = None,
        count_classes: Optional[List[str]] = None,
        count_threshold: int = 5,
        alert_cooldown: float = 30.0,
    ):
        """
        매개변수:
            zone_id:         구역 ID (예: 'zone_1')
            polygon:         폴리곤 좌표 [(x1, y1), (x2, y2), ...]
            name:            구역 이름 (예: '전기설비')
            mode:            ZoneMode (danger / object_watch / crowd_count)
            watch_classes:   [object_watch] 감시할 이벤트 타입 목록 (예: ['person', 'head'])
            count_classes:   [crowd_count] 카운트할 이벤트 타입 목록 (기본: ['person'])
            count_threshold: [crowd_count] 경고 임계값 (기본 5명)
            alert_cooldown:  [crowd_count/object_watch] 반복 경고 억제 간격(초)
        """
        super().__init__(zone_id, name)
        self.polygon = np.array(polygon, dtype=np.int32)
        self.mode = ZoneMode(mode) if isinstance(mode, str) else mode
        self.watch_classes = [c.lower() for c in (watch_classes or ["person"])]
        self.count_classes = [c.lower() for c in (count_classes or ["person"])]
        self.count_threshold = int(count_threshold)
        self.alert_cooldown = float(alert_cooldown)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """점이 폴리곤 내부에 있는지 확인"""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def intersects_bbox(self, bbox: Dict) -> bool:
        """바운딩박스와 폴리곤이 교차하는지 확인

        매개변수:
            bbox: {'x', 'y', 'width', 'height'} (좌상단 기준)
        """
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        if any(self.contains_point(c) for c in corners):
            return True

        return self.contains_point((int(x + w / 2), int(y + h / 2)))

    def draw(
        self,
        frame: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: int = 2,
    ):
        """프레임에 폴리곤 그리기 (모드별 색상 자동 적용)"""
        c = color if color is not None else _ZONE_MODE_COLORS.get(self.mode, (0, 255, 0))
        cv2.polylines(frame, [self.polygon], True, c, thickness)
        if len(self.polygon) > 0:
            x, y = self.polygon[0]
            mode_tag = f" [{self.mode.value}]" if self.mode != ZoneMode.DANGER else ""
            cv2.putText(
                frame, self.name + mode_tag,
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2,
            )

    def to_dict(self) -> Dict:
        d: Dict = {
            "id": self.zone_id,
            "name": self.name,
            "type": self.zone_type,
            "polygon": self.polygon.tolist(),
            "mode": self.mode.value,
        }
        if self.mode == ZoneMode.OBJECT_WATCH:
            d["watch_classes"] = self.watch_classes
            d["alert_cooldown"] = self.alert_cooldown
        elif self.mode == ZoneMode.CROWD_COUNT:
            d["count_classes"] = self.count_classes
            d["count_threshold"] = self.count_threshold
            d["alert_cooldown"] = self.alert_cooldown
        return d


class LineZone(Zone):
    """선분 교차 기반 위험 구역."""

    zone_type = "line"

    def __init__(
        self,
        zone_id: str,
        points: List[Tuple[int, int]],
        name: str = "",
        direction: str = "both",
    ):
        super().__init__(zone_id, name)
        self.points = np.array(points, dtype=np.int32)
        self.direction = direction or "both"

    def intersects_bbox(self, bbox: Dict) -> bool:
        """라인 구역은 bbox 점유 개념이 없어 항상 False를 반환한다."""
        return False

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ):
        if len(self.points) < 2:
            return
        start = tuple(int(v) for v in self.points[0])
        end = tuple(int(v) for v in self.points[1])
        cv2.line(frame, start, end, color, thickness)
        label_x = int((start[0] + end[0]) / 2)
        label_y = int((start[1] + end[1]) / 2) - 10
        cv2.putText(frame, self.name, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def to_dict(self) -> Dict:
        return {
            "id": self.zone_id,
            "name": self.name,
            "type": self.zone_type,
            "points": self.points.tolist(),
            "direction": self.direction,
        }


class ZoneManager:
    """구역 관리 및 이벤트 감지"""
    
    def __init__(self, zones_config: str = 'zones_config.json'):
        """
        매개변수:
            zones_config: 구역 설정 JSON 파일 경로
        """
        self.zones_config_path = zones_config
        self.zones: Dict[str, Dict[str, Zone]] = {}  # camera_id -> {zone_id -> Zone}
        self.object_states: Dict[str, Dict[Tuple[str, int], bool]] = {}  # camera_id -> {(zone_id, object_id) -> in_zone}
        self.object_enter_time: Dict[str, Dict[Tuple[str, int], float]] = {}  # camera_id -> {(zone_id, object_id) -> timestamp}
        self.object_positions: Dict[str, Dict[Tuple[str, int], Tuple[float, float]]] = {}  # camera_id -> {(zone_id, object_id) -> point}
        self.dwelling_threshold: float = 3.0  # 체류 시간 임계값 (초)
        self.sent_dwelling_events: Dict[Tuple[str, str, int], float] = {}  # (camera_id, zone_id, object_id) -> last_event_time
        # 군중/감시 구역 마지막 경보 시각: (camera_id, zone_id) -> timestamp
        self._crowd_last_alert: Dict[Tuple[str, str], float] = {}
        self._watch_last_alert: Dict[Tuple[str, str, int], float] = {}  # (cam, zone, obj_id)
        # 군중 구역 마지막 카운트 (오버레이 표시용): camera_id -> {zone_id -> count}
        self._last_crowd_counts: Dict[str, Dict[str, int]] = {}

        self._load_config()

    def _load_config(self):
        """JSON 설정 파일 로드"""
        try:
            config = json.loads(Path(self.zones_config_path).read_text(encoding='utf-8'))
            self.dwelling_threshold = config.get('dwelling_threshold_seconds', 3.0)
        except FileNotFoundError:
            logger.warning("%s 파일을 찾을 수 없습니다. 빈 설정으로 시작합니다.", self.zones_config_path)

    def load_zones(self, camera_id: str, zones_data: Optional[List[Dict]] = None):
        """카메라의 구역 로드
        
        매개변수:
            camera_id: 카메라 ID
            zones_data: [{'id': 'zone_1', 'name': '전기설비', 'polygon': [[x1,y1], ...]}]
        """
        if zones_data is None:
            try:
                config = json.loads(Path(self.zones_config_path).read_text(encoding='utf-8'))
                zones_data = config.get('cameras', {}).get(camera_id, {}).get('zones', [])
            except Exception as e:
                logger.error("구역 로드 오류 (%s): %s", camera_id, e)
                zones_data = []

        self.zones[camera_id] = {}
        for zone_def in zones_data:
            zone_id = zone_def['id']
            name = zone_def.get('name', zone_id)
            zone_type = zone_def.get('type', 'polygon')
            if zone_type == 'line':
                points = zone_def.get('points', [])
                direction = zone_def.get('direction', 'both')
                self.zones[camera_id][zone_id] = LineZone(zone_id, points, name, direction)
            else:
                polygon = zone_def['polygon']
                mode = zone_def.get('mode', ZoneMode.DANGER)
                self.zones[camera_id][zone_id] = PolygonZone(
                    zone_id, polygon, name,
                    mode=mode,
                    watch_classes=zone_def.get('watch_classes'),
                    count_classes=zone_def.get('count_classes'),
                    count_threshold=zone_def.get('count_threshold', 5),
                    alert_cooldown=zone_def.get('alert_cooldown', 30.0),
                )
        
        self.object_states[camera_id] = {}
        self.object_enter_time[camera_id] = {}
        self.object_positions[camera_id] = {}

        logger.info("%s에 %s개 구역 로드됨", camera_id, len(self.zones[camera_id]))

    def save_zones(
        self,
        camera_id: str,
        zones_data: List[Dict],
        cameras_config_path: Optional[str] = None,
    ) -> None:
        """카메라 구역을 파일에 저장하고 메모리를 즉시 갱신한다.

        cameras_config_path가 주어지면 cameras.json에 저장하고,
        없으면 zones_config.json 형식으로 저장한다.

        매개변수:
            camera_id: 카메라 ID
            zones_data: [{'id': ..., 'name': ..., 'polygon': [[x,y], ...]}]
            cameras_config_path: cameras.json 경로 (없으면 zones_config.json 사용)
        """
        if cameras_config_path:
            self._save_to_cameras_json(camera_id, zones_data, cameras_config_path)
        else:
            self._save_to_zones_config(camera_id, zones_data)
        self.load_zones(camera_id, zones_data)

    def _save_to_cameras_json(
        self, camera_id: str, zones_data: List[Dict], path: str
    ) -> None:
        """cameras.json의 해당 카메라 zones 필드를 업데이트한다."""
        p = Path(path)
        cameras = json.loads(p.read_text(encoding='utf-8'))
        for cam in cameras:
            if cam.get('id') == camera_id:
                cam['zones'] = zones_data
                break
        else:
            logger.warning("[%s] cameras.json에서 카메라를 찾을 수 없습니다", camera_id)
        tmp = p.with_suffix('.tmp')
        tmp.write_text(
            json.dumps(cameras, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        tmp.replace(p)
        logger.info(
            "[%s] zones를 cameras.json에 저장했습니다 (%d개)", camera_id, len(zones_data)
        )

    def _save_to_zones_config(self, camera_id: str, zones_data: List[Dict]) -> None:
        """zones_config.json의 해당 카메라 zones 필드를 업데이트한다."""
        p = Path(self.zones_config_path)
        try:
            config = json.loads(p.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            config = {
                "dwelling_threshold_seconds": self.dwelling_threshold,
                "cameras": {},
            }
        cam_entry = config.setdefault('cameras', {}).setdefault(camera_id, {})
        cam_entry['id'] = camera_id
        cam_entry['zones'] = zones_data
        tmp = p.with_suffix('.tmp')
        tmp.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        tmp.replace(p)
        logger.info(
            "[%s] zones를 zones_config.json에 저장했습니다 (%d개)", camera_id, len(zones_data)
        )

    def check_zones(
        self,
        camera_id: str,
        detections: List  # ai_analysis.DetectionEvent 리스트
    ) -> List[ZoneEvent]:
        """구역과 탐지 객체 교차 검사
        
        매개변수:
            camera_id: 카메라 ID
            detections: 현재 프레임의 탐지 결과 리스트

        반환값:
            ZoneEvent 리스트
        """
        events = []

        if camera_id not in self.zones or not self.zones[camera_id]:
            return events

        # 현재 프레임에서 탐지된 객체 ID 집합
        current_object_ids = set()

        # CROWD_COUNT 용: zone_id → zone내 객체 수 누적
        crowd_counts: Dict[str, int] = {}

        for detection in detections:
            object_id = detection.object_id or 0
            bbox_dict = detection.to_dict().get('bbox', {})
            current_object_ids.add(object_id)
            anchor_point = self._get_anchor_point(bbox_dict)
            # class_name 우선 사용, 없으면 event_type.value 로 fallback
            det_class = (
                getattr(detection, 'class_name', None)
                or detection.event_type.value.lower()
            )

            # 각 구역과 교차 검사
            for zone_id, zone in self.zones[camera_id].items():
                zone_key = (zone_id, object_id)

                if isinstance(zone, LineZone):
                    prev_point = self.object_positions[camera_id].get(zone_key)
                    event_type = self._check_line_crossing(zone, prev_point, anchor_point)
                    if event_type is not None:
                        events.append(ZoneEvent(
                            event_type=event_type,
                            zone_id=zone_id,
                            object_id=object_id,
                            camera_id=camera_id,
                            bbox=bbox_dict,
                            confidence=detection.confidence,
                        ))
                    self.object_positions[camera_id][zone_key] = anchor_point
                    continue

                # --- PolygonZone ---
                in_zone = zone.intersects_bbox(bbox_dict)
                mode = getattr(zone, 'mode', ZoneMode.DANGER)

                # CROWD_COUNT: 해당 클래스가 zone 안에 있으면 카운트만 누적
                if mode == ZoneMode.CROWD_COUNT:
                    count_classes = getattr(zone, 'count_classes', ['person'])
                    if in_zone and det_class in count_classes:
                        crowd_counts[zone_id] = crowd_counts.get(zone_id, 0) + 1
                    continue

                # OBJECT_WATCH: watch_classes에 속하지 않으면 무시
                if mode == ZoneMode.OBJECT_WATCH:
                    watch_classes = getattr(zone, 'watch_classes', ['person'])
                    if det_class not in watch_classes:
                        continue  # 감시 대상 아님 → 스킵

                # DANGER / OBJECT_WATCH 공통 진입·체류·퇴장 로직
                if in_zone and zone_key not in self.object_states[camera_id]:
                    self.object_states[camera_id][zone_key] = False

                prev_in_zone = self.object_states[camera_id].get(zone_key, False)

                if in_zone and not prev_in_zone:
                    ev_type = (ZoneEventType.OBJECT_DETECTED
                               if mode == ZoneMode.OBJECT_WATCH
                               else ZoneEventType.ENTERED)
                    events.append(ZoneEvent(
                        event_type=ev_type,
                        zone_id=zone_id,
                        object_id=object_id,
                        camera_id=camera_id,
                        bbox=bbox_dict,
                        confidence=detection.confidence,
                        metadata={"mode": mode.value, "class": det_class},
                    ))
                    self.object_states[camera_id][zone_key] = True
                    self.object_enter_time[camera_id][zone_key] = time.time()

                elif not in_zone and prev_in_zone:
                    events.append(ZoneEvent(
                        event_type=ZoneEventType.EXITED,
                        zone_id=zone_id,
                        object_id=object_id,
                        camera_id=camera_id,
                        bbox=bbox_dict,
                        confidence=detection.confidence,
                        metadata={"mode": mode.value},
                    ))
                    self.object_states[camera_id][zone_key] = False
                    self.object_enter_time[camera_id].pop(zone_key, None)

                elif in_zone and prev_in_zone:
                    enter_time = self.object_enter_time[camera_id].get(zone_key)
                    if enter_time:
                        now_ts = time.time()
                        dwelling_time = now_ts - enter_time
                        if dwelling_time >= self.dwelling_threshold:
                            key = (camera_id, zone_id, object_id)
                            last_event_time = self.sent_dwelling_events.get(key, 0)
                            if now_ts - last_event_time >= 1.0:
                                events.append(ZoneEvent(
                                    event_type=ZoneEventType.DWELLING,
                                    zone_id=zone_id,
                                    object_id=object_id,
                                    camera_id=camera_id,
                                    bbox=bbox_dict,
                                    confidence=detection.confidence,
                                    dwelling_seconds=dwelling_time,
                                    metadata={"mode": mode.value, "class": det_class},
                                ))
                                self.sent_dwelling_events[key] = now_ts

        # --- CROWD_COUNT 사후 처리 ---
        now_ts = time.time()
        cam_counts: Dict[str, int] = {}
        for zone_id, zone in self.zones[camera_id].items():
            if not isinstance(zone, PolygonZone):
                continue
            if getattr(zone, 'mode', ZoneMode.DANGER) != ZoneMode.CROWD_COUNT:
                continue
            count = crowd_counts.get(zone_id, 0)
            cam_counts[zone_id] = count
            threshold = getattr(zone, 'count_threshold', 5)
            cooldown = getattr(zone, 'alert_cooldown', 30.0)
            if count >= threshold:
                alert_key = (camera_id, zone_id)
                last_alert = self._crowd_last_alert.get(alert_key, 0.0)
                if now_ts - last_alert >= cooldown:
                    count_classes = getattr(zone, 'count_classes', ['person'])
                    events.append(ZoneEvent(
                        event_type=ZoneEventType.CROWD_WARNING,
                        zone_id=zone_id,
                        object_id=0,
                        camera_id=camera_id,
                        bbox={},
                        confidence=1.0,
                        metadata={"count": count, "threshold": threshold, "count_classes": count_classes},
                    ))
                    self._crowd_last_alert[alert_key] = now_ts
                    logger.info(
                        "[%s] 군중 경고: zone=%s 인원=%d명 (임계=%d)",
                        camera_id, zone_id, count, threshold,
                    )

        # 최신 카운트 저장 (오버레이 표시용)
        self._last_crowd_counts[camera_id] = cam_counts

        # 사라진 객체 정리
        for zone_key in list(self.object_states[camera_id].keys()):
            _, object_id = zone_key
            if object_id not in current_object_ids:
                self.object_states[camera_id].pop(zone_key, None)
                self.object_enter_time[camera_id].pop(zone_key, None)
        for zone_key in list(self.object_positions[camera_id].keys()):
            _, object_id = zone_key
            if object_id not in current_object_ids:
                self.object_positions[camera_id].pop(zone_key, None)

        return events

    def _get_anchor_point(self, bbox: Dict) -> Tuple[float, float]:
        """객체의 발 위치에 가까운 기준점(bottom-center)을 반환한다."""
        x = float(bbox.get('x', 0))
        y = float(bbox.get('y', 0))
        w = float(bbox.get('width', 0))
        h = float(bbox.get('height', 0))
        return (x + (w / 2.0), y + h)

    def _check_line_crossing(
        self,
        zone: LineZone,
        prev_point: Optional[Tuple[float, float]],
        curr_point: Tuple[float, float],
    ) -> Optional[ZoneEventType]:
        """이전/현재 기준점이 선분을 가로질렀는지 판정한다."""
        if prev_point is None or len(zone.points) < 2:
            return None

        line_start = tuple(float(v) for v in zone.points[0])
        line_end = tuple(float(v) for v in zone.points[1])
        prev_side = self._point_side(prev_point, line_start, line_end)
        curr_side = self._point_side(curr_point, line_start, line_end)

        if abs(prev_side) < 1e-6 or abs(curr_side) < 1e-6:
            return None
        if prev_side * curr_side >= 0:
            return None
        if not self._segments_intersect(prev_point, curr_point, line_start, line_end):
            return None

        moving_to_positive = prev_side < 0 < curr_side
        direction = zone.direction.lower()
        if direction in ("both", "bidirectional", "any"):
            return ZoneEventType.ENTERED if moving_to_positive else ZoneEventType.EXITED
        if direction in ("outside_to_inside", "forward"):
            return ZoneEventType.ENTERED if moving_to_positive else None
        if direction in ("inside_to_outside", "backward"):
            return ZoneEventType.EXITED if not moving_to_positive else None
        return ZoneEventType.ENTERED if moving_to_positive else ZoneEventType.EXITED

    def _point_side(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
    ) -> float:
        return (
            (line_end[0] - line_start[0]) * (point[1] - line_start[1])
            - (line_end[1] - line_start[1]) * (point[0] - line_start[0])
        )

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        q1: Tuple[float, float],
        q2: Tuple[float, float],
    ) -> bool:
        def orientation(a, b, c) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)
        return o1 * o2 <= 0 and o3 * o4 <= 0

    def draw_zones(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """프레임에 모든 구역 그리기"""
        if camera_id not in self.zones:
            return frame
        
        for zone in self.zones[camera_id].values():
            zone.draw(frame, color=(0, 255, 255), thickness=2)
        
        return frame


__all__ = ["Zone", "PolygonZone", "LineZone", "ZoneManager", "ZoneEvent", "ZoneMode", "ZoneEventType"]
