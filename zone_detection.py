"""
[zone_detection.py] 위험구역 진입/체류 탐지 모듈
제작일: 2025-11-20
설명:
  - 카메라별 위험 구역(폴리곤) 정의
  - 객체 바운딩박스와 위험 구역 교차 검사
  - 진입/퇴장/체류 이벤트 생성
  - 추적 ID 기반 중복 이벤트 방지

사용 예시:
    zone_mgr = ZoneManager(zones_config='zones_config.json')
    zone_mgr.load_zones('cam1')
    events = zone_mgr.check_zones('cam1', detections, tracking_ids)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from enum import Enum


class ZoneEventType(str, Enum):
    """구역 이벤트 타입"""
    ENTERED = "zone_entered"
    EXITED = "zone_exited"
    DWELLING = "zone_dwelling"


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

    def to_dict(self) -> Dict:
        data = asdict(self)
        # event_type을 문자열로 변환
        data['event_type'] = self.event_type.value
        return data


class Zone:
    """위험 구역 정의"""
    
    def __init__(self, zone_id: str, polygon: List[Tuple[int, int]], name: str = ""):
        """
        Args:
            zone_id: 구역 ID (예: 'zone_1')
            polygon: 폴리곤 좌표 [(x1, y1), (x2, y2), ...]
            name: 구역 이름 (예: '전기설비')
        """
        self.zone_id = zone_id
        self.polygon = np.array(polygon, dtype=np.int32)
        self.name = name or zone_id

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """점이 폴리곤 내부에 있는지 확인"""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def intersects_bbox(self, bbox: Dict) -> bool:
        """바운딩박스와 폴리곤이 교차하는지 확인
        
        Args:
            bbox: {'x', 'y', 'width', 'height'} (좌상단 기준)
        """
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        # bbox 좌상단 기준으로 코너 계산
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        
        # 바운딩박스의 4개 코너 중 하나라도 폴리곤 내부에 있으면 교차
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for corner in corners:
            if self.contains_point(corner):
                return True
        
        # 또는 중심점 확인
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        return self.contains_point((center_x, center_y))

    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        """프레임에 폴리곤 그리기"""
        cv2.polylines(frame, [self.polygon], True, color, thickness)
        # 구역 이름 표시
        if len(self.polygon) > 0:
            x, y = self.polygon[0]
            cv2.putText(frame, self.name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class ZoneManager:
    """구역 관리 및 이벤트 감지"""
    
    def __init__(self, zones_config: str = 'zones_config.json'):
        """
        Args:
            zones_config: 구역 설정 JSON 파일 경로
        """
        self.zones_config_path = zones_config
        self.zones: Dict[str, Dict[str, Zone]] = {}  # camera_id -> {zone_id -> Zone}
        self.object_states: Dict[str, Dict[str, bool]] = {}  # camera_id -> {object_id -> in_zone}
        self.object_enter_time: Dict[str, Dict[str, float]] = {}  # camera_id -> {object_id -> timestamp}
        self.dwelling_threshold: float = 3.0  # 체류 시간 임계값 (초)
        self.sent_dwelling_events: Dict[Tuple[str, int], float] = {}  # (camera_id, object_id) -> last_event_time
        
        self._load_config()

    def _load_config(self):
        """JSON 설정 파일 로드"""
        try:
            with open(self.zones_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.dwelling_threshold = config.get('dwelling_threshold_seconds', 3.0)
        except FileNotFoundError:
            print(f"⚠️ {self.zones_config_path} 파일을 찾을 수 없습니다. 빈 설정으로 시작합니다.")

    def load_zones(self, camera_id: str, zones_data: Optional[List[Dict]] = None):
        """카메라의 구역 로드
        
        Args:
            camera_id: 카메라 ID
            zones_data: [{'id': 'zone_1', 'name': '전기설비', 'polygon': [[x1,y1], ...]}]
        """
        if zones_data is None:
            try:
                with open(self.zones_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                zones_data = config.get('cameras', {}).get(camera_id, {}).get('zones', [])
            except Exception as e:
                print(f"❌ 구역 로드 오류 ({camera_id}): {e}")
                zones_data = []

        self.zones[camera_id] = {}
        for zone_def in zones_data:
            zone_id = zone_def['id']
            polygon = zone_def['polygon']
            name = zone_def.get('name', zone_id)
            self.zones[camera_id][zone_id] = Zone(zone_id, polygon, name)
        
        self.object_states[camera_id] = {}
        self.object_enter_time[camera_id] = {}
        
        print(f"✅ {camera_id}에 {len(self.zones[camera_id])}개 구역 로드됨")

    def check_zones(
        self,
        camera_id: str,
        detections: List,  # ai_analysis.DetectionEvent 리스트
        frame_shape: Tuple[int, int] = None  # (height, width)
    ) -> List[ZoneEvent]:
        """구역과 탐지 객체 교차 검사
        
        Args:
            camera_id: 카메라 ID
            detections: 현재 프레임의 탐지 결과 리스트
            frame_shape: 프레임 크기 (폴리곤 스케일 조정용)
        
        Returns:
            ZoneEvent 리스트
        """
        events = []
        
        if camera_id not in self.zones or not self.zones[camera_id]:
            return events

        # 현재 프레임에서 탐지된 객체 ID 집합
        current_object_ids = set()
        
        for detection in detections:
            object_id = detection.object_id or 0
            bbox_dict = detection.to_dict().get('bbox', {})
            current_object_ids.add(object_id)
            
            # 각 구역과 교차 검사
            for zone_id, zone in self.zones[camera_id].items():
                in_zone = zone.intersects_bbox(bbox_dict)
                
                # 진입 감지
                if in_zone and object_id not in self.object_states[camera_id]:
                    self.object_states[camera_id][object_id] = False
                
                prev_in_zone = self.object_states[camera_id].get(object_id, False)
                
                if in_zone and not prev_in_zone:
                    # 진입 이벤트
                    event = ZoneEvent(
                        event_type=ZoneEventType.ENTERED,
                        zone_id=zone_id,
                        object_id=object_id,
                        camera_id=camera_id,
                        bbox=bbox_dict,
                        confidence=detection.confidence
                    )
                    events.append(event)
                    self.object_states[camera_id][object_id] = True
                    self.object_enter_time[camera_id][object_id] = time.time()
                
                elif not in_zone and prev_in_zone:
                    # 퇴장 이벤트
                    event = ZoneEvent(
                        event_type=ZoneEventType.EXITED,
                        zone_id=zone_id,
                        object_id=object_id,
                        camera_id=camera_id,
                        bbox=bbox_dict,
                        confidence=detection.confidence
                    )
                    events.append(event)
                    self.object_states[camera_id][object_id] = False
                    self.object_enter_time[camera_id].pop(object_id, None)
                
                elif in_zone and prev_in_zone:
                    # 체류 중: 임계값 초과 시 이벤트
                    enter_time = self.object_enter_time[camera_id].get(object_id)
                    if enter_time:
                        dwelling_time = time.time() - enter_time
                        if dwelling_time >= self.dwelling_threshold:
                            # 중복 이벤트 방지 (동일 객체/카메라에서 1초마다만 전송)
                            key = (camera_id, object_id)
                            last_event_time = self.sent_dwelling_events.get(key, 0)
                            if time.time() - last_event_time >= 1.0:
                                event = ZoneEvent(
                                    event_type=ZoneEventType.DWELLING,
                                    zone_id=zone_id,
                                    object_id=object_id,
                                    camera_id=camera_id,
                                    bbox=bbox_dict,
                                    confidence=detection.confidence,
                                    dwelling_seconds=dwelling_time
                                )
                                events.append(event)
                                self.sent_dwelling_events[key] = time.time()
        
        # 사라진 객체 정리
        for object_id in list(self.object_states[camera_id].keys()):
            if object_id not in current_object_ids:
                self.object_states[camera_id].pop(object_id, None)
                self.object_enter_time[camera_id].pop(object_id, None)
        
        return events

    def draw_zones(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """프레임에 모든 구역 그리기"""
        if camera_id not in self.zones:
            return frame
        
        for zone in self.zones[camera_id].values():
            zone.draw(frame, color=(0, 255, 0), thickness=2)
        
        return frame


__all__ = ["Zone", "ZoneManager", "ZoneEvent"]
