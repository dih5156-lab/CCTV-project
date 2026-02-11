"""
cctv_device_service.py - CCTV 카메라 전용 Device Service
BaseDeviceService를 상속받아 CCTV 특화 기능 추가
"""

import logging
from typing import List

from .base_device_service import BaseDeviceService
from ..core.events import DetectionEvent

logger = logging.getLogger(__name__)


class CCTVDeviceService(BaseDeviceService):
    """
    CCTV 카메라 전용 Device Service
    
    AI 탐지 결과 (helmet, person, fall_detected)를 MQTT로 발행
    BaseDeviceService를 상속받아 표준화된 메시지 형식 사용
    
    사용 예:
        cctv_service = CCTVDeviceService(
            device_id="camera-1",
            mqtt_broker="mqtt-broker:1883"
        )
        
        # AI 탐지 결과 발행
        cctv_service.publish_detection_event(
            event_type="helmet",
            confidence=0.95,
            bbox={"x": 100, "y": 200, "width": 300, "height": 400},
            object_id=1
        )
    """
    
    def __init__(
        self,
        device_id: str = "camera-1",
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_topic_prefix: str = "edgex/events/device"
    ):
        """
        매개변수:
            device_id: CCTV 카메라 ID (예: camera-1, camera-2)
            mqtt_broker: MQTT 브로커 주소
            mqtt_port: MQTT 포트
            mqtt_topic_prefix: MQTT 토픽 접두사
        """
        super().__init__(
            device_id=device_id,
            device_type="cctv",
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            mqtt_topic_prefix=mqtt_topic_prefix
        )
    
    def publish_detection_event(
        self,
        event_type: str,
        confidence: float,
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
        object_id: int = None,
        timestamp: str = None
    ) -> bool:
        """
        AI 탐지 결과를 MQTT로 발행
        
        매개변수:
            event_type: 탐지 타입 (helmet, head, person, fall_detected)
            confidence: 신뢰도 (0.0~1.0)
            x, y, width, height: 바운딩 박스 좌표
            object_id: 추적 ID
            timestamp: 타임스탬프 (ISO 형식)
        
        반환값:
            발행 성공 여부
        """
        # 리소스명 결정
        if event_type in ["helmet", "head", "unsafe_behavior"]:
            resource_name = "helmet_detection"
        elif event_type in ["fall_detected", "not_fall"]:
            resource_name = "fall_detection"
        elif event_type == "person":
            resource_name = "person_detection"
        else:
            resource_name = "detection"
        
        event_data = {
            "type": event_type,
            "confidence": confidence,
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
            "object_id": object_id,
            "timestamp": timestamp
        }
        
        return self.publish_event(resource_name, event_data)
    
    def publish_detection_events(
        self,
        events: List[DetectionEvent]
    ) -> int:
        """
        여러 탐지 이벤트를 MQTT로 발행
        
        매개변수:
            events: DetectionEvent 리스트
        
        반환값:
            성공한 이벤트 개수
        """
        success_count = 0
        
        for event in events:
            try:
                # DetectionEvent를 딕셔너리로 변환
                event_dict = event.to_dict() if hasattr(event, 'to_dict') else event
                
                event_type = event_dict.get("type", "unknown")
                confidence = event_dict.get("confidence", 0.0)
                bbox = event_dict.get("bbox", {})
                object_id = event_dict.get("object_id")
                timestamp = event_dict.get("timestamp")
                
                if self.publish_detection_event(
                    event_type=event_type,
                    confidence=confidence,
                    x=bbox.get("x", 0),
                    y=bbox.get("y", 0),
                    width=bbox.get("width", 0),
                    height=bbox.get("height", 0),
                    object_id=object_id,
                    timestamp=timestamp
                ):
                    success_count += 1
            except Exception as e:
                logger.error(f"[CCTV] 이벤트 발행 실패: {e}")
        
        logger.info(f"[CCTV] {success_count}/{len(events)} 이벤트 발행 성공")
        return success_count
