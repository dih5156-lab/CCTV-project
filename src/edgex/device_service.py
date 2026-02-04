"""
EdgeX Device Service for CCTV
CCTV 카메라를 EdgeX Foundry 장치로 관리
"""

import json
import logging
import requests
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CCTVDeviceService:
    """EdgeX CCTV 장치 서비스"""
    
    def __init__(self, config: Dict):
        """
        매개변수:
            config: {
                "coreMetadataUrl": "http://localhost:48081",
                "coreDataUrl": "http://localhost:48080",
                "deviceServiceName": "cctv-device-service",
                "baseUrl": "http://localhost:59999"
            }
        """
        self.metadata_url = config.get("coreMetadataUrl", "http://localhost:48081")
        self.data_url = config.get("coreDataUrl", "http://localhost:48080")
        self.service_name = config.get("deviceServiceName", "cctv-device-service")
        self.base_url = config.get("baseUrl", "http://localhost:59999")
        self.devices: Dict[str, str] = {}  # camera_id -> device_id 매핑
        
        logger.info(f"EdgeX Device Service 초기화: {self.service_name}")
        logger.info(f"  - Metadata URL: {self.metadata_url}")
        logger.info(f"  - Data URL: {self.data_url}")
    
    async def initialize(self):
        """EdgeX 연결 확인 (비동기 호환)"""
        try:
            # Metadata 서비스 헬스체크
            response = requests.get(f"{self.metadata_url}/api/v2/ping", timeout=5)
            if response.status_code == 200:
                logger.info("✓ EdgeX Core Metadata 연결됨")
            else:
                logger.warning(f"EdgeX Metadata 상태: {response.status_code}")
            
            # Data 서비스 헬스체크
            response = requests.get(f"{self.data_url}/api/v2/ping", timeout=5)
            if response.status_code == 200:
                logger.info("✓ EdgeX Core Data 연결됨")
            else:
                logger.warning(f"EdgeX Data 상태: {response.status_code}")
        except Exception as e:
            logger.error(f"EdgeX 연결 실패: {e}")
    
    async def add_camera(self, camera_id: str, rtsp_source: str) -> Optional[str]:
        """
        카메라를 EdgeX 장치로 등록
        
        매개변수:
            camera_id: 카메라 ID (예: "camera_1")
            rtsp_source: RTSP URL (예: "rtsp://192.168.1.100:554/stream")
            
        반환값:
            device_id 또는 None
        """
        try:
            device_name = f"camera-{camera_id}"
            
            # Device 생성 페이로드
            device_payload = {
                "apiVersion": "v2",
                "device": {
                    "name": device_name,
                    "description": f"CCTV Camera {camera_id}",
                    "adminState": "UNLOCKED",
                    "operatingState": "UP",
                    "profileName": "CCTV-Camera-Profile",
                    "serviceName": self.service_name,
                    "protocols": {
                        "rtsp": {
                            "Address": rtsp_source.split("://")[1].split("/")[0] if "://" in rtsp_source else "localhost",
                            "Port": "554",
                            "URL": rtsp_source
                        }
                    },
                    "labels": [
                        "cctv",
                        f"camera_{camera_id}"
                    ]
                }
            }
            
            # Device 등록
            response = requests.post(
                f"{self.metadata_url}/api/v2/device",
                json=device_payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                device_id = response.json().get("id") or device_name
                self.devices[camera_id] = device_id
                logger.info(f"✓ 카메라 등록 성공: {camera_id} -> {device_name} (ID: {device_id})")
                logger.debug(f"  RTSP: {rtsp_source}")
                return device_id
            else:
                logger.warning(f"Device 등록 실패 ({camera_id}): {response.status_code}")
                logger.debug(f"응답: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"카메라 등록 오류 ({camera_id}): {e}")
            return None
    
    async def send_detection_event(self, camera_id: str, events: List) -> bool:
        """
        감지 이벤트를 EdgeX Event로 전송
        
        매개변수:
            camera_id: 카메라 ID
            events: DetectionEvent 리스트
            
        반환값:
            전송 성공 여부
        """
        if camera_id not in self.devices:
            logger.warning(f"등록되지 않은 카메라: {camera_id}")
            return False
        
        device_name = f"camera-{camera_id}"
        
        try:
            for event in events:
                # Event 데이터 구성
                event_data = {
                    "apiVersion": "v2",
                    "event": {
                        "deviceName": device_name,
                        "profileName": "CCTV-Camera-Profile",
                        "sourceName": event.event_type.value if hasattr(event, 'event_type') else "detection",
                        "readings": [
                            {
                                "deviceName": device_name,
                                "resourceName": event.event_type.value if hasattr(event, 'event_type') else "detection",
                                "value": json.dumps({
                                    "type": event.event_type.value if hasattr(event, 'event_type') else "unknown",
                                    "confidence": event.confidence if hasattr(event, 'confidence') else 0.0,
                                    "bbox": {
                                        "x": event.x if hasattr(event, 'x') else 0,
                                        "y": event.y if hasattr(event, 'y') else 0,
                                        "width": event.width if hasattr(event, 'width') else 0,
                                        "height": event.height if hasattr(event, 'height') else 0
                                    },
                                    "object_id": event.object_id if hasattr(event, 'object_id') else None,
                                    "timestamp": event.timestamp if hasattr(event, 'timestamp') else datetime.now().isoformat()
                                }),
                                "valueType": "String"
                            }
                        ]
                    }
                }
                
                # EdgeX Core Data로 전송
                response = requests.post(
                    f"{self.data_url}/api/v2/event",
                    json=event_data,
                    timeout=10,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code in [200, 201]:
                    logger.debug(f"✓ [{camera_id}] EdgeX 이벤트 전송: {event.event_type.value if hasattr(event, 'event_type') else 'detection'}")
                else:
                    logger.warning(f"Event 전송 실패 ({camera_id}): {response.status_code}")
                    logger.debug(f"응답: {response.text}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"이벤트 전송 오류 ({camera_id}): {e}")
            return False
    
    async def create_device_profile(self) -> bool:
        """
        CCTV 장치 프로필 생성 (필요시)
        """
        try:
            profile_payload = {
                "apiVersion": "v2",
                "profile": {
                    "name": "CCTV-Camera-Profile",
                    "description": "CCTV Camera Detection Profile",
                    "manufacturer": "CCTV",
                    "model": "Multi-Camera",
                    "deviceResources": [
                        {
                            "name": "helmet_detection",
                            "description": "헬멧 착용 감지",
                            "attributes": {"dataType": "String"},
                            "properties": {
                                "valueType": "String",
                                "readWrite": "R"
                            }
                        },
                        {
                            "name": "fall_detection",
                            "description": "낙상 감지",
                            "attributes": {"dataType": "String"},
                            "properties": {
                                "valueType": "String",
                                "readWrite": "R"
                            }
                        },
                        {
                            "name": "person_detection",
                            "description": "사람 감지",
                            "attributes": {"dataType": "String"},
                            "properties": {
                                "valueType": "String",
                                "readWrite": "R"
                            }
                        }
                    ]
                }
            }
            
            response = requests.post(
                f"{self.metadata_url}/api/v2/deviceprofile",
                json=profile_payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info("✓ Device Profile 생성: CCTV-Camera-Profile")
                return True
            else:
                logger.warning(f"Profile 생성 실패: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Profile 생성 오류: {e}")
            return False
