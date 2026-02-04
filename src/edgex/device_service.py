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
                "coreMetadataUrl": "http://localhost:59881",
                "coreDataUrl": "http://localhost:59880",
                "deviceServiceName": "cctv-device-service",
                "baseUrl": "http://localhost:59999"
            }
        """
        self.metadata_url = config.get("coreMetadataUrl", "http://localhost:59881")
        self.data_url = config.get("coreDataUrl", "http://localhost:59880")
        self.service_name = config.get("deviceServiceName", "cctv-device-service")
        self.base_url = config.get("baseUrl", "http://localhost:59999")
        self.devices: Dict[str, str] = {}  # camera_id -> device_id 매핑
        
        logger.info(f"EdgeX Device Service 초기화: {self.service_name}")
        logger.info(f"  - Metadata URL: {self.metadata_url}")
        logger.info(f"  - Data URL: {self.data_url}")
    
    async def initialize(self):
        """EdgeX 연결 확인 (비동기 호환)"""
        try:
            # Metadata 서비스 헬스체크 (v3 → v2 → v1 폴백)
            metadata_endpoints = [
                f"{self.metadata_url}/api/v3/ping",
                f"{self.metadata_url}/api/v2/ping",
                f"{self.metadata_url}/api/v1/ping",
                f"{self.metadata_url}/ping"
            ]
            
            metadata_ok = False
            for endpoint in metadata_endpoints:
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✓ EdgeX Core Metadata 연결됨 ({endpoint})")
                        metadata_ok = True
                        break
                except:
                    continue
            
            if not metadata_ok:
                logger.warning(f"EdgeX Metadata 연결 실패 - 시도한 엔드포인트:")
                for ep in metadata_endpoints:
                    logger.warning(f"  - {ep}")
            
            # Data 서비스 헬스체크 (v3 → v2 → v1 폴백)
            data_endpoints = [
                f"{self.data_url}/api/v3/ping",
                f"{self.data_url}/api/v2/ping",
                f"{self.data_url}/api/v1/ping",
                f"{self.data_url}/ping"
            ]
            
            data_ok = False
            for endpoint in data_endpoints:
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✓ EdgeX Core Data 연결됨 ({endpoint})")
                        data_ok = True
                        break
                except:
                    continue
            
            if not data_ok:
                logger.warning(f"EdgeX Data 연결 실패 - 시도한 엔드포인트:")
                for ep in data_endpoints:
                    logger.warning(f"  - {ep}")
                    
        except Exception as e:
            logger.error(f"EdgeX 연결 오류: {e}")
    
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
            
            # Device 등록 (v3 → v2 → v1 폴백)
            endpoints = [
                f"{self.metadata_url}/api/v3/device",
                f"{self.metadata_url}/api/v2/device",
                f"{self.metadata_url}/api/v1/device"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=device_payload,
                        timeout=10,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code in [200, 201]:
                        device_id = response.json().get("id") or device_name
                        self.devices[camera_id] = device_id
                        logger.info(f"✓ 카메라 등록 성공: {camera_id} -> {device_name} (ID: {device_id})")
                        logger.debug(f"  RTSP: {rtsp_source}")
                        logger.debug(f"  엔드포인트: {endpoint}")
                        return device_id
                    elif response.status_code == 404:
                        logger.debug(f"엔드포인트 없음: {endpoint}")
                        continue
                    else:
                        logger.warning(f"Device 등록 실패 ({camera_id}): {response.status_code}")
                        logger.debug(f"응답: {response.text}")
                        continue
                except Exception as e:
                    logger.debug(f"엔드포인트 {endpoint} 시도 실패: {e}")
                    continue
            
            logger.error(f"카메라 등록 실패: {camera_id} - 모든 엔드포인트 시도 완료")
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
                
                # EdgeX Core Data로 전송 (v3 → v2 → v1 폴백)
                endpoints = [
                    f"{self.data_url}/api/v3/event",
                    f"{self.data_url}/api/v2/event",
                    f"{self.data_url}/api/v1/event"
                ]
                
                success = False
                for endpoint in endpoints:
                    try:
                        response = requests.post(
                            endpoint,
                            json=event_data,
                            timeout=10,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        if response.status_code in [200, 201]:
                            logger.debug(f"✓ [{camera_id}] EdgeX 이벤트 전송: {event.event_type.value if hasattr(event, 'event_type') else 'detection'}")
                            success = True
                            break
                        elif response.status_code == 404:
                            logger.debug(f"엔드포인트 없음: {endpoint}")
                            continue
                        else:
                            logger.debug(f"Event 전송 실패 ({camera_id}): {response.status_code}")
                            logger.debug(f"응답: {response.text}")
                            continue
                    except Exception as e:
                        logger.debug(f"엔드포인트 {endpoint} 시도 실패: {e}")
                        continue
                
                if not success:
                    logger.warning(f"이벤트 전송 실패 ({camera_id}) - 모든 엔드포인트 시도 완료")
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
            
            # Profile 생성 (v3 → v2 → v1 폴백)
            endpoints = [
                f"{self.metadata_url}/api/v3/deviceprofile",
                f"{self.metadata_url}/api/v2/deviceprofile",
                f"{self.metadata_url}/api/v1/deviceprofile"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        json=profile_payload,
                        timeout=10,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code in [200, 201]:
                        logger.info(f"✓ Device Profile 생성: CCTV-Camera-Profile (엔드포인트: {endpoint})")
                        return True
                    elif response.status_code == 404:
                        logger.debug(f"엔드포인트 없음: {endpoint}")
                        continue
                    elif response.status_code == 409:
                        logger.info(f"✓ Device Profile 이미 존재: CCTV-Camera-Profile (엔드포인트: {endpoint})")
                        return True
                    else:
                        logger.warning(f"Profile 생성 실패: {response.status_code}")
                        logger.debug(f"응답: {response.text}")
                        continue
                except Exception as e:
                    logger.debug(f"엔드포인트 {endpoint} 시도 실패: {e}")
                    continue
            
            logger.warning("Profile 생성 실패 - 모든 엔드포인트 시도 완료")
            return False
                
        except Exception as e:
            logger.error(f"Profile 생성 오류: {e}")
            return False
