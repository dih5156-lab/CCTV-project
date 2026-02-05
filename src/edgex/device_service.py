"""
EdgeX Device Service for CCTV
CCTV 카메라를 EdgeX Foundry 장치로 관리
"""

import json
import logging
import requests
import time
import uuid
from typing import Dict, Optional, List
from datetime import datetime

import base64
import paho.mqtt.client as mqtt

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
        self.service_name = "cctv-device-service"  # CCTV Device Service
        self.mqtt_broker = config.get("mqttBroker", "localhost")
        self.mqtt_port = int(config.get("mqttPort", 1883))
        self.mqtt_topic_prefix = config.get("mqttTopicPrefix", "edgex/events/device")
        self._mqtt_client: Optional[mqtt.Client] = None
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
                    # EdgeX v3는 배열 형식 필요
                    payload = [device_payload] if "/v3/" in endpoint else device_payload
                    response = requests.post(
                        endpoint,
                        json=payload,
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
                    elif response.status_code == 207:  # Multi-Status (v3 배열 응답)
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            item_status = result[0].get("statusCode", 0)
                            if item_status in [200, 201]:
                                device_id = result[0].get("id") or device_name
                                self.devices[camera_id] = device_id
                                logger.info(f"✓ 카메라 등록 성공: {camera_id} -> {device_name} (ID: {device_id})")
                                logger.debug(f"  RTSP: {rtsp_source}")
                                logger.debug(f"  엔드포인트: {endpoint}")
                                return device_id
                            elif item_status == 409:
                                self.devices[camera_id] = device_name
                                logger.info(f"✓ 카메라 이미 존재: {camera_id} -> {device_name}")
                                logger.debug(f"  RTSP: {rtsp_source}")
                                logger.debug(f"  엔드포인트: {endpoint}")
                                return device_name
                        logger.warning(f"Device 등록 실패 ({camera_id}): 207 응답 - {response.text}")
                        continue
                    elif response.status_code == 404:
                        logger.debug(f"엔드포인트 없음: {endpoint}")
                        continue
                    elif response.status_code == 409:
                        self.devices[camera_id] = device_name
                        logger.info(f"✓ 카메라 이미 존재: {camera_id} -> {device_name}")
                        logger.debug(f"  RTSP: {rtsp_source}")
                        logger.debug(f"  엔드포인트: {endpoint}")
                        return device_name
                    else:
                        logger.warning(f"Device 등록 실패 ({camera_id}): {response.status_code}")
                        logger.warning(f"응답 내용: {response.text}")
                        logger.warning(f"엔드포인트: {endpoint}")
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
                # 이벤트 데이터 추출 (DetectionEvent 또는 dict 지원)
                if isinstance(event, dict):
                    event_type = event.get("type", "unknown")
                    confidence = event.get("confidence", 0.0)
                    bbox = event.get("bbox", {}) or {}
                    x = bbox.get("x", 0)
                    y = bbox.get("y", 0)
                    width = bbox.get("width", 0)
                    height = bbox.get("height", 0)
                    object_id = event.get("object_id")
                    timestamp = event.get("timestamp", datetime.now().isoformat())
                else:
                    event_type = event.event_type.value if hasattr(event, "event_type") else "unknown"
                    confidence = event.confidence if hasattr(event, "confidence") else 0.0
                    x = event.x if hasattr(event, "x") else 0
                    y = event.y if hasattr(event, "y") else 0
                    width = event.width if hasattr(event, "width") else 0
                    height = event.height if hasattr(event, "height") else 0
                    object_id = event.object_id if hasattr(event, "object_id") else None
                    timestamp = event.timestamp if hasattr(event, "timestamp") else datetime.now().isoformat()

                # EdgeX Device Profile 리소스에 맞게 매핑
                if event_type in ["helmet", "head", "unsafe_behavior"]:
                    resource_name = "helmet_detection"
                elif event_type in ["fall_detected", "not_fall"]:
                    resource_name = "fall_detection"
                elif event_type == "person":
                    resource_name = "person_detection"
                else:
                    resource_name = "helmet_detection"

                event_id = str(uuid.uuid4())
                base_event = {
                    "event": {
                        "apiVersion": "v3",
                        "id": event_id,
                        "deviceName": device_name,
                        "profileName": "CCTV-Camera-Profile",
                        "sourceName": resource_name,
                        "readings": [
                            {
                                "deviceName": device_name,
                                "resourceName": resource_name,
                                "value": json.dumps({
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
                last_status = None
                last_text = None
                last_endpoint = None
                for endpoint in endpoints:
                    try:
                        last_endpoint = endpoint
                        api_version = "v3" if "/v3/" in endpoint else "v2"
                        event_data = {"apiVersion": api_version, **base_event}
                        payload = [event_data] if api_version == "v3" else event_data
                        response = requests.post(
                            endpoint,
                            json=payload,
                            timeout=10,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        last_status = response.status_code
                        last_text = response.text

                        if response.status_code in [200, 201]:
                            logger.debug(f"✓ [{camera_id}] EdgeX 이벤트 전송: {event_type}")
                            success = True
                            break
                        elif response.status_code == 207:
                            result = response.json()
                            if isinstance(result, list) and len(result) > 0:
                                item_status = result[0].get("statusCode", 0)
                                if item_status in [200, 201]:
                                    logger.debug(f"✓ [{camera_id}] EdgeX 이벤트 전송: {event_type}")
                                    success = True
                                    break
                            logger.warning(f"Event 전송 실패 ({camera_id}): 207 응답 - {response.text}")
                            continue
                        elif response.status_code == 404:
                            logger.warning(f"엔드포인트 없음: {endpoint}")
                            continue
                        else:
                            logger.warning(f"Event 전송 실패 ({camera_id}): {response.status_code}")
                            logger.warning(f"응답: {response.text}")
                            logger.warning(f"엔드포인트: {endpoint}")
                            continue
                    except Exception as e:
                        last_text = str(e)
                        logger.warning(f"엔드포인트 {endpoint} 시도 실패: {e}")
                        continue
                
                if not success:
                    logger.warning(f"이벤트 전송 실패 ({camera_id}) - 모든 엔드포인트 시도 완료")
                    if last_endpoint:
                        logger.warning(f"마지막 엔드포인트: {last_endpoint}")
                    if last_status is not None:
                        logger.warning(f"마지막 상태 코드: {last_status}")
                    if last_text:
                        logger.warning(f"마지막 응답: {last_text}")
                    # REST 실패 시 MQTT로 폴백
                    if self._publish_event_mqtt(device_name, resource_name, event_type, confidence, x, y, width, height, object_id, timestamp):
                        logger.info(f"✓ [{camera_id}] MQTT 이벤트 전송: {event_type}")
                        return True
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"이벤트 전송 오류 ({camera_id}): {e}")
            return False

    def _ensure_mqtt_client(self) -> bool:
        if self._mqtt_client:
            return True
        try:
            client = mqtt.Client()
            client.connect(self.mqtt_broker, self.mqtt_port, 60)
            client.loop_start()
            self._mqtt_client = client
            logger.info(f"✓ MQTT 연결됨: {self.mqtt_broker}:{self.mqtt_port}")
            return True
        except Exception as e:
            logger.warning(f"MQTT 연결 실패: {e}")
            self._mqtt_client = None
            return False

    def _publish_event_mqtt(
        self,
        device_name: str,
        resource_name: str,
        event_type: str,
        confidence: float,
        x: int,
        y: int,
        width: int,
        height: int,
        object_id: Optional[int],
        timestamp: str
    ) -> bool:
        if not self._ensure_mqtt_client():
            return False

        try:
            logger.info(f"MQTT 이벤트 발행 시작: device={device_name}, resource={resource_name}, type={event_type}")
            
            try:
                origin = int(float(timestamp) * 1_000_000_000)
            except Exception:
                origin = int(time.time() * 1_000_000_000)

            event_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            correlation_id = str(uuid.uuid4())
            
            event_payload = {
                "apiVersion": "v3",
                "requestId": request_id,
                "event": {
                    "apiVersion": "v3",
                    "id": event_id,
                    "deviceName": device_name,
                    "profileName": "CCTV-Camera-Profile",
                    "sourceName": resource_name,
                    "origin": origin,
                    "readings": [
                        {
                            "origin": origin,
                            "deviceName": device_name,
                            "resourceName": resource_name,
                            "profileName": "CCTV-Camera-Profile",
                            "valueType": "String",
                            "value": json.dumps({
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
                            })
                        }
                    ]
                }
            }

            envelope = {
                "apiVersion": "v3",
                "receivedTopic": "",
                "correlationID": correlation_id,
                "requestID": "",
                "errorCode": 0,
                "payload": event_payload,
                "contentType": "application/json"
            }

            topic = f"{self.mqtt_topic_prefix}/{self.service_name}/CCTV-Camera-Profile/{device_name}/{resource_name}"
            logger.info(f"MQTT 토픽: {topic}")

            result = self._mqtt_client.publish(topic, json.dumps(envelope), qos=0)
            
            if result.rc == 0:
                logger.info(f"✓ MQTT 발행 성공: {topic} (mid={result.mid})")
                return True
            else:
                logger.error(f"MQTT 발행 실패: {topic} (rc={result.rc})")
                return False
        except Exception as e:
            logger.error(f"MQTT 전송 오류: {e}", exc_info=True)
            return False
    
    async def register_device_service(self) -> bool:
        """
        Device Service를 EdgeX에 등록
        """
        try:
            service_payload = {
                "apiVersion": "v2",
                "service": {
                    "name": self.service_name,
                    "description": "CCTV Detection Device Service",
                    "labels": ["cctv", "detection"],
                    "baseAddress": "http://edgex-device-virtual:59900",  # EdgeX 기본 서비스 사용
                    "adminState": "UNLOCKED"
                }
            }
            
            endpoints = [
                f"{self.metadata_url}/api/v3/deviceservice",
                f"{self.metadata_url}/api/v2/deviceservice",
                f"{self.metadata_url}/api/v1/deviceservice"
            ]
            
            for endpoint in endpoints:
                try:
                    payload = [service_payload] if "/v3/" in endpoint else service_payload
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=10,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code in [200, 201]:
                        logger.info(f"✓ Device Service 등록: {self.service_name}")
                        return True
                    elif response.status_code == 207:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            item_status = result[0].get("statusCode", 0)
                            if item_status in [200, 201]:
                                logger.info(f"✓ Device Service 등록: {self.service_name}")
                                return True
                            elif item_status == 409:
                                logger.info(f"✓ Device Service 이미 존재: {self.service_name}")
                                return True
                        logger.warning(f"Service 등록 실패: 207 응답 - {response.text}")
                        continue
                    elif response.status_code == 409:
                        logger.info(f"✓ Device Service 이미 존재: {self.service_name}")
                        return True
                    elif response.status_code == 404:
                        logger.debug(f"엔드포인트 없음: {endpoint}")
                        continue
                    else:
                        logger.warning(f"Service 등록 실패: {response.status_code}")
                        logger.debug(f"응답: {response.text}")
                        continue
                except Exception as e:
                    logger.debug(f"엔드포인트 {endpoint} 시도 실패: {e}")
                    continue
            
            logger.warning("Service 등록 실패 - 모든 엔드포인트 시도 완료")
            return False
            
        except Exception as e:
            logger.error(f"Service 등록 오류: {e}")
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
                    # EdgeX v3는 배열 형식 필요
                    payload = [profile_payload] if "/v3/" in endpoint else profile_payload
                    response = requests.post(
                        endpoint,
                        json=payload,
                        timeout=10,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code in [200, 201]:
                        logger.info(f"✓ Device Profile 생성: CCTV-Camera-Profile (엔드포인트: {endpoint})")
                        return True
                    elif response.status_code == 207:  # Multi-Status (v3 배열 응답)
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            item_status = result[0].get("statusCode", 0)
                            if item_status in [200, 201]:
                                logger.info(f"✓ Device Profile 생성: CCTV-Camera-Profile (엔드포인트: {endpoint})")
                                return True
                            elif item_status == 409:
                                logger.info(f"✓ Device Profile 이미 존재: CCTV-Camera-Profile (엔드포인트: {endpoint})")
                                return True
                        logger.warning(f"Profile 생성 실패: 207 응답 - {response.text}")
                        continue
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
