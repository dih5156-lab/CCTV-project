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

    def _describe_http_status(self, status_code: int) -> str:
        """HTTP 상태 코드에 대한 설명 반환"""
        status_map = {
            200: "OK: 요청 성공",
            201: "Created: 리소스 생성 성공",
            202: "Accepted: 요청 수락, 처리 중",
            204: "No Content: 성공적으로 처리됨 (본문 없음)",
            207: "Multi-Status: 부분 성공/실패 혼재",
            400: "Bad Request: 잘못된 요청 (데이터 형식 오류 또는 잘못된 데이터)",
            401: "Unauthorized: 인증 실패 (JWT 토큰 누락 또는 유효하지 않음)",
            403: "Forbidden: 접근 권한 없음",
            404: "Not Found: 요청한 리소스를 찾을 수 없음",
            405: "Method Not Allowed: 허용되지 않은 메서드",
            408: "Request Timeout: 요청 시간 초과",
            409: "Conflict: 리소스 충돌 (이미 존재 등)",
            415: "Unsupported Media Type: 지원되지 않는 콘텐츠 타입",
            422: "Unprocessable Entity: 의미 오류 (검증 실패)",
            423: "Locked: 디바이스 잠금 또는 운영 상태 비활성화",
            429: "Too Many Requests: 요청 과다",
            500: "Internal Server Error: 서비스 내부 오류",
            502: "Bad Gateway: 게이트웨이 오류",
            503: "Service Unavailable: 서비스 사용 불가 또는 연결 제한",
            504: "Gateway Timeout: 게이트웨이 시간 초과",
        }
        return status_map.get(status_code, "알 수 없는 오류")

    def _map_event_type_to_resource(self, event_type: str) -> str:
        if event_type in ["helmet", "head", "unsafe_behavior", "wearing_helmet"]:
            return "helmet_detection"
        if event_type in ["fall_detected", "not_fall"]:
            return "fall_detection"
        if event_type == "person":
            return "person_detection"
        return "helmet_detection"

    def _extract_event_fields(self, event: object) -> Dict[str, object]:
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

        return {
            "event_type": event_type,
            "confidence": confidence,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "object_id": object_id,
            "timestamp": timestamp,
        }

    def _build_value_payload(
        self,
        event_type: str,
        confidence: float,
        x: int,
        y: int,
        width: int,
        height: int,
        object_id: Optional[int],
        timestamp: str,
        device_name: str,
        resource_name: str,
    ) -> Dict[str, object]:
        return {
            "type": event_type,
            "device": device_name,
            "resource": resource_name,
            "confidence": confidence,
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
            "object_id": object_id,
            "timestamp": timestamp,
            "metadata": {
                "profile": "CCTV-Camera-Profile",
                "service": self.service_name,
                "version": "v1",
            },
        }

    def _build_event_payload(
        self,
        device_name: str,
        resource_name: str,
        origin: int,
        value_payload: Dict[str, object],
    ) -> Dict[str, object]:
        event_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        return {
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
                        "value": json.dumps(value_payload),
                    }
                ],
            },
        }

    def _build_envelope(self, event_payload: Dict[str, object]) -> Dict[str, object]:
        return {
            "apiVersion": "v3",
            "receivedTopic": "",
            "correlationID": str(uuid.uuid4()),
            "requestID": event_payload.get("requestId", ""),
            "errorCode": 0,
            "payload": event_payload,
            "contentType": "application/json",
        }
    
    @property
    def mqtt_client(self) -> Optional[mqtt.Client]:
        """MQTT 클라이언트 프로퍼티 (자동 초기화)"""
        if not self._mqtt_client:
            self._ensure_mqtt_client()
        return self._mqtt_client
    
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
                        logger.warning(
                            f"Device 등록 실패 ({camera_id}): {response.status_code} - "
                            f"{self._describe_http_status(response.status_code)}"
                        )
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
    
    def send_mqtt_event(self, camera_id: str, event_type: str, event_data: dict) -> bool:
        """
        이벤트를 MQTT로 직접 발행 (동기 메서드)
        EdgeX v3 형식으로 발행 (envelope + payload 구조)
        
        매개변수:
            camera_id: 카메라 ID
            event_type: 이벤트 타입 (person, head, fall_detected 등)
            event_data: 이벤트 데이터 딕셔너리
            
        반환값:
            발행 성공 여부
        """
        logger.info(f"[send_mqtt_event] 호출됨: camera_id={camera_id}, event_type={event_type}")
        
        try:
            device_name = f"camera-{camera_id}"
            
            resource_name = self._map_event_type_to_resource(event_type)

            event_fields = self._extract_event_fields(event_data)
            confidence = event_fields["confidence"]
            x = event_fields["x"]
            y = event_fields["y"]
            width = event_fields["width"]
            height = event_fields["height"]
            object_id = event_fields["object_id"]
            timestamp = event_fields["timestamp"]
            
            # _publish_event_mqtt()를 사용하여 발행
            # 이 메서드가 올바른 envelope 형식을 생성합니다
            result = self._publish_event_mqtt(
                device_name,
                resource_name,
                event_type,
                confidence,
                x,
                y,
                width,
                height,
                object_id,
                timestamp
            )
            
            if result:
                logger.info(f"✓ MQTT 이벤트 발행 성공: {device_name}/{resource_name} ({event_type})")
                return True
            else:
                logger.error(f"✗ MQTT 이벤트 발행 실패: {device_name}/{resource_name} ({event_type})")
                return False
                
        except Exception as e:
            logger.error(f"✗ MQTT 이벤트 발행 오류: {e}", exc_info=True)
            return False
    
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
                event_fields = self._extract_event_fields(event)
                event_type = event_fields["event_type"]
                confidence = event_fields["confidence"]
                x = event_fields["x"]
                y = event_fields["y"]
                width = event_fields["width"]
                height = event_fields["height"]
                object_id = event_fields["object_id"]
                timestamp = event_fields["timestamp"]

                resource_name = self._map_event_type_to_resource(event_type)

                try:
                    origin = int(float(timestamp) * 1_000_000_000)
                except Exception:
                    origin = int(time.time() * 1_000_000_000)

                value_payload = self._build_value_payload(
                    event_type,
                    confidence,
                    x,
                    y,
                    width,
                    height,
                    object_id,
                    timestamp,
                    device_name,
                    resource_name,
                )
                event_payload = self._build_event_payload(
                    device_name,
                    resource_name,
                    origin,
                    value_payload,
                )
                base_event = {"event": event_payload["event"]}
                
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
                            logger.warning(
                                f"Event 전송 실패 ({camera_id}): {response.status_code} - "
                                f"{self._describe_http_status(response.status_code)}"
                            )
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
                        logger.warning(
                            f"마지막 상태 코드: {last_status} - {self._describe_http_status(last_status)}"
                        )
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

            value_payload = self._build_value_payload(
                event_type,
                confidence,
                x,
                y,
                width,
                height,
                object_id,
                timestamp,
                device_name,
                resource_name,
            )
            event_payload = self._build_event_payload(
                device_name,
                resource_name,
                origin,
                value_payload,
            )
            envelope = self._build_envelope(event_payload)

            topic = f"{self.mqtt_topic_prefix}/{self.service_name}/{device_name}/{resource_name}"
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
                        logger.warning(
                            f"Service 등록 실패: {response.status_code} - "
                            f"{self._describe_http_status(response.status_code)}"
                        )
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
                        logger.warning(
                            f"Profile 생성 실패: {response.status_code} - "
                            f"{self._describe_http_status(response.status_code)}"
                        )
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

    def publish_device_event(
        self,
        device_id: str,
        device_type: str,
        resource_name: str,
        event_data: Dict
    ) -> bool:
        """
        범용 디바이스 이벤트 발행 메서드
        
        다양한 디바이스 타입 (CCTV, 열화상, 센서 등)을 지원하는 통합 인터페이스
        
        매개변수:
            device_id: 디바이스 ID (예: camera-1, thermal-1, sensor-1)
            device_type: 디바이스 타입 (예: cctv, thermal, sensor)
            resource_name: 리소스명 (예: helmet_detection, temperature, motion)
            event_data: 이벤트 데이터 딕셔너리
                {
                    "type": "detection type",
                    "confidence": 0.95,
                    "value": "measurement value",
                    "bbox": {"x": 100, "y": 200, "width": 300, "height": 400},  # 선택사항
                    "object_id": 1,  # 선택사항
                    "timestamp": "2026-02-05T06:00:00Z"
                }
        
        반환값:
            발행 성공 여부
        """
        if not self._ensure_mqtt_client():
            return False

        try:
            logger.info(f"범용 디바이스 이벤트 발행: {device_id}/{resource_name}")
            
            try:
                timestamp = event_data.get("timestamp", datetime.now().isoformat())
                origin = int(float(timestamp) * 1_000_000_000) if isinstance(timestamp, (int, float)) else int(time.time() * 1_000_000_000)
            except Exception:
                origin = int(time.time() * 1_000_000_000)

            event_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            correlation_id = str(uuid.uuid4())
            
            # 📊 표준화된 메시지 포맷 (모든 디바이스 타입에 공통)
            payload_value = {
                "type": event_data.get("type", "unknown"),
                "device": device_id,
                "device_type": device_type,
                "resource": resource_name,
                "confidence": event_data.get("confidence", 0.0),
                "value": event_data.get("value"),
                "bbox": event_data.get("bbox"),  # 선택사항 (detection 타입만 해당)
                "object_id": event_data.get("object_id"),  # 선택사항
                "timestamp": timestamp,
                "metadata": {
                    "service": self.service_name,
                    "version": "v1",
                    "device_type": device_type
                }
            }
            
            event_payload = {
                "apiVersion": "v3",
                "requestId": request_id,
                "event": {
                    "apiVersion": "v3",
                    "id": event_id,
                    "deviceName": device_id,
                    "sourceName": resource_name,
                    "origin": origin,
                    "readings": [
                        {
                            "origin": origin,
                            "deviceName": device_id,
                            "resourceName": resource_name,
                            "valueType": "String",
                            "value": json.dumps(payload_value)
                        }
                    ]
                }
            }

            envelope = {
                "apiVersion": "v3",
                "receivedTopic": "",
                "correlationID": correlation_id,
                "requestID": request_id,
                "errorCode": 0,
                "payload": event_payload,
                "contentType": "application/json"
            }

            # 확장성 있는 토픽 구조: edgex/events/device/{service}/{device_type}/{device_id}/{resource}
            topic = f"{self.mqtt_topic_prefix}/{self.service_name}/{device_type}/{device_id}/{resource_name}"
            logger.info(f"MQTT 토픽: {topic}")

            result = self._mqtt_client.publish(topic, json.dumps(envelope), qos=0)
            
            if result.rc == 0:
                logger.info(f"✓ 범용 디바이스 이벤트 발행 성공: {topic} (mid={result.mid})")
                return True
            else:
                logger.error(f"범용 디바이스 이벤트 발행 실패: {topic} (rc={result.rc})")
                return False
        except Exception as e:
            logger.error(f"범용 디바이스 이벤트 발행 오류: {e}", exc_info=True)
            return False