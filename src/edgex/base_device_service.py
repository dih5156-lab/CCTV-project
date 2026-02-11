"""
base_device_service.py - 모든 디바이스 서비스의 기본 클래스
각 디바이스(카메라, 센서 등)가 독립적으로 MQTT로 데이터를 발행하기 위한 기반 클래스
"""

import json
import logging
import time
import uuid
from typing import Dict, Optional
from datetime import datetime

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class BaseDeviceService:
    """
    모든 디바이스 서비스의 기본 클래스
    
    각 디바이스(CCTV, 열화상 카메라, 센서 등)는 이 클래스를 상속받아서
    자신의 데이터를 MQTT로 독립적으로 발행합니다.
    
    사용 예:
        camera_service = CameraDeviceService(config)
        thermal_service = ThermalDeviceService(config)
        sensor_service = SensorDeviceService(config)
    """
    
    def __init__(
        self,
        device_id: str,
        device_type: str,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_topic_prefix: str = "edgex/events/device"
    ):
        """
        매개변수:
            device_id: 디바이스 ID (예: camera-1, thermal-1)
            device_type: 디바이스 타입 (예: cctv, thermal, motion)
            mqtt_broker: MQTT 브로커 주소
            mqtt_port: MQTT 포트
            mqtt_topic_prefix: MQTT 토픽 접두사
        """
        self.device_id = device_id
        self.device_type = device_type
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic_prefix = mqtt_topic_prefix
        
        self._mqtt_client: Optional[mqtt.Client] = None
        self._connected = False
        
        logger.info(f"[{self.device_type}] Device Service 초기화: {device_id}")
        logger.info(f"  - MQTT Broker: {mqtt_broker}:{mqtt_port}")
        logger.info(f"  - Topic Prefix: {mqtt_topic_prefix}")
    
    def _ensure_mqtt_client(self) -> bool:
        """MQTT 클라이언트 초기화 및 연결 확인"""
        if self._mqtt_client is not None and self._connected:
            return True
        
        try:
            if self._mqtt_client is None:
                self._mqtt_client = mqtt.Client(
                    client_id=f"{self.device_type}-{self.device_id}-{uuid.uuid4().hex[:8]}",
                    clean_session=True
                )
                self._mqtt_client.on_connect = self._on_mqtt_connect
                self._mqtt_client.on_disconnect = self._on_mqtt_disconnect
                self._mqtt_client.on_publish = self._on_mqtt_publish
            
            if not self._connected:
                self._mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
                self._mqtt_client.loop_start()
                time.sleep(0.5)
            
            return self._connected
        except Exception as e:
            logger.error(f"[{self.device_type}] MQTT 연결 실패: {e}")
            return False
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT 연결 콜백"""
        if rc == 0:
            self._connected = True
            logger.info(f"[{self.device_type}] MQTT 연결 성공: {self.device_id}")
        else:
            logger.error(f"[{self.device_type}] MQTT 연결 실패 (rc={rc})")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT 연결 해제 콜백"""
        self._connected = False
        logger.warning(f"[{self.device_type}] MQTT 연결 해제: {self.device_id} (rc={rc})")
    
    def _on_mqtt_publish(self, client, userdata, mid):
        """MQTT 발행 완료 콜백"""
        logger.debug(f"[{self.device_type}] MQTT 발행 완료: mid={mid}")
    
    def publish_event(
        self,
        resource_name: str,
        event_data: Dict
    ) -> bool:
        """
        이벤트를 MQTT로 발행 (기본 구현)
        
        매개변수:
            resource_name: 리소스명 (예: helmet_detection, temperature, motion)
            event_data: 이벤트 데이터
                {
                    "type": "detection/measurement type",
                    "confidence": 0.95,
                    "value": "measurement value",
                    "bbox": {...},  # 선택사항
                    "timestamp": "2026-02-05T06:00:00Z"
                }
        
        반환값:
            발행 성공 여부
        """
        if not self._ensure_mqtt_client():
            return False
        
        try:
            logger.info(f"[{self.device_type}] 이벤트 발행: {resource_name}")
            
            timestamp = event_data.get("timestamp", datetime.now().isoformat())
            try:
                origin = int(float(timestamp) * 1_000_000_000) if isinstance(timestamp, (int, float)) else int(time.time() * 1_000_000_000)
            except Exception:
                origin = int(time.time() * 1_000_000_000)
            
            event_id = str(uuid.uuid4())
            request_id = str(uuid.uuid4())
            correlation_id = str(uuid.uuid4())
            
            # 📊 표준화된 페이로드 (모든 디바이스 타입이 이 구조를 따름)
            payload_value = {
                "type": event_data.get("type", "unknown"),
                "device": self.device_id,
                "device_type": self.device_type,
                "resource": resource_name,
                "confidence": event_data.get("confidence"),
                "value": event_data.get("value"),
                "bbox": event_data.get("bbox"),  # 선택사항
                "object_id": event_data.get("object_id"),  # 선택사항
                "timestamp": timestamp,
                "metadata": {
                    "version": "v1"
                }
            }
            
            # EdgeX v3 Envelope 포맷
            event_payload = {
                "apiVersion": "v3",
                "requestId": request_id,
                "event": {
                    "apiVersion": "v3",
                    "id": event_id,
                    "deviceName": self.device_id,
                    "sourceName": resource_name,
                    "origin": origin,
                    "readings": [
                        {
                            "origin": origin,
                            "deviceName": self.device_id,
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
            
            # 표준화된 토픽: edgex/events/device/{device_type}/{device_id}/{resource}
            topic = f"{self.mqtt_topic_prefix}/{self.device_type}/{self.device_id}/{resource_name}"
            logger.debug(f"MQTT 토픽: {topic}")
            
            result = self._mqtt_client.publish(topic, json.dumps(envelope), qos=0)
            
            if result.rc == 0:
                logger.info(f"✓ [{self.device_type}] 이벤트 발행 성공: {topic} (mid={result.mid})")
                return True
            else:
                logger.error(f"✗ [{self.device_type}] 이벤트 발행 실패: {topic} (rc={result.rc})")
                return False
        
        except Exception as e:
            logger.error(f"[{self.device_type}] 이벤트 발행 오류: {e}", exc_info=True)
            return False
    
    def disconnect(self):
        """MQTT 연결 종료"""
        try:
            if self._mqtt_client:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
                self._connected = False
                logger.info(f"[{self.device_type}] MQTT 연결 종료: {self.device_id}")
        except Exception as e:
            logger.error(f"[{self.device_type}] 연결 종료 실패: {e}")
