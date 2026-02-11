"""
thermal_device_service.py - 열화상 카메라 Device Service 예시
BaseDeviceService를 상속받아 열화상 이미지 데이터 발행
"""

import logging

from .base_device_service import BaseDeviceService

logger = logging.getLogger(__name__)


class ThermalDeviceService(BaseDeviceService):
    """
    열화상(Thermal) 카메라 전용 Device Service
    
    온도 데이터, 이상 감지 등을 MQTT로 발행
    
    사용 예:
        thermal_service = ThermalDeviceService(
            device_id="thermal-1",
            mqtt_broker="mqtt-broker:1883"
        )
        
        # 온도 데이터 발행
        thermal_service.publish_temperature_event(
            temperature=45.2,
            roi_name="furnace"
        )
        
        # 이상 감지 발행
        thermal_service.publish_anomaly_event(
            anomaly_type="high_temperature",
            confidence=0.92,
            max_temperature=48.5
        )
    """
    
    def __init__(
        self,
        device_id: str = "thermal-1",
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_topic_prefix: str = "edgex/events/device"
    ):
        super().__init__(
            device_id=device_id,
            device_type="thermal",
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            mqtt_topic_prefix=mqtt_topic_prefix
        )
    
    def publish_temperature_event(
        self,
        temperature: float,
        roi_name: str = "default",
        timestamp: str = None
    ) -> bool:
        """온도 데이터 발행"""
        event_data = {
            "type": "temperature",
            "value": temperature,
            "roi": roi_name,
            "timestamp": timestamp
        }
        return self.publish_event("temperature", event_data)
    
    def publish_anomaly_event(
        self,
        anomaly_type: str,
        confidence: float,
        max_temperature: float = None,
        timestamp: str = None
    ) -> bool:
        """이상 감지 발행"""
        event_data = {
            "type": anomaly_type,
            "confidence": confidence,
            "value": max_temperature,
            "timestamp": timestamp
        }
        return self.publish_event("anomaly_detection", event_data)


class SensorDeviceService(BaseDeviceService):
    """
    센서(Sensor) Device Service 예시
    
    동작 감지, 습도, 압력 등 다양한 센서 데이터 발행
    
    사용 예:
        sensor_service = SensorDeviceService(
            device_id="sensor-1",
            mqtt_broker="mqtt-broker:1883"
        )
        
        # 동작 감지 발행
        sensor_service.publish_motion_event(
            detected=True,
            confidence=0.95
        )
        
        # 환경 데이터 발행
        sensor_service.publish_environment_event(
            humidity=65.5,
            pressure=1013.25
        )
    """
    
    def __init__(
        self,
        device_id: str = "sensor-1",
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_topic_prefix: str = "edgex/events/device"
    ):
        super().__init__(
            device_id=device_id,
            device_type="sensor",
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            mqtt_topic_prefix=mqtt_topic_prefix
        )
    
    def publish_motion_event(
        self,
        detected: bool,
        confidence: float = 1.0,
        timestamp: str = None
    ) -> bool:
        """동작 감지 발행"""
        event_data = {
            "type": "motion",
            "value": detected,
            "confidence": confidence,
            "timestamp": timestamp
        }
        return self.publish_event("motion_detection", event_data)
    
    def publish_environment_event(
        self,
        humidity: float = None,
        pressure: float = None,
        light: float = None,
        timestamp: str = None
    ) -> bool:
        """환경 데이터 발행"""
        event_data = {
            "type": "environment",
            "humidity": humidity,
            "pressure": pressure,
            "light": light,
            "timestamp": timestamp
        }
        return self.publish_event("environment", event_data)
