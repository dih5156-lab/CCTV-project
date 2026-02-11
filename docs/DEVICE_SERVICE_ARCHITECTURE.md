# 🏗️ Device Service 아키텍처

## 개요

각 디바이스(카메라, 센서, 열화상 등)가 **독립적인 Device Service를 가지고 있으며**, 이들이 **MQTT를 통해 EdgeX와 통신**하는 마이크로서비스 아키텍처입니다.

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                    EdgeX Foundry (Core Services)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           MQTT Message Bus (Mosquitto)                   │   │
│  │  Topic: edgex/events/device/{device_type}/{id}/{resource}│   │
│  └──────────────────────────────────────────────────────────┘   │
│                         ▲  ▲  ▲                                  │
│     ┌───────────────────┘  │  └───────────────────┐              │
│     │                      │                      │              │
│  ┌──┴─────────────────┐ ┌──┴──────────────┐ ┌─────┴──────────┐  │
│  │   Core Data        │ │  Core Keeper   │ │ Core Metadata  │  │
│  │  (PostgreSQL)      │ │  (Registry)    │ │                │  │
│  └────────────────────┘ └────────────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          ▲                    ▲                    ▲
          │ MQTT Publish       │ MQTT Publish       │ MQTT Publish
          │                    │                    │
     ┌────┴──────────────┐ ┌──┴───────────────┐ ┌──┴──────────────┐
     │  CCTV Device      │ │ Thermal Device   │ │ Sensor Device   │
     │  Service          │ │ Service          │ │ Service         │
     │                  │ │                  │ │                 │
     │ Camera 1,2,3...  │ │ Thermal 1,2...   │ │ Motion, Env...  │
     │ (독립 컨테이너)   │ │ (독립 컨테이너)   │ │ (독립 컨테이너)  │
     └──────────────────┘ └──────────────────┘ └─────────────────┘
```

## 계층 구조

### 1. **BaseDeviceService** (기반 클래스)
```python
class BaseDeviceService:
    - device_id: 디바이스 ID
    - device_type: 디바이스 타입 (cctv, thermal, sensor, ...)
    - publish_event(): 기본 발행 메서드
    - MQTT 연결 관리
```

### 2. **특화 Device Service** (상속 클래스)
```
CCTVDeviceService (헬멧, 사람, 낙상 감지)
├── publish_detection_event()
├── publish_detection_events()
└── AI 분석 결과 발행

ThermalDeviceService (온도, 이상 감지)
├── publish_temperature_event()
├── publish_anomaly_event()
└── 열화상 데이터 발행

SensorDeviceService (동작, 환경 센서)
├── publish_motion_event()
├── publish_environment_event()
└── 센서 데이터 발행
```

## MQTT 토픽 구조

```
edgex/events/device/{device_type}/{device_id}/{resource_name}

예시:
edgex/events/device/cctv/camera-1/helmet_detection
edgex/events/device/cctv/camera-2/person_detection
edgex/events/device/thermal/thermal-1/temperature
edgex/events/device/sensor/motion-1/motion_detection
```

## 메시지 포맷 (표준화)

모든 디바이스가 다음의 표준화된 포맷을 사용:

```json
{
  "type": "detection_type",          // helmet, person, temperature, motion, etc.
  "device": "device-1",              // 디바이스 ID
  "device_type": "cctv",             // 디바이스 타입
  "resource": "helmet_detection",    // 리소스명
  "confidence": 0.95,                // 신뢰도 (선택사항)
  "value": null,                     // 측정값 (선택사항)
  "bbox": {                          // 바운딩 박스 (detection만)
    "x": 100,
    "y": 200,
    "width": 300,
    "height": 400
  },
  "object_id": 1,                    // 추적 ID (선택사항)
  "timestamp": "2026-02-05T06:00:00Z",
  "metadata": {
    "version": "v1"
  }
}
```

## 사용 예시

### CCTV 카메라 서비스
```python
# 초기화
cctv_service = CCTVDeviceService(
    device_id="camera-1",
    mqtt_broker="localhost:1883"
)

# 헬멧 감지 발행
cctv_service.publish_detection_event(
    event_type="helmet",
    confidence=0.95,
    x=100, y=200, width=300, height=400,
    object_id=1
)

# 여러 이벤트 발행
cctv_service.publish_detection_events(detection_events_list)
```

### 열화상 카메라 서비스
```python
thermal_service = ThermalDeviceService(
    device_id="thermal-1",
    mqtt_broker="localhost:1883"
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
```

### 센서 서비스
```python
sensor_service = SensorDeviceService(
    device_id="sensor-1",
    mqtt_broker="localhost:1883"
)

# 동작 감지
sensor_service.publish_motion_event(
    detected=True,
    confidence=0.95
)

# 환경 데이터
sensor_service.publish_environment_event(
    humidity=65.5,
    pressure=1013.25,
    light=500
)
```

## Docker 컨테이너 구조

각 Device Service는 **독립적인 컨테이너**로 배포:

```dockerfile
# CCTV Device Service
FROM python:3.9
COPY cctv_device_service.py /app/
CMD ["python", "/app/cctv_device_service.py"]

# Thermal Device Service
FROM python:3.9
COPY thermal_device_service.py /app/
CMD ["python", "/app/thermal_device_service.py"]

# Sensor Device Service
FROM python:3.9
COPY sensor_device_service.py /app/
CMD ["python", "/app/sensor_device_service.py"]
```

## docker-compose.yml 예시

```yaml
services:
  # CCTV Device Service
  cctv-device-service:
    build: ./cctv
    depends_on:
      - edgex-mqtt-broker
    environment:
      MQTT_BROKER: edgex-mqtt-broker
      MQTT_PORT: 1883
      DEVICE_ID: camera-1

  # Thermal Device Service
  thermal-device-service:
    build: ./thermal
    depends_on:
      - edgex-mqtt-broker
    environment:
      MQTT_BROKER: edgex-mqtt-broker
      MQTT_PORT: 1883
      DEVICE_ID: thermal-1

  # Sensor Device Service
  sensor-device-service:
    build: ./sensor
    depends_on:
      - edgex-mqtt-broker
    environment:
      MQTT_BROKER: edgex-mqtt-broker
      MQTT_PORT: 1883
      DEVICE_ID: sensor-1

  # EdgeX MQTT Broker
  edgex-mqtt-broker:
    image: eclipse-mosquitto:2.0
    ports:
      - "1883:1883"

  # EdgeX Core Data (MQTT 구독)
  core-data:
    image: nexus3.edgexfoundry.org:10004/core-data:latest
    depends_on:
      - edgex-mqtt-broker
      - edgex-postgres
    environment:
      MESSAGEBUS_TYPE: mqtt
      MESSAGEBUS_HOST: edgex-mqtt-broker
      MESSAGEBUS_PORT: 1883
```

## 데이터 흐름

```
1. AI 추론 (CCTV Device Service)
   └─→ 헬멧/사람/낙상 감지
        └─→ DetectionEvent 생성

2. 이벤트 발행
   └─→ publish_detection_event()
       └─→ BaseDeviceService.publish_event()
           └─→ MQTT 발행

3. MQTT 전송
   └─→ Topic: edgex/events/device/cctv/camera-1/helmet_detection
       └─→ EdgeX v3 Envelope 포맷

4. EdgeX 수신
   └─→ Core Data MQTT 구독
       └─→ PostgreSQL 저장
           └─→ REST API 조회 가능
```

## 확장성

새로운 디바이스 추가는 매우 간단:

```python
# 1. BaseDeviceService 상속
class CustomDeviceService(BaseDeviceService):
    def __init__(self, device_id, mqtt_broker="localhost"):
        super().__init__(
            device_id=device_id,
            device_type="custom",
            mqtt_broker=mqtt_broker
        )
    
    # 2. 특화 메서드 추가
    def publish_custom_event(self, data):
        return self.publish_event("custom_resource", data)

# 3. 사용
custom_service = CustomDeviceService("custom-1")
custom_service.publish_custom_event({"value": 123})
```

## 장점

✅ **독립성**: 각 디바이스가 자신의 서비스를 가짐  
✅ **확장성**: 새로운 디바이스 추가 쉬움  
✅ **표준화**: 모든 디바이스가 같은 메시지 포맷 사용  
✅ **분산화**: 마이크로서비스 패턴  
✅ **유지보수**: 각 서비스를 독립적으로 관리  
✅ **스케일링**: 필요에 따라 인스턴스 추가 가능
