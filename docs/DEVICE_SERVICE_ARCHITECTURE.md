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

### 1. **CCTV AI Engine**
- RTSP 수신
- 객체 감지
- JSON 이벤트 생성
- MQTT 발행 (`cctv/ai/events/...`)

### 2. **EdgeX Device Adapter**
- AI 이벤트 구독
- EdgeX 메타데이터(DeviceService/Profile/Device) 관리
- EdgeX 토픽 재발행 (`edgex/events/device/...`)

### 3. **Rule Engine (Kuiper)**
- intrusion/지속 감지/confidence 룰 적용
- 결과 라우팅 (`cctv/rules/...`)

### 4. **Action Layer (speaker-bridge)**
- 알람 재생
- DB 저장
- 외부 API 호출

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

## 실행 단위

현재 권장 실행 단위는 4개 프로세스입니다.

1. AI Engine (`main.py`)
2. EdgeX Adapter (`run_edgex_adapter.py`)
3. Rule Engine 배포 (`run_kuiper_rules.py`)
4. Action Layer (`run_action_bridge.py`)

## 데이터 흐름

```
1. AI Engine
   └─→ cctv/ai/events/{camera_id}/{event_type}

2. EdgeX Adapter
   └─→ 메타데이터(DeviceService/Profile/Device) 보장
   └─→ edgex/events/device/{service}/{device}/{resource}

3. Rule Engine (Kuiper)
   └─→ intrusion 필터/지속 감지/confidence 룰
   └─→ cctv/rules/intrusion/{filtered|persisted|critical}

4. Action Layer
   └─→ DB 저장 + 외부 API 호출 + 스피커 알람
```

## 장점

✅ **책임 분리**: 분석/어댑터/룰/액션이 독립적임  
✅ **운영 안정성**: 특정 계층 장애가 전체 파이프라인을 즉시 멈추지 않음  
✅ **확장성**: 룰/액션을 별도 증설 가능  
✅ **유지보수성**: 계층별 수정 범위가 명확함
