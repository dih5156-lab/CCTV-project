# EdgeX Device Service 아키텍처

## 결론

현재 저장소에서 직접 구현한 EdgeX Device Service는 CCTV 카메라용 `CCTVDeviceService`입니다. 열화상·환경 센서별 독립 Device Service 클래스는 현재 구현되어 있지 않습니다. AIoT 센서 데이터는 별도 `parser-python` 서비스가 수신·정규화한 뒤 EdgeX/MQTT 흐름으로 전달합니다.

## 현재 구성

```text
CCTV AI Engine
  -> cctv/ai/events/{camera_id}/{event_type}
  -> EdgeX Adapter / CCTVDeviceService
       -> 카메라 Device Service/Profile/Device 등록
       -> Redis Message Bus 또는 MQTT 발행
       -> 실패 시 SQLite outbox 저장 후 재전송
  -> EdgeX Core Data / MQTT
  -> eKuiper / ASC
  -> Action Layer

AIoT 센서
  -> parser-python
  -> PostgreSQL outbox / EdgeX / MQTT
  -> SensorRuleBridge / Action Layer
```

## CCTVDeviceService 책임

구현 위치는 `src/edgex/device_service.py`입니다.

- EdgeX Core Metadata와 Core Data 상태 확인
- `CCTV-Camera-Profile` 등록
- `camera-{camera_id}` 장치 등록
- Redis Message Bus 우선 또는 MQTT fallback 발행
- 선택적인 Core Data REST 전송
- 발행 실패 이벤트를 SQLite outbox에 저장하고 재전송

내부 책임은 다음 mixin으로 분리되어 있습니다.

| 파일 | 책임 |
|---|---|
| `src/edgex/_http_mixin.py` | EdgeX REST 호출과 API 버전 fallback |
| `src/edgex/_payload_mixin.py` | EdgeX v3 envelope와 reading 생성 |
| `src/edgex/_publisher_mixin.py` | Redis/MQTT 연결, 지수 backoff, 발행 |
| `src/edgex/_outbox_mixin.py` | SQLite store-and-forward |

## 카메라 이벤트 토픽

기본 MQTT prefix가 `edgex/events/device`일 때 CCTV 이벤트는 다음 형식입니다.

```text
edgex/events/device/{deviceServiceName}/{deviceName}/{resourceName}
```

예시:

```text
edgex/events/device/cctv-device-service/camera-camera_1/helmet_detection
edgex/events/device/cctv-device-service/camera-camera_1/fall_detection
edgex/events/device/cctv-device-service/camera-camera_1/person_detection
```

범용 `publish_device_event()`를 직접 사용할 때는 `device_type`이 한 단계 추가됩니다.

```text
edgex/events/device/{deviceServiceName}/{deviceType}/{deviceId}/{resourceName}
```

## 이벤트 payload

실제 발행값은 단순 JSON만 보내지 않고 EdgeX v3 envelope 안에 `event`와 `readings`를 포함합니다. reading의 `value`에는 아래와 같은 JSON 문자열이 들어갑니다.

```json
{
  "type": "fall_detected",
  "device": "camera-camera_1",
  "resource": "fall_detection",
  "confidence": 0.92,
  "bbox": {
    "x": 100,
    "y": 200,
    "width": 300,
    "height": 400
  },
  "object_id": 1,
  "timestamp": "2026-07-03T12:00:00+09:00",
  "metadata": {
    "profile": "CCTV-Camera-Profile",
    "service": "cctv-device-service",
    "version": "v1"
  }
}
```

전체 이벤트 계약은 [이벤트 표준 스키마](../features/EVENT_SCHEMA_STANDARD.md)를 참고합니다.

## 코드 사용 예시

현재 생성자는 개별 keyword 인자가 아니라 설정 `dict`를 받습니다.

```python
import asyncio

from src.edgex.device_service import CCTVDeviceService


async def main() -> None:
    service = CCTVDeviceService(
        {
            "coreMetadataUrl": "http://localhost:59881",
            "coreDataUrl": "http://localhost:59880",
            "deviceServiceName": "cctv-device-service",
            "messageBusType": "mqtt",
            "mqttBroker": "localhost",
            "mqttPort": 1883,
            "outboxDbPath": "data/runtime/event_outbox.db",
        }
    )
    await service.initialize()
    await service.register_device_service()
    await service.create_device_profile()
    await service.add_camera("camera_1", "rtsp://127.0.0.1:8554/camera_1")
    await service.send_detection_event(
        "camera_1",
        [
            {
                "type": "fall_detected",
                "confidence": 0.92,
                "bbox": {"x": 100, "y": 200, "width": 300, "height": 400},
                "object_id": 1,
            }
        ],
    )
    service.close()


asyncio.run(main())
```

## 실행 단위

주요 실행 프로세스는 다음과 같습니다.

1. AI Engine: `main.py`
2. EdgeX Adapter: `runners/run_edgex_adapter.py`
3. eKuiper 룰 배포: `runners/run_kuiper_rules.py`
4. Action Layer: `runners/run_action_bridge.py`

Compose에서는 각각 `cctv-ai-engine`, `cctv-edgex-adapter`, `cctv-kuiper-rule-loader`, `cctv-action-layer` 서비스로 실행합니다.
