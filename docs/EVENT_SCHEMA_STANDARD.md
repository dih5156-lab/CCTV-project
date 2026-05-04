# CCTV 이벤트 표준 스키마

## 결론

이 프로젝트의 이벤트 payload는 하위 호환을 위해 기존 평면 필드와 표준 중첩 필드를 함께 유지합니다.

기존 소비자는 `camera_id`, `type`, `confidence`, `severity`를 계속 읽을 수 있고,
신규 소비자는 `schema_version`, `device`, `event`, `decoded`, `raw`, `event_id`를 기준으로 처리합니다.

## 표준 payload 예시

```json
{
  "schema_version": "1.0",
  "message_type": "ai_detection_event",
  "message_id": null,
  "event_id": "evt_1234abcd5678ef90",
  "occurred_at": "2026-05-04T01:00:00+00:00",
  "camera_id": "camera_1",
  "type": "head",
  "severity": "normal",
  "confidence": 0.91,
  "device": {
    "camera_id": "camera_1"
  },
  "gateway": {},
  "event": {
    "event_type": "head",
    "severity": "normal",
    "source": "cctv-ai-engine",
    "source_type": "vision",
    "confidence": 0.91
  },
  "decoded": {},
  "raw": {
    "bbox": {
      "x": 120,
      "y": 80,
      "width": 64,
      "height": 96
    },
    "object_id": 3,
    "class_idx": 0,
    "class_name": "head",
    "keypoints": null,
    "metadata": {
      "camera_id": "camera_1"
    }
  }
}
```

## 필드 기준

| 필드 | 설명 | 비고 |
|---|---|---|
| `schema_version` | 표준 payload 버전 | 현재 `1.0` |
| `message_type` | 이벤트 출처별 메시지 타입 | `ai_detection_event`, `sensor_event`, `external_event` |
| `event_id` | 중복 처리 방지용 안정 ID | 없으면 payload 기반으로 생성 |
| `occurred_at` | 이벤트 발생 시각 | ISO 8601 UTC 권장 |
| `camera_id` | 기존 소비자용 카메라/장비 ID | 하위 호환 필드 |
| `type` | 기존 소비자용 이벤트 타입 | 하위 호환 필드 |
| `severity` | 기존 소비자용 심각도 | 하위 호환 필드 |
| `confidence` | 기존 소비자용 신뢰도 | 하위 호환 필드 |
| `device` | 장비 식별 정보 | 카메라, 센서, LoRa 장비 ID |
| `gateway` | 게이트웨이/수신기 정보 | LoRa/외부 MQTT에서 주로 사용 |
| `event` | 표준 이벤트 정보 | 타입, 심각도, 출처, 표시/음성 문구 |
| `decoded` | 파싱된 센서/외부 데이터 | 센서 값, 속성 값 |
| `raw` | 원본 또는 모델 상세 결과 | bbox, class, keypoints, 원본 payload |

## 소비자별 권장 사용 필드

### MQTT / Kuiper

기존 룰 호환을 위해 아래 필드를 우선 사용합니다.

```text
camera_id
type
confidence
severity
timestamp 또는 occurred_at
```

신규 룰에서는 표준 필드를 사용할 수 있습니다.

```text
device.camera_id
event.event_type
event.confidence
event.severity
occurred_at
```

### Action Layer / 디바이스

Action Layer는 레거시와 표준 필드를 모두 읽습니다.

우선순위:

```text
camera_id: device.camera_id -> camera_id -> device_id -> dev_eui
event_type: event.event_type -> event.type -> type -> event_type
severity: event.severity -> severity
confidence: event.confidence -> confidence
display_message: event.display_message -> event.message -> message
tts_message: event.tts_message -> event.message -> message
```

디바이스 출력은 최종적으로 다음처럼 변환됩니다.

```text
스피커: event.tts_message 또는 event_type_map.json의 tts_message
전광판: event.display_message 또는 event_type_map.json의 display_text
경광등: event_type/severity 기준 ON/OFF 명령
```

## 생성 위치

| 흐름 | 표준화 위치 |
|---|---|
| AI 이벤트 MQTT 발행 | `src/core/_event_publish.py`, `src/protocols/mqtt_publisher.py` |
| 센서 이벤트 | `src/core/sensor_detection.py` |
| 외부 MQTT 입력 | `src/services/external_ingest.py` |
| Action Layer 저장/디바이스 출력 | `src/services/action_bridge.py`, `src/services/_action_bridge_support.py` |

## 변경 원칙

- 기존 `camera_id/type/confidence/severity` 필드는 제거하지 않습니다.
- 신규 연동은 가능하면 `schema_version/event/device/event_id` 기준으로 구현합니다.
- 디바이스 표시 문구는 payload의 `event.display_message`와 `event.tts_message`가 있으면 그것을 우선합니다.
- payload에 문구가 없으면 `config/event_type_map.json`의 기본 문구를 사용합니다.
