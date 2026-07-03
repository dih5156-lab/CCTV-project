# CCTV 이벤트 표준 스키마

## 결론

현장/외부 연동에서 직접 보내는 payload는 단순하게 유지합니다.

기본 입력은 `camera_id`, `type`, `confidence`, `timestamp` 중심으로 받고,
성별/옷 색깔/가방 유무 같은 외형 정보는 선택 필드 `attributes`에 넣습니다.

`schema_version`, `device`, `event`, `raw`, `event_id` 같은 표준 중첩 필드는 내부 저장/전달 단계에서 자동 보강하는 구조로 봅니다.

## 최소 입력 payload

AI 엔진, 외부 MQTT, 테스트 스크립트에서 우선 맞춰야 하는 기본 형태입니다.

```json
{
  "camera_id": "camera_1",
  "type": "fall_detected",
  "confidence": 0.93,
  "timestamp": 1778032673.848273,
  "severity": "critical",
  "bbox": {
    "x": 120,
    "y": 80,
    "width": 64,
    "height": 96
  }
}
```

| 필드 | 필수 | 설명 |
|---|---:|---|
| `camera_id` | 예 | 카메라/장비 ID |
| `type` | 예 | 이벤트 타입. 예: `person`, `helmet`, `fall_detected`, `appearance_match` |
| `confidence` | 예 | 탐지 신뢰도. `0.0` ~ `1.0` |
| `timestamp` | 권장 | Unix seconds. 없으면 내부 수신 시각으로 보강 가능 |
| `severity` | 권장 | `normal`, `low`, `warning`, `critical` |
| `bbox` | 권장 | 화면 내 객체 위치. UI 표시/검색에 사용 |
| `object_id` | 선택 | tracker id 또는 객체 id |
| `metadata` | 선택 | backend, zone_id 등 부가 정보 |

## 외형 속성 payload

성별, 옷 색깔, 가방 유무 등은 일반 이벤트 필수 필드가 아니라 `attributes`에 넣습니다.
이 값들은 `AppearancePipeline`에서 추출되어 `appearance_log` DB에 저장되고, `/api/v1/search`에서 검색됩니다.

```json
{
  "camera_id": "camera_1",
  "type": "appearance_match",
  "confidence": 0.88,
  "timestamp": 1778032673.848273,
  "bbox": {
    "x": 120,
    "y": 80,
    "width": 180,
    "height": 420
  },
  "attributes": {
    "upper_color": "black",
    "lower_color": "gray",
    "has_helmet": true,
    "helmet_color": "white",
    "has_backpack": false,
    "has_handbag": true,
    "has_suitcase": false,
    "gender": "male",
    "age_group": "adult",
    "attribute_backend": "hsv"
  }
}
```

| 속성 필드 | 설명 | 생성/판정 기준 |
|---|---|---|
| `upper_color` | 상의 색상 | HSV 또는 PP-Human attribute backend |
| `lower_color` | 하의 색상 | HSV 또는 PP-Human attribute backend |
| `has_helmet` | 헬멧 착용 여부 | 헬멧/머리 bbox와 person bbox 관계 |
| `helmet_color` | 헬멧 색상 | 헬멧 영역 색상 분석 |
| `has_backpack` | 백팩 소지 여부 | 주변 객체 class: `backpack`, `rucksack` 등 |
| `has_handbag` | 핸드백 소지 여부 | 주변 객체 class: `handbag`, `purse` 등 |
| `has_suitcase` | 캐리어 소지 여부 | 주변 객체 class: `suitcase`, `luggage` 등 |
| `gender` | 성별 추정값 | face/attribute backend 결과. 확정 신원 정보가 아니라 추정값 |
| `age_group` | 나이대 추정값 | face/attribute backend 결과 |
| `attribute_backend` | 속성 분석 백엔드 | 예: `hsv`, `pphuman` |

운영 기준으로는 `attributes` 전체를 필수로 만들지 않습니다.
모델/백엔드가 못 채운 값은 `null`, `unknown`, 또는 `false`로 들어갈 수 있습니다.

## 내부 표준 payload 예시

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
| `attributes` | 외형 속성 정보 | 선택 필드. 색상/소지품/성별 등 |
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
attributes.upper_color 등 필요한 외형 속성
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
| 외형 속성 분석/저장 | `src/core/ai/_appearance_pipeline.py`, `src/core/ai/_appearance_analyzer.py`, `src/services/appearance_log.py` |
| 외형 속성 검색 API | `src/api/v1/search.py`, `src/api/v1/appearances.py` |
| 센서 이벤트 | `src/core/sensor_detection.py` |
| 외부 MQTT 입력 | `src/services/external_ingest.py` |
| Action Layer 저장/디바이스 출력 | `src/services/action_bridge.py`, `src/services/_action_bridge_support.py` |

## 변경 원칙

- 외부/현장 입력은 최소 payload를 우선 사용합니다.
- 기존 `camera_id/type/confidence/severity` 필드는 제거하지 않습니다.
- 성별/옷 색/가방 유무는 `attributes`에 넣고, 없으면 보내지 않아도 됩니다.
- 내부 저장/연동 단계에서는 `schema_version/event/device/raw/event_id`를 자동 보강합니다.
- 디바이스 표시 문구는 payload의 `event.display_message`와 `event.tts_message`가 있으면 그것을 우선합니다.
- payload에 문구가 없으면 `config/event_type_map.json`의 기본 문구를 사용합니다.
