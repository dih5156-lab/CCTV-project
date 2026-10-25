# CCTV 이벤트 표준 스키마

## 결론

현장/외부 연동에서 직접 보내는 payload는 단순하게 유지합니다.

기본 입력은 `camera_id`, `type`, `confidence`, `timestamp` 중심으로 받고,
성별/옷 색깔/가방 유무 같은 외형 정보는 선택 필드 `attributes`에 넣습니다.

`schema_version`, `device`, `event`, `raw`, `event_id` 같은 표준 중첩 필드는 내부 저장/전달 단계에서 자동 보강하는 구조로 봅니다.

## 추적 ID 계약 — 2026-10-12 / I63

**내부 추적 계산은 정수, Canonical/MQTT/EdgeX의 신규 AI 이벤트 JSON은 추적 ID를 크기와 관계없이 10진 문자열 또는 `null`로 통일한다.** 10-12 운영 AI/스트림에 반영했고 실시간 입력·ID 보존을 확인했다. [운영 반영 근거와 검증 한계](../planning/evidence/track_id_contract_20261012.json)를 따른다.

### 생성 방식과 의미

현재 DeepStream IOU 설정의 `useUniqueID: 1`은 초기화 때 스트림별 난수를 상위32비트에 두고, 하위32비트를 증가시키는 uint64 ID를 사용한다. 하위 번호 증가는 같은 트래커 인스턴스의 스트림들이 공유한다. [NVIDIA DeepStream 7.1 규격](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvtracker.html)에 근거하며 프로젝트는 이 비트를 새로 조합하지 않는다.

```text
관측 ID = 0xA83E89AA_00000027
        = (2822670762 << 32) | 39
        = 12123278610165399591
```

큰 값은 누적 사람 수가 아니다. int64 최대값 `9223372036854775807`보다 크므로 eKuiper의 정수 파서가 거부했고, JavaScript 숫자로도 정확한 비교를 보장할 수 없다. ID를 float로 변환하거나 상위 비트를 버리는 방식은 금지한다.

### 값·범위·전송 위치

| 항목 | 계약 |
|---|---|
| 유효 ID | `0`부터 `18446744073709551614`까지. `0`도 정상 ID |
| 외부 표기 | ASCII `0` 또는 `[1-9][0-9]{0,19}`이며 위 범위를 만족하는 문자열. 공백·부호·선행0·소수·지수 표기 불허 |
| 입력 호환 | 기존 Python 정수와 정규 10진 문자열을 수용. bool·float는 ID로 승격하지 않음 |
| 미추적·누락 | `None`은 `missing`; 기존 음수 정수와 uint64 최댓값 `18446744073709551615`는 `unknown`; 형식/범위 오류는 `invalid`. 외부 값은 모두 `null` |
| 품질 | `observation.track_id`의 `value`, `quality`, `source_field`로 구분. `valid`의 값은 반드시 문자열. 기존 null의 unknown/invalid 원인 보존 |
| 이벤트 | 최상위 및 `raw.object_id`, `metadata.person_object_id`, `raw.metadata.person_object_id`, 중첩 목록의 동일 필드 적용 |
| 관측·기타 추적 참조 | `observation.track_id.value`, 스칼라 `track_id`도 동일. 센서 이벤트의 `object_id`는 이 계약의 영상 추적 ID로 재해석하지 않음 |
| eKuiper | `object_id STRING`; 숫자 비교/계산/형변환 없이 전달. 송신 측 중첩 숫자 ID도 함께 바꿔야 JSON decode 단계가 통과함 |
| 추적기·DB | `DetectionEvent.object_id`는 내부 정수 유지. 원본 접수·기존 DB 행/해시/스키마는 재작성하지 않음 |
| 조회 API 호환 | 기존 `/api/v1/search`의 `track_id` 정수는 호환 필드. 정확한 외부 비교에는 기존 `track_id_text` 사용. 과거 DB의 음수·최댓값 행 보존과 신규 유효 추적 ID 범위는 구분 |

```json
{
  "object_id": "12123278610165399591",
  "raw": {"object_id": "12123278610165399591"},
  "observation": {
    "track_id": {
      "value": "12123278610165399591",
      "quality": "valid",
      "source_field": "object_id"
    }
  }
}
```

### 동일성 범위와 이벤트 식별자

- 추적 ID는 한 실행·카메라·추적 세대 안에서 객체를 구분하는 값이며 영구적인 사람 신원 ID가 아니다. 재진입·재연결·재부팅·다른 카메라에서 같은 사람이 같은 ID를 갖는다고 해석하지 않는다.
- `camera_id`와 기존 `metadata.frame_identity.generation`을 함께 사용한다. 세대 번호도 프로세스 재시작 뒤 재사용될 수 있으므로 ID·카메라·세대만으로 서로 다른 실행의 기록을 합치지 않는다. 현재 영속 실행 식별자가 없으므로 실행 간 동일인 병합은 지원하지 않는다.
- 영속 기록과 중복 방지는 기존 `event_id`/`observation_id`/`source_record_id`를 사용한다. 이미 생성된 ID를 보존하고, 기존 이벤트 ID 계산을 마친 뒤 추적 ID의 전송 표현만 바꾼다. ID 없는 서로 다른 입력 표현의 해시가 같다는 보장은 추가하지 않는다.

### 적용·복귀 경계

1. 공통 Canonical 변환·관측 생성·직접 EdgeX 전송을 같은 정규화 함수로 맞춘다. MQTT outbox의 이전 숫자형 메시지도 재전송 직전에 바꾸며 저장된 이벤트 식별자는 유지한다.
2. 기존 모니터링 전달의 저장된 wire/hash와 과거 DB는 보존한다. 새 이벤트부터 문자열을 사용하고, 과거 숫자형 기록은 기존 Python 수신 경로가 계속 읽는다.
3. 외곽 `schema_version: "1.0"`은 유지하지만 **추적 ID 필드 타입의 개정은 숫자형만 허용하던 소비자에게 호환 변경이 아니다.** eKuiper 스트림·송신 AI 이미지·관련 수신자를 함께 준비해 점검 시간에 적용한다. 필드 하나만 추가하거나 스트림만 STRING으로 바꾸어서는 중첩된 큰 정수의 decode 실패가 해결되지 않는다.
4. 운영 `ai_events_stream`과 AI 프로필을 먼저 보존한다. 규칙 중지/스트림 갱신/재시작에는 출력 영향이 있다. 로컬 검증 뒤 사용자 승인으로 10-12 운영에 함께 반영했다. 복귀 시 원래 AI 이미지·스트림 정의를 함께 복원하며 DB를 과거 백업으로 덮지 않는다.

### 규칙 경유 이벤트 ID — 2026-10-12 로컬 보완

`ai_events_stream`은 `event_id STRING`을 선언하고 단건 필터·고신뢰 라우팅은
원본 `event_id`를 그대로 내보낸다. 직접 수신과 규칙 경유 수신 모두 같은 이벤트이면
Action 명령은 `{event_id}:{장치 유형}`, EdgeX 요청은 여기에 `:{물리 장치 ID}`를 붙인다.
추적 ID의 값이나 MQTT 토픽을 새 이벤트 ID로 대신 사용하지 않는다.

5초 집계 규칙은 여러 입력을 합친 별도 결과이므로 특정 입력의 `event_id`를 붙이지 않는다.
원본 ID가 없는 구형 입력의 해시 생성은 기존 동작을 유지하며, 이때 경로 간 동일성을 보장하지 않는다.
장치의 성공 결과 캐시는 같은 프로세스에서 순차 재수신을 억제한다. 후속 로컬 보완에서
동시 요청을 같은 실행 결과로 합치고 미응답·부분 실패를 장치별로 보존했다(관련162 PASS).
재시작/캐시 축출 후의 중복 방지·요청 내용 충돌·응답 제한시간과 자동 재시도는 P3.3 잔여다.
이 규칙 보완은 격리 검증까지 완료했고
운영 스트림·규칙 로더 이미지는 아직 교체하지 않았다.

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
| `object_id` | 선택 | 위 추적 ID 계약의 10진 문자열/null. 기존 정수 입력은 전환 기간에 수용 |
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
    "object_id": "3",
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
