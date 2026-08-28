# MQTT·EdgeX 데이터 계약

최종 확인일: 2026-08-27

이 문서는 CCTV AI 이벤트, AIoT telemetry, EdgeX Reading, 장치 명령의 공통 형식을 정의한다. 운영 서비스는 기존 호환 필드를 유지하되, 신규 연동은 이 문서의 `schema_version`과 식별자 규칙을 사용한다.

## 1. 전송 계층과 책임

| 계층 | 역할 | 기본 위치 |
|---|---|---|
| Mosquitto MQTT | CCTV AI 이벤트와 외부 AIoT 메시지의 비동기 전달 | `edgex-mqtt-broker:1883` |
| EdgeX MessageBus | EdgeX 서비스 내부 이벤트 전달 | Redis 기반 |
| EdgeX Core Data | 장치 Reading 저장·조회 | `core-data` |
| EdgeX Core Metadata | 디바이스·리소스 등록 | `core-metadata` |
| EdgeX Core Command | 장치 제어 명령 라우팅 | `core-command` |
| Runtime DB / Outbox | AI 검색 이력과 실패 메시지 재처리 | `/app/data/runtime` |

MQTT 브로커와 EdgeX 내부 Redis MessageBus는 서로 다른 계층이므로 하나의 통신 채널로 취급하지 않는다.

## 2. CCTV AI 이벤트

토픽 형식:

```text
cctv/ai/events/{camera_id}/{event_type}
```

예시 이벤트:

```json
{
  "schema_version": "1.0",
  "message_type": "cctv.ai.event",
  "message_id": "evt-20260827-000123",
  "occurred_at": "2026-08-27T14:00:00+09:00",
  "device": {
    "camera_id": "cam-01"
  },
  "gateway": {
    "service": "cctv-ai-engine",
    "pipeline": "deepstream"
  },
  "event": {
    "event_type": "fall_detected",
    "severity": "high",
    "source": "deepstream",
    "confidence": 0.92
  },
  "decoded": {
    "camera_id": "cam-01",
    "track_id": 23,
    "fall": true,
    "helmet": false,
    "upper_color": "black",
    "lower_color": "gray"
  },
  "raw": {}
}
```

현재 코드의 하위 호환 입력(`camera_id`, `device_id`, `event_type`, `type`)은 `canonicalize_event_payload()`에서 표준 구조로 변환한다.

주요 구독 토픽:

```text
cctv/ai/events/+/fall_detected
cctv/ai/events/+/helmet
cctv/ai/events/+/head
cctv/ai/events/+/person
cctv/ai/events/+/face_unknown
cctv/ai/events/+/face_recognized
```

## 3. AIoT telemetry

센서 원본 토픽은 `aiot/rules/sensor/{sensor_type}` 형식을 사용한다.

```json
{
  "schema_version": "1.0",
  "message_type": "aiot.telemetry",
  "device_id": "sensor-01",
  "sensor_type": "temperature",
  "value": 24.8,
  "unit": "C",
  "timestamp": "2026-08-27T14:00:00+09:00",
  "metadata": {
    "site_id": "site-01"
  }
}
```

Sensor Rule Bridge는 원본 telemetry를 규칙 입력으로 정규화하고, AIoT Parser는 값·단위·디바이스 식별자를 PostgreSQL에 저장한다.

## 4. EdgeX Reading

`cctv-edgex-adapter`는 AI 이벤트를 EdgeX Core Data가 이해할 수 있는 장치·Reading 형태로 투영한다.

필수 의미 필드:

```text
device      : 장치 또는 카메라 식별자
resource    : fall_detected, helmet, appearance 등 측정 리소스
value       : 원시 값 또는 JSON 문자열
timestamp   : 이벤트 발생 시각
origin      : 원본 timestamp 또는 gateway 시각
```

## 5. 장치 제어 명령

Action Layer 또는 AIoT Command Service가 명령을 생성하고 EdgeX Core Command를 통해 Device REST/Virtual Device로 전달한다.

```json
{
  "schema_version": "1.0",
  "command_id": "cmd-20260827-000045",
  "device_id": "siren-01",
  "action": "siren",
  "duration_seconds": 10,
  "source_event_id": "evt-20260827-000123",
  "requested_at": "2026-08-27T14:00:02+09:00"
}
```

명령 결과 토픽:

```text
cctv/commands/result
```

결과에는 최소한 `command_id`, `device_id`, `status`, `error`, `completed_at`을 포함한다.

## 6. 전달·재처리 규칙

- `message_id`와 `command_id`는 재시도에도 유지한다.
- 소비자는 식별자를 기준으로 중복 처리를 방지한다.
- MQTT 발행 실패는 Runtime Outbox에 저장한 뒤 재시도한다.
- 장치 명령 실패는 Action HTTP Outbox에 저장한다.
- payload를 확장할 때 기존 필드를 삭제하지 않고 `schema_version`을 올린다.
- 운영 지표에는 publish 성공·실패, retry, outbox pending을 기록한다.
