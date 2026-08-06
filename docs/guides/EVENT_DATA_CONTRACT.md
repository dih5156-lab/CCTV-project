# 이벤트 데이터 계약

이 문서는 카메라 AI 이벤트가 MQTT, Canonical Event, EdgeX, Public API, Action Layer를 통과할 때 유지해야 하는 필드와 변환 규칙을 설명합니다.

## 전체 흐름

```text
AI/센서
  → MQTT topic + legacy JSON
  → canonicalize_event_payload()
  → EdgeX v3 event(readings)
  → Public API/JSONL 저장
  → Action Layer(알람)
```

## 1. AI 이벤트 원본 JSON

AI 엔진은 기존 소비자 호환을 위해 top-level 필드를 유지합니다.

```json
{
  "type": "fall_detected",
  "camera_id": "cam01",
  "severity": "critical",
  "confidence": 0.86,
  "timestamp": 1770000000.0,
  "bbox": {"x": 100, "y": 80, "width": 220, "height": 380},
  "object_id": 12,
  "metadata": {
    "fall_score": 5.2,
    "fall_reasons": ["torso_horizontal:142.1"],
    "fall_direction": "back",
    "fall_type": "뒤로 넘어짐",
    "fall_detail_status": "classified"
  }
}
```

낙상 방향은 `metadata`에 저장하며, 이벤트 타입은 항상 `fall_detected`로 통합합니다.

## 2. MQTT 계약

### AI 탐지 토픽

```text
cctv/ai/events/{camera_id}/{event_type}
```

예:

```text
cctv/ai/events/cam01/fall_detected
```

Action Layer는 다음 토픽을 구독합니다.

- `cctv/ai/events/+/person`
- `cctv/ai/events/+/fall_detected`
- `cctv/ai/events/+/helmet`
- `cctv/ai/events/+/head`
- `cctv/ai/events/+/face_unknown`
- `cctv/ai/events/+/face_recognized`

### 센서 룰 토픽

```text
aiot/rules/sensor/tilt
aiot/rules/sensor/temperature
aiot/rules/sensor/vibration
```

센서 이벤트도 `metadata`에 원본 센서 값과 수신 정보를 보존합니다.

## 3. Canonical Event

`src/canonical_event.py`의 `canonicalize_event_payload()`가 레거시 payload를 보강합니다.

```json
{
  "schema_version": "1.0",
  "message_type": "ai_detection_event",
  "message_id": "optional-id",
  "occurred_at": "2026-08-06T09:00:00+09:00",
  "device": {"camera_id": "cam01"},
  "gateway": {},
  "event": {
    "event_type": "fall_detected",
    "severity": "critical",
    "source": "cctv-ai-engine",
    "source_type": "vision",
    "confidence": 0.86,
    "display_message": "낙상 감지 - 즉시 확인",
    "tts_message": "낙상이 감지되었습니다. 즉시 확인 바랍니다."
  },
  "decoded": {},
  "raw": {"metadata": {"fall_direction": "back"}}
}
```

기존 `type`, `camera_id`, `confidence`, `severity`, `metadata` 필드는 하위 호환을 위해 제거하지 않습니다. 표준 소비자는 `event.event_type`와 `occurred_at`을 우선 사용합니다.

## 4. EdgeX v3 이벤트

EdgeX로 전달할 때는 value가 JSON 문자열인 reading으로 변환합니다.

```json
{
  "apiVersion": "v3",
  "requestId": "request-uuid",
  "event": {
    "apiVersion": "v3",
    "id": "event-uuid",
    "deviceName": "cctv-ai-engine",
    "profileName": "CCTV-AI-Profile",
    "sourceName": "fall_detection",
    "origin": 1770000000000000000,
    "readings": [{
      "resourceName": "fall_detection",
      "valueType": "String",
      "value": "{\"type\":\"fall_detected\",\"confidence\":0.86,\"metadata\":{...}}"
    }]
  }
}
```

이벤트 타입별 EdgeX resource 매핑:

| 이벤트 | resource |
|---|---|
| `fall_detected`, `not_fall` | `fall_detection` |
| `helmet`, `head` | `helmet_detection` |
| `person` | `person_detection` |

## 5. Public API 계약

```text
GET  /api/v1/events
POST /api/v1/alerts
```

Public API 응답은 `metadata`를 객체로 반환합니다. 방향 검색은 다음처럼 사용합니다.

```text
/api/v1/events?event_type=fall_detected&fall_direction=back
```

자세한 요청·응답 예시는 [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md)를 참고하세요.

## 6. Action Layer 계약

Action Layer는 `event_type`과 `severity`로 알람 종류를 결정합니다. `metadata.fall_direction`과 `fall_type`은 저장·조회용이며 장치 문구를 변경하지 않습니다.

```text
fall_detected
  → speaker: 낙상이 감지되었습니다. 즉시 확인 바랍니다.
  → signboard: 낙상 감지 - 즉시 확인
```

## 변경 규칙

- 새 필드는 optional로 추가합니다.
- 기존 `type`, `event_type`, `camera_id`, `metadata`는 제거하거나 타입을 바꾸지 않습니다.
- 새 MQTT 토픽은 구독 목록과 이 문서를 함께 수정합니다.
- EdgeX resource 이름을 변경할 때는 consumer와 프로파일을 함께 검증합니다.
- 민감정보(비밀번호, API 키, RTSP 인증정보)는 payload·문서·로그에 넣지 않습니다.
