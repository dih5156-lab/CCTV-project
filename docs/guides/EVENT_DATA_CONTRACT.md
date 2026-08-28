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

## 이벤트별 공통 구조

영상 이벤트는 `DetectionEvent.to_dict()`를 통해 공통 필드를 사용합니다.

| 이벤트 | 의미 | 주요 상세 필드 | 기본 성격 |
|---|---|---|---|
| `person` | 사람 검출 | `object_id`, `class_name`, `bbox` | 정보 |
| `helmet` | 헬멧 착용 검출 | `object_id`, `class_name`, `bbox` | 정보 |
| `head` | 안전모 미착용/머리 검출 | `object_id`, `helmet_reason` | 경고 |
| `fall_detected` | 낙상 판정 | `fall_score`, `fall_direction`, `fall_type`, `keypoints` | 긴급 |
| `face_recognized` | 등록 얼굴 인식 | `face_name`, `face_score`, `face_person_id` | 조회/경고 |
| `face_unknown` | 미등록 얼굴 인식 | `face_score`, `recognizer`, `face_decision` | 경고 |
| `danger_zone` / `intrusion` | 위험구역 침입 | `zone_id`, `zone_name`, `mode` | 긴급 |
| `zone_object` | 감시구역 객체 검출 | `zone_id`, `mode=object_watch` | 경고 |
| `crowd_warning` | 인원 임계치 초과 | `zone_id`, `person_count`, `threshold` | 경고 |
| `appearance_match` | 외형 조건 매칭 | 색상·헬멧·가방 속성 | 설정에 따름 |
| `unsafe_behavior` | 위험행동 검출 | `reason`, `behavior`, `score` | 긴급 |

공통 이벤트 예시는 다음과 같습니다.

```json
{
  "type": "helmet",
  "severity": "normal",
  "bbox": {"x": 100, "y": 80, "width": 220, "height": 380},
  "confidence": 0.93,
  "timestamp": 1770000000.0,
  "object_id": 12,
  "class_name": "helmet",
  "metadata": {
    "backend": "deepstream",
    "camera_id": "cam01",
    "source_id": 0,
    "frame_num": 1820
  }
}
```

`head`는 안전모 미착용 안내, `helmet`은 착용 확인에 사용합니다. 장치 문구와 우선순위는 `config/event_type_map.json`에서 결정됩니다.

### 낙상 이벤트

```json
{
  "type": "fall_detected",
  "severity": "critical",
  "confidence": 0.86,
  "object_id": 12,
  "keypoints": [[120, 90, 0.98], [130, 110, 0.95]],
  "metadata": {
    "fall_score": 5.2,
    "fall_direction": "back",
    "fall_type": "뒤로 넘어짐",
    "fall_detail_status": "classified",
    "skeleton_format": "coco17_xyc"
  }
}
```

스피커·전광판에는 통합 낙상 문구를 사용하고, 방향과 세부 유형은 DB/API 조회용으로 보존합니다.

### 얼굴 인식 이벤트

```json
{
  "type": "face_recognized",
  "severity": "normal",
  "confidence": 0.91,
  "object_id": 12,
  "metadata": {
    "person_object_id": 12,
    "face_name": "등록 사용자",
    "face_score": 0.91,
    "face_person_id": "worker-001",
    "face_category": "employee",
    "recognizer": "insightface"
  }
}
```

미등록 얼굴은 `type=face_unknown`, `face_decision=unknown`으로 저장합니다. 실제 얼굴 이름과 ID는 개인정보이므로 로그·문서에 넣지 않습니다.

### 구역 이벤트

```json
{
  "type": "danger_zone",
  "severity": "critical",
  "confidence": 0.88,
  "object_id": 12,
  "metadata": {
    "zone_id": "zone-a",
    "zone_name": "위험구역 A",
    "mode": "danger",
    "zone_event": "zone_entered"
  }
}
```

구역 상태 변화는 `zone_entered`, `zone_dwelling`, `zone_exited`, `zone_object_detected`로 기록될 수 있습니다. 화면 표시 시 사람은 `danger_zone`, 객체 감시 모드는 `zone_object`로 매핑됩니다.

### 외형 이벤트·외형 DB 레코드

외형 분석은 사람 track과 연결된 별도 DB 레코드로 저장됩니다.

```json
{
  "event_id": "appearance:cam01:12:1770000000000",
  "camera_id": "cam01",
  "track_id": 12,
  "upper_color": "blue",
  "lower_color": "black",
  "has_helmet": true,
  "helmet_color": "yellow",
  "has_backpack": false,

  
  "has_handbag": false,
  "gender": "unknown",
  "age_group": "adult",
  "face_name": null,
  "attribute_backend": "attribute_backend",
  "bbox_x": 100,
  "bbox_y": 80,
  "bbox_w": 220,
  "bbox_h": 380,
  "timestamp": 1770000000.0
}
```

외형 검색 조건과 일치해 운영 이벤트가 필요할 때만 `appearance_match`를 발행하고, 원본 속성은 appearance DB에서 조회합니다.

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

## 3. 센서 파서 구조

센서 파서는 **바이너리 수신값을 해석하는 계층**입니다. Kuiper처럼 SQL로 이벤트를 판단하는 계층이 아닙니다.

```text
LoRa/MQTT uplink
  → Base64 디코딩
  → LwM2M TLV 파싱(offset 8)
  → tableName별 typed 데이터(T34954, T34955, T34958 ...)
  → 공통 SensorData 원시 저장
  → EdgeX/MQTT/센서 이벤트 브리지
```

파서 입력은 `app_id`, `dev_eui`, Base64 `payload`, 채널·주파수·수신시각입니다. 파싱 결과는 `tableName`과 `data`로 분리합니다.

```json
{
  "dev_eui": "0102030405060708",
  "app_eui": "aabbccddeeff0011",
  "device_id": "worker-sensor-01",
  "table": "t34958",
  "data": {
    "acc_x_g": 0.02,
    "acc_y_g": 0.11,
    "acc_z_g": 1.01,
    "angle_x_deg": 52.0,
    "angle_y_deg": 4.0,
    "event_code": true
  },
  "received_at": 1770000000
}
```

이 데이터는 `aiot/sensors/{dev_eui}/{table}`로 발행되고, 동시에 EdgeX Core Data와 센서 로그 API로 전달됩니다. 파서가 알람을 직접 결정하지 않는 이유는 원시값 저장과 운영 임계치 판단을 분리하기 위해서입니다.

## 4. 센서 규칙 브리지와 Kuiper 룰

센서 규칙 브리지는 파싱된 센서 측정값을 `SensorReading`으로 정규화한 뒤, 기울기·온도·IMU 진동 detector를 실행합니다.

```text
aiot/sensors/#
  → SensorRuleBridge
  → SensorReading.from_decoded()
  → tilt_alert(30°/45°), temperature_alert(50℃/70℃), vibration_alert(Δ가속도 0.5g/1.0g)
  → aiot/rules/sensor/{event_type}
  → Action Layer(스피커·전광판·사이렌)
```

예를 들어 `angle_x_deg=52`이면 `tilt_alert`, `severity=critical`이 생성됩니다. MQTT 발행 실패 시 최대 500건을 메모리에 보류했다가 연결이 복구되면 재발행합니다.

`T34958`에 전용 진동값이나 샘플링 주파수가 포함되지 않는 경우, `acc_x_g`, `acc_y_g`, `acc_z_g` 3축의 벡터 크기와 기준 중력 1g의 편차를 계산해 `vibration_alert`를 생성합니다. 기본값은 편차 0.5g 이상 `warning`, 1.0g 이상 `critical`이며 `SensorRuleConfig`로 조정할 수 있습니다. 이 방식은 주파수·RMS 기반의 지속 진동 분석이 아니라 순간 충격/큰 가속도 변화 감지입니다. 프로파일에는 단위(`g`)만 정의되어 있고 센서 제조사 물리 측정 범위는 확인되지 않았으므로, 코드의 ±32g 입력 제한은 하드웨어 사양이 아닌 비정상 payload를 거르는 소프트웨어 sanity bound입니다.

Kuiper는 **이미 JSON으로 발행된 AI 이벤트를 SQL 스트림으로 필터·집계·라우팅하는 계층**입니다.

```sql
SELECT camera_id, type, confidence
FROM ai_events_stream
WHERE type IN ('fall_detected', 'unsafe_behavior')
  AND confidence >= 0.7;
```

현재 룰 묶음은 다음 세 가지입니다.

- `intrusion_confidence_filter`: 신뢰도 0.7 이상 이벤트를 필터링
- `intrusion_5s_persist`: 5초 tumbling window에서 5회 이상 지속된 이벤트를 라우팅
- `intrusion_high_confidence_routing`: 신뢰도 0.9 이상을 `critical`로 라우팅

Kuiper 스트림은 `cctv/ai/events/+/+`를 구독하고, 결과는 `cctv/rules/intrusion/filtered`, `/persisted`, `/critical` 토픽으로 발행합니다. `runners/run_kuiper_rules.py`는 MQTT 소스 설정 → 스트림 재생성 → 룰 삭제·재등록을 REST API로 수행하며, 임계값은 환경변수나 CLI로 주입됩니다.

즉, 센서 파서는 `바이트 → 측정값`, 센서 규칙 브리지는 `측정값 → 센서 알람`, Kuiper는 `JSON 이벤트 → 조건부 라우팅`을 담당합니다.

## 5. Canonical Event

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

### 원본 JSON과 Canonical Event가 비슷해 보이는 이유

둘은 서로 다른 이벤트가 아니라 **같은 사실을 두 소비자 층에 맞춰 표현한 것**입니다.

- AI 원본 JSON: 기존 MQTT 소비자와 Action Layer가 즉시 읽는 평평한 레거시 형식
- Canonical Event: 여러 생산자(AI·센서·외부 MQTT)를 Public API와 EdgeX에서 공통 처리하기 위한 표준 봉투

따라서 `type`과 `event.event_type`, `timestamp`와 `occurred_at`, `camera_id`와 `device.camera_id`처럼 값이 겹칩니다. Canonical 변환은 원본을 새로 생성하거나 버리는 작업이 아니라, 원본 필드를 보존하면서 표준 위치와 `schema_version`, `message_type`, `source`, `raw`를 추가하는 보강(enrichment)입니다. 이 중복이 있어야 기존 구독자는 계속 동작하고, 새 소비자는 표준 필드만 사용해 AI·센서 이벤트를 동일하게 처리할 수 있습니다.

## 6. EdgeX v3 이벤트

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

## 7. Public API 계약

```text
GET  /api/v1/events
POST /api/v1/alerts
```

Public API 응답은 `metadata`를 객체로 반환합니다. 방향 검색은 다음처럼 사용합니다.

```text
/api/v1/events?event_type=fall_detected&fall_direction=back
```

자세한 요청·응답 예시는 [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md)를 참고하세요.

## 8. Action Layer 계약

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
