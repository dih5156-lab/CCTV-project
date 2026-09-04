# 디바이스 이벤트 전송 Payload 계약

## 1. 기본 흐름

```text
AI/센서 이벤트
  → MQTT cctv/ai/events/... 또는 aiot/rules/sensor/...
  → Action Layer 정규화
  → 알람 대상·confidence·cooldown 확인
  → 스피커 / 전광판 / 사이렌별 명령 실행
```

장치에 제조사 API payload를 직접 보내는 것이 아니라, 먼저 공통 이벤트 payload를 보낸다. Action Layer가 공통 이벤트에서 장치별 문구와 명령을 만든다.

## 2. 공통 AI 이벤트

토픽: `cctv/ai/events/{camera_id}/{event_type}`

```json
{
  "schema_version": "1.0",
  "message_type": "cctv.ai.event",
  "message_id": "msg-20260903-0001",
  "event_id": "evt-20260903-0001",
  "occurred_at": "2026-09-03T14:00:00+09:00",
  "camera_id": "camera_1",
  "type": "fall_detected",
  "severity": "critical",
  "confidence": 0.86,
  "display_message": "낙상 사고가 감지되었습니다.",
  "tts_message": "낙상 사고가 감지되었습니다.",
  "object_id": 12,
  "metadata": {"fall_score": 5.2, "fall_direction": "back", "source": "deepstream"}
}
```

`display_message`는 전광판, `tts_message`는 스피커에 사용된다. 없으면 `message`와 이벤트 타입별 기본 문구를 사용한다. `confidence`는 사이트 설정 threshold와 비교될 수 있다.

## 3. 공통 센서 이벤트

토픽: `aiot/rules/sensor/{sensor_event_type}`

```json
{
  "schema_version": "1.0",
  "message_type": "aiot.sensor.alert",
  "message_id": "sensor-msg-001",
  "event_id": "sensor-evt-001",
  "device_id": "tilt-sensor-01",
  "camera_id": "factory-24",
  "type": "tilt_alert",
  "severity": "critical",
  "confidence": 1.0,
  "value": 52.0,
  "unit": "deg",
  "occurred_at": "2026-09-03T14:00:02+09:00",
  "metadata": {"sensor_type": "tilt", "angle_x_deg": 52.0, "dev_eui": "0D0D33330D0D3333"}
}
```

Action Layer는 `camera_id`가 없고 `device_id`가 있으면 `camera_id=device_id`로 보정하고, `type`이 없으면 topic 마지막 값에 `_alert`를 붙인다. 신규 센서는 이 보정보다 명시적 필드 사용을 권장한다.

## 4. 디바이스별 사용 필드

| 장치 | 사용하는 이벤트 필드 | 내부 처리 |
|---|---|---|
| 스피커 | `type`, `severity`, `tts_message`, `message`, `camera_id`, `event_id` | `{event_id}:speaker` 생성 후 TTS 생성·변환·재생 |
| 전광판 | `display_message`, `message`, `type`, `severity`, `camera_id` | `{event_id}:signboard` 생성 후 Dabit `display` |
| 사이렌 | `type`, `severity`, `camera_id`, `event_id` | `{event_id}:siren` 생성 후 ON, 자동 정지 후 OFF |

사이렌은 문구를 사용하지 않는다. 전광판의 `type`은 class/color 매핑에 사용될 수 있다.

## 5. 실행 결과 Payload

토픽: `cctv/status/action/events/executed`

```json
{
  "camera_id": "camera_1",
  "event_type": "fall_detected",
  "severity": "critical",
  "status": "executed",
  "alarm_played": true,
  "http_sent": true,
  "devices": ["speaker", "signboard", "siren"],
  "device_results": [
    {"device": "speaker", "command_id": "evt-20260903-0001:speaker", "status": "acknowledged"},
    {"device": "signboard", "command_id": "evt-20260903-0001:signboard", "status": "failed"},
    {"device": "siren", "command_id": "evt-20260903-0001:siren", "status": "acknowledged"}
  ]
}
```

`acknowledged`는 프로젝트 클라이언트가 제조사 요청 성공 응답을 받은 상태이며 실제 소리·화면·점등을 사람이 확인했다는 뜻은 아니다. `failed`는 해당 장치 실패이며 다른 장치와 이벤트 저장은 계속될 수 있다.

## 6. 필터·승인 대기 결과

```json
{"camera_id":"camera_1","event_type":"helmet","confidence":0.42,"threshold":0.7,"status":"filtered"}
```

```json
{"event_id":"pending-001","camera_id":"camera_1","site_id":"site-01","event_type":"fall_detected","status":"pending"}
```

## 7. 새 이벤트 추가 절차

1. `docs/guides/EVENT_DATA_CONTRACT.md`에 원본 예시를 추가한다.
2. `src/services/_action_bridge_topics.py`의 구독·알람 대상 여부를 정한다.
3. `config/event_type_map.json`에 전광판/TTS 기본 문구를 등록한다.
4. `display_message`, `tts_message`, `severity`, cooldown을 테스트한다.
5. 실행 결과 topic과 Action 이력에서 장치별 결과를 확인한다.

