# Action Layer 디바이스 제어 API

## 1. 문서 목적

스피커, 전광판, 경광등을 직접 호출하지 않고 CCTV 시스템의 공통 제어 API로 운영하는 방법을 설명합니다.

```text
Public API (/api/v1/control)
  → cctv-action-layer 내부 REST
  → Speaker / Signboard / Siren
```

개별 제조사 API는 다음 문서를 참고합니다.

- `SPEAKER_API.md`
- `SIGNBOARD_API.md`
- `SIREN_API.md`

## 2. 인증

### Public API

```text
http://<host>:9000/api/v1/control
```

운영 환경에서는 `X-API-Key: <PUBLIC_API_KEY>` 헤더가 필요합니다.

### Action Layer 내부 REST

```text
http://cctv-action-layer:8080
```

`INTERNAL_SERVICE_TOKEN`이 설정되어 있으면 `/health`, `/ping`, `/metrics`, `/` 외 요청에 `X-Internal-Token` 헤더가 필요합니다.

## 3. 출력 장치 상태 조회

```http
GET /api/v1/control/devices
X-API-Key: <PUBLIC_API_KEY>
```

응답 예시:

```json
{
  "success": true,
  "data": [
    {
      "device": "speaker",
      "label": "스피커",
      "configured": true,
      "reachable": true,
      "status": "up",
      "host": "<speaker-host>",
      "port": 80,
      "protocol": "HTTP Digest / InterM"
    },
    {
      "device": "signboard",
      "label": "전광판",
      "configured": true,
      "reachable": true,
      "status": "up",
      "host": "<signboard-host>",
      "port": 5000,
      "protocol": "TCP Socket / Dabit"
    },
    {
      "device": "siren",
      "label": "경광등",
      "configured": false,
      "reachable": null,
      "status": "disabled",
      "host": null,
      "port": 80,
      "protocol": "HTTP Digest / InterM"
    }
  ]
}
```

`reachable`는 TCP 연결 가능 여부이며 인증이나 실제 명령 성공까지 보장하지는 않습니다.

## 4. 제어 모드

### 조회

```http
GET /api/v1/control/mode
X-API-Key: <PUBLIC_API_KEY>
```

### 변경

```http
POST /api/v1/control/mode
Content-Type: application/json
X-API-Key: <PUBLIC_API_KEY>
```

자동 모드:

```json
{
  "mode": "auto",
  "alarm_devices": ["speaker", "signboard", "siren"],
  "confidence_threshold": 0.7,
  "display_message": "안전사고가 감지되었습니다.",
  "tts_message": "안전사고가 감지되었습니다."
}
```

수동 승인 모드:

```json
{"mode":"manual","site_id":"site-01"}
```

| 필드 | 설명 |
|---|---|
| `mode` | `auto` 또는 `manual` |
| `site_id` | 생략하면 전체 기본 설정, 지정하면 해당 사이트만 변경 |
| `alarm_devices` | `speaker`, `signboard`, `siren` 목록 |
| `confidence_threshold` | 이 값 미만 이벤트는 조치하지 않음 |
| `display_message` | 전광판 문구 |
| `tts_message` | 스피커 문구 |

## 5. 수동 승인

```http
GET  /api/v1/control/pending
POST /api/v1/control/approve/{event_id}
POST /api/v1/control/reject/{event_id}
X-API-Key: <PUBLIC_API_KEY>
```

수동 모드에서 `pending` 목록을 조회하고, 승인하면 장치 실행으로 전달합니다. 거부하면 장치 실행 없이 이력을 남깁니다.

## 6. 처리 이력과 명령 상태

```http
GET /api/v1/control/action-events?limit=20
GET /api/v1/control/commands?limit=50
X-API-Key: <PUBLIC_API_KEY>
```

장치별 결과는 다음처럼 확인할 수 있습니다.

```json
{
  "device_results": [
    {"device":"speaker","status":"acknowledged"},
    {"device":"signboard","status":"failed"},
    {"device":"siren","status":"acknowledged"}
  ]
}
```

## 7. 내부 REST 전체 경로

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/health`, `/ping` | Action Layer/MQTT 상태 |
| `GET` | `/metrics` | Prometheus metrics |
| `GET/POST` | `/sites` | 사이트 조회/추가 |
| `DELETE` | `/sites/{site_id}` | 사이트 삭제 |
| `GET/POST` | `/mode` | 모드 조회/변경 |
| `GET` | `/devices` | 장치 상태 |
| `GET` | `/pending` | 승인 대기 이벤트 |
| `GET` | `/events?limit=20` | 처리 이력 |
| `GET` | `/commands?limit=50` | 명령 상태 |
| `POST` | `/events` | 내부 이벤트 큐 입력 |
| `POST` | `/approve/{event_id}` | 이벤트 승인 |
| `POST` | `/reject/{event_id}` | 이벤트 거부 |

`POST /events`는 장치 처리 완료를 기다리지 않고 큐 적재 결과를 반환합니다.

```json
{"status":"ok","queued":true}
```

큐가 가득 차면 HTTP 503이 반환될 수 있습니다.

## 8. MQTT 제어 토픽

| 토픽 | payload |
|---|---|
| `cctv/commands/mode` | `{"mode":"auto|manual", "site_id":"..."}` |
| `cctv/commands/approve` | `{"event_id":"..."}` |
| `cctv/commands/reject` | `{"event_id":"..."}` |
| `cctv/status/action/...` | 처리 결과/상태 |

AI 이벤트 입력 토픽은 일반적으로 다음 형식입니다.

```text
cctv/ai/events/{camera_id}/{event_type}
```

기본 알람 대상에는 `fall_detected`, `helmet`, `head`, `unsafe_behavior`, 구역 이벤트, 센서 위험 이벤트 등이 포함됩니다. `person`은 저장 대상과 알람 대상이 다를 수 있습니다.

## 9. 장애 확인 순서

1. `/api/v1/control/devices`에서 `configured`, `reachable` 확인
2. `/api/v1/control/mode`에서 장치가 알람 대상으로 선택됐는지 확인
3. `/api/v1/control/action-events`에서 장치별 결과 확인
4. Action Layer 로그 확인
5. 장치별 API 문서에서 포트/인증/프로토콜 확인
6. MQTT 이벤트가 알람 토픽에 들어왔는지 확인
7. cooldown, 수동 승인 대기, confidence threshold 확인

## 10. 이벤트 Payload와 장치별 변환

자동 이벤트 입력은 `cctv/ai/events/{camera_id}/{event_type}` 또는 `aiot/rules/sensor/{event_type}`의 JSON입니다.

```json
{
  "event_id": "evt-20260903-0001",
  "camera_id": "camera_1",
  "type": "fall_detected",
  "severity": "critical",
  "confidence": 0.86,
  "display_message": "낙상 사고가 감지되었습니다.",
  "tts_message": "낙상 사고가 감지되었습니다."
}
```

Action Layer는 공통 이벤트를 정규화한 후 사이트 모드, confidence threshold, 알람 대상, cooldown을 확인합니다.

| 대상 | 변환 | 결과 |
|---|---|---|
| 스피커 | `tts_message` 또는 기본 문구 | InterM TTS 생성·재생 |
| 전광판 | `display_message` 또는 기본 문구 | Dabit display TCP/Device Service |
| 사이렌 | 문구 없이 event type·severity 사용 | InterM Warnbell ON/OFF |

장치별 제조사 API와 실행 결과 payload는 [디바이스 이벤트 Payload 계약](EVENT_PAYLOADS.md)을 기준으로 합니다. 결과의 `acknowledged`는 HTTP/TCP 요청 성공이며 현장 출력 확인과는 구분합니다.
