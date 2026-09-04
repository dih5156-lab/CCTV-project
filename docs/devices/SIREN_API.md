# 경광등/사이렌 API 인수인계 문서

## 1. 문서 목적

이 문서는 InterM 경광등/사이렌의 HTTP 제어 API와 CCTV Action Layer 연결 방식을 설명합니다.

## 2. 구현 위치

| 항목 | 경로 |
|---|---|
| 경광등 클라이언트 | `src/devices/siren.py` |
| Action Layer 연결 | `src/services/_action_bridge_executor.py` |
| 실행 진입점 | `runners/run_action_bridge.py` |
| 환경변수 | `.env.example`, `.env.jetson.example` |

## 3. 연결 정보와 인증

현재 클라이언트는 다음 주소를 사용합니다.

```text
http://{SIREN_HOST}/interm-api
```

- HTTP Digest 인증
- 기본 API 경로 `/interm-api`
- 설정상 기본 포트 `80`
- 연결 timeout 3초, 응답 timeout 7초
- host/계정/비밀번호가 모두 있어야 활성화

### 중요 확인사항

`SensorConfig`에는 `port`가 있지만 현재 `_SirenClient`는 base URL을 만들 때 host만 사용합니다. 80번이 아닌 포트를 사용한다면 실제 장치 테스트 전에 URL 조립 코드를 확인해야 합니다.

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `SIREN_HOST` | 빈 값 | 경광등 IP/호스트명 |
| `SIREN_PORT` | `80` | 설정값. 비표준 포트 사용 시 코드 확인 필요 |
| `SIREN_USER` | 빈 값 | Digest 계정 |
| `SIREN_PASSWORD` | 빈 값 | Digest 비밀번호 |
| `SIREN_AUTO_STOP` | `10` | 자동 정지 시간(초). 0 이하면 자동 정지 안 함 |

## 4. InterM API

### 켜기

```http
POST /interm-api/Warnbell/Control
Content-Type: application/json
Authorization: Digest ...
```

```json
{"Control": true, "Run": true}
```

### 끄기

```json
{"Control": true, "Run": false}
```

## 5. 프로젝트 내부 메서드

```python
siren.trigger(event_type="fall_detected", camera_id="camera_1")
siren.stop()
```

| 메서드 | 설명 |
|---|---|
| `trigger(event_type, camera_id)` | 경광등 ON 및 자동 정지 타이머 등록 |
| `stop()` | 경광등 OFF |

자동 정지 전에 `trigger()`가 다시 호출되면 기존 타이머를 취소하고 새 타이머를 등록합니다.

## 5-1. Action Layer 이벤트 Payload

MQTT 토픽은 `cctv/ai/events/{camera_id}/{event_type}` 또는 `aiot/rules/sensor/{event_type}`입니다.

```json
{
  "event_id": "evt-20260903-0001",
  "camera_id": "camera_1",
  "type": "fall_detected",
  "severity": "critical",
  "confidence": 0.86,
  "occurred_at": "2026-09-03T14:00:00+09:00"
}
```

사이렌은 `display_message`나 `tts_message`를 사용하지 않고, 알람 대상·severity·confidence 정책을 통과한 이벤트에서 ON됩니다. Action Layer는 `{event_id}:siren` command ID를 만들고 `trigger(event_type, camera_id)`를 호출한 뒤 `SIREN_AUTO_STOP` 후 OFF합니다. 전체 계약은 [디바이스 이벤트 Payload 계약](EVENT_PAYLOADS.md)을 참고합니다.

## 6. Action Layer 동작

```text
fall_detected
  → Action Layer 알람 조건 통과
  → siren.trigger()
  → SIREN_AUTO_STOP 초 후 siren.stop()
```

사이트의 `alarm_devices`에 `siren`이 포함되어 있어야 합니다. 사이렌 실패 여부와 관계없이 다른 장치와 이벤트 저장은 계속 처리됩니다.

## 7. 오류 처리

| 오류 | 프로젝트 동작 |
|---|---|
| host/계정 누락 | 비활성화, `False` 반환 |
| timeout/connection error | 네트워크 오류 로그, `False` 반환 |
| HTTP 오류 | 오류 로그, `False` 반환 |
| 자동 정지 실패 | 오류 로그. 장치 상태 별도 확인 필요 |

대표 로그:

```text
[Siren] 경광등 ON (camera=..., type=...)
[Siren] 자동 정지 예약: ...초 후
[Siren] 경광등 OFF
[Siren] 경광등 오프라인 (...:...) - trigger 건너뜀
```

## 8. 테스트 절차

1. `.env.jetson`에 host/계정/비밀번호를 설정합니다.
2. Action Layer를 재시작합니다.
3. `GET /api/v1/control/devices`로 `configured`, `reachable`을 확인합니다.
4. 테스트 이벤트를 발생시킵니다.
5. 실제 점등 후 자동 정지되는지 확인합니다.
6. `GET /api/v1/control/action-events`에서 결과를 확인합니다.

실제 장치 없이 네트워크 오류가 `failed`로 기록되는 것만 확인한 경우, 운영 성공으로 간주하지 않습니다.
