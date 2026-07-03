# Action Layer 모듈 구조와 장비 연동

## 결론

Action Layer는 MQTT 또는 내부 REST로 들어온 이벤트를 저장하고, 정책에 따라 스피커·전광판·경광등과 외부 HTTP API로 전달합니다. 현재 구현은 저장소, 알람 판단, 장비 실행, 사이트 설정, REST 큐를 보조 모듈로 분리합니다.

## 처리 흐름

```text
MQTT / 내부 REST
  -> payload 정규화
  -> 자동/수동 승인 정책
  -> 알람 대상 및 쿨다운 판단
  -> 스피커·전광판·경광등 실행
  -> 외부 HTTP 전달
  -> SQLite action_events 저장
  -> cctv/status/action/* 상태 발행
```

REST 요청은 별도 큐와 worker에서 처리하므로 HTTP 요청 스레드가 장비 제어나 외부 전송 완료를 기다리지 않습니다.

## 모듈별 책임

| 파일 | 책임 |
|---|---|
| `src/services/action_bridge.py` | 전체 생명주기, 승인 흐름, MQTT 연결과 조율 |
| `src/services/_action_bridge_topics.py` | 구독·알람·명령·상태 토픽 정의 |
| `src/services/_action_bridge_models.py` | 제어 모드, 장비 종류, 사이트 설정 모델 |
| `src/services/_action_bridge_repo.py` | `action_events` SQLite 저장과 조회 |
| `src/services/_action_bridge_alarm.py` | 알람 대상 판정과 카메라/이벤트별 쿨다운 |
| `src/services/_action_bridge_executor.py` | 장비 실행, 외부 HTTP 전달, 실행 결과 저장 |
| `src/services/_action_bridge_site_registry.py` | 사이트별 모드와 장비 설정 관리 |
| `src/services/_action_bridge_rest_queue.py` | REST 이벤트 큐와 worker 생명주기 |
| `src/services/_action_bridge_payloads.py` | 센서 payload 정규화 |
| `src/services/_action_bridge_support.py` | 하위 호환 import와 공통 지원 객체 |

## 입력 토픽

기본 구독 범위:

- `cctv/ai/events/+/person`
- `cctv/ai/events/+/fall_detected`
- `cctv/ai/events/+/helmet`
- `cctv/ai/events/+/head`
- `cctv/ai/events/+/zone_entered`
- `cctv/ai/events/+/zone_dwelling`
- `cctv/ai/events/+/zone_object_detected`
- `cctv/ai/events/+/crowd_warning`
- `cctv/rules/intrusion/filtered`
- `cctv/rules/intrusion/persisted`
- `cctv/rules/intrusion/critical`
- `aiot/rules/sensor/tilt`
- `aiot/rules/sensor/temperature`
- `aiot/rules/sensor/vibration`

구독 토픽과 실제 알람 토픽은 다릅니다. 예를 들어 `person` 이벤트는 기본적으로 저장하지만 알람 대상에는 포함하지 않습니다. 정확한 기본값은 `src/services/_action_bridge_topics.py`를 기준으로 합니다.

## 액션 순서

1. 이벤트 수신 및 payload 정규화
2. 자동 모드이면 바로 실행하고, 수동 모드이면 승인 대기열에 저장
3. 알람 토픽과 쿨다운 조건을 통과하면 설정된 장비 제어
4. 외부 API 호출 (`--external-api-url` 설정 시)
5. SQLite 저장 및 `cctv/status/action/events/executed` 상태 발행

## 장비 연동

현재 스피커 표준 경로는 `src/devices/speaker.py`의 HTTP 장비 연동입니다. 과거의 `speaker-code-root`, webhook, 임의 command 방식은 현재 runner 인자가 아니므로 사용하지 않습니다.

```bash
SPEAKER_HOST=<스피커_IP>
SPEAKER_PORT=80
SPEAKER_USER=<계정>
SPEAKER_PASSWORD=<비밀번호>
SPEAKER_VOLUME=1
```

전광판은 `SIGNBOARD_*`, 경광등은 `SIREN_*` 환경변수를 사용합니다. 호스트가 비어 있으면 해당 장비만 비활성화되고 이벤트 저장과 다른 출력은 계속 동작합니다.

## 내부 REST API

기본 포트는 `8080`입니다. `INTERNAL_SERVICE_TOKEN`이 설정되면 `/health`, `/ping`, `/metrics`, `/`를 제외한 요청에 `X-Internal-Token` 헤더가 필요합니다.

| 메서드 | 경로 | 역할 |
|---|---|---|
| `GET` | `/health`, `/ping` | Action Layer와 MQTT 연결 상태 |
| `GET` | `/metrics` | Prometheus metric |
| `GET/POST` | `/sites` | 사이트 조회·추가 |
| `DELETE` | `/sites/{site_id}` | 사이트 삭제 |
| `GET/POST` | `/mode` | 기본 또는 사이트 제어 모드 조회·변경 |
| `GET` | `/pending` | 수동 승인 대기 이벤트 |
| `GET` | `/devices` | 출력 장비 설정·연결 상태 |
| `GET` | `/events?limit=20` | 최근 처리 이벤트 이력 |
| `POST` | `/events` | 내부 이벤트 비동기 큐 입력 |
| `POST` | `/approve/{event_id}` | 대기 이벤트 승인 |
| `POST` | `/reject/{event_id}` | 대기 이벤트 거부 |

`POST /events`는 장비 동작 완료를 기다리지 않고 큐 적재 성공 시 `{"status":"ok","queued":true}`를 반환합니다. 큐가 가득 차면 `503`을 반환합니다.

## 실행

장비 없이 이벤트 저장과 외부 전송만 확인:

```bash
python runners/run_action_bridge.py \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --db-path data/runtime/action_events.db \
  --external-api-url http://localhost:8000/api/alerts \
  --alarm-cooldown 10
```

스피커 연동:

```bash
python runners/run_action_bridge.py \
  --mqtt-broker localhost \
  --db-path data/runtime/action_events.db \
  --speaker-host "${SPEAKER_HOST}" \
  --speaker-port "${SPEAKER_PORT:-80}" \
  --speaker-user "${SPEAKER_USER}" \
  --speaker-password "${SPEAKER_PASSWORD}" \
  --speaker-volume 1 \
  --alarm-cooldown 10
```

## 저장소

- 이벤트 이력: `action_events(id, event_id, received_at, topic, camera_id, event_type, confidence, severity, alarm_played, http_sent, payload_json)`
- 외부 API 재시도 outbox: `ACTION_HTTP_OUTBOX_DB`
- Compose 기본 outbox 경로: `/app/data/runtime/action_http_outbox.db`

외부 HTTP 전송은 메모리 재시도 큐에서 지수 backoff로 재시도하고, outbox가 설정되면 pending 항목을 SQLite에도 기록합니다. worker가 유휴 상태일 때 영속 pending 항목을 다시 읽어 전송합니다.

민감한 장비 계정과 비밀번호는 명령행에 직접 기록하지 말고 `.env` 또는 배포 secret으로 전달합니다.
