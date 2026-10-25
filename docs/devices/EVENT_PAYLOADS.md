# 디바이스 이벤트 전송 Payload 계약

10-12 추적 ID 계약 개정: 신규 AI 이벤트의 `object_id`와 중첩 추적 참조는 10진 문자열/null을 사용한다. 내부 정수·기존 DB/조회 호환 필드는 유지하며, 생성 방식·미추적 값·적용 범위는 [공통 ID 규격](../features/EVENT_SCHEMA_STANDARD.md#추적-id-계약--2026-10-12--i63)을 따른다. 운영 반영 여부는 진행 기록과 구분한다.

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
  "object_id": "12",
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
| 전광판 | `display_message`, `message`, `type`, `severity`, `camera_id`, `event_id` | `{event_id}:signboard` 생성 후 Dabit `display` |
| 사이렌 | `type`, `severity`, `camera_id`, `event_id` | `{event_id}:siren` 생성 후 ON, 자동 정지 후 OFF |

사이렌은 문구를 사용하지 않는다. 전광판의 `type`은 class/color 매핑에 사용될 수 있다.

10-12 로컬 매핑 보완: `config/output_devices.json`의 `camera_1`은 실물 확인된
`cctv-signboard-01`에만 연결한다. 미가동 스피커·미보유 사이렌은 추가하지 않으며
기존 `cam-01`/`sample_eval` 연결을 유지한다. Action Layer는 시작할 때 매핑을 읽는다.
현재 운영 Action 이미지에는 이 파일이 포함되어 있어 소스 수정만으로 반영되지 않는다.
새 이미지 반영 후에는 실제 영상 경보가 전광판에 출력될 수 있으므로 규칙 ID 보완과
배포·복귀 검증을 함께 진행한다. 이번 단위는 로컬·격리 시험이며 실물 출력은 하지 않았다.

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

10-12 로컬 보완에서는 EdgeX MQTT **발행 접수**를 `accepted`로 반환하고, 명령 DB/Action 응답은
고정 대상의 실제 결과가 오기 전까지 `pending`으로 표시한다. `alarm_played`는 발행 접수만으로
참이 되지 않는다. `events/executed`는 실행 시점의 스냅샷이며 이후 결과는 `devices/result`와 명령 조회로 확인한다.
장치별 `device_results`에는 `request_id`, `event_id`, `device_id`, `status`, `error_code`를 보존한다.
`devices/result.status`는 해당 장치 상태, 추가된 `command_status`는 논리 명령의 집계 상태다.

| 고정 대상의 결과 | 명령 집계 |
|---|---|
| 발행 접수 또는 미응답 장치가 남음 | `pending` |
| 실제 응답이 모두 성공 | `acknowledged` |
| 실제 성공과 실패가 섞임 | `partial_failure` |
| 전송 예외 또는 실패와 미응답이 섞임 | `uncertain` |
| 모두 실패 / 모두 모의 성공 | `failed` / `simulated` |

대상 목록 저장 실패 시 발행하지 않는다. 응답을 받지 못했다고 자동 재실행하지 않으며,
같은 프로세스에서 성공 응답을 잃은 뒤 같은 요청을 재수신하면 저장된 결과를 반환한다.
동시 수신은 같은 실행 결과를 공유하고 다른 요청은 공통 잠금으로 막지 않는다.
10-12 후속 보완: 장치3종 러너는 SQLite `device_requests`에 실행 전 요청을 선점하고 결과를 확정 저장한다.
메모리4096건 축출·서비스 재시작 뒤에도 완료 결과를 재사용한다. 기존 Action DB 스키마는 유지한다.

- 같은 `request_id`의 `event_id/device/device_id/action/payload/dry_run`을 정렬 JSON의 SHA-256으로 대조한다. `issued_at/attempt_id/source` 변경은 같은 재전송으로 취급한다.
- 내용·대상이 다르면 `failed/request_id_conflict`로 거부하고 원본 기록을 보존한다. HTTP는409, `X-Command-Id` 누락은400이다. HTTP 본문의 event_id는 공통 메타데이터로 분리해 MQTT와 같은 내용을 비교한다.
- 실행 선점 후 결과 없는 재수신은 `uncertain/execution_unconfirmed`다. 통신 실패도 실제 미출력의 증거가 없어 영속 모드에서는 uncertain으로 보존한다. 결과 DB 오류는 `uncertain/request_store_unavailable`이며 재출력하지 않는다.
- 러너는 `DEVICE_REQUEST_DB_PATH` 또는 `data/device_requests/<장치종류>.sqlite3`를 사용한다. 전광판 Compose는 `/app/data/device_requests`에 호스트 영속 디렉터리를 연결한다. 새 이미지 반영 전 UID1000 쓰기 권한·가용 공간·백업/복구를 확인해야 한다.
- DB가 없던 과거 명령은 소급 복원되지 않는다. 이력 자동 삭제/TTL은 두지 않으며 DB 삭제·과거 백업 복원은 중복 출력 위험이 있다. 운영 전환 시 구형 요청 유입과 DB 보존을 확인해야 한다.

영속 경로 없이 서비스 클래스를 직접 생성하면 기존 메모리 전용 동작이다. 중지된 구형 Dabit HTTP 러너의
직접 `execute`와 Direct transport는 이번 영속 보호 범위 밖이며 새 경로와 동시에 활성화하지 않는다.
Shadow의 발행 접수는 `awaiting_device_result`다. 위 영속 구현은249개 검증을 거쳤고, 아래 응답 기한/조회까지 포함한 최신 회귀는282개 PASS다.

### 응답 기한·실행 이력 조회 — 10-12 로컬 보완

Action Layer는 발행 등록 때 최초 `response_deadline`을 `action_commands`의 새 열에 보존한다.
`COMMAND_RESPONSE_TIMEOUT_SECONDS` 또는 `--command-response-timeout`으로 지정하며 기본30초는
잠정 운영값이다. 현장 합격 지연 기준이 아니며 유한한 양수만 허용한다. 기한은 UTC로 조회되고
재수신/재시작·설정 변경으로 늘어나지 않는다. 기존 대기 행은 최초 requested_at을 기준으로 이관한다.
장치별 결과나 유효한 시각이 없는 구형 기록은 자동 복구 근거가 없어 별도 확인이 필요하다.

- 매초 서비스 루프와 `/commands` 조회에서 기한이 지난 미응답만 `uncertain/response_timeout`으로 저장한다. DB 확정 저장 후 `devices/result`를 알리고 물리 명령을 재발행하지 않는다.
- 늦은 ACK는 정상 반영하고 `response_timed_out_at`을 남긴다. 이미 확인한 성공은 시간 초과나 뒤늦은 발행 상태로 덮지 않는다.
- 동시 중복 대기자는 기본30초 후 `uncertain/response_wait_timeout`을 반환한다. 원래 장치 호출은 취소하지 않으며 소유권·실행 이력을 유지해 두 번째 실행을 막는다.
- 장치 서비스의 `GET /requests/{URL-인코딩한 request_id}`는 **저장된 실행 이력**만 조회한다. 완료/미확정은200, 이력 없음은404, DB 조회 실패는503이다. 완료 이력이 없는 것은 미출력의 증명이 아니다.
- 조회는 DB나 실행 선점을 새로 만들지 않고 payload/인증값을 반환하지 않는다. `durable`은 영속 경로 사용 여부이며 DB 정상 여부는 status/error_code로 확인한다.
- 장치별 결과와 조회에 `retry_allowed=false`를 제공한다. 완료는 `retry_reason=already_completed`, 나머지는 `requires_reconciliation`이다. 현재 자동 재시도 허용 상태는 없다. 미발행 증거 또는 실물 상태 확인·감사 가능한 승인 재시도는 후속 구현 범위다.

예시(기존 장치 서비스 주소에서 읽기만 수행):

```bash
rtk curl -fsS 'http://127.0.0.1:59993/requests/event-1%3Asignboard%3Acctv-signboard-01'
```

새 API/DB 이관은 운영 미반영이다. 기한 점검은 호스트 시각과 DB 접근 상태에 의존하고 실행 중 장치를 강제 중지하지 않는다.
자동으로 원격 장치 이력을 조회해 Action DB와 대조하는 기능·attempt 관리·실물 상태 조회·승인 재시도·실제 전원 차단 시험은 남아 있다.

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


### 완료 증거·MQTT 이력 대조 — 10-13 로컬 보완

전광판 결과에 `execution_completed`, `resource_idle`, `completion_evidence`를 추가한다. 기본값은 false/false/null이며 단순 명령 ACK와 구분한다.
원래 경보와 유휴 화면 전체의 확인된 Dabit 응답을 같은 요청에 연결한 뒤 장치 DB에 저장하고 완료 결과를 발행한다.
`dabit_idle_frame_ack`는 컨트롤러 응답 근거이며 실제 LED 가시성 확인은 아니다. 스피커/사이렌 결과는 종료 증거를 만들어 내지 않는다.
기존 명령 토픽에서 `action=request_status`는 같은 request_id/event_id/device/device_id와 빈 payload를 사용하고, 장치3종에서 출력 없이 저장 이력만 읽는다.
HTTP `GET /requests/{request_id}`에도 보존된 완료 필드가 포함된다. 이력 없음·조회 실패는 재실행 허가가 아니다.
물리 큐 활성 시 Action은 5초마다 미확정 점유를 조회하고, 완료 증거가 없으면 유지한다. 운영/준비 이미지 미반영, 실물 종료 검증 잔여.
[상세 계약·복귀 제약](../planning/ACTION_POLICY_CONTRACT.md#3-장치별-순서와-선점) · [검증](../planning/evidence/action_completion_20261013.json)

## MQTT 장치 작업자 분리 옵션 (10-13 로컬 검증)

`DEVICE_WORKERS_ENABLED=true`일 때 기존 요청 DB의 `device_mqtt_inbox`에 원본을 commit한 후 QoS1 ACK한다. 물리 device_id별 호출과 읽기 전용 request_status를 분리하며, 결과/완료 메시지의 기존 ID 계약은 유지한다. 기본값은 OFF다.
`DEVICE_WORKER_TIMEOUT_SECONDS` 기본30초 경과는 `uncertain/device_worker_timeout`을 기록할 뿐 실행을 취소하거나 점유를 해제하지 않는다. 재시작 전 대기/실행 명령과 동일 요청 재전달은 `worker_recovery_requires_review`로 보류한다. 완료 이력이 있으면 기존 lookup 결과를 우선한다.
`DEVICE_MQTT_CLIENT_ID`는 서비스 간 유일하고 재시작에 안정적이어야 한다(미지정 시 Jetson ID+장치 종류). 새 요청을 만들기 전에도 Action의 물리 장치 점유/완료 조건을 따라야 한다. 직접 HTTP 명령은 이번 MQTT 격리 범위에 포함하지 않는다. [상세 설정·한계](../planning/ACTION_POLICY_CONTRACT.md#10-13-p32-장치별-작업자느린-장치-격리).

## Action REST 영속 접수 (10-13 로컬 검증)

영속 수신 DB와 REST를 함께 켜면 `POST /events`는 event_id가 있는 단일 JSON 본문을 저장한 뒤 **202**와 ingest_id를 반환한다. 재시도는 같은 event_id·내용·topic을 유지한다. 기본 topic은 rest/inbound이며 명시된 만료 규칙이 없으면 출력은 보류한다. 본문은 Content-Length가 필요하고 최대1MiB, chunked는 지원하지 않는다.
저장 실패는503, 잘못된 입력은400, 초과 본문은413이다. `GET /receipts/{ingest_id}`는 같은 내부 인증으로 접수 처리/보류 상태만 조회한다. stage=complete도 실제 출력 완료가 아니며 장치 결과는 기존 /commands에서 대조한다. 비영속 REST는 기존200 계약을 유지한다. [설정·복구·호환 범위](../planning/ACTION_POLICY_CONTRACT.md#10-13-p32-rest-영속-접수실행-분리).

공개 API 경보 입력의 선택 event_id는 Action까지 보존한다. ID가 없으면 기존 timestamp·내용으로 생성한다. 센서 중계도 원본 ID 또는 시각으로 ID를 고정하며, ID·시각이 모두 없는 센서는 영속 REST에서 거부된다. 재전달 때 현재 시각을 새로 넣지 않는다.
