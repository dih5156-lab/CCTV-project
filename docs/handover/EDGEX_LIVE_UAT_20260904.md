# EdgeX 전광판 실장치 UAT 기록

## 시험 정보

| 항목 | 내용 |
|---|---|
| 시험 일시 | 2026-09-04 16:50 KST 전후 |
| 시험 장치 | `cctv-signboard-01` |
| Device Service | `cctv-device-signboard` |
| 명령 경로 | MQTT `edgex/commands/cctv/jetson-01/signboard/cctv-signboard-01` |
| 결과 경로 | MQTT `edgex/results/cctv/jetson-01/signboard` |
| 실행 환경 | `SIGNBOARD_DRY_RUN=false` |

## 시험 명령

```json
{
  "version": "1",
  "request_id": "uat-20260904-event-06:signboard:cctv-signboard-01",
  "event_id": "uat-20260904-event-06",
  "source": "manual-uat",
  "device": "signboard",
  "device_id": "cctv-signboard-01",
  "action": "display",
  "payload": {
    "text": "Action 상태 추적 확인",
    "title": "테스트",
    "event_type": "uat",
    "severity": "info"
  }
}
```

## 확인 결과

### 1. Device Service health

`cctv-device-signboard:59993/health` 응답:

```json
{"service": "cctv-device-signboard", "status": "up"}
```

### 2. 물리 장치 실행 로그

Device Service 로그에서 다음 동작을 확인했다.

```text
[Signboard] 표시: title='테스트' text='Action 상태 추적 확인' color=7 back=0
[Signboard] 방송 시작 완료
```

로그 기준으로 실제 TCP 전송 경로가 호출되었으며, `SIGNBOARD_DRY_RUN=false` 상태였다. 별도 육안 촬영 사진은 아직 등록하지 않았다.

### 3. EdgeX 결과 payload

```json
{
  "command_id": "uat-20260904-event-06:signboard:cctv-signboard-01",
  "device_id": "cctv-signboard-01",
  "event_id": "uat-20260904-event-06",
  "status": "acknowledged",
  "error_code": null,
  "request_id": "uat-20260904-event-06:signboard:cctv-signboard-01"
}
```

### 4. 공통 결과 DB

`/app/data/runtime/edgex_command_results.db`에서 다음 행을 확인했다.

```text
request_id=uat-20260904-event-06:signboard:cctv-signboard-01
event_id=uat-20260904-event-06
device_id=cctv-signboard-01
status=acknowledged
error_code=null
```

### 5. Action Layer 명령 DB

물리 요청 ID의 접미사를 제거하고 논리 명령 ID를 갱신하는 것을 확인했다.

```text
command_id=uat-20260904-event-06:signboard
status=acknowledged
event_id=uat-20260904-event-06
```

## 판정

| 검증 항목 | 판정 |
|---|---|
| Device Service health | PASS |
| MQTT Command 수신 | PASS |
| 전광판 실제 호출 로그 | PASS |
| `request_id` 포함 결과 발행 | PASS |
| 공통 결과 DB 저장 | PASS |
| 논리 `command_id` 상태 갱신 | PASS |
| 육안 촬영 증빙 | 미등록 |

## 미검증 범위

- Action Layer의 운영 기본 명령 모드는 아직 Direct이며, 이번 시험은 Device Service MQTT 경계와 결과 연계를 검증했다.
- 스피커·사이렌은 health만 확인했으며 실제 물리 동작은 검증하지 않았다.
- timeout·재시도·장치 장애 격리는 별도 UAT가 필요하다.
