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

## 6. Action Layer 자동 이벤트 전체 경로 검증

2026-09-04 17:09 KST경, UAT용 Compose override를 적용해 Action Layer만
일시적으로 EdgeX 명령 모드로 전환한 뒤 테스트 이벤트를 발행했다.

### 입력 이벤트

```json
{
  "camera_id": "camera_1",
  "event_id": "uat-action-edgex-20260904",
  "type": "fall_detected",
  "severity": "critical",
  "event": {
    "display_message": "Action EdgeX 전체 경로 확인"
  }
}
```

### 확인된 경로

`AI 이벤트 MQTT 입력 → Action Layer 이벤트 처리 → EdgeX 명령 MQTT →
cctv-device-signboard → 전광판 출력 → EdgeX 결과 MQTT → 결과 DB 및
Action DB 상태 갱신` 순서로 처리됐다.

### 결과

전광판 Device Service 로그에서 다음 출력이 확인됐다.

```text
[Signboard] 표시: title='경고!' text='Action EdgeX 전체 경로 확인' color=7 back=0
[Signboard] 방송 시작 완료
```

공통 결과 DB와 Action Layer DB에는 다음 상태가 저장됐다.

```text
request_id=uat-action-edgex-20260904:signboard
event_id=uat-action-edgex-20260904
device_id=cctv-signboard-01
status=acknowledged
command_id=uat-action-edgex-20260904:signboard
```

### 판정

| 검증 항목 | 판정 |
|---|---|
| Action Layer EdgeX 모드 기동 | PASS |
| AI 이벤트 입력 처리 | PASS |
| EdgeX 전광판 명령 변환·전송 | PASS |
| 전광판 물리 출력 로그 | PASS |
| EdgeX 결과 수신 | PASS |
| 공통 결과 DB 저장 | PASS |
| Action DB 상태 갱신 | PASS |
| UAT 후 Direct 모드 복구 | PASS |

이번 검증은 실제 전광판만 대상으로 수행했다. 스피커·사이렌은 물리 장치
제약으로 명령을 전송하지 않았으며, 해당 장치의 실물 동작은 미검증 상태다.

## 7. 카메라-물리 장치 ID 매핑 검증

앞선 테스트의 `camera_1`은 `config/output_devices.json`의 `cam-01`과
일치하지 않아 물리 장치 ID 없는 기본 경로로 처리됐다. 이 상태를
성공으로 간주하지 않기 위해 `camera_id=cam-01`로 재시험했다.

재시험 결과, 레지스트리 매핑에 따라 다음 물리 장치 ID가 명령에 포함됐다.

```text
request_id=uat-action-edgex-mapped-20260904:signboard:cctv-signboard-01
event_id=uat-action-edgex-mapped-20260904
device_id=cctv-signboard-01
status=acknowledged
```

따라서 현재 레지스트리 기반 다중 장치 라우팅은 `camera_id`가
`camera_ids`와 정확히 일치할 때 검증된 상태다. 현재 `cameras.json`의
활성 카메라 ID인 `sample_eval`을 세 출력 장치의 매핑에 추가했으며,
기존 `cam-01` 매핑도 호환성을 위해 유지했다. `camera_1`은 별도 매핑이
없으므로 운영 입력에서 사용하지 않도록 주의한다.

## 미검증 범위

- Action Layer의 운영 기본 명령 모드는 아직 Direct이며, 이번 시험은 Device Service MQTT 경계와 결과 연계를 검증했다.
- 스피커·사이렌은 health만 확인했으며 실제 물리 동작은 검증하지 않았다.
- timeout·재시도·장치 장애 격리는 별도 UAT가 필요하다.

## 8. 후속 장애 재검증 기록

후속 검증에서 `sample_eval` 이벤트를 동일한 전광판 물리 ID로 전송했을 때
명령 상관관계는 정상적으로 유지됐지만, 결과 상태가 다음과 같이 반환됐다.

```text
request_id=uat-action-edgex-safety-20260904:signboard:cctv-signboard-01
event_id=uat-action-edgex-safety-20260904
device_id=cctv-signboard-01
status=failed
error_code=device_unreachable
```

확인된 사실은 다음과 같다.

- `cctv-device-signboard` 컨테이너 health는 `up`이고 컨테이너도 healthy였다.
- 결과 수집기는 `device_unreachable` 결과를 정상 저장했다.
- 해당 시험 구간에는 전광판 실제 출력 로그가 남지 않았다.
- 따라서 컨테이너 정상 상태를 전광판 TCP 연결 정상으로 간주하면 안 된다.

이 상태에서는 스피커·사이렌을 포함한 추가 물리 제어 시험을 진행하지 않는다.
전광판 네트워크 주소와 포트(`192.168.88.91:5000`)를 현장에서 확인한 뒤,
동일한 UAT를 재실행해야 한다.

## 9. 네트워크 회복 후 재시험 기록

2026-09-07 08:46 KST경 네트워크 진단에서 전광판 TCP 포트 접근이
정상으로 확인된 후 동일한 전체 경로를 재시험했다.

```text
event_id=uat-action-edgex-retry-20260907
request_id=uat-action-edgex-retry-20260907:signboard:cctv-signboard-01
device_id=cctv-signboard-01
status=acknowledged
error_code=null
```

이 결과로 다음 항목을 재확인했다.

- `sample_eval` 카메라에서 출력 장치 레지스트리 매핑 성공
- 물리 장치 ID가 포함된 EdgeX 요청 발행
- EdgeX 결과 수집 및 `acknowledged` 저장
- Action Layer 논리 명령 상태 갱신

따라서 직전 `device_unreachable`은 고정된 라우팅 오류가 아니라 당시
장치 응답 상태에 따른 일시 실패로 판단한다. 단, 현장 장치의 전원·망 상태가
변하면 동일 오류가 재발할 수 있으므로 UAT 시 TCP 연결 확인을 선행한다.

## 10. Docker 런타임 복구 기록

2026-09-07 09:37 KST경 Docker 런타임 변경 후 Action Layer 컨테이너를
기본 Compose 설정으로 재생성했다.

```text
Docker Server: 29.8.0
EDGEX_COMMAND_MODE: 미설정(기본 Direct)
Action Layer: healthy
MQTT: connected
```

`cctv-device-signboard`, `cctv-edgex-command-result`, `edgex-mqtt-broker`도
함께 실행 상태임을 확인했다. 이전의 `containerd shim/TTRPC` 기동 오류는
현재 재현되지 않는다.

## 11. 최신 이미지 기준 최종 canary

2026-09-07 10:36 KST경 최신 `aiot-parser` 이미지 재기동 이후 전광판만
허용한 EdgeX canary를 다시 수행했다.

```text
event_id=uat-final-edgex-20260907
request_id=uat-final-edgex-20260907:signboard:cctv-signboard-01
device_id=cctv-signboard-01
status=acknowledged
error_code=null
```

전광판 출력 로그도 확인했다.

```text
[Signboard] 표시: title='경고!' text='최종 EdgeX canary 확인' color=7 back=0
[Signboard] 방송 시작 완료
```

결과 DB와 Action DB 모두 동일한 `event_id`와 물리 장치 ID를 기준으로
`acknowledged` 상태를 기록했다. 시험 종료 후 기본 Direct 운영 모드로
복구했으며 Action Layer health와 MQTT 연결도 정상이다.

## 12. 9월 2주차 전광판 단독 canary 재검증

2026-09-07 13:57~14:03 KST에 `--device signboard` 방식으로 전광판만
대상으로 EdgeX canary를 재검증했다. 스피커와 사이렌은 허용 목록에 넣지
않았으며 실제 명령을 전송하지 않았다.

```text
event_id=uat-20260907-edgex-signboard-001
request_id=uat-20260907-edgex-signboard-001:signboard:cctv-signboard-01
device_id=cctv-signboard-01
status=acknowledged
```

확인된 전광판 Device Service 로그:

```text
[Signboard] 표시: title='경고!' text='[cam-01] 낙상 감지 - 즉시 확인' color=7 back=0
[Signboard] 방송 시작 완료
```

시험 전 Action Layer를 `EDGEX_COMMAND_MODE=edgex`,
`EDGEX_ALLOWED_DEVICES=signboard`로 재기동했고, 시험 후 UAT override를
제거했다. 복귀 후 환경변수는 비어 있었고 Action Layer health는 `up`,
MQTT는 `connected`였다.

`/commands` 내부 조회는 별도 인증 토큰 없이 호출해 HTTP 401이 반환됐으므로
해당 API 응답은 증빙으로 사용하지 않았다. 물리 출력 로그와 EdgeX 결과 DB를
기준으로 전광판 canary를 PASS 처리한다.

## 13. 실제 CCTV 이벤트 관찰 결과

2026-09-07 14:03~14:05 KST에 물리 명령을 추가로 발생시키지 않고
`cctv-ai-engine` 로그와 Action Layer 이벤트 DB를 대조했다. 영상 파이프라인은
계속 처리 중이었으며 최근 30분 집계는 다음과 같다.

```text
person=48, head=10, helmet=3, fall_detected=4
```

조회된 `fall_detected` 4건의 event_id는 모두 `uat-` 접두어를 가진 합성
UAT 이벤트였다. 따라서 이번 관찰만으로는 실제 CCTV 영상에서 생성된
새로운 낙상 이벤트의 Action Layer·물리 장치 전체 경로를 PASS로 판정하지
않는다. 실제 낙상 장면이 포함된 영상 replay 또는 현장 발생 이벤트를
별도 식별자로 수집한 뒤 재검증해야 한다.

## 14. 실제 낙상 영상 replay와 물리 출력 차단 검증

2026-09-07 14:26~14:52 KST에 `manual_old_overlay.jsonl`의
`old_demo_overlay_20260902_142824` 영상을 `sample_eval` 카메라로 replay했다.
재생 전 Action Layer를 `ACTION_DEVICE_DRY_RUN=true`로 기동하여 AI Engine과
MQTT 이벤트 흐름은 유지하고 물리 장치 호출·EdgeX Command 발행은 차단했다.

Action Layer DB에서 새로 확인된 `fall_detected` 명령은 6건이며 모두
`status=simulated`로 기록됐다. 이 결과는 실제 영상 → AI Engine → MQTT →
Action Layer까지 연결된 증거이지만, 영상 전체 구간의 정밀도·재현율을
산출한 결과는 아니다.

주의: replay가 완전히 종료되기 전에 운영 모드로 복원하면서 14:53:54에
잔여 이벤트 1건이 전광판으로 실제 출력됐다. 이후 AI Engine을 복원된 RTSP
카메라 설정으로 재시작했고, 14:54 이후 추가 전광판 출력은 확인되지 않았다.
향후 replay 도구는 입력 종료와 Action Layer 운영 복원을 별도 확인한 뒤
복원하도록 운영 절차를 보강해야 한다.

## 15. 다중 디바이스 격리 UAT 재검증

2026-09-07 15:20 KST에 Action Layer 컨테이너 내부의
`ACTION_DEVICE_DRY_RUN=true` 적용을 확인한 뒤 `cam-01` 합성 낙상 이벤트를
전송했다.

```text
event_id=uat-20260907-multi-device-safe-002
request_id=uat-20260907-multi-device-safe-002:input
command_id=uat-20260907-multi-device-safe-002:signboard
status=simulated
```

스피커는 네트워크 연결 불가로 실행 대상에서 제외됐고, 사이렌은 미등록
상태로 실행되지 않았다. 전광판·스피커·사이렌 관련 물리 출력 로그는
발생하지 않았다. 시험 후 Action Layer는 `ACTION_DEVICE_DRY_RUN=false`인
운영 설정으로 복원했다.

같은 날 15:17에 수행한 이전 시도(`...safe-001`)는 dry-run 환경변수 적용
확인 전에 컨테이너를 재생성해 전광판이 실제 출력됐다. 따라서 safe-001은
안전 UAT 증빙으로 사용하지 않고, safe-002만 안전 UAT 결과로 인정한다.

## 16. 장치별 EdgeX API 계약 재검증

2026-09-07 KST에 실행 중인 Device Service의 API 계약을 장치별로
검증했다. 정상·오류 payload 검증을 포함한 스크립트 실행 결과는 다음과 같다.

```text
speaker   HTTP 200  result_status=simulated  error_code=null
siren     HTTP 200  result_status=simulated  error_code=null
signboard HTTP 200  result_status=simulated  error_code=null
```

이번 검증은 API 계약과 dry-run 응답만 확인한 것이며, 스피커·사이렌의 실제
물리 동작을 증명하지 않는다. 실제 장치 연결 후에는 동일 계약에 대해
`acknowledged`와 timeout·장치 오류 결과를 별도로 검증해야 한다.

## 17. request_id 멱등성 코드 런타임 반영

2026-09-07 KST에 `request_id` 중복 실행 방지 코드가 포함된 이미지를
전광판·스피커·사이렌 Device Service에 빌드하고 컨테이너를 재생성했다.
세 컨테이너 모두 healthy이며 전광판 서비스는 다음 Command 토픽 구독 로그를
남겼다.

```text
edgex/commands/cctv/jetson-01/signboard
```

중복 요청에 대한 물리 메서드 1회 실행 보장은 단위 테스트로 검증했고,
실제 장치가 연결된 뒤에는 각 장치에 동일 `request_id`를 재전송하는 런타임
UAT를 추가해야 한다.

## 18. request_id 멱등성 런타임 UAT

2026-09-07 KST에 세 Device Service를 일시적으로 dry-run으로 실행한 뒤
각 HTTP Command API에 동일 `request_id`를 2회 전송했다.

```text
signboard: simulated, simulated / same_result=true
speaker:   simulated, simulated / same_result=true
siren:     simulated, simulated / same_result=true
```

세 서비스 모두 동일 결과와 request_id를 반환했으며, 물리 장치 호출은
발생하지 않았다. 시험 후 `SIGNBOARD_DRY_RUN=false`, `SPEAKER_DRY_RUN=false`,
`SIREN_DRY_RUN=false`로 복원하고 세 컨테이너 health를 확인했다.
