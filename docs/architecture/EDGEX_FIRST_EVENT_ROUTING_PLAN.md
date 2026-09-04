# EdgeX 중심 Edge AIoT 전환 계획

## 목표

이 프로젝트의 목적은 AI·센서가 감지한 현장 상황을 엣지 장비에서 빠르게 판단해 물리 장치를 제어하는 것이다.

```text
현장 입력 → Edge AIoT 분석 → 규칙 판단 → EdgeX Command → Device Service → 장치
```

EdgeX는 단순 Reading 저장 연동이 아니라 모든 현장 장치 제어의 표준 중심이 되는 것을 목표로 한다.

## 현재 상태

현재 세 장치 모두 EdgeX 명령 계약을 처리하는 Device Service 경계를 갖추었지만, 운영 기본값은 기존 직접 제어를 유지하는 shadow 전환 단계다. 따라서 아직 모든 운영 제어가 EdgeX Core Command를 통과한다고 표현하면 안 된다.

현재 경로:

```text
AI/MQTT → Action Layer → 기존 직접 제어
AI/MQTT → EdgeX Command(MQTT 또는 HTTP) → 장치별 Device Service → 실제 장치
```

## 단계적 전환 진행 상태

1단계로 장치 명령의 공통 계약 모듈 `src/edgex/command_contract.py`와 단위 테스트를 추가했다. 2단계에서는 `EDGEX_SHADOW_ENABLED=false`를 기본값으로 유지하면서 Action Layer에 비교용 shadow Command 발행 경계를 연결했다. 기존 장치 직접 호출은 아직 운영 제어 경로로 유지된다.

스피커는 `src/edgex/speaker_device_service.py`에서 공통 Command를 InterM 호출로 변환하며, `runners/run_speaker_device_service.py`가 `edgex/commands/cctv/{jetson_id}/speaker`를 구독한다. 처리 결과는 `edgex/results/cctv/{jetson_id}/speaker`로 발행한다. 현재는 Core Command API를 직접 호출하는 최종 Device Service가 아니라, MQTT 기반 전환 검증 경계다.

스피커가 연결되지 않은 개발·검증 환경에서는 `SPEAKER_DRY_RUN=true`로 설정한다. 이때 장치 호출 없이 결과 상태를 `simulated`로 발행하므로 실제 장치 성공(`acknowledged`)과 혼동하지 않는다. 운영 환경에서는 반드시 `false`로 두고 실제 InterM 응답을 확인한다.

사이렌도 `src/edgex/siren_device_service.py`와 `runners/run_siren_device_service.py`를 통해 같은 구조로 동작한다. 명령은 `trigger`와 `stop`을 지원하며, 미연결 검증은 `SIREN_DRY_RUN=true`로 수행한다.

전광판은 기존 Dabit TCP 변환기 `DabitDeviceService`를 재사용해 `runners/run_signboard_device_service.py`가 MQTT 명령을 처리한다. `display`, `clear`, `power_on`, `power_off` 명령을 지원하며, 미연결 검증은 `SIGNBOARD_DRY_RUN=true`로 수행한다.

세 장치의 결과는 `runners/run_command_result_collector.py`가 `edgex/results/cctv/{jetson_id}/#`로 구독한다. `request_id`를 기본 키로 `data/runtime/edgex_command_results.db`에 저장하므로 동일 명령의 진행 상태가 갱신되며 중복 행이 생성되지 않는다.

Public API는 `GET /api/v1/command-results`와 `GET /api/v1/command-results/{request_id}`로 이 감사 결과를 읽기 전용 조회한다. 기존 API 인증 정책을 그대로 적용하며, `device_id`, `status`, `limit` 조건으로 목록을 필터링할 수 있다.

Action Layer도 `edgex/results/cctv/+/+`를 구독해 `request_id`에 해당하는 `action_commands` 상태를 갱신한다. 따라서 공통 결과 DB는 전체 장치 감사용으로 사용하고, Action Layer DB는 이벤트 실행 이력과 연결된 상태 조회용으로 사용한다. 결과 상태가 없는 명령은 기존 `sent` 상태로 남는다.

세 장치 Device Service는 MQTT 경계와 별도로 EdgeX Core Command 호환 HTTP 경계도 제공한다. 스피커는 `PUT /api/v3/device/name/cctv-speaker-01/play` 및 59991 포트, 사이렌은 `PUT /api/v3/device/name/cctv-siren-01/trigger` 및 59992 포트, 전광판은 `PUT /api/v3/device/name/cctv-signboard-01/display` 및 59993 포트를 사용한다. 장치가 연결되지 않은 환경에서는 각 `*_DRY_RUN=true`로 HTTP 라우팅과 결과 형식을 검증할 수 있다.

실제 장치가 없는 개발 환경에서는 `python scripts/ops/check_edgex_device_service_contracts.py --json`을 실행한다. 이 점검은 세 Device Service를 `dry-run`으로 생성하고 공통 HTTP 변환 경계를 통과시켜 HTTP 200과 `simulated` 결과를 확인한다. 실제 네트워크 연결, MQTT 브로커, 장치 동작까지 검증하는 현장 UAT를 대체하지는 않는다.

Jetson에서 Device Service 컨테이너가 실행된 뒤에는 `python scripts/ops/run_edgex_device_service_uat.py --mode dry-run --json`으로 세 HTTP 엔드포인트의 health와 명령 응답을 점검한다. 실제 장치 검증은 `--mode real --confirm-physical-control`을 함께 지정해야 하며, 이 경우 스피커 안내·사이렌 동작·전광판 표시가 발생할 수 있으므로 현장 승인 후 실행한다. 하나라도 health가 아니거나 기대한 결과 상태가 아니면 종료 코드 1을 반환한다.

EdgeX Metadata 등록은 `python edgex/register_output_devices.py --metadata-url http://127.0.0.1:59881`로 실행한다. 이 스크립트는 스피커·사이렌·전광판 프로파일을 업로드하고, 각 Device Service를 등록한 뒤 장치와 서비스의 연결을 확인한다. 기존에 `cctv-device-dabit`로 등록된 전광판이 있으면 새 `cctv-device-signboard` 서비스 연결로 갱신한다.

2026-09-03 실행 환경에서 세 서비스와 장치 등록을 완료했다. 등록 검증 결과는 `cctv-speaker-01 → cctv-device-speaker`, `cctv-siren-01 → cctv-device-siren`, `cctv-signboard-01 → cctv-device-signboard`이며, 세 Device Service의 `validate/device` 응답기도 정상 기동했다. 실제 장치 명령은 운영 환경의 `DRY_RUN=false` 설정 때문에 이번 검증에서 호출하지 않았다.

다중 장치 라우팅은 `config/output_devices.json`에 장치별 `device_id`, `device_type`, `site_id`, `zone_id`, `camera_ids`, `connection.host`, `connection.port`를 등록하고 `EDGEX_DEVICE_REGISTRY_PATH=/app/config/output_devices.json`로 활성화한다. 인증정보는 레지스트리에 저장하지 않고 기존 환경변수에서 읽는다. 레지스트리 경로를 비워두면 기존 단일 장치 호환 동작을 유지한다.

다중 장치 shadow 토픽은 `edgex/commands/cctv/{jetson_id}/{device_type}/{device_id}` 형식이며, Command payload에도 `device_id`를 포함한다. 동일 이벤트가 여러 장치를 대상으로 하면 장치별 `request_id`를 생성해 결과를 분리한다. Device Service 러너는 레지스트리가 활성화되면 장치별 클라이언트 풀을 만들고, MQTT 하위 토픽과 HTTP 경로의 `device_id`를 허용 목록과 대조한다.

Device Service는 `devices={device_id: client}` 형태의 클라이언트 풀을 받고 요청의 `device_id`에 해당하는 클라이언트만 실행한다. 미등록 ID는 `device_not_found`로 반환해 다른 장치에 잘못 전달되지 않는다. 장치별 연결 실패는 해당 장치의 실행 결과만 `device_unreachable` 또는 `device_error`가 되며, 다른 장치 클라이언트의 생성·실행은 계속된다. 남은 검증은 실제 2대 이상 장치와 장애 장치를 포함한 현장 UAT다.

코드 주석은 전환 대상부터 한글로 통일하며, 새 함수는 함수 바로 위에 기능을 설명하는 한글 주석을 최대 2줄로 작성한다. 기존 전체 코드의 주석 전환은 기능 영역별로 나누어 테스트와 함께 진행한다.

## 목표 구조

```text
AI Engine / AIoT Parser
  → Mosquitto MQTT
  → eKuiper 또는 Edge Rule
  → Action Layer 정책·승인
  → EdgeX Core Command
  → 장치별 Device Service
  → 실제 장치
```

장치별 제조사 protocol은 Device Service 안에만 둔다.

| 구성 | 책임 |
|---|---|
| AI Engine / Parser | 입력 분석·이벤트 생성 |
| MQTT | 엣지 이벤트 전달 |
| eKuiper | 조건·window·라우팅 |
| Action Layer | 정책·승인·cooldown·감사 이력 |
| EdgeX Core Command | 표준 Command 라우팅 |
| Device Service | HTTP/TCP 변환·timeout·응답 mapping |

## 이벤트 경로 재설정 원칙

### 실시간 제어 경로

DB 저장·Public API·Grafana를 기다리지 않고 다음 경로로 장치를 제어한다.

```text
입력 → AI/센서 분석 → MQTT → Rule → Action Policy → EdgeX Command → 장치
```

DB·API·모니터링은 같은 이벤트의 병렬 감사 경로로 동작한다.

### 이벤트 책임

| 단계 | 책임 |
|---|---|
| AI Engine | CCTV 분석과 AI event 생성 |
| AIoT Parser | Base64/TLV 해석과 원시값 생성 |
| eKuiper | threshold·시간 window 판단 |
| Action Layer | 사이트 모드·승인·중복·cooldown |
| EdgeX | 표준 Reading·Command·장치 관리 |
| Device Service | 실제 제조사 장치 통신 |
| Public API/Grafana | 조회·운영·시각화 |

## eKuiper와 Python Rule 중복 방지

eKuiper와 `SensorRuleBridge`가 같은 조건을 각각 최종 알람으로 발행하면 중복 장치 동작이 생길 수 있다. 다음 중 하나를 기준으로 확정한다.

- 권장: eKuiper는 stream/window 규칙의 기준, Python은 TLV 정규화·복잡한 물리 계산
- 대안: Python은 최종 센서 판정, eKuiper는 단순 전달·집계

어느 쪽이든 같은 센서 조건에 대해 최종 alarm event를 두 곳에서 발행하지 않는다.

## 장치 전환 순서

1. 스피커·사이렌·전광판 Device Profile과 Command 이름 확정
2. `device_id`, `resource`, `event_id`, `command_id` 계약 확정
3. 스피커 Device Service 구축
4. 사이렌 Device Service 구축
5. 전광판 직접 TCP와 EdgeX 경로 중 운영 기본 경로 하나 선택
6. Action Layer를 직접 client 호출자가 아닌 EdgeX Command 발행자로 변경
7. 기존 직접 경로는 feature flag와 rollback용으로 유지
8. EdgeX shadow command와 기존 결과 비교
9. 단일 장치·단일 카메라에서 전환
10. 전체 현장 전환 후 직접 제어 코드 제거

## 실시간성 측정

다음 전체 구간을 측정한다.

```text
event 발생 → MQTT publish → rule decision → EdgeX Command → Device Service → 장치 응답
```

필수 지표:

- event-to-command latency p50/p95/p99
- rule decision latency
- EdgeX Command latency
- Device Service timeout·retry
- command duplicate count
- 장치별 실제 출력 성공률

cloud·관제 화면이 끊겨도 local MQTT, eKuiper, EdgeX, Device Service가 동작해야 한다.

## 전환 완료 기준

- [ ] 세 장치가 EdgeX Device Service로 등록됨
- [ ] 모든 운영 장치 제어가 Core Command를 통과함
- [ ] Action Layer에 제조사 직접 HTTP/TCP 호출이 없음
- [ ] event/command 중복 실행 방지
- [ ] local network에서 독립 제어 가능
- [ ] event-to-command latency와 장치 성공률 측정
- [ ] 장애·복구·rollback 검증
- [ ] 단일 장치→단일 카메라→전체 현장 순서로 전환
