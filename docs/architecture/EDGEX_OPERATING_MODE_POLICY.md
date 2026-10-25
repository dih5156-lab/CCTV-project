# EdgeX 운영 모드 전환 정책

10-12 로컬 후속: MQTT 발행 접수를 `accepted`, 장치 응답 대기를 `pending`으로 구분하고
실제 장치 응답의 `acknowledged`와 분리했다. 대상별 부분 실패·불확실 상태를 보존하며
Shadow도 발행 접수만으로 직접 출력과 일치했다고 표시하지 않는다. 관련162 PASS,
운영 미반영이므로 아래10-08 운영 관측의 과거 `acknowledged` 의미와 구별한다.
[상태 계약](../devices/EVENT_PAYLOADS.md#5-실행-결과-payload) · [검증 근거](../planning/evidence/action_concurrency_20261012.json)

> **2026-10-08 현재 확인:** 실행 모드는 `edgex`다. AI 이벤트의 Core Data 저장은 확인했지만,
> 운영 컨테이너에서 `camera_1`의 전광판/스피커/사이렌 대상을 조회하면 모두0개다.
> 설정은 `cam-01`/`sample_eval`만 연결하며 최신 장치 결과는09-17 시험 기록이다.
> 다음은 전광판1대의 대상·명령 결과 사전검증이다. 사용자 확인상 전광판·스피커 각1대와
> 온습도·기울기 센서 시험이 가능하나, 실제 표시/음성·센서 유입은 별도 검증한다.
> MQTT 발행 성공의 `acknowledged`와 장치 결과 메시지·실제 작동을 구분한다.
> [현재 근거](../planning/evidence/smartpole_direction_20261008.json) · [작업 순서](../planning/README.md#smartpole-direction-20261008)

> **2026-09-18 현행 대조:** 아래는 Direct 기본값을 전제로 작성된 정책·명령 예시다.
> 현재 실행은 `edgex`이며 `speaker,signboard,siren`이 허용 목록에 있다.
> 아래 재생성 명령을 현행 Direct 복구 명령으로 확인한 것은 아니다.
> 정책·Compose·검사·테스트 정합성은 [I06](../planning/STATUS.md#3-미해결-이슈-대장)으로
> 관리한다. 현행 계약과 복구 범위를 확인하기 전 이 과거 예시로 운영을 변경하지 않는다.
> 이번 최신화는 운영 설정 변경이나 미검증 장치의 승인을 뜻하지 않는다.

> **2026-10-06 검사 구분:** 배포 구성 검사의 기본 프로필은 현재 `env_file` 전달과
> 기록된 `edgex`/`speaker,signboard,siren` 값의 일치 여부를 확인한다.
> `--edgex-profile direct-rollback`은 아래 과거 Direct 계약을 계속 검사하며,
> 현재 구성에서는 실패한다. 기본 검사 통과는 Direct 복구·실물 장치 합격을 뜻하지 않는다.
> [I46 구성 대조 근거](../planning/VIDEO_INFERENCE_CONTRACT.md#deployment-config-v107).

## 1. 목적

이 문서는 Direct, Shadow, EdgeX 세 가지 장치 제어 모드의 사용 기준과
전환 조건을 정의한다. 이 프로젝트의 최종 목적은 Edge AIoT 환경에서 AI
판단 결과를 EdgeX를 통해 현장 장치로 빠르게 전달하고 제어하는 것이다.

운영 모드 전환은 기능이 동작한다는 이유만으로 진행하지 않고, 장치별
실제 증거와 복구 절차가 준비된 경우에만 진행한다.

## 2. 모드별 역할

| 모드 | 장치 제어 경로 | 용도 | 현재 판단 |
|---|---|---|---|
| Direct | Action Layer → 장치 클라이언트 | 기존 운영 호환, 장애 시 복구 | 과거 기본 정책. 현재 운영 아님·현행 복구 미검증 |
| Shadow | Direct 실행 + EdgeX Command 비교 발행 | EdgeX payload·라우팅 사전 검증 | 제한된 시험용 |
| EdgeX | Action Layer → 프로젝트 MQTT 명령 계약 → Device Service → 장치 | 현재 실행 경로 | camera_1 대상 매핑 누락, 장치별 실물 검증 잔여. Core Data 저장 경로와 구분 |

## 3. 전환 조건

### Direct → Shadow

다음 조건을 모두 만족하면 카메라 또는 사이트 단위로 Shadow 시험을
진행할 수 있다.

- `config/output_devices.json`의 물리 장치 ID와 `camera_ids`가 일치한다.
- `scripts/health/check_cameras_config.py`가 통과한다.
- EdgeX Command payload에 `event_id`, `request_id`, `device_id`가 포함된다.
- Direct 장치 동작 결과와 EdgeX 발행 결과를 비교할 로그 위치가 정해져 있다.
- 시험 종료 후 Direct로 복귀할 Compose 명령을 준비한다.

### Shadow → EdgeX

다음 장치별 증거가 있어야 해당 장치를 EdgeX 모드로 전환한다.

- Device Service health가 정상이다.
- 실제 EdgeX Command 수신과 결과 MQTT 발행을 확인했다.
- 결과 DB에 `request_id`, `event_id`, `device_id`, `status`가 저장된다.
- Action DB의 논리 `command_id`가 정상 갱신된다.
- 장치의 실제 출력 또는 장치 응답을 현장 증거로 남겼다.
- 실패 시 Direct 복귀와 재시험 절차가 문서화되어 있다.

현재 전광판은 `sample_eval → cctv-signboard-01` 매핑과 EdgeX 결과
`acknowledged`까지 확인했다. 스피커와 사이렌은 실제 물리 동작 증거가
없으므로 EdgeX 모드 전환 대상에 포함하지 않는다.

## 4. EdgeX canary 운영 방법

처음부터 모든 장치를 EdgeX로 전환하지 않고, 실제 연결이 확인된 전광판
한 대만 대상으로 canary 시험을 진행한다.

```bash
docker compose \
  -f docker-compose.jetson.yml \
  -f docker-compose.edgex-uat.yml \
  up -d --no-deps cctv-action-layer
```

canary 시험 중에는 `EDGEX_ALLOWED_DEVICES=signboard`를 유지한다. 시험이
끝나면 즉시 기본 Direct 설정으로 복귀한다.

```bash
docker compose -f docker-compose.jetson.yml \
  up -d --no-deps cctv-action-layer
```

복귀 후 다음을 확인한다.

```bash
docker exec cctv-action-layer sh -lc \
  'printf "EDGEX_COMMAND_MODE="; printenv EDGEX_COMMAND_MODE; curl -fsS http://localhost:8080/health'
```

`EDGEX_COMMAND_MODE`가 비어 있고 health의 `mqtt`가 `connected`이면 기본
Direct 모드로 복귀한 상태다.

배포 전에는 다음 readiness 검사로 기본 Compose와 UAT override의 모드 계약을
확인한다. 이 검사는 파일만 읽으며 실행 중인 컨테이너나 장치 상태를 변경하지 않는다.

```bash
rtk python scripts/health/check_compose_runtime_assumptions.py
```

검사 기준은 기본 Jetson Compose의 `EDGEX_COMMAND_MODE`·
`EDGEX_ALLOWED_DEVICES`가 빈 기본값이고, UAT override만 `edgex`와
`signboard`를 지정하는 것이다.

## 5. 실패 및 rollback 기준

다음 중 하나라도 발생하면 해당 장치의 EdgeX 시험을 중단하고 Direct로
복귀한다.

- `device_unreachable`, `device_not_found`, `invalid_request` 결과 발생
- `request_id` 또는 `device_id`가 없는 결과 수신
- 물리 장치 출력이 없는데 결과가 성공으로 기록됨
- 카메라 ID가 레지스트리 `camera_ids`와 불일치함
- EdgeX 결과 DB와 Action DB의 상태가 서로 다름
- Docker/containerd 오류로 Action Layer health가 내려감

대상 장치가 레지스트리에 없을 때는 물리 ID 없는 공통 토픽으로 발행하지
않고 실패 처리한다. 이는 다른 장치가 잘못 동작하는 것을 막기 위한
안전장치다.

## 6. 변경 승인 기준

운영 기본값을 Direct에서 EdgeX로 변경하려면 다음 내용을 함께 기록한다.

1. 대상 장치와 카메라 ID
2. 시험 일시와 담당자
3. 성공·실패 결과 payload
4. 물리 장치 동작 증거
5. rollback 명령과 예상 영향
6. 스피커·사이렌처럼 아직 검증하지 않은 장치의 제외 사유

이 기록이 없는 상태에서는 Compose 기본값을 EdgeX로 변경하지 않는다.
