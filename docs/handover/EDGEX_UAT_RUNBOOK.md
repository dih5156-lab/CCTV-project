# EdgeX 운영 전환 UAT 런북

## 목적

운영 기본 Direct 경로를 영구 변경하지 않고, 전광판 1대를 대상으로 Action Layer부터 EdgeX Device Service까지의 실제 경로를 검증한다.

## 사전 확인

- 전광판 현장 제어가 가능한 시간인지 확인한다.
- `cctv-device-signboard`가 health 상태인지 확인한다.
- 스피커·사이렌은 이 절차의 대상이 아니다.
- 테스트 문구가 실제 전광판에 표시된다.

```bash
rtk docker exec cctv-device-signboard curl -fsS http://localhost:59993/health
rtk docker exec cctv-action-layer curl -fsS http://localhost:8080/health
```

## EdgeX UAT 프로파일 적용

기본 Compose 파일은 수정하지 않고 UAT override를 함께 사용한다.

```bash
rtk docker compose -f docker-compose.jetson.yml -f docker-compose.edgex-uat.yml up -d --build --no-deps cctv-action-layer
```

적용 후 다음 값을 확인한다.

```bash
rtk docker exec cctv-action-layer sh -lc 'for key in EDGEX_COMMAND_MODE EDGEX_ALLOWED_DEVICES EDGEX_DEVICE_REGISTRY_PATH; do printf "%s=" "$key"; printenv "$key"; done'
```

기대값:

```text
EDGEX_COMMAND_MODE=edgex
EDGEX_ALLOWED_DEVICES=signboard
EDGEX_DEVICE_REGISTRY_PATH=/app/config/output_devices.json
```

## 이벤트 시험

실제 이벤트 입력은 Action Layer의 REST 또는 MQTT 시험 계약을 사용한다. 이벤트에는 고유한 `event_id`를 넣고 전광판 테스트 문구를 지정한다.

```text
입력 이벤트
  → ActionPolicy
  → DeviceResolver(signboard만 허용)
  → EdgeX Command MQTT
  → cctv-device-signboard
  → 전광판 출력
  → EdgeX 결과 MQTT
  → Action Layer command_id 갱신
```

확인할 값:

- 명령 토픽: `edgex/commands/cctv/jetson-01/signboard/...`
- 결과 토픽: `edgex/results/cctv/jetson-01/signboard`
- 결과 상태: `acknowledged` 또는 명확한 `failed`
- 결과 필드: `request_id`, `event_id`, `device_id`
- Action Layer 논리 명령: `event_id:signboard`

실제 낙상 이벤트 경로를 반복해서 확인할 때는 다음 도구를 사용한다.
기본 실행은 미리보기만 수행하며, 실제 전송은 `--send`를 명시해야 한다.

```bash
rtk python scripts/ops/run_action_layer_fall_trace.py \
  --event-id fall-trace-YYYYMMDD-001 \
  --request-id fall-trace-YYYYMMDD-001:input

rtk python scripts/ops/run_action_layer_fall_trace.py \
  --event-id fall-trace-YYYYMMDD-001 \
  --request-id fall-trace-YYYYMMDD-001:input \
  --confirm-physical-control \
  --send
```

`--send` 실행 전에는 전광판·스피커 등 현재 연결된 출력 장치와 운영 모드를
확인한다. 도구는 토큰을 출력하지 않고 `event_id`, `request_id`, 명령 이력만 반환한다.

현재처럼 일부 장치만 연결된 현장에서는 장치 선택 옵션으로 해당 장치만
검증한다. 예를 들어 전광판만 연결된 경우 다음 명령을 사용한다.

```bash
rtk python scripts/ops/run_edgex_device_service_uat.py \
  --device signboard --mode dry-run --json
```

`--device`를 생략하면 기존처럼 스피커·사이렌·전광판을 모두 점검한다.
`--mode dry-run`은 물리 장치 명령을 보내지 않고 Device Service의 health만
확인한다. 실제 명령 전송은 현장 승인 후 `--mode real
--confirm-physical-control`을 함께 지정한 경우에만 수행한다.

## 결과 확인

```bash
rtk docker logs --since 2m cctv-device-signboard
rtk docker exec cctv-edgex-command-result python -c 'import sqlite3; c=sqlite3.connect("/app/data/runtime/edgex_command_results.db"); print(c.execute("select request_id,event_id,device_id,status from command_results order by received_at desc limit 5").fetchall())'
```

전광판 로그의 `[Signboard] 표시`와 `방송 시작 완료`는 장치 호출 로그다. 별도 육안 사진이 없으면 물리 출력 증빙은 로그 기준이라고 명시한다.

## Direct 복귀

UAT가 끝나면 override를 사용하지 않고 기본 Compose 파일만 적용해 Direct 기본값으로 복귀한다.

```bash
rtk docker compose -f docker-compose.jetson.yml up -d --no-deps cctv-action-layer
```

복귀 확인:

```bash
rtk docker exec cctv-action-layer sh -lc 'printf "EDGEX_COMMAND_MODE="; printenv EDGEX_COMMAND_MODE; curl -fsS http://localhost:8080/health'
```

`EDGEX_COMMAND_MODE`가 비어 있으면 현재 러너의 호환 기본값에 따라 Direct가 선택된다. 명시적으로 확인할 필요가 있으면 `.env.jetson`의 운영 설정을 확인하되, UAT override 파일을 운영 기본 파일로 복사하지 않는다.

## 판정 기준

| 항목 | PASS 기준 |
|---|---|
| Action Layer | health 정상, EdgeX 모드 확인 |
| 장치 선택 | signboard만 선택 |
| EdgeX 발행 | Command 토픽 발행 성공 |
| 물리 출력 | 전광판 표시 로그 또는 현장 사진 |
| 결과 수신 | `request_id`·`event_id`·`device_id` 포함 |
| 상태 연결 | `event_id:signboard`가 `acknowledged` 또는 원인 있는 `failed` |

## 주의사항

- 이 프로파일은 실제 전광판을 제어한다.
- 스피커와 사이렌을 허용 목록에 추가하지 않는다.
- `--no-deps`를 사용해 의존 서비스의 불필요한 재생성을 막는다.
- UAT 성공은 운영 기본값 전환 승인과 동일하지 않다. 승인 후에만 `.env.jetson`의 명시적 운영 설정을 변경한다.
