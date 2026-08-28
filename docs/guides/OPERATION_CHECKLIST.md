# CCTV 운영 체크리스트

## 결론

운영/데모 실행 전에는 아래 순서로 확인합니다.

```text
설정 확인 -> 컨테이너 상태 확인 -> health 확인 -> smoke test -> DeepStream 안정성 관찰 -> 로그 보관
```

## 1. 실행 전 설정

- 기본 스택은 `.env`, Jetson 통합 스택은 `.env.jetson`이 존재하는지 확인
- `PUBLIC_API_KEY`, `INTERNAL_SERVICE_TOKEN`, `MQTT_USER`, `MQTT_PASSWORD`, `AIOT_DB_PASSWORD`가 운영값인지 확인
- `known_faces.json`이 존재하고 JSON 형식이 깨지지 않았는지 확인
- Compose nginx 데모 설정은 `config/nginx/public-demo.conf.template` 기준으로 확인
- `/stream-api/`는 활성 프록시이며, `/hls/`와 과거 정적 미디어 asset 경로만 `410`을 반환
- 변수 기준표는 `docs/guides/DEPLOYMENT_ENVIRONMENT_VARIABLES.md`를 따른다

## 2. 컨테이너 상태

Docker 권한이 없는 계정에서는 같은 터미널에서 먼저 sudo 인증을 열어둡니다.

이미 사용자가 docker 그룹에 속하지만 현재 셸이 변경을 반영하지 못한 경우에는
재로그인 전까지 아래 래퍼를 사용합니다.

```bash
./scripts/ops/with_docker_group.sh docker compose ps
./scripts/ops/with_docker_group.sh docker inspect cctv-ai-engine --format 'Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} RestartCount={{.RestartCount}} Runtime={{.HostConfig.Runtime}}'
```

```bash
sudo -v
```

그 다음 상태를 확인합니다.

```bash
docker compose ps
sudo docker inspect cctv-ai-engine --format 'Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} RestartCount={{.RestartCount}} Runtime={{.HostConfig.Runtime}}'
```

통과 기준:

- 핵심 컨테이너가 `Up` 또는 `running`
- `cctv-ai-engine` health가 `healthy`
- `RestartCount=0`
- Jetson/DeepStream 환경에서는 runtime이 `nvidia`

## 3. Health 확인

```bash
curl -fsS http://localhost:9000/api/v1/health
curl -fsS http://localhost:8765/health
curl -fsS http://localhost:8766/health
curl -fsS http://localhost:8767/health
curl -fsS http://localhost:8769/health
```

API key가 필요한 운영 환경에서는 Public API 호출에 헤더를 붙입니다.

```bash
curl -fsS -H "X-API-Key: ${PUBLIC_API_KEY}" http://localhost:9000/api/v1/health
```

## 4. 표준 운영 점검

DeepStream 장시간 관찰 전에는 wrapper로 기본 점검을 먼저 실행합니다.

```bash
./scripts/ops/run_operation_check.sh
RUNTIME_ENV_FILE=.env.jetson ./scripts/ops/run_operation_check.sh
./scripts/ops/with_docker_group.sh ./scripts/ops/run_operation_check.sh
RUNTIME_ENV_FILE=.env.jetson ./scripts/ops/with_docker_group.sh ./scripts/ops/run_operation_check.sh
```

통과 기준:

- `deployment smoke` PASS
- `data flow smoke` PASS
- `public api fd stability` PASS
- 최종 `result: PASS`

개별 데이터 흐름만 따로 확인할 때는 아래 명령을 사용합니다.

```bash
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
```

통과 기준:

- `passed=true`
- alert api 수신 통과
- action layer event 수신 통과
- public api metrics 확인 통과

Public API FD 누적 여부만 따로 확인할 때는 아래 명령을 사용합니다.

```bash
.venv/bin/python scripts/health/check_public_api_fd_stability.py
```

통과 기준:

- readiness 반복 호출이 모두 `ready`
- 반복 호출 전후 FD 증가량이 기본 허용치 `32` 이하
- FD 최대값이 readiness가 보고한 `soft_limit` 이하
- `/api/v1/metrics`의 `cctv_public_api_open_file_descriptors`가 장시간 계속 증가하지 않음

## 5. DeepStream 안정성 관찰

짧은 확인:

```bash
sudo -v
./scripts/ops/run_deepstream_stability_watch.sh 30 30
```

운영/데모 전 권장 확인:

```bash
sudo -v
./scripts/ops/run_deepstream_stability_watch.sh 720 60
```

결과 위치:

```text
reports/deepstream-stability/*.log
reports/deepstream-stability/*.summary
```

통과 기준:

- summary의 `result=pass`
- `fail=0`
- `failure_rate=0.0%`
- DeepStream 컨테이너 `RestartCount=0`

## 6. 실패 시 우선 확인

Public API readiness가 `503`이고 로그에 `Too many open files`가 보이면 우선 Public API만 재시작해서 회복 여부를 확인합니다.

```bash
sudo docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build --force-recreate cctv-public-api
RUNTIME_ENV_FILE=.env.jetson ./scripts/ops/run_operation_check.sh
```

그 외에는 아래 순서로 확인합니다.

1. `docker inspect`로 컨테이너 health와 재시작 횟수 확인
2. `docker stats`로 메모리/CPU 급증 확인
3. `sudo docker logs --tail 160 cctv-ai-engine`로 DeepStream 로그 확인
4. `sudo docker logs --tail 160 cctv-action-layer`로 이벤트 처리 timeout 확인
5. `scripts/smoke/smoke_test_data_flow.py`를 단독 재실행해서 주변 데이터 경로만 분리 확인

## 7. 기록 보관

- 안정성 관찰 로그와 summary는 `reports/deepstream-stability/`에 보관
- 장애가 있었으면 실패 샘플 번호, 실패 항목, 해당 시각의 컨테이너 로그를 함께 남김
- 최종 판단은 `docs/guides/DEEPSTREAM_PERFORMANCE_STABILITY_2026-05-26.md`의 표준 안정성 통과 기준을 따름

런타임 데이터 정리 대상은 먼저 미리보기로 확인합니다.

```bash
./scripts/cleanup/cleanup_runtime_data.sh
```

기본 정책은 7일이 지난 외형 crop 삭제, 삭제된 crop의 DB 참조 정리,
7일이 지난 `sent` 상태의 HTTP/MQTT outbox 행 정리, 200MB를 넘은
이벤트/센서 JSONL 로그 회전입니다. outbox는 DB별 1회 최대 25,000건만
삭제하며 `pending` 행과 `action_events.db` 운영 이력은 삭제하지 않습니다.
확인 후 실제로 반영할 때만 `--apply`를 사용합니다.
컨테이너가 생성한 crop은 `nobody` 소유일 수 있으므로 운영 장비에서는 `sudo`로 실행합니다.

```bash
sudo ./scripts/cleanup/cleanup_runtime_data.sh --apply
```

SQLite 행 삭제 후 파일 크기는 즉시 줄지 않을 수 있지만 빈 페이지를 이후
쓰기에서 재사용하므로 지속적인 증가를 억제합니다. 디스크 파일 자체를 줄이는
`VACUUM`은 서비스 정지와 충분한 여유 공간을 확보한 별도 유지보수 시간에만
수행합니다.

Docker socket 권한이 제한된 장비에서는 표준 운영 점검 wrapper도 `sudo`로 실행합니다.

```bash
sudo ./scripts/ops/run_operation_check.sh
sudo env RUNTIME_ENV_FILE=.env.jetson ./scripts/ops/run_operation_check.sh
```

운영 장비에서는 아래 명령으로 매일 실행되는 systemd 타이머를 설치합니다.
기본 실행 시각은 한국시간 기준 매일 09:00 정각입니다.

```bash
./scripts/ops/install_runtime_cleanup_timer.sh --dry-run
sudo ./scripts/ops/install_runtime_cleanup_timer.sh
systemctl list-timers --all cctv-runtime-cleanup.timer
```
