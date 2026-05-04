# CCTV 운영 점검 Runbook

## 결론

이 문서는 Docker Compose로 실행 중인 CCTV 플랫폼을 운영자가 빠르게 점검하고 복구하기 위한 절차입니다.

기본 확인 순서는 아래처럼 잡으면 됩니다.

```text
컨테이너 상태
  -> health endpoint
  -> smoke test
  -> 로그 확인
  -> 필요한 서비스만 재시작
```

## 현재 확인된 상태

2026-05-04 기준 로컬 Compose 상태 확인 결과, 핵심 서비스는 아래처럼 실행 중이었습니다.

```text
cctv-alert-api      Up, healthy, 127.0.0.1:8000->8000
cctv-action-layer   Up, healthy, 127.0.0.1:8080->8080
cctv-public-api     Up, healthy, 0.0.0.0:9000->9000
edgex-mqtt-broker   Up, healthy, 127.0.0.1:1883->1883
cctv-prometheus     Up, 127.0.0.1:9090->9090
cctv-grafana        Up, 127.0.0.1:3001->3000
```

주의:

- `cctv-public-api`만 호스트 외부 접근이 가능한 `0.0.0.0:9000`으로 열려 있습니다.
- `alert-api`, `action-layer`, `mqtt`, `prometheus`, `grafana`는 로컬 호스트에만 바인딩되어 있습니다.
- 운영 환경에서는 `PUBLIC_API_KEY`, `INTERNAL_SERVICE_TOKEN`, `GRAFANA_ADMIN_PASSWORD`를 반드시 실제 값으로 설정해야 합니다.

## 서비스 구성 요약

| 서비스 | 역할 | 포트 | Health |
|---|---|---:|---|
| `edgex-mqtt-broker` | MQTT broker | `127.0.0.1:1883` | Compose healthcheck |
| `cctv-alert-api` | 이벤트 JSONL 수신/저장 | `127.0.0.1:8000` | `/health` |
| `cctv-action-layer` | 알람/외부전송/수동승인 | `127.0.0.1:8080` | `/health` |
| `cctv-public-api` | 대시보드/서버팀 공개 API | `0.0.0.0:9000` | `/api/v1/health`, `/api/v1/readiness` |
| `cctv-prometheus` | 메트릭 수집 | `127.0.0.1:9090` | `/-/ready` |
| `cctv-grafana` | 모니터링 대시보드 | `127.0.0.1:3001` | `/api/health` |

브라우저 확인용 주소:

```text
http://127.0.0.1:9000/      Public API 안내
http://127.0.0.1:9000/docs  Public API Swagger
http://127.0.0.1:8000/      Alert API 안내
http://127.0.0.1:8080/      Action Layer 안내
```

## 빠른 상태 확인

일반 Docker 권한이 있는 환경:

```bash
docker compose ps
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

현재 개발 장비처럼 Docker socket 권한이 막혀 있으면 sudo로 확인합니다.

```bash
sudo docker compose ps
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

## Compose 실행 범위 선택

Public API, Alert API, Action Layer 중심으로 운영/개발 확인만 할 때는 전체 스택보다 필요한 서비스만 올리는 편이 안전합니다.

```bash
docker compose up -d cctv-alert-api cctv-action-layer cctv-public-api prometheus grafana edgex-mqtt-broker
```

전체 EdgeX 스택까지 올릴 때:

```bash
docker compose up -d
```

arm64/Jetson 계열 호스트에서 기본 compose 전체 스택을 올릴 때:

```bash
docker compose -f docker-compose.yml -f docker-compose.arm64.yml up -d
```

주의:

- arm64/Jetson 계열 호스트에서 `docker-compose.yml`의 일부 EdgeX 이미지는 `linux/amd64`일 수 있습니다.
- 이 경우 `exec format error`로 `core-data`, `core-metadata`, `device-rest`, `ui`가 재시작 루프에 들어갈 수 있습니다.
- 기본 compose 전체 스택을 arm64에서 실행하려면 `docker-compose.arm64.yml` override를 함께 적용합니다.
- EdgeX UI 이미지는 이 환경에서 ARM64 manifest가 확인되지 않아 `docker-compose.arm64.yml` 기본 실행에서 제외됩니다.
- AIoT parser는 PostgreSQL 설정이 필요합니다. 기본 compose에서는 `aiot-parser-db` 서비스와 전용 volume을 사용합니다.
- Jetson 현장 배포는 가능하면 `docker-compose.jetson.yml` 기준으로 확인합니다.

## Health endpoint 확인

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:9000/api/v1/health
curl -fsS http://localhost:9000/api/v1/readiness
curl -fsS http://localhost:9090/-/ready
curl -fsS http://localhost:3001/api/health
```

API key가 설정된 운영 환경에서는 Public API 호출에 헤더를 붙입니다.

```bash
curl -fsS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/health
```

운영 배포 확인에는 하위 서비스 연결까지 보는 readiness를 권장합니다.

```bash
curl -fsS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/readiness
```

## 자동 smoke test

로컬 서비스가 떠 있는 상태에서 아래 스크립트를 실행합니다.

```bash
.venv/bin/python scripts/smoke_test_deployment.py
.venv/bin/python scripts/smoke_test_data_flow.py
```

기대 결과:

```text
"passed": true
```

`smoke_test_deployment.py`는 health/readiness 중심 점검입니다.

- alert api health
- action layer health
- public api health
- public api readiness
- prometheus readiness
- grafana health
- prometheus scrape targets

`smoke_test_data_flow.py`는 실제 데이터 흐름에 가까운 점검입니다.

- alert api alert POST
- alert api sensor reading POST
- action layer event POST
- action layer metrics
- public api metrics

## 로그 확인

전체 로그:

```bash
docker compose logs --tail 120
```

주요 서비스별 로그:

```bash
docker compose logs --tail 120 cctv-public-api
docker compose logs --tail 120 cctv-action-layer
docker compose logs --tail 120 cctv-alert-api
docker compose logs --tail 120 edgex-mqtt-broker
docker compose logs --tail 120 prometheus
docker compose logs --tail 120 grafana
```

실시간 추적:

```bash
docker compose logs -f cctv-public-api
```

## 재시작 절차

### Public API만 재시작

대시보드 API만 이상할 때 우선 시도합니다.

```bash
docker compose restart cctv-public-api
docker compose ps cctv-public-api
curl -fsS http://localhost:9000/api/v1/health
curl -fsS http://localhost:9000/api/v1/readiness
```

### Action Layer만 재시작

사이트/제어/수동승인/알람 동작이 이상할 때 시도합니다.

```bash
docker compose restart cctv-action-layer
docker compose ps cctv-action-layer
curl -fsS http://localhost:8080/health
```

### Alert API만 재시작

이벤트 로그 저장이나 `/api/alerts` 수신이 이상할 때 시도합니다.

```bash
docker compose restart cctv-alert-api
docker compose ps cctv-alert-api
curl -fsS http://localhost:8000/health
```

### MQTT broker 재시작

MQTT broker 재시작은 영향 범위가 큽니다. AI 이벤트, Kuiper, Action Layer 구독 흐름이 일시적으로 끊길 수 있습니다.

```bash
docker compose restart edgex-mqtt-broker
docker compose ps edgex-mqtt-broker
```

재시작 후에는 Action Layer와 AI 관련 서비스 로그를 같이 확인합니다.

```bash
docker compose logs --tail 120 cctv-action-layer
docker compose logs --tail 120 cctv-ai-engine
```

## 장애별 확인 순서

### Public API가 401/403을 반환함

원인 후보:

- `PUBLIC_API_KEY`가 서버에 설정되어 있는데 클라이언트가 `X-API-Key`를 보내지 않음
- 클라이언트 key가 서버 key와 다름
- 임시 쿼리 인증을 기대하지만 `PUBLIC_API_ALLOW_QUERY_KEY=1`이 설정되지 않음

확인 방법:

```bash
docker compose exec cctv-public-api env | grep PUBLIC_API
curl -i http://localhost:9000/api/v1/health
curl -i -H "X-API-Key: ${PUBLIC_API_KEY}" http://localhost:9000/api/v1/health
```

수정 방법:

- 운영에서는 클라이언트에 `X-API-Key` 헤더를 추가합니다.
- 서버의 `PUBLIC_API_KEY`는 `.env` 또는 배포 secret으로 관리합니다.

### `/api/v1/sites` 또는 `/api/v1/control/*`이 502/504를 반환함

원인 후보:

- `cctv-action-layer`가 내려감
- `ACTION_LAYER_URL` 설정 오류
- Action Layer health는 살아 있지만 내부 REST 응답 지연

확인 방법:

```bash
docker compose ps cctv-action-layer
curl -i http://localhost:8080/health
docker compose logs --tail 120 cctv-action-layer
docker compose exec cctv-public-api env | grep ACTION_LAYER_URL
curl -i http://localhost:9000/api/v1/readiness
```

수정 방법:

```bash
docker compose restart cctv-action-layer
docker compose restart cctv-public-api
```

### `/api/v1/events`가 비어 있음

원인 후보:

- 아직 alert-api에 이벤트가 들어오지 않음
- `ALERT_LOG_PATH`가 Public API와 Alert API 사이에서 공유되지 않음
- Alert API는 받았지만 다른 로그 파일에 저장 중

확인 방법:

```bash
docker compose logs --tail 120 cctv-alert-api
docker compose exec cctv-public-api env | grep ALERT_LOG_PATH
docker compose exec cctv-alert-api sh -lc 'ls -lh /app/logs && tail -n 5 /app/logs/alert_api_events.jsonl'
```

수정 방법:

- `alert-logs` volume이 두 서비스에 모두 연결되어 있는지 확인합니다.
- `ALERT_LOG_PATH`가 `/app/logs/alert_api_events.jsonl`인지 확인합니다.

### 브라우저에서 `{"error":"not found"}`가 보임

원인 후보:

- Alert API `8000`번에서 정의되지 않은 경로를 열었음
- Public API인데 `/api/v1` prefix를 빼고 호출했음
- 오래된 컨테이너에서 `POST` 전용 API를 브라우저 `GET`으로 열었음

최신 코드에서는 `GET /api/alerts`처럼 경로는 맞지만 method가 틀린 경우 404 대신
`405 method not allowed`와 `allowed=POST` 안내를 반환합니다.

확인 방법:

```bash
curl -i http://localhost:8000/
curl -i http://localhost:8000/health
curl -i http://localhost:9000/
curl -i http://localhost:9000/api/v1/health
```

수정 방법:

- Public API 문서는 `http://localhost:9000/docs`를 엽니다.
- Public API health는 `http://localhost:9000/api/v1/health`를 사용합니다.
- Alert API health는 `http://localhost:8000/health`를 사용합니다.
- Alert API 이벤트 수신은 브라우저가 아니라 `POST /api/alerts`로 호출합니다.

### 외형 검색 결과가 비어 있음

원인 후보:

- AI 엔진에서 `appearance` 감지가 활성화되지 않음
- `APPEARANCES_DB`가 Public API와 AI 엔진 사이에서 공유되지 않음
- 아직 저장된 appearance log가 없음

확인 방법:

```bash
curl -sS -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/appearances/status

docker compose exec cctv-public-api env | grep APPEARANCE
```

수정 방법:

- 카메라 설정의 `detections`에 `appearance`가 포함되어 있는지 확인합니다.
- compose volume에서 `/app/data/appearances.db`가 유지되는지 확인합니다.

### 이벤트는 감지되지만 스피커/전광판/경광등이 동작하지 않음

원인 후보:

- 현장 디바이스 전원이 꺼져 있거나 장비가 부팅 중입니다.
- 디바이스 IP가 바뀌었거나 CCTV 서버와 같은 네트워크에 없습니다.
- 방화벽, 스위치, VLAN, 포트 설정 문제로 장비 포트에 접근할 수 없습니다.
- `SPEAKER_*`, `SIREN_*`, `SIGNBOARD_*` 환경변수가 비어 있어 Action Layer가 장비를 비활성화했습니다.
- Action Layer의 알람 쿨다운 때문에 같은 카메라/이벤트가 잠시 스킵되었습니다.

확인 방법:

```bash
.venv/bin/python scripts/check_alarm_devices.py
docker compose logs --tail 120 cctv-ai-engine
docker compose logs --tail 120 cctv-action-layer
docker compose exec cctv-action-layer env | grep -E 'SPEAKER|SIREN|SIGNBOARD'
```

Docker socket 권한이 막혀 있으면 `sudo`로 확인합니다.

```bash
sudo docker compose logs --tail 120 cctv-action-layer
sudo docker compose exec cctv-action-layer env | grep -E 'SPEAKER|SIREN|SIGNBOARD'
```

장비 네트워크 연결까지 보지 않고 설정 누락만 먼저 확인하려면:

```bash
.venv/bin/python scripts/check_alarm_devices.py --skip-network
```

장비 전원과 네트워크를 장비별로 확인합니다.

```bash
ping <스피커_IP>
ping <경광등_IP>
ping <전광판_IP>

nc -vz <스피커_IP> 80
nc -vz <경광등_IP> 80
nc -vz <전광판_IP> 5000
```

스피커/경광등은 InterM HTTP 장비이므로 80번 포트가 열려 있어야 합니다.
전광판은 Dabit TCP 장비이므로 기본 5000번 포트가 열려 있어야 합니다.

수정 방법:

- 디바이스 전원, LAN 케이블, PoE/어댑터, 장비 IP를 먼저 확인합니다.
- `.env` 또는 compose 환경변수에 실제 장비 값을 넣습니다.

```bash
SPEAKER_HOST=<스피커_IP>
SPEAKER_USER=<스피커_계정>
SPEAKER_PASSWORD=<스피커_비밀번호>

SIREN_HOST=<경광등_IP>
SIREN_USER=<경광등_계정>
SIREN_PASSWORD=<경광등_비밀번호>

SIGNBOARD_HOST=<전광판_IP>
SIGNBOARD_PORT=5000
```

환경변수를 바꾼 뒤 Action Layer를 재시작합니다.

```bash
docker compose restart cctv-action-layer
docker compose logs --tail 120 cctv-action-layer
curl -fsS http://localhost:8080/health
```

정상 로그 기준:

```text
Action Layer MQTT 연결 성공
구독: cctv/ai/events/+/...
```

비정상 로그 예:

```text
[Speaker] host/username/password 미설정 - 스피커 비활성화
[Siren] host/username/password 미설정 - 경광등 비활성화
[Signboard] host 미설정 - 비활성화
```

주의:

- 스피커/경광등/전광판이 꺼져 있어도 AI 이벤트 감지와 MQTT 전송은 정상일 수 있습니다.
- 알람 장비 문제는 `cctv-ai-engine`보다 `cctv-action-layer` 로그를 우선 봅니다.
- 같은 이벤트가 짧은 시간 반복되면 `ACTION_ALARM_COOLDOWN` 때문에 장비 출력이 스킵될 수 있습니다.

### Prometheus/Grafana는 뜨지만 데이터가 없음

원인 후보:

- Prometheus target scrape 실패
- Public API 또는 Action Layer metrics endpoint 오류
- Grafana datasource provisioning 문제

확인 방법:

```bash
curl -fsS http://localhost:9090/-/ready
curl -fsS http://localhost:9090/api/v1/targets
curl -fsS http://localhost:9000/api/v1/metrics | head
curl -fsS http://localhost:8080/metrics | head
docker compose logs --tail 120 prometheus
docker compose logs --tail 120 grafana
```

수정 방법:

- `monitoring/prometheus.yml`의 target 주소가 compose service name 기준인지 확인합니다.
- Prometheus 재시작 후 target 상태를 다시 확인합니다.

```bash
docker compose restart prometheus
```

### EdgeX 서비스가 `exec format error`로 재시작함

원인 후보:

- 현재 호스트는 arm64인데, 실행한 EdgeX 이미지가 amd64 전용입니다.
- 이전 다른 compose 프로젝트의 중지 컨테이너가 같은 `container_name`을 점유하고 있습니다.

확인 방법:

```bash
docker compose ps
docker logs --tail 80 edgex-core-data
docker inspect edgex-core-data --format '{{json .Config.Image}}'
docker ps -a --filter label=com.docker.compose.project=edgex-jetson
```

수정 방법:

- Public API/Action/Alert 검증만 필요하면 실패 중인 EdgeX 부가 서비스를 중지합니다.

```bash
docker compose stop core-data core-metadata device-rest ui
```

- 같은 arm64 호스트에서 기본 compose 전체 스택을 계속 써야 하면 ARM64 override를 함께 적용합니다.

```bash
docker compose -f docker-compose.yml -f docker-compose.arm64.yml up -d
```

- EdgeX UI가 필요하면 ARM64 장비에서 직접 띄우기보다 x86_64 서버/PC에서 UI를 실행하거나, EdgeX REST API와 Grafana를 우선 사용합니다.
- Jetson/arm64 운영은 `docker-compose.jetson.yml` 사용을 우선 검토합니다.
- 이전 compose 프로젝트의 중지 컨테이너가 이름을 점유한다면, 해당 컨테이너가 실행 중이 아닌지 확인한 뒤 제거합니다.

### AIoT parser가 PostgreSQL/Outbox 오류로 재시작함

원인 후보:

- `aiot-parser-db`가 떠 있지 않거나 healthcheck가 실패했습니다.
- `parser-python/.env`의 DB 설정이 compose override와 다르게 직접 실행되고 있습니다.
- 컨테이너 안에서 `localhost`는 호스트가 아니라 자기 자신입니다.
- `/data/event_outbox.db`가 bind mount 권한 문제로 쓰기 불가입니다.

확인 방법:

```bash
docker logs --tail 120 aiot-parser
docker compose ps aiot-parser aiot-parser-db
```

수정 방법:

- 기본 compose에서는 `aiot-parser-db`와 `aiot-parser`를 함께 올립니다.

```bash
docker compose up -d aiot-parser-db aiot-parser
```

- outbox 쓰기 권한 문제를 피하기 위해 `/data`는 `aiot-parser-data` named volume을 사용합니다.
- 외부 PostgreSQL을 쓰려면 `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`를 compose 네트워크 기준으로 맞춥니다.

## 배포 전 체크리스트

배포 전에는 아래를 한 번에 확인합니다.

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/check_compose_runtime_assumptions.py --json
.venv/bin/python scripts/check_deployment_readiness.py
.venv/bin/python scripts/check_alarm_devices.py --skip-network
.venv/bin/python scripts/check_sensitive_defaults.py
.venv/bin/python scripts/check_dockerfile_sources.py
.venv/bin/python scripts/smoke_test_deployment.py
.venv/bin/python scripts/smoke_test_data_flow.py
```

기준:

- 전체 테스트가 통과해야 합니다.
- `check_compose_runtime_assumptions.py`가 실패하면 full compose 실행 전에 호스트 아키텍처, EdgeX 이미지, AIoT parser DB 설정을 먼저 맞춥니다.
- `check_sensitive_defaults.py`에서 민감 기본값이 없어야 합니다.
- smoke test 두 개가 모두 `"passed": true`여야 합니다.

## 관련 문서

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md): 전체 구조와 데이터 흐름
- [EVENT_SCHEMA_STANDARD.md](EVENT_SCHEMA_STANDARD.md): AI/센서/디바이스 이벤트 표준 스키마
- [PUBLIC_API_GUIDE.md](PUBLIC_API_GUIDE.md): Public API 사용 가이드
- [PUBLIC_API_EXAMPLES.md](PUBLIC_API_EXAMPLES.md): Public API 복붙용 샘플
- [JETSON_EDGEX_FIELD_CHECKLIST.md](JETSON_EDGEX_FIELD_CHECKLIST.md): Jetson/EdgeX 현장 점검
- [MLOPS_MODEL_EVALUATION.md](MLOPS_MODEL_EVALUATION.md): 모델 교체 전 평가 절차
