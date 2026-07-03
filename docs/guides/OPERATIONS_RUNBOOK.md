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

## 과거 확인 기록과 현재 확인 방법

아래 출력은 2026-05-26 당시 로컬 Compose 확인 기록이며 현재 실행 상태를 보장하지 않습니다. 현재 상태는 반드시 `docker compose ps`와 health endpoint로 다시 확인합니다.

```text
cctv-alert-api      Up, healthy, 127.0.0.1:8000->8000
cctv-action-layer   Up, healthy, 127.0.0.1:8080->8080
cctv-public-api     Up, healthy, 0.0.0.0:9000->9000
edgex-mqtt-broker   Up, healthy, 127.0.0.1:1883->1883
```

> **Prometheus / Grafana**는 `docker-compose.yml`의 `monitoring` profile로 분리되어 있습니다. 필요할 때만 별도로 실행합니다 (아래 "모니터링 옵션" 섹션 참조).

주의:

- `cctv-public-api`만 호스트 외부 접근이 가능한 `0.0.0.0:9000`으로 열려 있습니다.
- `alert-api`, `action-layer`, `mqtt`는 로컬 호스트에만 바인딩되어 있습니다.
- 운영 환경에서는 `PUBLIC_API_KEY`, `INTERNAL_SERVICE_TOKEN`, `GRAFANA_ADMIN_PASSWORD`를 반드시 실제 값으로 설정해야 합니다.
- MQTT 브로커는 익명 접속을 허용하지 않습니다. 실행 전 `mosquitto/passwd`를 생성하고 `MQTT_USER`, `MQTT_PASSWORD`를 설정해야 합니다.

## 서비스 구성 요약

| 서비스 | 역할 | 포트 | Health |
|---|---|---:|---|
| `edgex-mqtt-broker` | MQTT broker | `127.0.0.1:1883` | Compose healthcheck |
| `cctv-alert-api` | 이벤트 JSONL 수신/저장 | `127.0.0.1:8000` | `/health` |
| `cctv-action-layer` | 알람/외부전송/수동승인 | `127.0.0.1:8080` | `/health` |
| `cctv-public-api` | 대시보드/서버팀 공개 API | `0.0.0.0:9000` | `/api/v1/health`, `/api/v1/readiness` |
| `cctv-ai-engine` | 영상 입력, AI 추론, 이벤트 발행 | 구성에 따라 `8769` | Stream API `/health` |
| `cctv-prometheus` ⚙️ | 메트릭 수집 (옵션) | `127.0.0.1:9090` | `/-/ready` |
| `cctv-grafana` ⚙️ | 모니터링 대시보드 (옵션) | `127.0.0.1:3001` | `/api/health` |

⚙️ 옵션 서비스 — `docker-compose.yml`의 `monitoring` profile로 분리됨

브라우저 확인용 주소:

```text
http://127.0.0.1:9000/      Public API 안내
http://127.0.0.1:9000/docs  Public API Swagger
http://127.0.0.1:8000/      Alert API 안내
http://127.0.0.1:8080/      Action Layer 안내
http://127.0.0.1:8769/health Stream API 상태와 현재 송출 설정
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
docker compose up -d cctv-alert-api cctv-action-layer cctv-public-api edgex-mqtt-broker
```

### 모니터링 옵션 (Prometheus + Grafana)

평상시엔 끄고, 필요할 때만 켭니다.

```bash
# 메인 스택과 함께 올리기
docker compose --profile monitoring up -d prometheus grafana

# 모니터링 컨테이너만 중지 (메인 스택과 볼륨 유지)
docker compose stop prometheus grafana
```

전체 EdgeX 스택까지 올릴 때:

```bash
docker run --rm -v "$PWD/mosquitto:/mosquitto/config" eclipse-mosquitto:2.0 \
  mosquitto_passwd -c /mosquitto/config/passwd "${MQTT_USER}"
docker compose up -d
```

arm64/Jetson 계열 호스트에서 전체 스택을 올릴 때:

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d
```

주의:

- arm64/Jetson 계열 호스트에서 `docker-compose.yml`의 일부 EdgeX 이미지는 `linux/amd64`일 수 있습니다.
- 이 경우 `exec format error`로 `core-data`, `core-metadata`, `device-rest`, `ui`가 재시작 루프에 들어갈 수 있습니다.
- Jetson 현장 배포는 `docker-compose.jetson.yml` 기준으로 확인합니다.
- Jetson compose의 external volume은 첫 실행 전에 생성해야 합니다. 목록과 생성 명령은 [배포 환경변수 문서](DEPLOYMENT_ENVIRONMENT_VARIABLES.md)를 참고합니다.
- AIoT parser는 PostgreSQL 설정이 필요합니다. 기본 compose에서는 `aiot-parser-db` 서비스와 전용 volume을 사용합니다.

## Health endpoint 확인

핵심 서비스:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:9000/api/v1/health
curl -fsS http://localhost:9000/api/v1/readiness
curl -fsS http://localhost:8769/health
```

모니터링 서비스 (옵션, 실행 중일 때만):

```bash
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
.venv/bin/python scripts/smoke/smoke_test_deployment.py
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
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
- prometheus readiness (모니터링 스택 실행 중일 때만 PASS)
- grafana health (모니터링 스택 실행 중일 때만 PASS)
- prometheus scrape targets (모니터링 스택 실행 중일 때만 PASS)

> 모니터링 스택을 끈 상태에서 실행하면 prometheus/grafana 항목은 FAIL로 표시됩니다. 데이터 흐름 자체는 `smoke_test_data_flow.py`로 별도 확인하세요.

반복 안정성 확인 (30분~1시간):

```bash
bash scripts/smoke/run_smoke_loop.sh 60 30   # 60분, 30초 간격
```

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
# 모니터링 스택 실행 중일 때:
docker compose --profile monitoring logs --tail 120 prometheus
docker compose --profile monitoring logs --tail 120 grafana
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

- `.env`에 키가 없으면 `./scripts/ops/ensure_public_api_key.sh .env`로 생성합니다.
- 데모 UI(`/public-api/*`)는 `public-demo-ui` nginx가 `X-API-Key`를 서버 쪽에서 주입합니다.
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
docker compose exec cctv-alert-api sh -lc 'ls -lh /app/data/logs && tail -n 5 /app/data/logs/alert_api_events.jsonl'
```

수정 방법:

- `./data` 볼륨이 두 서비스에 모두 연결되어 있는지 확인합니다.
- `ALERT_LOG_PATH`가 `/app/data/logs/alert_api_events.jsonl`인지 확인합니다.

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
- compose volume에서 `/app/data/runtime/appearances.db`가 유지되는지 확인합니다.

### Stream API 화면이 끊기거나 CPU 사용량이 높음

원인 후보:

- `STREAM_FPS`, 해상도, JPEG 품질이 장비 성능이나 동시 접속 수보다 높음
- H.264 POC 보정이 Python buffer handoff를 사용해 추가 복사를 발생시킴
- DeepStream preview frame이 아직 준비되지 않았거나 source가 재연결 중임

확인 방법:

```bash
curl -fsS http://localhost:8769/health
docker compose logs --tail 200 cctv-ai-engine
docker compose exec cctv-ai-engine env | grep -E 'STREAM_|DS_H264_|DS_PREVIEW_'
```

수정 방법은 MJPEG 부하부터 단계적으로 낮춥니다.

```bash
STREAM_FPS=10
STREAM_WIDTH=640
STREAM_HEIGHT=360
STREAM_JPEG_QUALITY=60
```

환경변수를 바꾼 뒤에는 컨테이너를 재생성해야 합니다.

```bash
docker compose up -d --force-recreate cctv-ai-engine
```

POC 보정이 필요 없는 스트림이면 `DS_H264_POC_FIX_ENABLED=false`를 별도로 검증할 수 있습니다. 이 값은 H.264/WebRTC 호환성에 영향을 줄 수 있으므로 MJPEG 튜닝과 한 번에 바꾸지 않습니다.

### DeepStream 카메라 source가 반복 재시작됨

현재 source 장애 처리는 `_deepstream_source_health.py`에서 오류 기록, backoff, 재시작 요청 중복 방지, 파이프라인 재시작을 담당합니다.

확인 방법:

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml logs --tail 300 cctv-ai-engine
docker compose --env-file .env.jetson -f docker-compose.jetson.yml logs cctv-ai-engine | grep -E 'source|restart|backoff|ERROR|WARNING'
```

우선 확인 순서:

1. RTSP 주소와 카메라 네트워크 연결
2. 특정 카메라만 실패하는지 전체 source가 실패하는지
3. `DS_SOURCE_FAILURE_BACKOFF_SEC`가 너무 짧아 재시작이 반복되는지
4. `DS_PIPELINE_RESTART_MIN_INTERVAL_SEC`가 운영 환경에 적절한지
5. 카메라 모델 설정 변경으로 추론 topology가 바뀌었는지

### 이벤트는 감지되지만 스피커/전광판/경광등이 동작하지 않음

원인 후보:

- 현장 디바이스 전원이 꺼져 있거나 장비가 부팅 중입니다.
- 디바이스 IP가 바뀌었거나 CCTV 서버와 같은 네트워크에 없습니다.
- 방화벽, 스위치, VLAN, 포트 설정 문제로 장비 포트에 접근할 수 없습니다.
- `SPEAKER_*`, `SIREN_*`, `SIGNBOARD_*` 환경변수가 비어 있어 Action Layer가 장비를 비활성화했습니다.
- Action Layer의 알람 쿨다운 때문에 같은 카메라/이벤트가 잠시 스킵되었습니다.

확인 방법:

```bash
.venv/bin/python scripts/health/check_field_network.py --allow-unconfigured
.venv/bin/python scripts/health/check_alarm_devices.py
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
.venv/bin/python scripts/health/check_alarm_devices.py --skip-network
```

서버 라우팅이 현장 장비망으로 잡히는지 먼저 확인하려면:

```bash
.venv/bin/python scripts/health/check_field_network.py \
  --allow-unconfigured \
  --expected-interface eno1 \
  --expected-subnet 192.168.88.0/24
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
# 모니터링 스택 실행 중일 때:
docker compose --profile monitoring logs --tail 120 prometheus
docker compose --profile monitoring logs --tail 120 grafana
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

- 같은 arm64 호스트에서 전체 스택을 계속 써야 하면 Jetson compose 기준으로 재기동합니다.

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d
```

- EdgeX UI가 필요하면 ARM64 장비에서 직접 띄우기보다 x86_64 서버/PC에서 UI를 실행하거나, EdgeX REST API와 Grafana를 우선 사용합니다.
- Jetson/arm64 운영은 `docker-compose.jetson.yml` 사용을 우선합니다.
- 이전 compose 프로젝트의 중지 컨테이너가 이름을 점유한다면, 해당 컨테이너가 실행 중이 아닌지 확인한 뒤 제거합니다.

### AIoT parser가 PostgreSQL/Outbox 오류로 재시작함

원인 후보:

- `aiot-parser-db`가 떠 있지 않거나 healthcheck가 실패했습니다.
- `parser-python/.env`의 DB 설정이 compose override와 다르게 직접 실행되고 있습니다.
- 컨테이너 안에서 `localhost`는 호스트가 아니라 자기 자신입니다.
- `/data/runtime/event_outbox.db`가 bind mount 권한 문제로 쓰기 불가입니다.

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
.venv/bin/python scripts/health/check_offline_readiness.py
.venv/bin/python -m pytest
.venv/bin/python scripts/health/check_compose_runtime_assumptions.py --json
.venv/bin/python scripts/health/check_deployment_readiness.py
.venv/bin/python scripts/health/check_alarm_devices.py --skip-network
.venv/bin/python scripts/health/check_sensitive_defaults.py
.venv/bin/python scripts/health/check_dockerfile_sources.py
.venv/bin/python scripts/smoke/smoke_test_deployment.py
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
```

기준:

- 장비 전원이 아직 안 들어온 상태에서는 `check_offline_readiness.py`를 먼저 봅니다. 이 스크립트는 카메라/스피커/전광판 네트워크 연결을 요구하지 않고 배포 설정, 핵심 API, AI/DeepStream 단위, parser/outbox 테스트를 확인합니다.
- 전체 테스트가 통과해야 합니다.
- `check_compose_runtime_assumptions.py`가 실패하면 full compose 실행 전에 호스트 아키텍처, EdgeX 이미지, AIoT parser DB 설정을 먼저 맞춥니다.
- `check_sensitive_defaults.py`에서 민감 기본값이 없어야 합니다.
- smoke test 두 개가 모두 `"passed": true`여야 합니다.

## 관련 문서

- [../modules/PROJECT_STRUCTURE.md](../modules/PROJECT_STRUCTURE.md): 전체 구조와 데이터 흐름
- [../features/EVENT_SCHEMA_STANDARD.md](../features/EVENT_SCHEMA_STANDARD.md): AI/센서/디바이스 이벤트 표준 스키마
- [../features/PUBLIC_API_GUIDE.md](../features/PUBLIC_API_GUIDE.md): Public API 사용 가이드
- [../features/PUBLIC_API_EXAMPLES.md](../features/PUBLIC_API_EXAMPLES.md): Public API 복붙용 샘플
- [JETSON_EDGEX_FIELD_CHECKLIST.md](JETSON_EDGEX_FIELD_CHECKLIST.md): Jetson/EdgeX 현장 점검
- [MLOPS_MODEL_EVALUATION.md](MLOPS_MODEL_EVALUATION.md): 모델 교체 전 평가 절차
