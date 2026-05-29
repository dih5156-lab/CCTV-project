# CCTV 운영 체크리스트

## 결론

운영/데모 실행 전에는 아래 순서로 확인합니다.

```text
설정 확인 -> 컨테이너 상태 확인 -> health 확인 -> smoke test -> DeepStream 안정성 관찰 -> 로그 보관
```

## 1. 실행 전 설정

- `.env`가 존재하는지 확인
- `PUBLIC_API_KEY`, `INTERNAL_SERVICE_TOKEN`, `MQTT_USER`, `MQTT_PASSWORD`가 운영값인지 확인
- `known_faces.json`이 존재하고 JSON 형식이 깨지지 않았는지 확인
- nginx 데모 설정은 `config/nginx/public-demo.conf` 기준으로 확인
- HLS/Stream API 구 경로는 현재 데모 표준 경로에서 제외됨 (`410` 반환)

## 2. 컨테이너 상태

Docker 권한이 없는 계정에서는 같은 터미널에서 먼저 sudo 인증을 열어둡니다.

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
```

API key가 필요한 운영 환경에서는 Public API 호출에 헤더를 붙입니다.

```bash
curl -fsS -H "X-API-Key: ${PUBLIC_API_KEY}" http://localhost:9000/api/v1/health
```

## 4. 표준 운영 점검

DeepStream 장시간 관찰 전에는 wrapper로 기본 점검을 먼저 실행합니다.

```bash
./scripts/run_operation_check.sh
```

통과 기준:

- `deployment smoke` PASS
- `data flow smoke` PASS
- 최종 `result: PASS`

개별 데이터 흐름만 따로 확인할 때는 아래 명령을 사용합니다.

```bash
.venv/bin/python scripts/smoke_test_data_flow.py
```

통과 기준:

- `passed=true`
- alert api 수신 통과
- action layer event 수신 통과
- public api metrics 확인 통과

## 5. DeepStream 안정성 관찰

짧은 확인:

```bash
sudo -v
./scripts/run_deepstream_stability_watch.sh 30 30
```

운영/데모 전 권장 확인:

```bash
sudo -v
./scripts/run_deepstream_stability_watch.sh 720 60
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
sudo docker restart cctv-public-api
./scripts/run_operation_check.sh
```

그 외에는 아래 순서로 확인합니다.

1. `docker inspect`로 컨테이너 health와 재시작 횟수 확인
2. `docker stats`로 메모리/CPU 급증 확인
3. `sudo docker logs --tail 160 cctv-ai-engine`로 DeepStream 로그 확인
4. `sudo docker logs --tail 160 cctv-action-layer`로 이벤트 처리 timeout 확인
5. `scripts/smoke_test_data_flow.py`를 단독 재실행해서 주변 데이터 경로만 분리 확인

## 7. 기록 보관

- 안정성 관찰 로그와 summary는 `reports/deepstream-stability/`에 보관
- 장애가 있었으면 실패 샘플 번호, 실패 항목, 해당 시각의 컨테이너 로그를 함께 남김
- 최종 판단은 `docs/DEEPSTREAM_PERFORMANCE_STABILITY_2026-05-26.md`의 표준 안정성 통과 기준을 따름
