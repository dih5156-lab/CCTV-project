# CCTV 프로젝트 리뷰 준비 - 2026년 6월

## 결론

현재는 운영 보안 강화보다 리뷰 데모 안정화가 우선입니다.
개발단계 기준으로는 민감정보 유출 방지, 핵심 기능 회귀 테스트, 데모 시나리오 고정에 집중합니다.

## 현재 검증 상태

- 전체 테스트: `771 passed, 75 skipped`
- 핵심 데모 경로 테스트: `138 passed`
- 민감 기본값 검사: `No sensitive defaults found.`
- 현재 실행 스택 확인: Public API, Alert API, Action Layer, Stream API, Demo UI, EdgeX core-data 응답 정상
- Stream API 스냅샷 확인: `camera_1` JPEG 1920x1080 생성 정상
- Public API 데이터 확인: 카메라 목록 조회 정상, 최근 이벤트 조회 정상
- Jetson/DeepStream 확인: `cctv-ai-engine` healthy, `Runtime=nvidia`, `RestartCount=0`, 컨테이너 내부 `gi+pyds import ok`
- DeepStream 처리 로그 확인: 약 3분 구간에서 frames/events 지속 증가, `dropped=0`, `cameras=1`
- Jetson 리소스 확인: Jetson Linux R36.5.0, `tegrastats` 기준 RAM 약 16GB/62GB, GPU 온도 약 48~49도, CPU/TJ 약 53~54도
- Jetson/DeepStream 30분 안정성 관찰: 30/30 샘플 `healthy`, Stream API `ok`, `RestartCount=0`, `dropped=0`
- 30분 관찰 상세: frames `1,916,816 -> 1,956,999` (+40,183), events `298,157 -> 336,218` (+38,061)
- 30분 리소스 상세: `cctv-ai-engine` 메모리 `694.5~694.8MiB`, Jetson RAM `16,169~17,010MB`, GPU 온도 `47.9~49.2C`, TJ `51.2~53.8C`

실행한 명령:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_api_auth.py tests/test_public_api.py tests/test_mqtt.py tests/test_stream_api.py tests/test_device_service.py
.venv/bin/python scripts/check_sensitive_defaults.py
curl -fsS http://localhost:9000/api/v1/health
curl -fsS http://localhost:9000/api/v1/readiness
curl -fsS http://localhost:9000/api/v1/cameras
curl -fsS "http://localhost:9000/api/v1/events?limit=5"
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8769/health
curl -fsS -o /tmp/camera_1_snapshot_check.jpg http://localhost:8769/snapshot/camera_1
curl -fsS -I http://localhost:7000/
curl -fsS http://localhost:59880/api/v3/ping
.venv/bin/python scripts/check_jetson_edgex_stack.py --host localhost --public-api-port 9000 --deepstream --check-appearance-status
sudo docker inspect cctv-ai-engine --format 'Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} RestartCount={{.RestartCount}} StartedAt={{.State.StartedAt}} Runtime={{.HostConfig.Runtime}}'
sudo docker exec cctv-ai-engine python -c 'import gi; import pyds; print("gi+pyds import ok")'
sudo docker stats cctv-ai-engine --no-stream
sudo docker logs --tail 80 cctv-ai-engine
tegrastats --interval 1000
```

30분 안정성 관찰 로그:

```text
/tmp/deepstream_stability_20260514_110527.log
```

## 리뷰 데모 흐름

1. Docker Compose 또는 Jetson Compose 실행
2. 카메라 또는 샘플 영상 입력
3. 사람/헬멧/낙상/침입/얼굴/외형 조건 중 1~2개 이벤트 발생
4. MQTT 및 EdgeX 이벤트 전달 확인
5. Public API에서 이벤트, 카메라, 상태 조회
6. Stream API 또는 Demo UI에서 화면 확인
7. 스피커/경광등/전광판은 실제 장비 또는 Mock으로 호출 흐름 확인

## 개발단계 보안 기준

- `.env`, `.env.jetson`, `cameras.json`, `known_faces.json`, `mosquitto/passwd`는 커밋하지 않습니다.
- `.env.example`, `.env.jetson.example`에는 실제 비밀번호, API key, RTSP 계정을 넣지 않습니다.
- `PUBLIC_API_KEY`, `INTERNAL_SERVICE_TOKEN`, `CORS_ORIGINS` 강제는 운영 전환 시점에 적용합니다.
- 개발 중 외부 노출 포트는 허용하되, 리뷰 자료에는 개발 편의 설정임을 명시합니다.

## 남은 작업 우선순위

### P0 - 실행 차단

- 현재 자동 테스트 기준 P0는 발견되지 않았습니다.
- Jetson 실기기, 카메라, 스피커/경광등 장비 연결은 별도 현장 검증이 필요합니다.

### P1 - 데모 흐름 안정화

- 리뷰용 샘플 영상 또는 카메라 입력 1개를 고정합니다.
- 이벤트 발생 조건을 미리 정합니다.
- 데모 중 사용할 API 호출 명령을 `docs/PUBLIC_API_EXAMPLES.md` 기준으로 선별합니다.

### P2 - 설명 가능성 보강

- 어떤 이벤트가 어떤 MQTT topic과 Public API 응답으로 이어지는지 1장 흐름도로 정리합니다.
- Jetson/DeepStream 실행 실패 시 확인할 로그 위치를 문서 앞부분에 모읍니다.
- 얼굴/외형 검색은 정확도보다 POC 동작 범위를 명확히 설명합니다.

### P3 - 운영 전환 TODO

- Public API key 필수화
- 내부 API token 필수화
- CORS 도메인 제한
- Stream API 접근 제한
- Grafana/MQTT/DB 비밀번호 강제
- 외부 노출 포트 최소화

## 리뷰 전 체크 명령

```bash
.venv/bin/python scripts/check_sensitive_defaults.py
.venv/bin/python -m pytest
docker compose ps
```

Jetson 현장 확인 시:

```bash
.venv/bin/python scripts/check_jetson_edgex_stack.py --host localhost --public-api-port 9000 --deepstream --check-appearance-status
.venv/bin/python scripts/smoke_test_data_flow.py
```

주의:

- `check_jetson_edgex_stack.py --deepstream`의 `GStreamer Python`, `DeepStream pyds` 검사는 호스트 Python 기준입니다.
  현재 배포 형태에서는 DeepStream이 `cctv-ai-engine` 컨테이너 내부에서 실행되므로, 최종 판단은 컨테이너 내부 `gi`, `pyds` import와 DeepStream 처리 로그를 함께 봅니다.
- `AIoT Parser DB`, `AIoT Parser`는 로컬 호스트 포트 기준 검사에서 실패할 수 있습니다.
  컨테이너 네트워크 내부 서비스로 운영 중인지, 리뷰 데모에 해당 경로가 필요한지 별도 확인합니다.

## 발표 시 한 줄 요약

현재 프로젝트는 개발단계 POC 기준으로 핵심 AI 분석, 이벤트 전달, Public API, 장비 액션 흐름이 자동 테스트를 통과했습니다.
운영 보안 항목은 리뷰 이후 운영 전환 단계에서 별도 체크리스트로 잠글 예정입니다.
