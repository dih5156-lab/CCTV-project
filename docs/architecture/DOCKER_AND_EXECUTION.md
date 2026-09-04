# Docker 구성과 실행 방법

## 1. Compose 선택

| 파일 | 대상 | 핵심 |
|---|---|---|
| `docker-compose.yml` | 일반 PC·개발 검증 | OpenCV 경로와 기본 서비스 |
| `docker-compose.jetson.yml` | Jetson 운영·DeepStream | GPU, TensorRT, DeepStream, 외부 영속 볼륨 |

실제 배포 전에는 `docker compose ... config --quiet`로 Compose 렌더링 오류를 먼저 잡는다.

## 2. Jetson 기본 실행

```bash
cp .env.jetson.example .env.jetson
# .env.jetson에서 비밀번호·IP·카메라·모델 경로를 환경에 맞게 수정
docker compose --env-file .env.jetson -f docker-compose.jetson.yml config --quiet
docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build
docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps
```

기본 확인:

```bash
curl -fsS http://127.0.0.1:9000/health
curl -fsS http://127.0.0.1:8765/health
curl -fsS http://127.0.0.1:8769/health
docker logs --tail 120 cctv-ai-engine
```

## 3. 주요 서비스와 포트

| 서비스 | 포트 | 확인 목적 |
|---|---:|---|
| Public API | 9000 | 외부 조회·제어 |
| Alert API | 8000 | 알림·이벤트 처리 |
| Action Layer | 8080 | 장치 제어·승인 |
| Stream API | 8769 | 영상 스트림 |
| EdgeX Core Data | 48080 | Reading |
| EdgeX Core Metadata | 48081 | 장치 등록 |
| MQTT | 1883 | 내부 이벤트 |
| Prometheus | 9090 | 메트릭 |
| Grafana | 3000 | 대시보드 |

## 4. 운영 데이터와 중지

Jetson Compose는 EdgeX DB, AIoT DB, parser 데이터, eKuiper 데이터, TensorRT cache 등을 named volume에 보관한다. 컨테이너를 다시 만들더라도 volume을 삭제하지 않으면 데이터가 유지된다.

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml logs --tail 200 cctv-edgex-adapter aiot-parser cctv-action-layer
docker compose --env-file .env.jetson -f docker-compose.jetson.yml restart cctv-ai-engine
docker compose --env-file .env.jetson -f docker-compose.jetson.yml down
```

`down -v`는 영속 volume을 삭제할 수 있으므로 운영 데이터 백업과 승인 없이 사용하지 않는다.

## 5. 배포 후 검증

```bash
RUNTIME_ENV_FILE=.env.jetson ./scripts/ops/run_operation_check.sh
./scripts/ops/run_deepstream_stability_watch.sh 60 60
```

검증 결과는 “health 응답”, “영상 프레임 처리”, “MQTT 이벤트”, “장치 응답”, “DB 저장”을 따로 기록한다. 컨테이너가 `Up`이라고 해서 전체 파이프라인이 정상인 것은 아니다.

