# CCTV 헬멧 착용 및 낙상 감지 시스템

YOLOv8 기반 실시간 안전 관리 시스템으로, 다중 카메라 환경에서 헬멧 착용 여부, 낙상 사고, 위험 구역 침입을 자동 감지합니다.
Windows PC와 NVIDIA Jetson Orin 모두 동작합니다.

## 주요 기능

- **헬멧 착용 감지**: 커스텀 YOLOv8 모델로 헬멧 착용/미착용 실시간 탐지
- **낙상 감지**: YOLOv8-pose 모델 기반 사람 자세 분석으로 낙상 사고 탐지
- **다중 카메라**: RTSP/웹캠 동시 처리 및 자동 재연결
- **위험 구역 관리**: 실시간 폴리곤 그리기·저장·삭제 (GUI 인터랙션 지원)
- **얼굴 인식 확장 구조**: Windows 개발/디버깅, Jetson 운영 배포를 전제로 한 선택형 백엔드
- **Zone API**: REST API로 외부에서 구역 설정 조회·수정
- **EdgeX Foundry 연동**: MQTT 기반 표준 EdgeX v3 이벤트 발행
- **이벤트 포워딩**: Public API 수신 이벤트를 Action Layer로 전달해 알람/이력 처리를 통합
- **센서 위험 분류**: TLV 센서 온도·기울기·event_code 기반 경고/위험 이벤트 자동 생성
- **Action Layer**: 스피커 알람·외부 API 호출·SQLite 이벤트 저장
- **데이터셋 수집**: YOLO 형식 자동 라벨링 및 학습 데이터 생성
- **Jetson 가속**: GStreamer NVDec 하드웨어 디코딩 + TensorRT `.engine` 모델 자동 인식

## 프로젝트 구조

```
CCTV-project/
├── main.py                         # CCTV AI 엔진 기본 실행 진입점
├── run_external_ingest.py           # 외부 MQTT/NC 수신 진입점
├── src/                            # 핵심 애플리케이션 코드
│   ├── api/                        # FastAPI 공개 API (/api/v1)
│   ├── bootstrap/                  # CLI, 런타임 초기화, 프로세서 생성
│   ├── config/                     # 중앙화된 설정 (ENV 오버라이드 지원)
│   ├── core/                       # 영상 처리, AI 추론, 이벤트 생성
│   │   ├── ai/                     # YOLO, 낙상, 얼굴, 외형 분석
│   │   ├── processor.py            # OpenCV + YOLO 기반 처리기
│   │   └── deepstream_processor.py # NVIDIA DeepStream 기반 처리기
│   ├── services/                   # ActionBridge, API 서버, 로그/검색 서비스
│   ├── edgex/                      # EdgeX 디바이스 서비스/어댑터
│   ├── protocols/                  # MQTT, HTTP, REST, TLV 통신 계층
│   ├── devices/                    # 스피커, 전광판, 경광등 제어
│   ├── storage/                    # SQLite 저장소
│   └── utils/                      # 카메라 입력, 구역, 시각화, geometry
├── runners/                        # 서비스별 단독 실행 진입점
├── parser-python/                  # AIoT TLV 센서 파서 서비스
├── web/                            # 시연/관제용 HTML 대시보드
├── config/                         # DeepStream/외형 분석 설정
├── models/                         # YOLO, PP-Human, TensorRT 모델 파일
├── data/                           # SQLite DB, crop 이미지, 런타임 데이터
├── known_faces/                    # 얼굴 인식용 등록/샘플 이미지
├── edgex/                          # EdgeX device profile 및 ASC 설정
├── kuiper/                         # eKuiper 룰 파일
├── monitoring/                     # Prometheus/Grafana 설정
├── mosquitto/                      # MQTT broker 설정
├── scripts/                        # 점검, 변환, smoke test, 모델 평가
├── tests/                          # pytest 테스트
├── docker-compose.yml              # 일반 Docker/EdgeX 통합 배포
├── docker-compose.arm64.yml        # ARM64 EdgeX 호환 override
├── docker-compose.jetson.yml       # Jetson/DeepStream 운영 배포
└── docs/architecture/PROJECT_STRUCTURE.md # 상세 프로젝트 구조 문서
```

더 자세한 디렉터리별 역할과 데이터 흐름은
[docs/architecture/PROJECT_STRUCTURE.md](docs/architecture/PROJECT_STRUCTURE.md)를 참고하세요.
운영 중 상태 확인과 복구 절차는
[docs/operations/OPERATIONS_RUNBOOK.md](docs/operations/OPERATIONS_RUNBOOK.md)에 정리되어 있습니다.
현장 점검 순서와 DeepStream 장시간 안정성 확인은
[docs/operations/OPERATION_CHECKLIST.md](docs/operations/OPERATION_CHECKLIST.md),
[docs/operations/DEEPSTREAM_PERFORMANCE_STABILITY_2026-05-26.md](docs/operations/DEEPSTREAM_PERFORMANCE_STABILITY_2026-05-26.md)를 참고하세요.

## 포트별 역할

| 포트 | 서비스 | 브라우저 확인 주소 | 비고 |
|------|--------|-------------------|------|
| `9000` | Public API | `http://127.0.0.1:9000/` | Swagger 문서는 `/docs`, API는 `/api/v1/*` |
| `8000` | Alert API | `http://127.0.0.1:8000/` | 이벤트 수신용 내부 API, `/api/alerts`는 POST 전용 |
| `8080` | Action Layer | `http://127.0.0.1:8080/` | 사이트/제어/알람 액션 처리 |
| `9090` | Prometheus | `http://127.0.0.1:9090/-/ready` | 메트릭 수집 상태 |
| `3001` | Grafana | `http://127.0.0.1:3001/api/health` | 모니터링 UI |
| `1883` | MQTT broker | TCP only | 브라우저 확인 대상 아님 |
| `8769` | Stream API | `http://127.0.0.1:8769/health` | 카메라 MJPEG 미리보기 |

브라우저에서 `{"error":"not found"}`가 보이면 서버가 죽은 것이 아니라,
대부분 포트나 경로가 맞지 않은 경우입니다. 예를 들어 `8000`번의 루트는 Alert API 안내용이고,
Public API 문서는 `http://127.0.0.1:9000/docs`에서 확인합니다.

## 지원 플랫폼

| 플랫폼 | Python | CUDA | 비고 |
|--------|--------|------|------|
| Windows 10/11 | 3.10+ | 선택 | 개발·테스트 환경 |
| Ubuntu 22.04 | 3.10+ | 선택 | 서버 배포 |
| NVIDIA Jetson Orin | 3.10 (L4T) | 필수 | `USE_GSTREAMER=1` 설정 필요 |

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/dih5156-lab/CCTV-project.git
cd CCTV-project
```

### 2. 가상 환경 생성

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Jetson
source .venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

> **Jetson Orin**: PyTorch/OpenCV는 L4T 이미지에 이미 포함되어 있어 별도 설치 불필요.
> `torch`, `torchvision`, `opencv-python*` 라인은 설치를 건너뛰세요.

### 얼굴 인식 환경 분리

- Windows: 개발/디버깅용
- Jetson: 실제 얼굴 인식 운영용

Jetson에서 실사용 얼굴 인식을 켜려면 추가로 아래 파일을 설치합니다.

```bash
pip install -r requirements-face-jetson.txt
```

자세한 내용은 [docs/operations/FACE_RECOGNITION_SETUP.md](docs/operations/FACE_RECOGNITION_SETUP.md)를 참고하세요.

### 4. 모델 파일 준비

`models/` 폴더에 아래 파일을 배치합니다. 없으면 YOLOv8 공식 모델이 자동 다운로드됩니다.

```
models/
├── helmet_model_ver0.5.pt   # 헬멧 감지 (커스텀)
├── yolov8n-pose.pt          # 낙상 감지 (포즈)
├── yolov8n.pt               # 사람 감지
└── pphuman_attribute.onnx   # 선택: 외형 속성 분석
```

외형 속성 분석은 기본적으로 HSV 색상 기반으로 동작합니다. PP-Human 계열 ONNX 모델을
사용하려면 아래 환경변수를 추가합니다.

```bash
APPEARANCE_ENABLED=true
APPEARANCE_BACKEND=pphuman
APPEARANCE_MODEL_PATH=models/pphuman_attribute.onnx
APPEARANCE_LABEL_MAP_PATH=config/appearance_pphuman_labels.example.json
APPEARANCE_RUNTIME=onnxruntime
```

Jetson에서 ONNX Runtime wheel 호환 문제가 있으면 Paddle 원본 모델을 직접 지정할 수 있습니다.

```bash
pip install -r requirements-appearance-paddle.txt

APPEARANCE_BACKEND=pphuman
APPEARANCE_MODEL_PATH=models/pphuman_attribute_src/PP-LCNet_x1_0_pedestrian_attribute_infer
APPEARANCE_LABEL_MAP_PATH=config/appearance_pphuman_labels.example.json
APPEARANCE_RUNTIME=paddle
```

카메라 설정의 `detections`에 `appearance`를 포함하면 `YOLO person bbox → 속성 모델
crop → SQLite appearance_log 저장/검색` 흐름으로 연결됩니다.

**Jetson TensorRT 가속 (선택사항):**

```python
# 모델을 TensorRT .engine으로 변환 (Jetson에서 1회 실행)
from ultralytics import YOLO
YOLO("models/yolov8n.pt").export(format="engine", device=0)
# → models/yolov8n.engine 자동 생성 후 우선 사용됨
```

## 설정

### 카메라 설정 (cameras.json)

```json
[
  {
    "id": "camera_1",
    "name": "현장 카메라 1",
    "source": "rtsp://user:pass@192.168.1.100:554/stream",
    "enabled": true,
    "detections": ["person", "helmet", "fall"],
    "zones": [
      {
        "id": "zone_1",
        "name": "위험 구역",
        "polygon": [[100,100],[500,100],[500,400],[100,400]]
      }
    ]
  },
  {
    "id": "webcam",
    "source": 0,
    "enabled": true
  }
]
```

### 환경 변수 오버라이드

주요 설정은 환경변수로 재정의할 수 있습니다.

| 환경변수 | 설명 | 예시 |
|---------|------|------|
| `DEVICE` | 추론 장치 | `cuda`, `cuda:0`, `cpu` |
| `HELMET_MODEL_PATH` | 헬멧 모델 경로 | `/models/helmet.pt` |
| `PERSON_MODEL_PATH` | 사람 모델 경로 | `/models/yolov8n.pt` |
| `POSE_MODEL_PATH` | 포즈 모델 경로 | `/models/yolov8n-pose.pt` |
| `APPEARANCE_ENABLED` | 외형 분석 활성화 | `true` / `false` |
| `APPEARANCE_BACKEND` | 외형 분석 방식 | `hsv` / `pphuman` |
| `APPEARANCE_MODEL_PATH` | 속성 모델 경로 | `models/pphuman_attribute.onnx` |
| `APPEARANCE_LABEL_MAP_PATH` | 속성 라벨 맵 경로 | `config/appearance_pphuman_labels.example.json` |
| `APPEARANCE_RUNTIME` | 속성 모델 런타임 | `auto` / `onnxruntime` / `paddle` |
| `APPEARANCES_DB` | 외형 로그 SQLite 경로 | `/app/data/runtime/appearances.db` |
| `DISPLAY_ENABLED` | 화면 출력 | `true` / `false` |
| `MQTT_BROKER` | MQTT 브로커 호스트 | `localhost` |
| `USE_GSTREAMER` | Jetson NVDec 하드웨어 디코딩 | `1` (Jetson 전용) |

## 실행

### 기본 실행 (Windows)

```bash
# 웹캠 + 화면 표시
python main.py --display

# 다중 RTSP 카메라 + Zone API
python main.py --cameras cameras.json --display --api-port 8765

# CUDA 사용
python main.py --cameras cameras.json --device cuda --display
```

### Jetson Orin

```bash
# GStreamer 하드웨어 디코딩 + CUDA 추론
USE_GSTREAMER=1 DEVICE=cuda:0 python main.py --cameras cameras.json

# TensorRT 모델 자동 사용 (models/*.engine 파일이 있으면 우선 로드)
USE_GSTREAMER=1 DEVICE=cuda:0 python main.py --cameras cameras.json
```

### EdgeX 전체 파이프라인

```bash
# 1) AI 엔진 (MQTT 이벤트 발행)
python main.py --cameras cameras.json \
  --mqtt-broker localhost --mqtt-port 1883 \
  --mqtt-topic-prefix cctv/ai/events

# 2) EdgeX 어댑터 (AI 이벤트 → EdgeX Core Data)
python runners/run_edgex_adapter.py \
  --ai-mqtt-broker localhost --ai-topic-prefix cctv/ai/events \
  --edgex-metadata-url http://localhost:59881 \
  --edgex-data-url http://localhost:59880

# 3) Kuiper 룰 배포 (침입 필터링 / 고신뢰 라우팅)
python runners/run_kuiper_rules.py \
  --kuiper-api http://localhost:9081 \
  --intrusion-confidence 0.7 --critical-confidence 0.9

# 4) Action Layer (스피커 알람 + 외부 API + DB)
python runners/run_action_bridge.py \
  --mqtt-broker localhost \
  --external-api-url http://localhost:8000/api/alerts \
  --speaker-host 192.168.88.92 --speaker-port 5000 \
  --speaker-user admin --speaker-password YOUR_PASSWORD
```

### Docker (EdgeX 통합)

#### Compose 파일 구성

| 파일 | 용도 |
|------|------|
| `docker-compose.yml` | **기본 구성** — 공개 API, Action Layer, EdgeX 계열 서비스를 같은 서버/PC에서 실행할 때 사용 |
| `docker-compose.arm64.yml` | **ARM64 override** — Jetson/ARM64에서 기본 compose의 EdgeX 이미지를 ARM64용으로 보정할 때 사용 |
| `docker-compose.jetson.yml` | **Jetson 구성** — Jetson에서 AI 엔진, alert-api, 내부 관리 API를 함께 실행할 때 사용 |

#### 기본 실행 (Windows PC에서 모두 실행)

```bash
# 환경변수 파일 생성 (.env.example 복사 후 편집)
copy .env.example .env
# .env 에서 SPEAKER_HOST, SPEAKER_PASSWORD 등 설정

docker compose up -d --build

# 로그 확인
docker compose logs -f cctv-action-layer

# 중지
docker compose down
```

Public API, Alert API, Action Layer 중심으로만 확인할 때는 필요한 서비스만 올릴 수 있습니다.

```bash
docker compose up -d cctv-alert-api cctv-action-layer cctv-public-api prometheus grafana edgex-mqtt-broker
```

AIoT parser까지 확인할 때는 PostgreSQL 보조 서비스도 함께 올립니다.

```bash
docker compose up -d aiot-parser-db aiot-parser
```

> arm64/Jetson 계열 호스트에서 기본 `docker-compose.yml`의 일부 EdgeX 이미지는 `exec format error`가 날 수 있습니다.
> 같은 장비에서 기본 compose 전체 스택을 올릴 때는 `docker-compose.arm64.yml`을 함께 적용하고,
> Jetson 현장 배포는 `docker-compose.jetson.yml`을 우선 사용하세요. 자세한 운영 대응은
> [docs/operations/OPERATIONS_RUNBOOK.md](docs/operations/OPERATIONS_RUNBOOK.md)를 참고하세요.
> EdgeX UI 이미지는 이 환경에서 ARM64 manifest가 확인되지 않아 ARM64 override 기본 실행에서 제외됩니다.

```bash
docker compose -f docker-compose.yml -f docker-compose.arm64.yml up -d
```

배포 전 런타임 전제 조건은 아래 명령으로 미리 확인할 수 있습니다.

```bash
.venv/bin/python scripts/health/check_compose_runtime_assumptions.py --json
```

운영 투입 전에는 비밀값, compose 전제 조건, API readiness, DeepStream 안정성 점검을 한 번에 확인할 수 있습니다.

```bash
.venv/bin/python scripts/health/check_runtime_secret_consistency.py --env-file .env --json
.venv/bin/python scripts/health/check_deepstream_env.py
./scripts/ops/run_operation_check.sh
./scripts/ops/run_deepstream_stability_watch.sh 300 60
```

#### Jetson 분리 실행 (Jetson Orin + 서버/PC 분리)

```bash
# 서버/PC (공개 API + Action Layer + EdgeX)
docker compose -f docker-compose.yml up -d --build

# Jetson Orin (AI 엔진 계열 컨테이너 실행)
docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build

# 또는 Jetson에서 단일 프로세스로 AI 엔진 실행 시
USE_GSTREAMER=1 DEVICE=cuda:0 python main.py \
  --cameras cameras.json \
  --mqtt-broker <SERVER_IP> \
  --mqtt-port 1883 \
  --mqtt-topic-prefix cctv/ai/events
```

#### 개별 서비스 확인

```bash
# 실행 중인 서비스 확인
docker compose ps

# 특정 서비스 로그
docker compose logs -f cctv-ai-engine
docker compose logs -f cctv-action-layer
docker compose logs -f cctv-public-api
docker compose logs -f edgex-kuiper

# 서비스 재시작
docker compose restart cctv-action-layer
```

#### 운영 환경 변수 기준

기본 스택은 `.env`, Jetson 통합 스택은 `.env.jetson`을 기준 파일로 사용합니다.
Jetson은 반드시 `docker compose --env-file .env.jetson -f docker-compose.jetson.yml ...` 형태로 실행하세요.

운영 기준 변수 표와 우선순위는 [docs/operations/DEPLOYMENT_ENVIRONMENT_VARIABLES.md](docs/operations/DEPLOYMENT_ENVIRONMENT_VARIABLES.md)에 모아두었습니다.

운영에서 비워두면 안 되는 핵심 값:
- `MQTT_USER`, `MQTT_PASSWORD`, `AIOT_DB_PASSWORD`
- `PUBLIC_API_KEY`, `INTERNAL_SERVICE_TOKEN`, `GRAFANA_ADMIN_PASSWORD`
- 현장 장비를 붙일 경우 `SPEAKER_*`, `SIGNBOARD_*`, `SIREN_*`

`.env`와 `.env.jetson`은 Git에 커밋하지 않고, 실제 비밀번호와 토큰은 예시 파일에 넣지 않습니다.

### 주요 CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--cameras FILE` | 카메라 목록 JSON | 없음 (웹캠 0번) |
| `--video FILE` | 비디오 파일 경로 | 없음 |
| `--device` | 추론 장치 (`cpu`/`cuda`) | `cpu` |
| `--confidence` | 헬멧 감지 신뢰도 | `0.5` |
| `--pose-confidence` | 포즈/사람 감지 신뢰도 | `0.3` |
| `--fps` | 목표 FPS | `30` |
| `--frame-skip N` | 매 N프레임마다 AI 추론 | `3` |
| `--display` | GUI 화면 표시 | off |
| `--api-port PORT` | Zone REST API 포트 | off |
| `--zone-presets FILE` | 구역 프리셋 저장 파일 | `zone_presets.json` |
| `--zone-detection` | 위험 구역 감지 활성화 | off |
| `--mqtt-broker HOST` | MQTT 브로커 | `localhost` |
| `--mqtt-port PORT` | MQTT 포트 | `1883` |
| `--mqtt-topic-prefix` | MQTT 토픽 prefix | `cctv/ai/events` |
| `--no-debounce` | 이벤트 디바운싱 비활성화 | off |
| `--debounce SEC` | 디바운싱 간격(초) | `3.0` |
| `--collect-dataset` | 데이터셋 자동 수집 | off |
| `--dataset-dir DIR` | 수집 데이터 저장 경로 | `./collected_data` |

## 공개 API

공개 API는 `runners/run_public_api.py`로 실행합니다.
인증, 요청/응답 예시, 대시보드 연동 기준은
[docs/api/PUBLIC_API_GUIDE.md](docs/api/PUBLIC_API_GUIDE.md)에 정리되어 있습니다.
바로 호출해볼 `curl` 예시는 [docs/api/PUBLIC_API_EXAMPLES.md](docs/api/PUBLIC_API_EXAMPLES.md)를 참고하세요.
네트워크가 제한된 현장에서는 API 프로세스가 제공하는 로컬 문서 엔드포인트로도 기본 사용법을 확인할 수 있습니다.

```bash
python runners/run_public_api.py --host 0.0.0.0 --port 9000
```

내부 관리 API는 AI 엔진 프로세스 또는 Jetson compose에서 함께 뜹니다.

- Zone API: 위험구역 조회/수정
- Camera Model API: 카메라별 모델 on/off
- Face API: 등록 얼굴 관리
- Stream API: MJPEG 스트림

운영 환경에서는 `INTERNAL_SERVICE_TOKEN`을 설정해 내부 관리 API를 보호하는 것을 권장합니다.

주요 엔드포인트:

- `GET /api/v1/health`
- `GET /api/v1/readiness`
- `GET /api/v1/docs`
- `GET /api/v1/docs/openapi.json`
- `GET /api/v1/events`
- `GET /api/v1/cameras`
- `GET /api/v1/search`
- `GET /api/v1/appearances/status`
- `GET/POST/DELETE /api/v1/sites`
- `GET/POST /api/v1/control/*`

응답 형식:

- 성공/실패 응답은 `{ success, data, error, timestamp }`
- 목록 조회는 페이지네이션 응답에서 `{ success, items, total, limit, offset, timestamp }`
- `/api/v1/health`는 Public API 프로세스 자체 상태, `/api/v1/readiness`는 Action Layer와 Alert API 연결까지 확인합니다.
- `/api/v1/events`, `/api/v1/alerts`, `/api/v1/sensor-readings`로 들어온 위험 이벤트는 Action Layer `/events`로 포워딩되어 동일한 알람/저장 경로를 탑니다.

외형 검색 상태 API:

- `GET /api/v1/appearances/status`
- 대시보드에서 `enabled`, `ready`, `warnings`, `next_steps`를 기준으로 필터 활성/비활성 및 운영 진단을 표시할 때 사용
- 상세 계약 문서: [docs/api/APPEARANCES_STATUS_API.md](docs/api/APPEARANCES_STATUS_API.md)

카메라 / 사이트 / 제어 API 해석 기준:

- `GET /api/v1/cameras`
  - `url`은 자격증명을 제거한 값만 내려갑니다.
  - `zones`는 `cameras.json` 기준 구역 설정입니다.
- `GET /api/v1/sites`
  - `camera_ids`, `control_mode`, `alarm_devices`를 함께 반환합니다.
  - 사이트 제어 기준 화면에서는 이 응답을 기준 source of truth로 쓰는 것을 권장합니다.
- `GET /api/v1/control/pending`
  - 현재는 Action Layer 원본 payload를 거의 그대로 전달합니다.
  - 프론트에서는 `event_id`, `camera_id`, `type`, `timestamp` 존재 여부를 우선적으로 방어적으로 처리하는 것을 권장합니다.

## 실사용 시연 UI

내부 발표나 현장 시연에서는 `web/public-demo.html`을 사용합니다.
이 화면은 Public API와 Stream API를 직접 호출해서 운영 흐름을 빠르게 보여주기 위한
가벼운 HTML 대시보드입니다.

브라우저에서 아래 파일을 엽니다.

```text
web/public-demo.html
```

기본 연결 주소:

| 항목 | 기본값 | 설명 |
|------|--------|------|
| Public API | `http://localhost:9000` | 상태, readiness, 카메라 목록, 이벤트 전송 |
| Stream API | `http://localhost:8769` | 카메라 MJPEG 화면 |
| Grafana | `http://localhost:3001` | 선택: 운영 메트릭 대시보드 |

이번 주 기능 진행 기준:

- TLV 센서 로그는 조회만 하지 않고 `src/services/sensor_classifier.py` 기준으로 위험 상태를 함께 계산합니다.
  - `temperature >= 50`: `temperature_alert` / `warning`
  - `temperature >= 70`: `temperature_alert` / `critical`
  - `angle_x` 또는 `angle_y` 절댓값 `>= 30`: `tilt_alert` / `warning`
  - `event_code != 0`: `sensor_event` / `warning`
- 위험 센서 입력은 Public API가 Action Layer `/events`로 함께 전달합니다.
- 시연 UI의 운영 요약은 최근 CCTV 이벤트, TLV 센서 이상, 승인 대기 건수를 같이 봅니다.
- 이벤트 포워딩 실패 시에도 API 응답은 실패 원인을 포함하고, 운영자는 readiness와 Action Layer 로그를 기준으로 확인합니다.

시연 전 확인:

```bash
docker compose ps
curl -fsS http://localhost:9000/api/v1/health
curl -fsS http://localhost:9000/api/v1/readiness
curl -fsS http://localhost:9000/api/v1/docs/openapi.json
curl -fsS http://localhost:8769/health
curl -fsS http://localhost:8769/cameras
.venv/bin/python scripts/smoke/smoke_test_deployment.py
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
.venv/bin/python scripts/health/check_public_api_fd_stability.py
.venv/bin/python scripts/health/check_runtime_secret_consistency.py --json
```

Prometheus/Grafana는 운영 모니터링용 선택 구성입니다. 핵심 시연 검증은 기본 smoke test만으로
Alert API, Action Layer, Public API readiness를 확인합니다. 모니터링까지 함께 검증할 때만 아래처럼 실행합니다.

```bash
.venv/bin/python scripts/smoke/smoke_test_deployment.py --include-monitoring
```

런타임 로그 정책:

- 새로 생성되는 ISO 8601 시각 문자열은 한국 표준시 오프셋 `+09:00`을 사용합니다.
- Unix epoch 숫자는 시간대와 무관한 절대시각이므로 기존 형식을 유지합니다.
- `data/*.jsonl`은 시연/운영 중 계속 누적되는 로컬 로그이므로 git 추적 대상에서 제외합니다.
- 공유가 필요한 샘플 데이터는 실제 런타임 파일 대신 별도 예제 파일로 분리해서 추가합니다.
- 런타임 로그와 외형 crop은 먼저 `./scripts/cleanup/cleanup_runtime_data.sh`로 미리보기합니다.
- 확인 후 `sudo ./scripts/cleanup/cleanup_runtime_data.sh --apply`를 실행하면 기본 7일 crop 보존, 삭제된 crop의 DB 참조 정리, 200MB JSONL 로그 회전 정책을 반영합니다.
- 운영 장비에서는 `./scripts/ops/install_runtime_cleanup_timer.sh --dry-run`으로 한국시간 매일 09:00 예약 내용을 확인한 뒤
  `sudo ./scripts/ops/install_runtime_cleanup_timer.sh`로 일일 정리 타이머를 설치합니다.
- Docker socket 권한이 제한된 장비에서 표준 운영 점검을 실행할 때는
  `sudo ./scripts/ops/run_operation_check.sh`를 사용합니다.

Docker socket 권한이 막힌 장비에서는 `docker` 명령 앞에 `sudo`를 붙입니다.

```bash
sudo docker compose ps
sudo docker compose up -d --force-recreate cctv-ai-engine
```

카메라 화면이 `스트림 연결 실패`로 보이면 아래 순서로 확인합니다.

1. `curl -fsS http://localhost:8769/health`
2. `curl -fsS http://localhost:8769/cameras`
3. `sudo docker logs --tail 120 cctv-ai-engine`

`cctv-ai-engine` 로그에 `MJPEG 스트리밍 서버 시작`이 없으면 Stream API가 뜨지 않은 상태입니다.
이 경우 `docker-compose.yml`의 `STREAM_API_ENABLED`, `STREAM_PORT`, `8769` 포트 publish 설정을 확인하고
`cctv-ai-engine` 컨테이너를 재생성합니다.

내일 설명할 때는 아래 순서가 가장 안전합니다.

```text
1. 전체 구조: AI Engine → MQTT/Alert API → Action Layer → Public API → Demo UI
2. Public API health/readiness로 서비스 상태 확인
3. 카메라 목록과 Stream API 화면으로 실제 입력 확인
4. 데모 UI에서 낙상/헬멧 미착용/위험구역 이벤트 전송
5. TLV 정상값/위험값을 전송해 센서 이벤트와 Action Layer 이력 확인
6. Metrics/Swagger와 선택형 Grafana로 운영 연동 가능성 확인
```

## 테스트

GitHub Actions와 같은 기본 검사를 push 전에 로컬에서 실행:

```bash
./scripts/dev/check_before_push.sh
```

이 저장소는 `.githooks/pre-push`를 사용하도록 설정하면 push 전에 위 검사를 자동 실행합니다.

```bash
git config --local core.hooksPath .githooks
```

긴급히 hook을 건너뛸 때만 아래처럼 실행합니다.

```bash
SKIP_PRE_PUSH_CHECKS=1 git push
```

공개 API 회귀 테스트:

```bash
python -m pytest tests/test_public_api.py -q
```

이벤트 포워딩 / 센서 분류 회귀 테스트:

```bash
python -m pytest tests/test_event_forwarding.py tests/test_sensor_classifier.py tests/test_sensor_device.py -q
```

스트림 API 회귀 테스트:

```bash
python -m pytest tests/test_stream_api.py -q
```

## 위험 구역 GUI 조작 (`--display` 모드)

| 키 / 동작 | 기능 |
|-----------|------|
| `d` | 구역 그리기 모드 ON/OFF |
| 마우스 좌클릭 | 구역 꼭짓점 추가 |
| `Enter` 또는 `c` | 현재 구역 완성 및 저장 |
| `z` | 마지막 꼭짓점 삭제 |
| `ESC` | 현재 그리기 취소 |
| 마우스 hover | 기존 구역 위에 올리면 주황색 하이라이트 |
| `x` (hover 중) | hover 중인 구역 삭제 |

저장된 구역은 `cameras.json`에 자동으로 반영됩니다.

## Zone REST API

`--api-port 8765` 실행 시 아래 엔드포인트를 사용할 수 있습니다.

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/cameras` | 전체 카메라 + 구역 목록 |
| `GET` | `/cameras/{id}/zones` | 특정 카메라 구역 목록 |
| `POST` | `/cameras/{id}/zones` | 구역 전체 교체 |
| `DELETE` | `/cameras/{id}/zones/{zone_id}` | 특정 구역 삭제 |
| `GET` | `/zone-presets` | 저장된 프리셋 목록 |
| `POST` | `/zone-presets` | 새 프리셋 저장 |
| `DELETE` | `/zone-presets/{preset_id}` | 프리셋 삭제 |
| `POST` | `/cameras/{id}/zones/from-preset/{pid}` | 프리셋 적용 |

## EdgeX Foundry 연동

EdgeX UI: `http://localhost:4000`
- Device Center > Device List → `camera-camera_1` 선택 후 실시간 이벤트 확인

EdgeX REST API:
```bash
# 최근 이벤트 조회
curl http://localhost:59880/api/v3/event/all?limit=10

# 특정 카메라 이벤트
curl http://localhost:59880/api/v3/event/device/camera-camera_1?limit=10
```

MQTT 토픽 구조:
```
edgex/events/device/cctv-device-service/CCTV-Camera-Profile/{device}/{resource}
```

상세 내용: `docs/architecture/DEVICE_SERVICE_ARCHITECTURE.md`, `docs/architecture/ASC_RULE_ENGINE.md`

## 주요 모듈

### `config.py` — 중앙화된 설정

- `ModelPaths`: `.engine` (TensorRT) → `.pt` 우선순위로 자동 탐지
- `DetectionConfig`: `device` 필드가 `cpu`, `cuda`, `cuda:0` 등 모두 허용
- `ENV_OVERRIDES`: 환경변수로 모든 설정 재정의 가능

### `src/core/ai/analyzer.py` — 다중 모델 AI 추론

- **사람 모델** (YOLOv8n fallback): 640px 입력, `person_confidence=0.4`
- **포즈 모델** (YOLOv8n-pose): 낙상 감지, 어깨-엉덩이 각도 분석
- **헬멧 모델** (커스텀): 320px, `helmet_confidence=0.7`
- `track(persist=True)`로 프레임 간 객체 ID 유지
- IoU 기반 중복 박스 제거

참고:
- 기존 `src/core/ai_analysis.py` 경로는 제거되었고, 현재 표준 경로는 `src/core/ai/analyzer.py` 입니다.

### `processor.py` — 파이프라인 오케스트레이터

- 카메라 스레드(프레임 획득) ↔ AI 추론 스레드 분리
- 최신 프레임만 유지하는 프레임 큐 (지연 최소화)
- 이벤트 큐 가득 시 로컬 JSON 백업 (손실 방지)
- 연결 실패 카메라 자동 백그라운드 재시도

### `camera_input.py` — RTSP/웹캠 관리

| 환경 | 방식 | 활성화 |
|------|------|--------|
| Windows / Linux | `cv2.CAP_FFMPEG` + TCP transport | 기본 |
| Jetson Orin | GStreamer `nvv4l2decoder` NVDec | `USE_GSTREAMER=1` |

### `zone_drawer.py` — GUI 구역 편집기

`_DisplayGrid`의 디스플레이 루프와 연동. `cameras.json`에 폴리곤 자동 저장.

### `zone_api.py` — 구역 REST API

표준 라이브러리(`http.server`)만 사용하는 경량 HTTP 서버. 데몬 스레드로 동작.

### `action_bridge.py` — Action Layer

MQTT 알람 수신 → 스피커 TCP 전송 + 외부 REST API 호출 + SQLite 이벤트 저장.

## 주요 설정 파라미터

### 감지 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `person_confidence` | 0.4 | 사람 감지 최소 신뢰도 |
| `helmet_confidence` | 0.7 | 헬멧 감지 최소 신뢰도 |
| `pose_confidence` | 0.5 | 포즈 감지 최소 신뢰도 |
| `iou_threshold` | 0.3 | YOLO NMS IoU |
| `fall_angle_threshold` | 45.0 | 낙상 감지 수평 각도 임계값 (도) |
| `fall_height_ratio` | 0.3 | 낙상 머리 높이 비율 |

### 시스템 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `debounce_seconds` | 3.0 | 동일 이벤트 재전송 간격 |
| `queue_max_size` | 500 | 이벤트 큐 최대 크기 |
| `frame_queue_size` | 1 | 카메라당 프레임 큐 크기 |
| `event_retention_hours` | 24 | 이벤트 보관 시간 |

## 데이터셋 수집

```bash
python main.py --cameras cameras.json --collect-dataset --dataset-dir ./my_data
```

수집 구조:
```
collected_data/
├── images/train/   ← 학습 이미지 (80%)
├── images/val/     ← 검증 이미지 (20%)
├── labels/train/   ← YOLO 라벨
└── labels/val/
```

## 모델 재학습

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
cp runs/detect/train/weights/best.pt models/helmet_model_ver0.5.pt
```

## 모델 평가

모델 교체 전에는 `models/model_manifest.json`의 기준과 고정 평가 데이터셋으로
precision/recall/latency를 확인합니다.

먼저 manifest에 기록된 모델 파일이 실제로 존재하는지 확인합니다.

```bash
python scripts/health/check_model_report.py --check-artifacts
```

```bash
python scripts/ops/evaluate_detection.py \
  --model models/helmet_model_ver0.5.onnx \
  --dataset data/eval/helmet \
  --output reports/eval/helmet_model_ver0.5.json \
  --imgsz 320 \
  --conf 0.35 \
  --iou 0.5 \
  --warmup 1 \
  --target-classes helmet,head
```

상세 절차는 `docs/operations/MLOPS_MODEL_EVALUATION.md`를 참고하세요.

## 문제 해결

| 증상 | 해결 방법 |
|------|-----------|
| 헬멧이 감지되지 않음 | `--confidence 0.3`으로 낮추기 |
| 중복 박스 표시 | `iou_threshold`를 0.4로 증가 |
| 카메라 연결 실패 | RTSP URL 및 네트워크 확인 |
| Jetson에서 FFMPEG 오류 | `USE_GSTREAMER=1` 환경변수 추가 |
| `cuda:0` 인식 안 됨 | `DEVICE=cuda:0` 환경변수로 지정 |
| Windows 로그 한글 깨짐 | `chcp 65001` 후 실행, 또는 `PYTHONUTF8=1` |

## 변경 이력

### v1.11.0 (2026-06-08) - 문서/스크립트 구조 정리 및 운영 정리 자동화

- **문서 구조 재분류**
  - API, 운영, 아키텍처, 보고서 문서를 `docs/api`, `docs/operations`, `docs/architecture`, `docs/reports`로 분류
  - README와 주요 보고서의 문서 링크를 새 경로 기준으로 갱신
  - 문서 진입점 `docs/README.md`를 추가해 운영/리뷰 자료를 빠르게 찾을 수 있게 정리

- **스크립트 카테고리 정리**
  - health, smoke, ops, cleanup, convert 기준으로 `scripts/` 하위 구조를 정리
  - 운영 점검 명령을 `scripts/health/*`, `scripts/ops/*`, `scripts/smoke/*` 경로로 통일
  - 런타임 정리 스크립트와 systemd timer 설치 스크립트를 추가

- **런타임 데이터 정리 정책**
  - 루트 `action_events.db`를 `data/runtime/action_events.db` 기준으로 정리
  - 회전 로그 `data/alert_api_events.jsonl.1`을 gzip 백업해 `data/archive/`로 이동
  - appearance crop 보존기간과 JSONL 로그 회전 기준을 README에 반영

- **검증 기록**
  - `bash -n` 기준 shell script 문법 검사를 통과
  - `tests/test_runtime_cleanup_scripts.py`, `tests/test_check_public_api_fd_stability.py` 통과
  - 문서 내 구 `scripts/*` 경로 잔여 패턴 0건 확인

### v1.10.0 (2026-05-26) - 이벤트 포워딩 및 운영 점검 강화

- **Public API 이벤트 전달 경로 정리**
  - `/api/v1/events`, `/api/v1/alerts`, `/api/v1/sensor-readings` 수신 이벤트를 Action Layer `/events`로 포워딩
  - 공통 포워딩 모듈(`src/api/_event_forwarding.py`)로 중복 로직 축소
  - 로컬 API 문서(`src/api/_local_docs.py`)와 `/api/v1/docs/openapi.json` 추가

- **TLV 센서 위험 분류 추가**
  - `src/services/sensor_classifier.py`에서 온도, 기울기, event_code 기준으로 warning/critical 이벤트 생성
  - 센서 payload fixture와 회귀 테스트 추가

- **운영 점검 자동화 보강**
  - 런타임 비밀값 일관성 점검: `scripts/health/check_runtime_secret_consistency.py`
  - Jetson DeepStream 런타임/GStreamer/nvinfer 설정 점검: `scripts/health/check_deepstream_env.py`
  - 현장 운영 점검 래퍼: `scripts/ops/run_operation_check.sh`
  - DeepStream 장시간 안정성 관찰: `scripts/ops/run_deepstream_stability_watch.sh`
  - 운영 체크리스트와 DeepStream 안정성 기록 문서 추가

### v1.9.0 (2026-05-22) - 성능 최적화 및 안정성 강화

- **DeepStream 파이프라인 안정성 개선**
  - `_restart_pipeline` finally 블록에서 `_pipeline_restart_pending = False` 리셋 누락 버그 수정
    → 재시작 완료 후에도 API가 계속 "재시작 중"을 반환하던 문제 해결
  - `cameras.json` 동시 R/W 레이스 컨디션 수정: `_cameras_json_lock` + `os.replace` 원자적 쓰기 적용
  - `docker-compose.yml`의 `cctv-ai-engine` cameras.json 마운트에서 `read_only: true` 제거
    (모델 설정 저장 API 사용 시 컨테이너 내 쓰기가 실패하던 문제)

- **GStreamer 메인 루프 블로킹 제거 (고우선순위)**
  - InsightFace CNN 얼굴 인식 및 AppearancePipeline을 GLib 메인 루프 스레드에서 동기 호출하던 문제 해결
  - `_face_work_queue (maxsize=8)` + `ds-face-worker` 데몬 스레드 도입으로 완전 비동기화
  - 30fps 기준 per-frame 블로킹 ~200ms 제거 → 파이프라인 프레임 드롭 방지

- **메모리 누수 수정**
  - `_last_event_emit_at` dict: nvtracker object_id 기반 키가 무한 증가하는 문제
    → 1000 프레임마다 만료 키 일괄 정리 (쿨다운 10배 초과 항목 제거)
  - `_pphuman_attrs_by_frame` dead code 제거 (선언만 있고 미사용)

- **성능 최적화**
  - `_on_pad_probe` 내 `pad_to_camera` dict 매 프레임 재생성 제거
    → `_pad_to_camera` 캐시 필드 도입, `_build_pipeline` 완료 시 1회 갱신
  - `_on_preview_sample` BGRx→BGR 변환: `reshape[:,:,:3].copy()` 2단계 →
    `np.ascontiguousarray` 단일 패스로 통합 (30fps 기준 메모리 복사 1회 제거)
  - `import tempfile` 함수 내부 위치에서 모듈 최상단으로 이동

- **SyntheticObjectIdAssigner O(n²) IoU 개선**
  - `_find_best_track()` 에 AABB 비중첩 사전 검사 추가 (IoU 계산보다 10배 이상 저렴)
  - IoU > 0.85 조기 종료 추가
  - 탐지 50개 × 트랙 50개 기준 최대 95% 계산 감소

- **JSONL 이벤트 API 이중 스캔 제거**
  - `GET /api/v1/events`에서 `camera_id` 필터를 전체 수집 후 2nd pass로 처리하던 로직 수정
  - 파싱 루프 내 즉시 필터링으로 통합 → 최대 5000줄 기준 메모리 사용량 절반 감소

- **카메라 모델 설정 API 응답 개선**
  - `POST /api/v1/cameras/{id}/model_settings` 응답에 `pipeline_restarting` 필드 추가
  - 프론트엔드(`public-demo.html`)에서 파이프라인 재시작 여부 감지 후 32초 후 자동 iframe 갱신

### v1.8.0 (2026-05-06) - 실사용 시연 UI 및 운영 데모 준비

- **시연용 Public API 대시보드 추가**
  - `web/public-demo.html` 추가
  - Public API 상태, readiness, 카메라 목록, metrics 상태를 한 화면에서 확인
  - 낙상 감지, 헬멧 미착용, 위험구역 침입 이벤트를 UI 버튼으로 전송 가능
  - Swagger(`http://localhost:9000/docs`)와 Grafana(`http://localhost:3001`) 바로가기 제공
  - `/api/v1/events` 조회가 실패해도 시연 중 전송한 이벤트를 화면에 남기는 fallback 처리 추가

- **카메라 스트림 시연 지원**
  - `Stream API` 기본 주소를 `http://localhost:8769`로 정리
  - `web/public-demo.html`에서 카메라 목록을 기반으로 MJPEG 화면 자동 표시
  - 스트림 연결 실패 시 `8769` 포트와 Stream API 상태를 확인하도록 UI 메시지 개선
  - `docker-compose.yml`의 `cctv-ai-engine`에 `STREAM_API_ENABLED`, `STREAM_PORT`, `8769` 포트 publish 설정 추가

- **README 운영/시연 문서 보강**
  - 프로젝트 구조에 `web/` 대시보드 디렉터리 설명 추가
  - 포트 표에 `8769` Stream API 항목 추가
  - `실사용 시연 UI` 섹션 추가
  - 시연 전 점검 명령, Stream API 장애 확인 순서, 컨테이너 재생성 명령 정리
  - 내일 발표용 설명 순서 추가: 전체 구조 → health/readiness → 카메라 화면 → 이벤트 전송 → Grafana/Swagger 확인
  - README 내 흩어져 있던 CLI 옵션 표 조각을 `주요 CLI 옵션` 섹션으로 정리

### v1.7.0 (2026-03-09) - 크로스 플랫폼 지원 (Windows + Jetson Orin)

- **Jetson Orin 지원**
  - `camera_input.py`: `USE_GSTREAMER=1` 환경변수로 GStreamer NVDec 하드웨어 디코딩 선택
  - `config.py` `MODEL_CANDIDATES`: `.engine` (TensorRT) 경로를 `.pt` 앞에 우선 배치 → Jetson에서 자동 가속
  - `DetectionConfig.device` 검증 완화: `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"` 모두 허용

- **Windows 로그 한글 깨짐 수정**
  - `main.py`: `SetConsoleOutputCP(65001)` + `SetConsoleCP(65001)` + `reconfigure(encoding='utf-8')` 조합

### v1.6.0 (2026-03-09) - 위험 구역 GUI 완성

- **zone_drawer.py 기능 추가**
  - 재시작 후 저장된 구역 폴리곤 화면 자동 복원 (그리드 좌표 변환)
  - 마우스 hover로 구역 주황색 하이라이트 + `[x=delete]` 라벨 표시
  - `x` 키로 hover 중인 구역 즉시 삭제 → `cameras.json` 자동 저장
  - `_zone_counter` 초기화를 `cameras.json` 기존 최대 번호 기준으로 변경 (중복 방지)
  - 이중 렌더링 제거: `draw_zones()` 호출 제거, `overlay()`로 시각화 일원화
  - Zone 이름 라벨 중복 표시 버그 수정

- **Zone 프리셋 시스템 추가** (`zone_presets.py`)
  - 구역 폴리곤을 이름 있는 프리셋으로 저장·재사용

- **Zone REST API 확장** (`zone_api.py`)
  - 프리셋 관련 엔드포인트 추가: `/zone-presets`, `/cameras/{id}/zones/from-preset/{pid}`
  - `--api-port`, `--zone-presets` CLI 옵션 추가

### v1.5.0 (2026-02-12) - 모델 업그레이드 및 감지 파라미터 최적화

- 사람 감지 모델: YOLOv8n → YOLOv8s, 입력 해상도 800px
- `person_confidence` 0.5 → 0.4, 누적 감지 히스토리 5 → 3
- 디스플레이 파이프라인과 서버 전송 파이프라인 분리

### v1.4.0 (2026-02-05) - EdgeX Foundry v3 통합 완성

- EdgeX Core Metadata / Core Data MQTT 연동
- 표준 EdgeX v3 envelope 형식 (requestId, correlationId)
- Action Layer (speaker-bridge) 구현: TCP 스피커 + 외부 REST API + SQLite

### v1.3.0 (2026-01-30) - 코드베이스 전면 개선

- 멀티스레드 아키텍처 재설계 (카메라 / AI 스레드 분리)
- `predict()` → `track(persist=True)` 전환
- 이벤트 손실 방지 로컬 JSON 백업 시스템

### v1.2.0 (2026-01-07) - 프로젝트 구조 개선

- `src/` 모듈화 패키지 구조 도입
- `config.py` 중앙화된 설정 관리

### v1.0.0 (2025-11-12) - 초기 버전

- YOLOv8 기반 헬멧·낙상·구역 감지 초기 구현

## 라이선스

MIT License

## 문의

- GitHub: https://github.com/dih5156-lab/CCTV-project
- Issues: https://github.com/dih5156-lab/CCTV-project/issues
