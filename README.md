# CCTV 헬멧 착용 및 낙상 감지 시스템

YOLOv8 기반 실시간 안전 관리 시스템으로, 다중 카메라 환경에서 헬멧 착용 여부, 낙상 사고, 위험 구역 침입을 자동 감지합니다.
Windows PC와 NVIDIA Jetson Orin 모두 동작합니다.

## 주요 기능

- **헬멧 착용 감지**: 커스텀 YOLOv8 모델로 헬멧 착용/미착용 실시간 탐지
- **낙상 감지**: YOLOv8-pose 모델 기반 사람 자세 분석으로 낙상 사고 탐지
- **다중 카메라**: RTSP/웹캠 동시 처리 및 자동 재연결
- **위험 구역 관리**: 실시간 폴리곤 그리기·저장·삭제 (GUI 인터랙션 지원)
- **Zone API**: REST API로 외부에서 구역 설정 조회·수정
- **EdgeX Foundry 연동**: MQTT 기반 표준 EdgeX v3 이벤트 발행
- **Action Layer**: 스피커 알람·외부 API 호출·SQLite 이벤트 저장
- **데이터셋 수집**: YOLO 형식 자동 라벨링 및 학습 데이터 생성
- **Jetson 가속**: GStreamer NVDec 하드웨어 디코딩 + TensorRT `.engine` 모델 자동 인식

## 프로젝트 구조

```
CCTV-project/
├── src/
│   ├── config/
│   │   └── config.py              # 중앙화된 설정 (ENV 오버라이드 지원)
│   ├── core/
│   │   ├── events.py              # 이벤트 타입 정의
│   │   ├── event_filters.py       # 누적 감지 필터 / 트랙 관리
│   │   ├── ai_analysis.py         # 다중 YOLO 모델 추론
│   │   └── processor.py           # 비디오 파이프라인 오케스트레이터
│   ├── utils/
│   │   ├── camera_input.py        # RTSP/웹캠 연결 (GStreamer 지원)
│   │   ├── geometry.py            # 좌표 변환 유틸리티
│   │   ├── visualizer.py          # 감지 결과 시각화
│   │   ├── zone_detection.py      # 폴리곤 구역 침입 판정
│   │   ├── zone_drawer.py         # GUI 구역 그리기 / 삭제
│   │   ├── zone_presets.py        # 구역 프리셋 저장소
│   │   └── dataset_collector.py   # YOLO 형식 데이터셋 수집
│   ├── services/
│   │   ├── zone_api.py            # 위험구역 REST API 서버
│   │   └── action_bridge.py       # 스피커·외부 API·DB 액션 레이어
│   ├── protocols/
│   │   ├── mqtt.py                # MQTT 이벤트 발행
│   │   ├── http.py                # HTTP 이벤트 전송
│   │   └── rest.py                # REST 클라이언트 공통
│   ├── edgex/
│   │   ├── device_service.py      # EdgeX Foundry v3 디바이스 서비스
│   │   └── adapter_service.py     # EdgeX 어댑터 (AI→EdgeX 브릿지)
│   └── devices/
│       ├── speaker.py             # TCP 스피커 제어
│       ├── signboard.py           # 전광판 제어
│       └── sensor.py              # 센서 추상화
├── main.py                        # 메인 진입점
├── run_edgex_adapter.py           # EdgeX 어댑터 단독 실행
├── run_action_bridge.py           # Action Layer 단독 실행
├── run_kuiper_rules.py            # Kuiper 룰 배포
├── run_alert_api.py               # Alert REST API 서버
├── cameras.json                   # 카메라 목록 및 구역 설정
├── zones_config.json              # 전역 위험 구역 설정
├── Dockerfile                     # 컨테이너 빌드 (x86)
├── requirements.txt               # Python 의존성
└── docs/
    ├── ACTION_LAYER_SPEAKER_BRIDGE.md
    ├── ASC_RULE_ENGINE.md
    ├── DEVICE_SERVICE_ARCHITECTURE.md
    └── KUIPER_RULE_ENGINE.md
```

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

### 4. 모델 파일 준비

`models/` 폴더에 아래 파일을 배치합니다. 없으면 YOLOv8 공식 모델이 자동 다운로드됩니다.

```
models/
├── helmet_model_ver0.5.pt   # 헬멧 감지 (커스텀)
├── yolov8n-pose.pt          # 낙상 감지 (포즈)
└── yolov8n.pt               # 사람 감지
```

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
python run_edgex_adapter.py \
  --ai-mqtt-broker localhost --ai-topic-prefix cctv/ai/events \
  --edgex-metadata-url http://localhost:59881 \
  --edgex-data-url http://localhost:59880

# 3) Kuiper 룰 배포 (침입 필터링 / 고신뢰 라우팅)
python run_kuiper_rules.py \
  --kuiper-api http://localhost:9081 \
  --intrusion-confidence 0.7 --critical-confidence 0.9

# 4) Action Layer (스피커 알람 + 외부 API + DB)
python run_action_bridge.py \
  --mqtt-broker localhost \
  --external-api-url http://localhost:8000/api/alerts \
  --speaker-host 192.168.88.92 --speaker-port 5000 \
  --speaker-user admin --speaker-password YOUR_PASSWORD
```

### Docker (EdgeX 통합)

#### Compose 파일 구성

| 파일 | 용도 |
|------|------|
| `docker-compose.yml` | **기본 구성** — AI 엔진이 같은 PC에서 실행될 때 사용 |
| `docker-compose.server.yml` | **서버 모드 override** — Jetson Orin이 AI를 담당하고 Windows PC가 EdgeX 서버 역할을 할 때 기본 파일 위에 덧씌움 |

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

#### 서버 모드 실행 (Jetson Orin + Windows PC 분리)

```bash
# Windows PC (EdgeX 서버 + Action Layer)
docker compose -f docker-compose.yml -f docker-compose.server.yml up -d --build

# Jetson Orin (AI 엔진만 실행, 이 PC의 IP를 MQTT 브로커로 지정)
USE_GSTREAMER=1 DEVICE=cuda:0 python main.py \
  --cameras cameras.json \
  --mqtt-broker <Windows_PC_IP> \   # ipconfig 로 확인 (예: 192.168.0.10)
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
docker compose logs -f edgex-app-rules-engine

# 서비스 재시작
docker compose restart cctv-action-layer
```

#### 필수 환경변수 (.env 파일)

`.env.example`을 복사하여 `.env`를 만들고 아래 값을 설정합니다.

| 변수 | 설명 | 예시 |
|------|------|------|
| `SPEAKER_HOST` | 스피커 IP | `192.168.0.100` |
| `SPEAKER_USER` | 스피커 인증 사용자명 | `admin` |
| `SPEAKER_PASSWORD` | 스피커 인증 비밀번호 | _(보안상 직접 입력)_ |
| `SIGNBOARD_HOST` | 전광판 IP (없으면 비활성화) | `192.168.0.101` |
| `SIREN_HOST` | 경광등 IP (없으면 비활성화) | `192.168.0.102` |
| `ACTION_ALARM_COOLDOWN` | 알람 재발生 억제 간격(초) | `10` |

> **주의**: `.env` 파일은 `.gitignore`에 포함되어 Git에 커밋되지 않습니다.
> 실제 비밀번호를 `.env.example`에 직접 쓰지 마세요.

필수 환경값:
- `ACTION_SPEAKER_PASSWORD` — 스피커 인증 비밀번호
- `EDGEX_METADATA_URL`, `EDGEX_DATA_URL` — EdgeX 외부 연결 시

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

상세 내용: `docs/DEVICE_SERVICE_ARCHITECTURE.md`, `docs/ASC_RULE_ENGINE.md`

## 주요 모듈

### `config.py` — 중앙화된 설정

- `ModelPaths`: `.engine` (TensorRT) → `.pt` 우선순위로 자동 탐지
- `DetectionConfig`: `device` 필드가 `cpu`, `cuda`, `cuda:0` 등 모두 허용
- `ENV_OVERRIDES`: 환경변수로 모든 설정 재정의 가능

### `ai_analysis.py` — 다중 모델 AI 추론

- **사람 모델** (YOLOv8s): 800px 입력, `person_confidence=0.4`
- **포즈 모델** (YOLOv8n-pose): 낙상 감지, 어깨-엉덩이 각도 분석
- **헬멧 모델** (커스텀): 640px, `helmet_confidence=0.7`
- `track(persist=True)`로 프레임 간 객체 ID 유지
- IoU 기반 중복 박스 제거

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
