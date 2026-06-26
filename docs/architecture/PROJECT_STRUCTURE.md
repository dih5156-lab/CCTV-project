# CCTV-project 프로젝트 구조

## 결론

이 프로젝트는 CCTV 영상 AI, AIoT 센서 파서, EdgeX/Kuiper 룰 엔진, Action Layer, Public API, Grafana/시연 UI를 한 Compose 환경에서 묶은 엣지 관제 플랫폼입니다.

현재 구조는 아래 흐름으로 이해하면 가장 쉽습니다.

```text
카메라/RTSP/비디오
  -> AI Engine(VideoProcessor 또는 DeepStreamProcessor)
  -> AI 이벤트 생성
  -> MQTT / Alert API
  -> EdgeX / Kuiper Rule / Action Layer
  -> Public API / Grafana / web 시연 UI / 외부 시스템
```

내일 시연 기준으로는 아래 4개를 중심으로 설명하면 됩니다.

```text
AI Engine        영상 입력과 AI 추론
Action Layer     알람, 외부 전송, 승인/거절, 이벤트 저장
Public API       서버팀/대시보드가 붙는 표준 API
web UI/Grafana   실사용 화면과 운영 모니터링
```

## 최상위 구조

```text
CCTV-project/
├── main.py                         # CCTV AI 엔진 실행 진입점
├── run_external_ingest.py           # 외부 MQTT/NC 수신 실행 진입점
├── runners/                        # 서비스별 단독 실행 진입점
├── src/                            # 핵심 애플리케이션 코드
├── parser-python/                  # AIoT TLV 센서 파서 서비스
├── web/                            # 브라우저 기반 관제/시연 UI
├── docs/                           # 설계, 운영, API, 배포 문서
├── scripts/                        # 점검, smoke test, 모델 변환/평가 스크립트
├── deploy/systemd/                 # 운영 정리용 systemd service/timer 템플릿
├── tests/                          # pytest 테스트
├── config/                         # DeepStream, 라벨, 이벤트 타입, 외형 분석 설정
├── models/                         # YOLO, pose, helmet, PP-Human, TensorRT 캐시
├── data/                           # SQLite DB, crop 이미지, 런타임 데이터
├── external/                       # 외부 학습 저장소/공개 데이터셋 연결 지점
├── falldata/                       # 공공 낙상 데이터 패키지, git 제외
├── known_faces/                    # 얼굴 인식용 샘플/등록 이미지
├── edgex/                          # EdgeX device profile, ASC 설정, 등록 스크립트
├── kuiper/                         # eKuiper 룰 파일
├── monitoring/                     # Prometheus/Grafana 설정
├── mosquitto/                      # MQTT broker 설정
├── tooling/                        # 앱 실행과 분리된 개발 도구 원본 자료
├── Dockerfile                      # 일반 AI/API/Action/Parser 멀티 타깃 이미지
├── Dockerfile.jetson               # Jetson/DeepStream 이미지
├── docker-compose.yml              # 일반 Docker/EdgeX 통합 실행
├── docker-compose.jetson.yml       # Jetson/DeepStream 운영 실행
├── cameras.example.json            # 카메라 설정 예시
├── cameras.json                    # 실제 카메라 설정
├── zones_config.json               # 구역 설정
├── known_faces.example.json        # 얼굴 등록 설정 예시
├── known_faces.json                # 실제 얼굴 등록 설정
├── COMMANDS.md                     # 실행 명령 모음
└── README.md                       # 프로젝트 소개/실행/시연 안내
```

제외해도 되는 실행 산출물:

```text
.venv/
.venv-falldata/
.venv-mediapipe/
.venv-par/
.pytest_cache/
__pycache__/
tmp_test_dirs/
falldata/
data/runtime/appearance_crops/
data/crops/
```

## 실행 진입점

### `main.py`

AI Engine의 기본 진입점입니다.

역할:

- OpenCV/콘솔/Jetson 관련 런타임 환경 초기화
- CLI 인자와 환경변수 설정 로드
- `cameras.json`, 웹캠, 비디오 파일 입력 구성
- `src/bootstrap/runtime.py`를 통해 실제 프로세서 실행

실행 모드:

```text
일반 개발/서버
  -> OpenCV + Ultralytics 기반 VideoProcessor

Jetson 운영
  -> USE_DEEPSTREAM=1일 때 DeepStreamProcessor 우선 사용
  -> 실패 시 VideoProcessor fallback
```

### `runners/`

Compose에서 각 기능을 독립 서비스로 띄우기 위한 실행 파일입니다.

```text
runners/
├── run_public_api.py               # Public API, 기본 포트 9000
├── run_action_bridge.py            # Action Layer, 기본 포트 8080
├── run_alert_api.py                # Alert API, 기본 포트 8000
├── run_edgex_adapter.py            # AI MQTT 이벤트 -> EdgeX 이벤트 변환
├── run_kuiper_rules.py             # eKuiper 룰 등록
├── run_sensor_rule_bridge.py       # 센서 룰 결과 MQTT 브리지
└── _shared.py                      # runner 공통 sys.path/logging 유틸
```

## `src/` 핵심 구조

```text
src/
├── api/                            # FastAPI Public API
├── bootstrap/                      # CLI, 런타임 초기화, 프로세서 생성
├── config/                         # 중앙 설정과 이벤트 타입 매핑
├── core/                           # 영상 처리, AI 추론, 이벤트 생성
├── devices/                        # 스피커, 전광판, 경광등, 센서 장치 모델
├── edgex/                          # EdgeX 연동 서비스
├── protocols/                      # MQTT/HTTP/REST/TLV 통신 계층
├── services/                       # ActionBridge, 관리 API, 검색/로그 서비스
├── storage/                        # SQLite 저장소
├── utils/                          # 카메라 입력, geometry, zone, visualizer
└── canonical_event.py              # 표준 이벤트 변환/정규화
```

### `src/bootstrap/`

```text
src/bootstrap/
├── cli.py                          # CLI 옵션 정의
└── runtime.py                      # 런타임 구성, 카메라 로드, 프로세서 실행
```

중요한 책임:

- `AppConfig` 생성
- 카메라 설정 로드
- `VideoProcessor` 또는 `DeepStreamProcessor` 선택
- Zone API, Camera Model API, Face API, Stream API 시작
- display 모드 또는 headless loop 실행

### `src/config/`

```text
src/config/
├── config.py                       # AppConfig, 모델/MQTT/EdgeX/Action 설정
└── event_type_map.py               # 이벤트 타입 매핑 로더
```

운영에서 자주 쓰는 환경변수:

```text
DEVICE
USE_DEEPSTREAM
USE_GSTREAMER
HELMET_MODEL_PATH
PERSON_MODEL_PATH
POSE_MODEL_PATH
MQTT_BROKER
MQTT_PORT
APPEARANCE_ENABLED
STREAM_API_ENABLED
STREAM_PORT
PUBLIC_API_KEY
INTERNAL_SERVICE_TOKEN
```

### `src/core/`

영상 처리와 이벤트 생성의 중심입니다.

```text
src/core/
├── base_processor.py               # Processor 공통 인터페이스
├── processor.py                    # OpenCV + YOLO 기반 처리기
├── deepstream_processor.py         # NVIDIA DeepStream 기반 처리기
├── events.py                       # EventType, DetectionEvent
├── event_filters.py                # 이벤트 필터/디바운스 보조
├── sensor_detection.py             # 센서 기반 이벤트 판정
├── _adaptive_governor.py           # 성능/부하 적응 제어
├── _camera_registry.py             # 카메라 등록, 스레드, 재연결
├── _deepstream_event_factory.py    # DeepStream event 생성
├── _deepstream_face_context.py     # DeepStream 얼굴 컨텍스트
├── _display_event_mapper.py        # 표시용 이벤트 매핑
├── _display_grid.py                # 다중 카메라 화면 표시
├── _event_context.py               # 이벤트 생성 컨텍스트
├── _event_publish.py               # MQTT/외부 발행 헬퍼
├── _face_snapshot.py               # 얼굴/이벤트 스냅샷
├── _inference_pipeline.py          # AI 추론과 이벤트 큐 적재
├── _synthetic_object_ids.py        # object_id 보정
├── _yolo_postprocess.py            # YOLO 후처리
└── ai/                             # 실제 AI 모델 파이프라인
```

핵심 흐름:

```text
CameraInput
  -> Processor frame queue
  -> AIAnalyzer 또는 DeepStream probe
  -> DetectionEvent
  -> zone/event filter/debounce
  -> MQTT publish
```

### `src/core/ai/`

```text
src/core/ai/
├── analyzer.py                     # AIAnalyzer 오케스트레이터
├── _constants.py                   # AI 계층 상수
├── _object_detection_pipeline.py   # 사람/헬멧/객체 탐지
├── _object_tracker.py              # track/object_id 관리
├── _fall_detector.py               # 낙상 판정
├── _falldata_aux.py                # 공공 falldata RF 모델 기반 낙상 보조 검증
├── _face_recognition_pipeline.py   # 얼굴 인식
├── _appearance_pipeline.py         # 외형 속성 분석 흐름
├── _appearance_analyzer.py         # HSV/PP-Human 분석
├── _attribute_backend.py           # 속성 백엔드 인터페이스
├── _attribute_backends.py          # backend 구현
├── _attribute_runtimes.py          # onnxruntime/paddle runtime
└── _yolo_helpers.py                # YOLO 결과 추출 유틸
```

지원하는 주요 AI 기능:

- 사람 감지
- 헬멧/머리 감지
- 낙상 감지
- 낙상 보조 검증: pose 후보를 MediaPipe feature + RF 모델로 shadow/confirm 확인
- 위험 구역 침입/체류/객체 감지
- 얼굴 인식
- 외형 속성 분석과 검색

## Public API 구조

```text
src/api/
├── app.py                          # FastAPI 앱, middleware, router 등록
├── _action_proxy.py                # Action Layer 프록시
├── dependencies/
│   ├── _settings.py                # URL/path/env 설정
│   ├── auth.py                     # X-API-Key 인증
│   └── rate_limit.py               # slowapi rate limit
├── schemas/
│   ├── common.py                   # BaseResponse/PaginatedResponse
│   ├── event.py                    # 이벤트/알림 schema
│   └── site.py                     # site schema
└── v1/
    ├── health.py                   # /api/v1/health, /readiness
    ├── alerts.py                   # /api/v1/alerts
    ├── events.py                   # /api/v1/events
    ├── cameras.py                  # /api/v1/cameras
    ├── sites.py                    # /api/v1/sites
    ├── control.py                  # /api/v1/control/*
    ├── event_reviews.py            # /api/v1/event-reviews*
    ├── sensor_readings.py          # /api/v1/sensor-readings
    ├── appearances.py              # /api/v1/appearances*
    ├── search.py                   # /api/v1/search*
    └── metrics.py                  # /api/v1/metrics
```

Public API 특징:

- 기본 포트: `9000`
- Swagger: `http://localhost:9000/docs`
- Health: `GET /api/v1/health`
- Readiness: `GET /api/v1/readiness`
- 이벤트 검수: `POST /api/v1/event-reviews`, `GET /api/v1/event-reviews/summary`
- 응답 형식: `{ success, data, error, timestamp }` 또는 pagination wrapper
- 운영 인증: `X-API-Key`
- CORS: `CORS_ORIGINS`

## 서비스 계층

```text
src/services/
├── action_bridge.py                # 알람/외부전송/승인/SQLite 저장
├── _action_bridge_support.py       # ActionBridge 지원 타입/저장소/실행기
├── appearance_log.py               # 외형 기록 SQLite
├── appearance_conditions.py        # 외형 검색 조건 저장
├── appearance_status.py            # 외형 검색 준비 상태 계산
├── camera_model_api.py             # 카메라별 모델 on/off API
├── cctv_metrics.py                 # Prometheus metric
├── event_review.py                 # 이벤트 맞음/오탐/애매함 검수 저장
├── external_ingest.py              # 외부 MQTT/NC ingest
├── face_api.py                     # 얼굴 등록/삭제/조회 API
├── sensor_classifier.py            # TLV 센서 위험 상태 분류
├── sensor_bridge.py                # 센서 이벤트 bridge
├── sensor_rule_bridge.py           # 센서 룰 bridge
├── stream_api.py                   # MJPEG 카메라 스트림 API
└── zone_api.py                     # 위험구역/프리셋 REST API
```

운영에서 중요한 내부 API:

```text
Zone API          /cameras, /cameras/{id}/zones
Camera Model API  /cameras/{id}/models
Face API          /faces
Stream API        /health, /cameras, /stream/{camera_id}
```

Stream API는 현재 시연 UI에서 카메라 화면을 보여줄 때 사용합니다.

```text
http://localhost:8769/health
http://localhost:8769/cameras
http://localhost:8769/stream/camera_1
```

## EdgeX / Device / Protocol 계층

### `src/edgex/`

```text
src/edgex/
├── adapter_service.py              # AI MQTT 이벤트 -> EdgeX 이벤트
├── device_service.py               # EdgeX device service
├── _http_mixin.py                  # EdgeX HTTP helper
├── _outbox_mixin.py                # 발행 실패 outbox
├── _payload_mixin.py               # payload 변환
└── _publisher_mixin.py             # publish helper
```

### `src/devices/`

```text
src/devices/
├── sensor_device.py                # 센서 도메인 모델
├── siren.py                        # 경광등/사이렌
├── speaker.py                      # TCP 스피커
└── signboard.py                    # 전광판
```

### `src/protocols/`

```text
src/protocols/
├── mqtt_publisher.py               # MQTT publish
├── mqtt_subscriber.py              # MQTT subscribe
├── _mqtt_factory.py                # MQTT client factory
├── http.py                         # HTTP forwarding
├── rest.py                         # REST event receiver
└── tlv_decoder.py                  # TLV decode
```

## 유틸과 저장소

### `src/utils/`

```text
src/utils/
├── camera_input.py                 # RTSP/webcam/file 입력, GStreamer 지원
├── dataset_collector.py            # YOLO 데이터셋 수집
├── face_recognition.py             # 얼굴 인식 유틸
├── geometry.py                     # bbox/좌표 계산
├── visualizer.py                   # 화면 overlay
├── zone_detection.py               # zone 판정
├── zone_drawer.py                  # GUI 구역 편집
└── zone_presets.py                 # zone preset 저장
```

### `src/storage/`

```text
src/storage/
└── sqlite.py                       # SQLite 저장소 공통 유틸
```

## `parser-python/` AIoT 센서 파서

AIoT 센서의 TLV payload를 파싱하고 DB/MQTT/EdgeX로 전달하는 별도 서비스입니다.

```text
parser-python/
├── main.py                         # parser 서비스 진입점
├── live_receiver.py                # live 수신 도구
├── GO_PYTHON_COMPARISON.md         # Go/Python 비교 문서
├── config/
│   ├── config.py                   # parser 설정
│   └── validation.py               # 설정 검증
├── tlv/
│   ├── parser.py                   # TLV parser
│   ├── transformer_v0.py
│   └── transformer_v1.py
├── mqtt/
│   ├── manager.py                  # MQTT manager
│   ├── edgex_forwarder.py          # EdgeX forward
│   ├── classifier.py
│   ├── base_publisher.py
│   └── interfaces.py
├── database/
│   ├── connection.py
│   ├── models.py
│   ├── queries.py
│   ├── processor.py
│   └── edgex_outbox.py
├── service/
│   ├── sensor_service.py
│   ├── event_service.py
│   └── device_info_service.py
├── batch/
│   ├── manager.py
│   └── devices_batch.py
└── tests/
```

센서 흐름:

```text
AIoT sensor payload
  -> TLV parser/transformer
  -> DB 저장
  -> EdgeX/MQTT forward
  -> Kuiper rule 또는 SensorRuleBridge
  -> Action Layer
```

## UI와 운영 화면

```text
web/
├── edgex-status.html               # EdgeX 상태 확인용 경량 UI
└── public-demo.html                # Public API + Stream API 시연 UI
```

현재 시연에서는 `web/public-demo.html`을 우선 사용합니다.

시연 UI가 보는 주소:

```text
Public API   http://localhost:9000
Stream API   http://localhost:8769
Grafana      http://localhost:3001
Swagger      http://localhost:9000/docs
```

역할:

- 서비스 상태 확인
- readiness 확인
- 카메라 목록 표시
- MJPEG 카메라 화면 표시
- 낙상/헬멧 미착용/위험구역 침입 이벤트 전송
- Grafana/Swagger로 이동

## 설정, 모델, 데이터

### `config/`

```text
config/
├── event_type_map.json
├── appearance_pa100k_labels.json
├── appearance_pphuman_labels.example.json
├── sensor_devices.example.json
├── sensor_devices.json
├── nginx/
│   ├── edgex-status.conf
│   ├── public-demo.conf
│   └── public-demo.conf.template
└── deepstream/
    ├── config_infer_primary.txt
    ├── config_infer_helmet.txt
    ├── config_infer_pa100k.txt
    ├── config_infer_pphuman.txt
    ├── config_tracker.txt
    ├── config_tracker_nvdcf.yml
    ├── config_streammux.txt
    ├── labels.txt
    └── labels_helmet.txt
```

### `models/`

```text
models/
├── helmet_model.pt
├── helmet_model_ver0.5.pt
├── yolov8n.pt
├── yolov8n-pose.pt
├── yolov8m-pose.pt
├── pa100k_resnet50_attr.onnx
├── pa100k_resnet50_attr.engine
├── model_manifest.json
├── pphuman_attribute_src/
└── trt_cache/
```

### `data/`

```text
data/
├── runtime/
│   ├── appearances.db
│   ├── action_events.db
│   ├── action_http_outbox.db
│   ├── event_outbox.db
│   ├── mqtt_event_outbox.db
│   ├── appearance_crops/
│   └── face_snapshots/
├── logs/
│   ├── alert_api_events.jsonl
│   └── sensor_readings.jsonl
└── insightface/
```

`data/`는 런타임 산출물이 많으므로, 문서/커밋에 실제 운영 이미지나 민감 데이터가 들어가지 않도록 주의합니다.

## EdgeX / Kuiper / Monitoring

### `edgex/`

```text
edgex/
├── device-profiles/
│   ├── aiot-t34950-profile.yaml
│   ├── aiot-t34955-profile.yaml
│   ├── aiot-t34957-profile.yaml
│   └── aiot-t34958-profile.yaml
├── asc/
│   ├── aiot-external-http/configuration.yaml
│   ├── cctv-external-http/configuration.yaml
│   ├── cctv_asc.env
│   └── cctv_asc.env.example
└── register_aiot_devices.py
```

### `kuiper/`

```text
kuiper/rules/
├── aiot_sensor_rules.json
└── cctv_intrusion_rules.json
```

### `monitoring/`

```text
monitoring/
├── prometheus.yml
└── grafana/provisioning/
    ├── dashboards/
    └── datasources/
```

### `mosquitto/`

```text
mosquitto/
└── mosquitto.conf
```

## Docker Compose 서비스

### 일반 Compose: `docker-compose.yml`

주요 서비스:

```text
edgex-mqtt-broker                 MQTT broker
edgex-core-consul                 EdgeX config/registry
edgex-redis                       EdgeX DB/message bus
edgex-core-data                   EdgeX Core Data
edgex-core-metadata               EdgeX Core Metadata
edgex-device-rest                 EdgeX Device REST
edgex-kuiper                      eKuiper rule engine

cctv-ai-engine                    영상 AI 처리
cctv-edgex-adapter                AI 이벤트 -> EdgeX 변환
cctv-kuiper-rule-loader           Kuiper 룰 등록
cctv-alert-api                    alert/sensor JSONL 수신
cctv-action-layer                 알람/외부전송/승인/SQLite 저장
cctv-public-api                   Public API
aiot-parser                       AIoT TLV parser
aiot-parser-db                    parser PostgreSQL

cctv-prometheus                   Prometheus
cctv-grafana                      Grafana
```

현재 일반 compose의 `cctv-ai-engine`에는 시연을 위해 Stream API 포트가 열려 있습니다.

```text
STREAM_API_ENABLED=1
STREAM_PORT=8769
8769:8769
```

### Jetson Compose: `docker-compose.jetson.yml`

Jetson/DeepStream 운영용 compose입니다.

특징:

- `Dockerfile.jetson`
- `runtime: nvidia`
- `USE_DEEPSTREAM=1`
- `USE_GSTREAMER=1`
- TensorRT/DeepStream/GStreamer 경로 설정
- Stream API 기본 포트 `8769`

## 스크립트 구조

```text
scripts/
├── cleanup/                        # 런타임 로그/crop 정리
├── convert/                        # ONNX/TensorRT/export 변환
├── datasets/                       # 공개 데이터셋 점검/manifest/feature 변환
├── dev/                            # push 전 로컬 검사
├── health/                         # 배포/런타임/모델 readiness 점검
├── ops/                            # 운영 점검, 장비 확인, 평가
└── smoke/                          # 배포/데이터 흐름 smoke test
```

falldata 관련 주요 명령:

```text
scripts/datasets/check_falldata_package.py
scripts/datasets/extract_falldata_mediapipe_features.py
scripts/datasets/smoke_falldata_video_model.py
scripts/health/check_falldata_aux.py
```

## 주요 데이터 흐름

### 영상 AI 이벤트

```text
cameras.json / RTSP / webcam
  -> main.py
  -> src/bootstrap/runtime.py
  -> VideoProcessor 또는 DeepStreamProcessor
  -> AIAnalyzer 또는 DeepStream probe
  -> pose 낙상 후보 생성
  -> 선택: falldata 보조 검증 shadow/confirm
  -> DetectionEvent
  -> event filter / zone detection / debounce
  -> MQTT cctv/ai/events/...
```

### 룰과 알람

```text
cctv/ai/events/... MQTT
  -> eKuiper rules
  -> cctv/rules/... MQTT
  -> ActionBridge
  -> speaker/signboard/siren
  -> external HTTP
  -> SQLite action_events.db
```

### Public API와 시연 UI

```text
web/public-demo.html
  -> Public API :9000
  -> Stream API :8769
  -> Grafana :3001
  -> Swagger :9000/docs
```

### 외형 검색

```text
YOLO person bbox
  -> AppearancePipeline
  -> HSV 또는 PP-Human/PA100K backend
  -> AppearanceLog SQLite
  -> /api/v1/search
  -> crop 이미지 조회
```

### AIoT 센서

```text
AIoT TLV/MQTT
  -> parser-python
  -> PostgreSQL/outbox
  -> EdgeX/MQTT
  -> Kuiper/SensorRuleBridge
  -> ActionBridge
```

## 문서 구조

```text
docs/
├── README.md
├── api/
│   ├── APPEARANCES_STATUS_API.md
│   ├── EVENT_SCHEMA_STANDARD.md
│   ├── EXTERNAL_INGEST.md
│   ├── PUBLIC_API_EXAMPLES.md
│   └── PUBLIC_API_GUIDE.md
├── architecture/
│   ├── ACTION_LAYER_SPEAKER_BRIDGE.md
│   ├── ASC_RULE_ENGINE.md
│   ├── COMPATIBILITY_SHIMS.md
│   ├── DEVICE_SERVICE_ARCHITECTURE.md
│   ├── EDGEX_SQLITE_DATA_ARCHITECTURE.md
│   ├── KUIPER_RULE_ENGINE.md
│   ├── PROJECT_OVERVIEW.md
│   └── PROJECT_STRUCTURE.md
├── falldata_integration_notes.md
├── operations/
│   ├── DEEPSTREAM_PERFORMANCE_STABILITY_2026-05-26.md
│   ├── DEVICE_INTEGRATION_CHECK_2026-05-29.md
│   ├── FACE_RECOGNITION_SETUP.md
│   ├── JETSON_EDGEX_FIELD_CHECKLIST.md
│   ├── MLOPS_MODEL_EVALUATION.md
│   ├── OPERATION_CHECKLIST.md
│   ├── OPERATIONS_RUNBOOK.md
│   └── PPHUMAN_ATTRIBUTE_INTEGRATION.md
├── reports/
│   ├── CODE_REVIEW_REPORT.md
│   ├── PROJECT_REVIEW_2026-04.md
│   ├── PROJECT_REVIEW_2026-06.md
│   └── TECHNICAL_WORK_REPORT_GUIDE.md
└── tooling/
  └── CODEX_MULTI_AGENT_SETUP.md
```

```text
.codex/skills/   # Codex가 실제로 자동 발견하는 repo-local skill 정의 (필수 도구)
.github/agents/  # 저장소와 함께 관리하는 UI-facing agent 정의 (필수 도구)
tooling/agents/  # skill 작성 시 참고하는 원본 agent 문서 (참고 자료)
```

## 테스트와 점검 스크립트

### `tests/`

대표 테스트 영역:

- Public API, auth, health, metrics
- ActionBridge, SensorBridge, EdgeX adapter
- AI analyzer, YOLO postprocess, DeepStream factory
- falldata 낙상 보조 검증
- appearance/search/face recognition
- zone detection, geometry
- Dockerfile/compose/runtime assumption
- smoke test

### `scripts/`

```text
scripts/
├── cleanup/                        # cleanup_runtime_data.sh 등
├── convert/                        # convert_pt_to_onnx.py, export_rethinking_par_onnx.py 등
├── datasets/                       # PAR/falldata 데이터셋 점검과 변환
├── dev/                            # check_before_push.sh
├── health/                         # readiness, 보안, 모델, falldata 점검
├── ops/                            # 운영 점검, 평가, 장비 확인
└── smoke/                          # smoke_test_deployment.py, smoke_test_data_flow.py
```

내일 시연 전 최소 확인:

```bash
docker compose ps
curl -fsS http://localhost:9000/api/v1/health
curl -fsS http://localhost:9000/api/v1/readiness
curl -fsS http://localhost:8769/health
curl -fsS http://localhost:8769/cameras
.venv/bin/python scripts/smoke/smoke_test_deployment.py
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
```

## 처음 코드를 볼 때 추천 순서

### 전체 흐름

1. `README.md`
2. `docs/architecture/PROJECT_STRUCTURE.md`
3. `docker-compose.yml`
4. `main.py`
5. `src/bootstrap/runtime.py`
6. `src/config/config.py`
7. `src/core/processor.py`
8. `src/core/ai/analyzer.py`
9. `src/services/action_bridge.py`
10. `src/api/app.py`
11. `web/public-demo.html`

### Jetson/DeepStream

1. `docker-compose.jetson.yml`
2. `Dockerfile.jetson`
3. `src/core/deepstream_processor.py`
4. `config/deepstream/config_infer_primary.txt`
5. `config/deepstream/config_infer_helmet.txt`
6. `scripts/health/check_jetson_edgex_stack.py`

### AIoT parser

1. `parser-python/main.py`
2. `parser-python/config/config.py`
3. `parser-python/tlv/parser.py`
4. `parser-python/service/sensor_service.py`
5. `parser-python/mqtt/manager.py`
6. `parser-python/database/edgex_outbox.py`

### UI/API 시연

1. `web/public-demo.html`
2. `src/api/app.py`
3. `src/api/v1/health.py`
4. `src/api/v1/cameras.py`
5. `src/api/v1/alerts.py`
6. `src/services/stream_api.py`

## 주의할 점

- `cameras.json`, `known_faces.json`, `.env`, `parser-python/.env`, `edgex/asc/cctv_asc.env`에는 현장 정보나 민감값이 들어갈 수 있습니다.
- 운영 API key, 내부 토큰, 장비 비밀번호는 코드에 하드코딩하지 않습니다.
- `web/public-demo.html`은 시연용 단일 HTML입니다. 운영 제품 UI가 아니라 Public API/Stream API 동작을 보여주는 용도입니다.
- 카메라 화면이 안 뜨면 `8769` 포트, `STREAM_API_ENABLED`, `cctv-ai-engine` 로그의 `MJPEG 스트리밍 서버 시작` 문구를 먼저 확인합니다.
- Jetson 운영은 코드보다 런타임 호환성이 중요합니다. L4T, CUDA, TensorRT, DeepStream, GStreamer 버전이 맞아야 합니다.
- `data/`, `known_faces/`, crop 이미지 폴더는 커밋 전에 민감 정보 포함 여부를 반드시 확인합니다.
