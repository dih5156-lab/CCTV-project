# CCTV-project 프로젝트 구조 설명

## 결론

이 프로젝트는 CCTV 영상에서 안전 이벤트를 감지하고, EdgeX/Kuiper/Action Layer/Public API까지 연결하는 엣지 AI 플랫폼입니다.

큰 흐름은 아래처럼 이해하면 됩니다.

```text
카메라/영상 입력
  -> AI 추론(VideoProcessor 또는 DeepStreamProcessor)
  -> 이벤트 생성/필터링/구역 판정
  -> MQTT 발행
  -> EdgeX / Kuiper 룰 / Action Layer / Alert API
  -> Public API, 대시보드, 외부 시스템 연동
```

이벤트 payload 표준은 [EVENT_SCHEMA_STANDARD.md](EVENT_SCHEMA_STANDARD.md)에 정리되어 있습니다.
현재 구조는 기존 `camera_id/type/confidence` 필드를 유지하면서
`schema_version/device/event/raw/event_id` 표준 필드를 함께 싣는 하위 호환 방식입니다.

현실적으로는 두 가지 실행 모드를 함께 갖고 있습니다.

- 개발/일반 실행: `main.py`에서 OpenCV + Ultralytics YOLO 기반 `VideoProcessor`를 실행합니다.
- Jetson 운영 실행: `USE_DEEPSTREAM=1`일 때 NVIDIA DeepStream 기반 `DeepStreamProcessor`를 사용합니다.

## 최상위 구조

```text
CCTV-project/
├── main.py                         # CCTV AI 엔진 기본 실행 진입점
├── src/                            # 핵심 애플리케이션 코드
├── runners/                        # 서비스별 단독 실행 진입점
├── parser-python/                  # AIoT TLV 센서 파서 서비스
├── config/                         # DeepStream, 외형 분석 등 설정 파일
├── models/                         # YOLO/PP-Human/TensorRT 모델 파일
├── edgex/                          # EdgeX 디바이스 프로파일 및 ASC 설정
├── kuiper/                         # eKuiper 룰 파일
├── monitoring/                     # Prometheus/Grafana 설정
├── docs/                           # 설계/운영 문서
├── scripts/                        # 점검, 변환, smoke test, 모델 평가 스크립트
├── tests/                          # pytest 테스트
├── Dockerfile*                     # 서비스별 컨테이너 이미지 정의
├── docker-compose.yml              # 일반 Docker/EdgeX 통합 배포
└── docker-compose.jetson.yml       # Jetson/DeepStream 운영 배포
```

## 실행 진입점

### `main.py`

기본 CCTV AI 엔진 진입점입니다.

주요 순서는 다음과 같습니다.

1. OpenCV, 콘솔, Jetson 관련 런타임 환경을 초기화합니다.
2. CLI 인자를 읽고 `AppConfig.from_env()`로 환경변수 기반 설정을 만듭니다.
3. `cameras.json` 또는 단일 웹캠/비디오 입력을 로드합니다.
4. `src/bootstrap/runtime.py`의 `start_processor_runtime()`으로 실제 프로세서를 시작합니다.

### `runners/`

Docker Compose나 운영 환경에서 각 기능을 독립 서비스로 띄우기 위한 실행 파일입니다.

```text
runners/
├── run_public_api.py               # FastAPI 공개 API 서버, 기본 포트 9000
├── run_action_bridge.py            # 알람/외부 전송/SQLite 저장 Action Layer
├── run_alert_api.py                # 내부 Alert API, JSONL 이벤트 수신/저장
├── run_edgex_adapter.py            # AI 이벤트를 EdgeX 이벤트로 변환
├── run_kuiper_rules.py             # eKuiper 룰 배포 도구
├── run_sensor_rule_bridge.py       # AIoT 센서 룰 MQTT 브리지
└── _shared.py                      # runner 공통 로깅/sys.path 유틸
```

## `src/` 핵심 구조

```text
src/
├── api/                            # 서버팀/대시보드용 FastAPI 공개 API
├── bootstrap/                      # CLI, 런타임 초기화, 프로세서 생성
├── config/                         # 중앙 설정 객체와 환경변수 오버라이드
├── core/                           # 영상 처리, AI 추론, 이벤트 생성 핵심
├── devices/                        # 스피커, 전광판, 경광등 등 현장 장치 제어
├── edgex/                          # EdgeX 디바이스 서비스/어댑터 구현
├── protocols/                      # MQTT, HTTP, REST, TLV 통신 유틸
├── services/                       # 서비스 계층: ActionBridge, API 서버, 로그 저장 등
├── storage/                        # SQLite 공통 저장소
└── utils/                          # 카메라 입력, geometry, zone, visualizer 유틸
```

### `src/bootstrap/`

애플리케이션 시작 흐름을 담당합니다.

- `cli.py`: CLI 인자 파싱과 설정 반영
- `runtime.py`: 런타임 환경 초기화, 카메라 목록 로드, 프로세서 생성, 종료 핸들러 등록

중요한 분기점은 `create_processor()`입니다.

```text
USE_DEEPSTREAM=1
  -> DeepStreamProcessor 시도
  -> DeepStream 사용 불가 시 VideoProcessor로 fallback

USE_DEEPSTREAM 미설정
  -> VideoProcessor 사용
```

### `src/config/`

프로젝트 전체 설정의 중심입니다.

- `config.py`: `AppConfig`, 모델 경로, MQTT, EdgeX, ActionBridge, 카메라, 외형 분석, 이벤트 디바운스 설정
- `event_type_map.py`: 이벤트 타입 매핑

특징은 환경변수 오버라이드가 많다는 점입니다. 예를 들어 `HELMET_MODEL_PATH`, `PERSON_MODEL_PATH`, `POSE_MODEL_PATH`, `DEVICE`, `MQTT_BROKER`, `APPEARANCE_ENABLED`, `USE_DEEPSTREAM` 같은 값으로 운영 환경을 바꿀 수 있습니다.

### `src/core/`

실시간 영상 처리의 핵심입니다.

```text
src/core/
├── base_processor.py               # VideoProcessor/DeepStreamProcessor 공통 인터페이스
├── processor.py                    # OpenCV + YOLO 기반 기본 영상 처리기
├── deepstream_processor.py         # NVIDIA DeepStream 기반 Jetson 처리기
├── events.py                       # DetectionEvent, EventType 정의
├── event_filters.py                # 누적 감지/트랙 기반 필터링
├── sensor_detection.py             # 센서 기반 이벤트 판정
├── _inference_pipeline.py          # AI 추론, 구역 판정, 이벤트 큐 적재
├── _camera_registry.py             # 카메라/스레드/재연결 관리
├── _display_grid.py                # 다중 카메라 표시
├── _event_publish.py               # 이벤트 발행 헬퍼
├── _yolo_postprocess.py            # YOLO 후처리
└── ai/                             # 실제 AI 분석 컴포넌트
```

`VideoProcessor`는 전체 오케스트레이터이고, 내부 상세 책임은 여러 보조 클래스로 나뉘어 있습니다.

- 카메라 등록/재연결: `_CameraRegistry`
- AI 추론과 이벤트 큐 적재: `_InferencePipeline`
- 이벤트 중복 전송 방지: `_EventDebouncer`
- 다중 화면 표시: `_DisplayGrid`
- 통계: `ProcessorStats`

### `src/core/ai/`

AI 모델 추론과 후처리 계층입니다.

```text
src/core/ai/
├── analyzer.py                     # AIAnalyzer, 멀티 모델 오케스트레이터
├── _object_detection_pipeline.py   # 객체 탐지 파이프라인
├── _face_recognition_pipeline.py   # 얼굴 인식 파이프라인
├── _appearance_pipeline.py         # 외형 속성 분석 파이프라인
├── _appearance_analyzer.py         # 색상/PP-Human 기반 외형 분석
├── _attribute_backends.py          # 외형 속성 백엔드
├── _attribute_runtimes.py          # onnxruntime/paddle 등 런타임
├── _fall_detector.py               # 낙상 판정
├── _object_tracker.py              # object_id/track 관리
└── _yolo_helpers.py                # YOLO 결과 추출 유틸
```

현재 구조상 `AIAnalyzer`가 모델 로딩과 추론 오케스트레이션을 담당하고, 세부 로직은 전담 모듈로 분리되어 있습니다.

주요 감지 대상은 다음입니다.

- 사람 탐지
- 헬멧 착용/미착용
- 낙상
- 얼굴 인식
- 외형 속성 검색: 상의/하의 색, 헬멧, 가방류 등

### `src/api/`

서버팀이나 대시보드가 사용할 수 있는 FastAPI 공개 API입니다.

```text
src/api/
├── app.py                          # FastAPI 앱 생성, 미들웨어, 라우터 등록
├── _action_proxy.py                # Action Layer REST 프록시
├── dependencies/                   # 인증, rate limit, URL/path 설정
├── schemas/                        # Pydantic 응답/요청 스키마
└── v1/                             # /api/v1 라우터
```

`src/api/app.py` 기준 공통 정책은 다음입니다.

- 모든 주요 API는 `/api/v1` prefix를 사용합니다.
- 응답은 공통 래퍼 형식을 사용합니다.
- `X-API-Key` 인증을 사용합니다.
- `slowapi` rate limit을 사용합니다.
- CORS는 `CORS_ORIGINS` 환경변수로 제어합니다.

주요 API는 다음과 같습니다.

```text
GET    /api/v1/health
GET    /api/v1/metrics

GET    /api/v1/events
GET    /api/v1/cameras
GET    /api/v1/cameras/{camera_id}

GET    /api/v1/sites
POST   /api/v1/sites
DELETE /api/v1/sites/{site_id}

GET    /api/v1/control/mode
POST   /api/v1/control/mode
GET    /api/v1/control/pending
POST   /api/v1/control/approve/{event_id}
POST   /api/v1/control/reject/{event_id}

GET    /api/v1/appearances
GET    /api/v1/appearances/status
POST   /api/v1/appearances
DELETE /api/v1/appearances/{condition_id}

GET    /api/v1/search
GET    /api/v1/search/crops/{filename}
POST   /api/v1/alerts
```

### `src/services/`

도메인 서비스 계층입니다.

```text
src/services/
├── action_bridge.py                # MQTT/REST 이벤트 수신 후 현장 액션 실행
├── _action_bridge_support.py       # ActionBridge 내부 도메인/저장소/실행 헬퍼
├── appearance_log.py               # 외형 기록 SQLite 저장/검색
├── appearance_conditions.py        # 외형 검색 조건 저장
├── appearance_status.py            # 외형 검색 준비 상태 계산
├── cctv_metrics.py                 # Prometheus 메트릭
├── external_ingest.py              # 외부 MQTT/NC 수신
├── sensor_bridge.py                # 센서 이벤트 브리지
├── sensor_rule_bridge.py           # 센서 룰 결과 변환/발행
├── zone_api.py                     # 구역 설정 REST API
├── camera_model_api.py             # 카메라별 모델 설정 API
├── face_api.py                     # 얼굴 등록/삭제/조회 API
└── stream_api.py                   # 프레임/스트림 API
```

특히 `ActionBridge`는 운영에서 중요한 역할을 합니다.

- MQTT 룰 결과를 구독합니다.
- REST 이벤트를 받을 수 있습니다.
- 스피커/전광판/경광등 제어를 수행합니다.
- 외부 플랫폼 HTTP 전송을 수행합니다.
- 이벤트를 SQLite에 저장합니다.
- 자동/수동 승인 모드를 관리합니다.

### `src/edgex/`

EdgeX Foundry 연동 계층입니다.

```text
src/edgex/
├── device_service.py               # EdgeX 디바이스 서비스 구현
├── adapter_service.py              # AI MQTT 이벤트를 EdgeX 이벤트로 변환
├── _http_mixin.py
├── _payload_mixin.py
├── _publisher_mixin.py
└── _outbox_mixin.py                # EdgeX 발행 실패 시 outbox 계층
```

`run_edgex_adapter.py`는 AI 엔진의 MQTT 이벤트를 구독해서 EdgeX Core Data/Metadata 쪽으로 넘기는 역할입니다.

### `src/devices/`

현장 장치 제어 코드입니다.

```text
src/devices/
├── speaker.py                      # TCP 스피커 제어
├── signboard.py                    # 전광판 문구 생성/전송
├── siren.py                        # 경광등/사이렌 제어
└── sensor_device.py                # 센서 도메인 모델
```

이 계층은 Action Layer에서 호출됩니다.

### `src/protocols/`

외부 통신 프로토콜 유틸입니다.

```text
src/protocols/
├── mqtt_publisher.py               # MQTT 이벤트 발행
├── mqtt_subscriber.py              # MQTT 구독
├── _mqtt_factory.py                # MQTT 클라이언트 생성
├── http.py                         # HTTP 이벤트 포워딩
├── rest.py                         # REST 이벤트 수신
└── tlv_decoder.py                  # TLV 디코딩
```

### `src/utils/`

영상 처리 보조 유틸입니다.

```text
src/utils/
├── camera_input.py                 # RTSP/웹캠 입력, GStreamer 지원
├── geometry.py                     # bbox/좌표/헬멧 착용 판정 유틸
├── visualizer.py                   # 이벤트 시각화
├── zone_detection.py               # 폴리곤 구역 침입 판정
├── zone_drawer.py                  # GUI 기반 구역 그리기
├── zone_presets.py                 # 구역 프리셋 저장
├── dataset_collector.py            # YOLO 형식 데이터셋 저장
└── face_recognition.py             # 얼굴 인식 엔진
```

## `parser-python/`

AIoT 센서 TLV 파서 서비스입니다. CCTV 영상 AI와 별도로, 센서 데이터를 받아 파싱하고 DB/EdgeX/Alert API로 전달하는 쪽에 가깝습니다.

```text
parser-python/
├── main.py                         # 파서 서비스 진입점
├── config/                         # DB/MQTT/서버 설정
├── tlv/                            # TLV parser/transformer
├── mqtt/                           # MQTT manager, EdgeX forwarder
├── database/                       # DB 연결, query, model, outbox
├── service/                        # Sensor/Event/DeviceInfo 서비스
├── batch/                          # 디바이스 배치 갱신
└── tests/                          # parser 전용 테스트
```

실행 흐름은 다음입니다.

```text
.env 로드
  -> config 로드
  -> DB 연결
  -> SensorService 생성
  -> BatchManager로 디바이스 목록 갱신
  -> MQTT Manager 시작
  -> Flask health 서버 시작
```

## 설정 파일과 모델 파일

### `config/`

```text
config/
├── event_type_map.json
├── appearance_pphuman_labels.example.json
└── deepstream/
    ├── config_infer_primary.txt
    ├── config_infer_helmet.txt
    ├── config_tracker.txt
    ├── config_tracker_nvdcf.yml
    ├── config_streammux.txt
    ├── labels.txt
    └── labels_helmet.txt
```

DeepStream 실행 시 `config/deepstream/` 파일들이 중요합니다.

### `models/`

```text
models/
├── helmet_model.pt
├── helmet_model_ver0.5.pt
├── yolov8n.pt
├── yolov8n-pose.pt
├── yolov8m-pose.pt
├── model_manifest.json
├── pphuman_attribute_src/
└── trt_cache/
```

Jetson 운영에서는 `.pt`보다 TensorRT `.engine`을 우선 사용하는 구성이 들어가 있습니다. 실제 `.engine` 생성은 `scripts/convert_pt_to_onnx.py`, `scripts/convert_onnx_to_engine.py` 같은 스크립트를 통해 준비하는 흐름으로 보입니다.

## EdgeX / Kuiper / Monitoring

### `edgex/`

```text
edgex/
├── device-profiles/                # AIoT 디바이스 프로파일 YAML
├── asc/                            # App Service Configurable 설정
└── register_aiot_devices.py        # EdgeX 디바이스 등록 스크립트
```

### `kuiper/`

```text
kuiper/rules/
├── cctv_intrusion_rules.json       # CCTV 침입/위험구역 룰
└── aiot_sensor_rules.json          # 센서 기반 룰
```

AI 엔진이 MQTT로 이벤트를 발행하면 eKuiper가 룰을 적용하고, 결과를 Action Layer로 넘기는 구조입니다.

### `monitoring/`

```text
monitoring/
├── prometheus.yml
└── grafana/provisioning/
```

Prometheus와 Grafana 대시보드 설정이 포함되어 있습니다.

## Docker Compose 기준 서비스 구조

`docker-compose.yml`에는 EdgeX 기본 구성과 CCTV 서비스가 함께 들어 있습니다.

주요 서비스는 다음입니다.

```text
edgex-mqtt-broker                 # Mosquitto MQTT broker
edgex-core-consul                 # EdgeX registry/config
edgex-redis                       # EdgeX database/message bus
edgex-core-data                   # EdgeX Core Data
edgex-core-metadata               # EdgeX Core Metadata
edgex-device-rest                 # EdgeX REST device service
edgex-kuiper                      # eKuiper rule engine

cctv-ai-engine                    # 영상 AI 처리 엔진
cctv-edgex-adapter                # AI 이벤트 -> EdgeX 변환
cctv-kuiper-rule-loader           # Kuiper 룰 등록
cctv-alert-api                    # Alert 수신/로그 저장 API
cctv-action-layer                 # 알람/외부전송/수동승인/SQLite 저장
aiot-parser                       # AIoT TLV 센서 파서
cctv-public-api                   # 외부/대시보드용 FastAPI API

cctv-prometheus                   # 메트릭 수집
cctv-grafana                      # 대시보드
edgex-ui-go                       # EdgeX UI
```

`docker-compose.jetson.yml`은 Jetson/DeepStream 운영에 맞춰 `cctv-ai-engine`, EdgeX, parser, action layer를 구성합니다.

## 주요 데이터 흐름

### 영상 AI 이벤트 흐름

```text
cameras.json / RTSP / webcam
  -> main.py
  -> bootstrap.runtime
  -> VideoProcessor 또는 DeepStreamProcessor
  -> AIAnalyzer 또는 DeepStream probe
  -> DetectionEvent
  -> zone_detection / event_filters / debouncer
  -> MqttEventPublisher
  -> cctv/ai/events/... MQTT topic
```

### 룰/알람 흐름

```text
cctv/ai/events/... MQTT
  -> eKuiper rules
  -> cctv/rules/... MQTT
  -> ActionBridge
  -> 스피커/전광판/경광등
  -> 외부 HTTP API
  -> SQLite action_events.db
```

### Public API 조회 흐름

```text
대시보드/서버팀
  -> cctv-public-api(FastAPI)
  -> JSONL alert log, cameras.json, appearances.db 조회
  -> Action Layer REST proxy
```

### 외형 검색 흐름

```text
YOLO person bbox
  -> AppearancePipeline
  -> HSV 또는 PP-Human attribute backend
  -> AppearanceLog(SQLite)
  -> /api/v1/search
  -> /api/v1/search/crops/{filename}
```

### AIoT 센서 흐름

```text
AIoT sensor MQTT/TLV
  -> parser-python
  -> TLV parser/transformer
  -> DB 저장
  -> EdgeX forwarder / Alert API
  -> sensor_rule_bridge 또는 Kuiper rule
  -> ActionBridge
```

## 테스트와 점검 스크립트

### `tests/`

프로젝트 전반의 pytest 테스트가 있습니다.

대표 영역은 다음입니다.

- API 인증/이벤트/카메라/사이트/제어 API
- AI 분석기, YOLO 후처리, DeepStream 이벤트 factory
- 외형 검색/외형 조건/얼굴 인식
- Zone detection
- ActionBridge, SensorBridge, AdapterService
- Dockerfile, 배포 준비 상태, 민감 설정 점검
- smoke test

### `scripts/`

```text
scripts/
├── check_deployment_readiness.py
├── check_dockerfile_sources.py
├── check_jetson_edgex_stack.py
├── check_model_report.py
├── check_monitoring_config.py
├── check_sensitive_defaults.py
├── convert_pt_to_onnx.py
├── convert_onnx_to_engine.py
├── evaluate_detection.py
├── smoke_test_data_flow.py
└── smoke_test_deployment.py
```

운영 전 점검, 모델 변환, 배포 smoke test에 쓰는 도구들입니다.

운영 중 상태 확인, 로그 확인, 서비스별 재시작 절차는
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md)를 참고하세요.

## 처음 코드를 볼 때 추천 순서

1. `README.md`
2. `main.py`
3. `src/bootstrap/runtime.py`
4. `src/config/config.py`
5. `src/core/base_processor.py`
6. `src/core/processor.py`
7. `src/core/ai/analyzer.py`
8. `src/api/app.py`
9. `src/services/action_bridge.py`
10. `docker-compose.yml`

Jetson/DeepStream 쪽을 볼 때는 아래 순서가 좋습니다.

1. `docker-compose.jetson.yml`
2. `Dockerfile.jetson`
3. `src/core/deepstream_processor.py`
4. `config/deepstream/config_infer_primary.txt`
5. `config/deepstream/config_infer_helmet.txt`
6. `scripts/check_jetson_edgex_stack.py`

AIoT 센서 파서 쪽은 아래 순서가 좋습니다.

1. `parser-python/main.py`
2. `parser-python/config/config.py`
3. `parser-python/tlv/parser.py`
4. `parser-python/service/sensor_service.py`
5. `parser-python/mqtt/manager.py`
6. `parser-python/mqtt/edgex_forwarder.py`

## 주의할 점

- 현재 git 상태상 많은 파일이 이미 수정/추가되어 있습니다. 이 문서는 현재 워크스페이스 파일 기준으로 정리했습니다.
- `core.12`, `tmp_test_dirs/`, `.pytest_cache/`, `.venv/`는 코드 구조 설명에서는 제외해도 되는 실행/테스트 산출물 성격입니다.
- `known_faces.json`, `cameras.json`, `.env*` 계열에는 운영 정보가 들어갈 수 있으므로 문서나 커밋에 민감값이 포함되지 않도록 주의해야 합니다.
- Jetson 운영에서는 DeepStream, TensorRT, GStreamer, CUDA/L4T 버전 호환이 중요합니다. 코드만 맞아도 런타임 이미지와 드라이버가 맞지 않으면 실행이 실패할 수 있습니다.
- Public API는 `X-API-Key` 인증을 사용하므로, 운영 배포 시 API key와 내부 토큰은 반드시 환경변수/시크릿으로 주입해야 합니다.
