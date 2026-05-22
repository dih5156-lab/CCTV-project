# CCTV 프로젝트 업무 기술문서 작성 가이드

## 결론

이 문서는 지금까지 수행한 CCTV 프로젝트 업무를 기술문서로 정리하기 위한 가이드와 초안입니다.

기술문서는 단순히 "무엇을 만들었다"보다 아래 흐름으로 작성하는 것이 좋습니다.

```text
문제 정의
  -> 시스템 구조
  -> 내가 구현한 기술
  -> 구현 과정에서의 판단 근거
  -> 검증 결과
  -> 남은 과제
```

현재 프로젝트는 아래 5개 축으로 정리하면 가장 자연스럽습니다.

1. EdgeX 기반 이벤트 연동 구조
2. CCTV AI 모델 구현과 추론 파이프라인
3. AI 학습/평가/MLOps 관리
4. Jetson/DeepStream 엣지 배포
5. Public API, Action Layer, 운영 점검 자동화

주의할 점:

- 정확한 학습 데이터 수, 모델 성능 수치, 실제 현장 검증 결과는 확인된 값만 작성합니다.
- 아직 측정하지 않은 성능은 "예정", "평가 필요", "추정"으로 구분합니다.
- `.env`, `.env.jetson`, RTSP 주소, API key, 장비 비밀번호 같은 민감정보는 문서에 넣지 않습니다.

## 문서 작성 원칙

### 1. 결론을 먼저 쓴다

읽는 사람이 가장 먼저 알고 싶은 것은 "그래서 이 프로젝트가 어떤 수준까지 되었는가"입니다.

좋은 예:

```text
본 프로젝트는 CCTV 영상에서 헬멧 미착용, 낙상, 위험 구역 침입을 감지하고,
감지 이벤트를 MQTT, EdgeX, Public API, Action Layer로 전달하는 엣지 관제 PoC 시스템이다.
현재 개발 단계 기준으로 AI 분석, 이벤트 표준화, EdgeX 연동, Jetson 배포, 운영 점검 스크립트까지 구현했다.
```

피해야 할 예:

```text
YOLOv8을 사용했고 EdgeX를 사용했으며 Docker도 사용했다.
```

기술 나열만 있으면 업무 범위와 성과가 잘 보이지 않습니다.

### 2. 기술을 기능 단위로 묶는다

기술문서에서는 라이브러리 이름보다 "왜 썼고, 어디에 붙였는지"가 중요합니다.

예시:

| 기술 | 사용 위치 | 목적 |
|---|---|---|
| YOLOv8 | `src/core/ai/*` | 사람/헬멧/포즈 기반 이벤트 감지 |
| DeepStream | `src/core/deepstream_processor.py` | Jetson에서 GPU 기반 영상 처리 |
| EdgeX Foundry | `src/edgex/*`, `edgex/*` | AI/센서 이벤트 표준화 및 외부 연동 |
| MQTT | `src/protocols/mqtt_publisher.py` | 서비스 간 이벤트 전달 |
| SQLite | `src/storage/*`, `src/services/*` | 엣지 로컬 이벤트 저장과 검색 |
| FastAPI | `src/api/*` | 관제/서버 연동용 Public API |
| Docker Compose | `docker-compose*.yml` | 개발/운영/Jetson 배포 구성 |

### 3. 구현 내용과 학습 내용을 분리한다

AI 모델 관련 문서는 보통 아래처럼 분리하는 편이 읽기 좋습니다.

```text
AI 모델 구현
  - 모델이 서비스에 어떻게 붙었는지
  - 입력/출력/후처리 구조
  - 이벤트 생성 방식

AI 학습 및 평가
  - 어떤 데이터로 학습했는지
  - 어떤 모델 버전을 관리하는지
  - precision/recall/latency 기준
  - 아직 평가가 필요한 부분
```

### 4. 검증 결과는 명령어와 함께 남긴다

검증 결과는 "잘 됨"보다 아래처럼 남기는 것이 좋습니다.

```text
검증 명령:
- .venv/bin/python -m pytest
- .venv/bin/python scripts/check_sensitive_defaults.py
- .venv/bin/python scripts/check_jetson_edgex_stack.py --host localhost --public-api-port 9000 --deepstream --check-appearance-status

검증 결과:
- 전체 테스트 771 passed, 75 skipped
- 핵심 데모 경로 테스트 138 passed
- 민감 기본값 검사 통과
- Jetson DeepStream 컨테이너 healthy, Runtime=nvidia, RestartCount=0
```

검증 시점이 바뀌면 결과도 바뀔 수 있으므로, 문서에는 날짜를 같이 적습니다.

## 제출용 문서 추천 목차

아래 목차를 그대로 사용하면 됩니다.

```text
1. 프로젝트 개요
2. 개발 배경 및 문제 정의
3. 전체 시스템 아키텍처
4. 주요 개발 업무
   4.1 EdgeX 이벤트 연동
   4.2 CCTV AI 모델 구현
   4.3 AI 학습 및 모델 관리
   4.4 Jetson/DeepStream 엣지 배포
   4.5 Public API 및 Action Layer
   4.6 운영 점검 및 테스트 자동화
5. 데이터 흐름
6. 검증 결과
7. 기술적으로 고민한 점
8. 한계 및 향후 개선 사항
9. 참고 문서
```

## 업무 기술문서 초안

아래 내용은 현재 저장소 기준으로 작성한 초안입니다.
확인된 학습데이터 수, 학습환경, 모델 성능은 수치로 반영했고,
사진·링크·라벨링 화면처럼 사용자가 별도 첨부할 자료는 `첨부 예정` 공간으로 분리했습니다.

---

# CCTV AI 엣지 관제 시스템 업무 기술문서

## 1. 프로젝트 개요

본 프로젝트는 CCTV 영상과 AIoT 센서 데이터를 활용하여 산업 현장의 안전 이벤트를 감지하고, 이를 EdgeX Foundry 기반 이벤트 파이프라인과 관제 API로 전달하는 엣지 관제 시스템이다.

주요 감지 대상은 다음과 같다.

- 헬멧 착용/미착용 감지
- 사람 감지
- 낙상 감지
- 위험 구역 침입 감지
- 얼굴 인식 및 외형 속성 분석 확장
- AIoT 센서 이벤트 수신 및 룰 기반 알람 연동

프로젝트는 Windows 개발 환경과 NVIDIA Jetson Orin 운영 환경을 모두 고려하여 구성했다.
일반 개발 환경에서는 OpenCV와 Ultralytics 기반 프로세서를 사용하고, Jetson 환경에서는 DeepStream과 TensorRT 기반의 GPU 가속 처리를 목표로 한다.

## 2. 개발 배경 및 문제 정의

현장 CCTV 시스템은 영상을 단순 저장하거나 사람이 직접 확인하는 방식만으로는 실시간 안전 대응에 한계가 있다.
특히 헬멧 미착용, 낙상, 위험 구역 침입 같은 이벤트는 빠른 감지와 외부 시스템 연동이 중요하다.

본 프로젝트의 목표는 다음과 같다.

1. CCTV 영상에서 안전 관련 이벤트를 자동 감지한다.
2. AI 이벤트를 표준 payload로 정규화한다.
3. MQTT와 EdgeX를 통해 다른 서비스와 연동 가능하게 만든다.
4. Action Layer를 통해 스피커, 전광판, 경광등 등 출력 장비와 연결한다.
5. Jetson 기반 엣지 장비에서도 배포 가능한 구조를 만든다.
6. 테스트와 점검 스크립트를 통해 시연 및 현장 점검 가능성을 높인다.

## 3. 전체 시스템 아키텍처

현재 시스템은 아래 흐름으로 동작한다.

```text
카메라/RTSP/비디오
  -> AI Engine
  -> AI 이벤트 생성
  -> MQTT / Alert API
  -> EdgeX Adapter / Kuiper Rule / Action Layer
  -> Public API / Grafana / web UI / 외부 시스템
```

주요 실행 단위는 다음과 같다.

| 구성요소 | 역할 |
|---|---|
| AI Engine | 영상 입력, AI 추론, 이벤트 생성 |
| EdgeX Adapter | AI 이벤트를 EdgeX 이벤트 구조로 변환 및 발행 |
| Kuiper Rule Engine | 이벤트 필터링, 지속 감지, 조건 기반 룰 처리 |
| Action Layer | 알람, 외부 API 호출, SQLite 이벤트 저장 |
| Public API | 대시보드/서버 연동용 REST API |
| Stream API | 카메라 미리보기 및 스냅샷 제공 |
| Grafana/Prometheus | 운영 모니터링 |

관련 구현 위치:

- AI 처리: `src/core/`, `src/core/ai/`
- EdgeX 연동: `src/edgex/`, `edgex/`
- Action Layer: `src/services/action_bridge.py`
- Public API: `src/api/`
- 배포 구성: `docker-compose.yml`, `docker-compose.jetson.yml`, `Dockerfile.jetson`

## 4. 주요 개발 업무

### 4.1 EdgeX 이벤트 연동

EdgeX 연동은 AI 이벤트를 외부 시스템이 이해하기 쉬운 표준 이벤트로 변환하기 위해 구현했다.

구현 내용:

- AI Engine에서 발생한 이벤트를 MQTT로 발행
- EdgeX Adapter에서 AI MQTT 이벤트를 구독
- EdgeX DeviceService, DeviceProfile, Device 등록 구조 정리
- EdgeX Core Data로 전달 가능한 이벤트 payload 구성
- CCTV, 센서, 외부 입력 이벤트를 표준 payload 구조로 통일

핵심 데이터 흐름:

```text
AI Engine
  -> cctv/ai/events/{camera_id}/{event_type}
  -> EdgeX Adapter
  -> edgex/events/device/{service}/{device}/{resource}
  -> Kuiper / Action Layer / 외부 시스템
```

관련 문서:

- `docs/DEVICE_SERVICE_ARCHITECTURE.md`
- `docs/EVENT_SCHEMA_STANDARD.md`
- `docs/EDGEX_SQLITE_DATA_ARCHITECTURE.md`
- `docs/JETSON_EDGEX_FIELD_CHECKLIST.md`

### 4.2 CCTV AI 모델 구현

AI 모델은 영상에서 객체와 행동 이벤트를 감지하기 위해 구성했다.

현재 모델 구성:

| 모델 | 목적 | 주요 파일 |
|---|---|---|
| `helmet_model_ver0.5` | 헬멧/머리 감지 | `models/helmet_model_ver0.5.pt` |
| `helmet_model` | 헬멧 감지 이전/대체 모델 | `models/helmet_model.pt` |
| `yolov8n` | 사람 감지 fallback | `models/yolov8n.pt` |
| `yolov8n-pose` | 낙상 감지용 포즈 추정 | `models/yolov8n-pose.pt` |
| `yolov8m-pose` | 정확도 기준 포즈 모델 | `models/yolov8m-pose.pt` |
| `pphuman_attribute` | 사람 외형 속성 분석 | `models/pphuman_attribute_src/*` |

구현한 AI 처리 흐름:

```text
프레임 입력
  -> 사람/헬멧/포즈 모델 추론
  -> bbox, confidence, class_id 후처리
  -> zone 조건 및 이벤트 필터 적용
  -> DetectionEvent 생성
  -> MQTT/Public API/Action Layer로 전달
```

주요 구현 포인트:

- `AIAnalyzer`를 중심으로 객체 탐지, 낙상 감지, 얼굴 인식, 외형 분석 흐름을 분리
- 카메라별 detections 설정으로 필요한 분석만 선택 가능하게 구성
- 헬멧 감지는 현장 안전 리스크를 고려해 recall을 중요 지표로 설정
- 낙상 감지는 pose keypoint 기반 판단 구조로 확장
- 외형 분석은 기본 HSV 기반으로 동작하고, `APPEARANCE_BACKEND=pphuman` 설정 시 PP-Human 백엔드로 확장 가능

관련 구현 위치:

- `src/core/ai/analyzer.py`
- `src/core/ai/_object_detection_pipeline.py`
- `src/core/ai/_fall_detector.py`
- `src/core/ai/_appearance_analyzer.py`
- `src/core/ai/_attribute_runtimes.py`
- `src/core/_yolo_postprocess.py`

### 4.3 AI 학습 데이터, 학습 환경 및 모델 관리

헬멧 감지 모델은 공개 안전모 데이터셋과 추가 정제 데이터를 기반으로 YOLO 형식 학습데이터를 구성하고,
Jetson CUDA 환경에서 YOLO 학습을 진행했다.

학습데이터와 모델 결과는 단순 파일 보관이 아니라, 데이터 출처, 라벨링 방식, 학습 조건, 성능 지표를 함께 남기는 방식으로 정리했다.
정부과제 제출 시에는 아래 표를 기준으로 수치와 증빙자료를 함께 제시한다.

#### 4.3.1 학습데이터 구성

확인된 학습데이터 경로:

- 원천/백업 데이터: `/media/sawwave/ESD-USB/헬멧 학습 데이터`
- 실제 YOLO 학습 사용 데이터: `/media/sawwave/Learning11/데이터 학습 자료/datasets`
- 학습 후 정리 데이터: `/media/sawwave/Learning11/데이터 학습 자료/학습 후 데이터/헬멧데이터 1`

확인된 학습데이터 수량:

| 구분 | 경로 | 이미지 수 | 라벨 수 | 비고 |
|---|---|---:|---:|---|
| 원천 데이터 1 | `ESD-USB/헬멧 학습 데이터/헬멧1` | 7,581 | 7,581 | Pascal VOC XML 라벨, train 6,281 / val 1,300 |
| 원천 데이터 2 | `ESD-USB/헬멧 학습 데이터/헬멧2` | 7,035 | 7,035 | YOLOv8 라벨, train 5,269 / test 1,766 |
| 실제 학습 데이터 | `데이터 학습 자료/datasets` | 7,035 | 7,035 | YOLOv8 라벨, train 5,269 / test 1,766 |
| 학습 후 정리 데이터 | `학습 후 데이터/헬멧데이터 1` | 5,000 | 5,000 | 학습/정리 산출 데이터 |

실제 YOLO 라벨 파일 분석 기준 class 구성:

| class id | class name | train 객체 수 | test 객체 수 | 합계 |
|---:|---|---:|---:|---:|
| 0 | `head` | 14,884 | 4,863 | 19,747 |
| 1 | `helmet` | 4,874 | 1,803 | 6,677 |
| 2 | `person` | 473 | 142 | 615 |

프로젝트 학습 설정 파일 기준 데이터셋 설정:

```yaml
train: train/images
val: test/images
nc: 2
names: ["helmet", "head"]
```

주의:

- 원천 YOLO 데이터에는 `head`, `helmet`, `person` 3개 클래스가 포함되어 있다.
- 현재 프로젝트의 헬멧 감지 운영 모델은 `helmet`, `head` 중심으로 사용한다.
- 제출 전 최종 문서에서는 실제 학습 시 `person` 클래스를 제외했는지, 또는 라벨 변환 과정에서 매핑했는지 작업 이력을 한 줄로 보완하는 것이 좋다.

#### 4.3.2 데이터 출처 및 라벨링 증빙

현재 확인된 공개 데이터셋 출처:

- Dataset: Hard Hat Workers
- 제공 경로: Roboflow Universe
- 원 제공 기관: Northeastern University - China
- 라이선스: Public Domain
- Roboflow 프로젝트 URL: `https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/2`
- 데이터셋 설명 파일: `/media/sawwave/ESD-USB/헬멧 학습 데이터/헬멧2/README.dataset.txt`

제출용 첨부자료 공간:

| 첨부 항목 | 첨부 예정 자료 | 상태 |
|---|---|---|
| 원천 데이터 출처 | Roboflow/공개 데이터셋 링크, 다운로드 화면 캡처 | 첨부 예정 |
| 라벨링 기준 | `head`, `helmet`, `person` class 정의 화면 또는 설명 이미지 | 첨부 예정 |
| 라벨링 예시 | 라벨링 화면 캡처 또는 bbox 표시 이미지 | 첨부 예정 |
| 데이터 정제 과정 | XML -> YOLO 변환, class 정리, train/test 구성 설명 | 첨부 예정 |
| 학습 실행 증빙 | Jetson 터미널 학습 명령, 학습 로그, `args.yaml` 캡처 | 첨부 예정 |
| 모델 결과 증빙 | `results.png`, confusion matrix, PR curve, F1 curve 이미지 | 첨부 예정 |

#### 4.3.3 학습 환경

학습은 Jetson 장비에서 CUDA GPU를 사용하여 진행했다.

확인된 대표 학습 설정:

| 항목 | 내용 |
|---|---|
| 학습 장비 | NVIDIA Jetson 계열 장비 |
| 가속 환경 | CUDA GPU 사용 |
| 학습 프레임워크 | Ultralytics YOLO |
| 작업 유형 | Object Detection |
| 대표 모델 | `helmet_model7_data_plus2` |
| 초기/기반 모델 | `helmet_model_ver0.5.pt` |
| epoch | 100 |
| batch size | 4 |
| image size | 768 |
| optimizer | AdamW |
| device | `0` |
| workers | 0 |
| AMP | enabled |
| 주요 augmentation | HSV, translate, scale, flip, mosaic, mixup, copy-paste |

대표 학습 설정 파일:

- `/media/sawwave/Learning11/데이터 학습 자료/runs/helmet_model7_data_plus2/args.yaml`

대표 학습 조건 요약:

```text
model=helmet_model_ver0.5.pt
epochs=100
batch=4
imgsz=768
device=0
optimizer=AdamW
amp=True
save_period=10
```

모델 관리 파일:

- `models/model_manifest.json`

관리 기준:

- 모델명
- 수행 task
- PyTorch/ONNX/TensorRT artifact 경로
- input size
- class 목록
- 배포 target
- acceptance criteria
- latest evaluation

현재 acceptance criteria 예시:

| 모델 | 기준 |
|---|---|
| 헬멧 감지 | precision 0.85 이상, recall 0.90 이상, 평균 latency 50ms 이하 |
| 낙상/포즈 감지 | precision 0.80 이상, recall 0.85 이상, 평균 latency 60ms 이하 |
| 사람 감지 fallback | precision 0.80 이상, recall 0.85 이상, 평균 latency 50ms 이하 |
| 외형 속성 분석 | attribute accuracy 0.80 이상, 평균 latency 35ms 이하 |

#### 4.3.4 모델 성능

학습 결과는 Ultralytics YOLO의 `results.csv` 기준으로 정리했다.
대표 모델은 `helmet_model7_data_plus2`이며, 마지막 epoch 기준 성능은 다음과 같다.

| 모델 | epoch | precision(B) | recall(B) | mAP50(B) | mAP50-95(B) | 비고 |
|---|---:|---:|---:|---:|---:|---|
| `helmet_model7_data_plus2` | 100 | 0.95471 | 0.94161 | 0.97835 | 0.69082 | 대표 제출 후보 |

학습 이력 비교:

| 모델/run | epoch | precision(B) | recall(B) | mAP50(B) | mAP50-95(B) |
|---|---:|---:|---:|---:|---:|
| `helmet_model3` | 216 | 0.91810 | 0.87972 | 0.93194 | 0.61432 |
| `helmet_model4` | 100 | 0.91373 | 0.86853 | 0.92772 | 0.57953 |
| `helmet_model5` | 30 | 0.91002 | 0.87742 | 0.92716 | 0.56164 |
| `helmet_model6_1024` | 20 | 0.91752 | 0.89786 | 0.94372 | 0.57273 |
| `helmet_model7_first` | 137 | 0.93009 | 0.89881 | 0.94591 | 0.54339 |
| `helmet_model7_data_plus2` | 100 | 0.95471 | 0.94161 | 0.97835 | 0.69082 |

결과 해석:

- `helmet_model7_data_plus2`는 이전 학습 run 대비 precision, recall, mAP50, mAP50-95가 모두 가장 높다.
- 헬멧 미착용 감지 업무 특성상 현장 안전 리스크를 줄이기 위해 recall을 중요 지표로 본다.
- 현재 대표 모델의 recall은 0.94161로, manifest의 헬멧 감지 기준인 recall 0.90 이상을 만족한다.
- 단, 위 성능은 학습/검증 데이터셋 기준이며, 실제 현장 CCTV에서는 조명, 카메라 각도, 안전모 색상, 가림 현상에 따른 별도 검증이 필요하다.

평가 워크플로우:

```bash
python scripts/evaluate_detection.py \
  --model models/helmet_model_ver0.5.onnx \
  --dataset data/eval/helmet \
  --output reports/eval/helmet_model_ver0.5.json \
  --imgsz 320 \
  --conf 0.35 \
  --iou 0.5 \
  --target-classes helmet,head
```

리포트 기준 확인:

```bash
python scripts/check_model_report.py \
  --model-name helmet_model_ver0.5 \
  --report reports/eval/helmet_model_ver0.5.json
```

아직 보완이 필요한 부분:

- 학습데이터 출처/라벨링 화면 캡처 첨부
- 실제 현장 CCTV 영상 기준 false positive per hour 측정
- Jetson TensorRT `.engine` 기준 성능 리포트 분리
- 모델별 end-to-end latency, FPS, GPU/RAM 사용량 측정

관련 문서:

- `docs/MLOPS_MODEL_EVALUATION.md`
- `models/model_manifest.json`

### 4.4 Jetson/DeepStream 엣지 배포

운영 환경은 NVIDIA Jetson Orin을 고려하여 구성했다.

구현 내용:

- Jetson 전용 Dockerfile 구성
- Jetson 전용 Docker Compose 구성
- DeepStream processor 구현
- TensorRT `.engine` 모델 자동 인식 구조
- GStreamer 기반 하드웨어 디코딩 옵션 지원
- Jetson/EdgeX 현장 점검 스크립트 작성

관련 파일:

- `Dockerfile.jetson`
- `docker-compose.jetson.yml`
- `src/core/deepstream_processor.py`
- `config/deepstream/*`
- `scripts/check_jetson_edgex_stack.py`

검증된 내용:

- `cctv-ai-engine` 컨테이너 healthy
- Docker runtime이 `nvidia`로 동작
- 컨테이너 내부 `gi`, `pyds` import 확인
- DeepStream 처리 로그에서 frames/events 증가 확인
- 30분 안정성 관찰에서 `RestartCount=0`, `dropped=0` 확인

주의:

- 위 검증 결과는 `docs/PROJECT_REVIEW_2026-06.md` 기준이다.
- 장비, JetPack 버전, 모델 파일, 카메라 입력이 바뀌면 재검증이 필요하다.

### 4.5 Public API 및 Action Layer

Public API는 대시보드와 외부 서버가 CCTV 이벤트, 카메라, 상태 정보를 조회할 수 있도록 구성했다.

주요 기능:

- Health/readiness 확인
- 카메라 목록 조회
- 최근 이벤트 조회
- 사이트/구역/알람 제어 API
- 외형 검색 및 상태 API
- 내부 서비스와 Action Layer 프록시 연동

Action Layer는 이벤트를 받아 실제 알람과 외부 전송으로 연결하는 계층이다.

주요 기능:

- 이벤트 저장
- 스피커 알람
- 전광판/경광등 연동 확장
- 외부 API 호출
- 수동 승인/거절 흐름
- 메트릭 제공

관련 구현 위치:

- `src/api/`
- `src/services/action_bridge.py`
- `src/services/stream_api.py`
- `src/storage/sqlite.py`

### 4.6 운영 점검 및 테스트 자동화

운영 및 시연 안정성을 위해 점검 스크립트와 테스트를 구성했다.

주요 점검 스크립트:

| 스크립트 | 목적 |
|---|---|
| `scripts/smoke_test_deployment.py` | health/readiness 중심 배포 점검 |
| `scripts/smoke_test_data_flow.py` | Alert API, Action Layer, Public API 데이터 흐름 점검 |
| `scripts/check_sensitive_defaults.py` | 민감 기본값 유출 점검 |
| `scripts/check_jetson_edgex_stack.py` | Jetson/EdgeX/Action Layer 현장 점검 |
| `scripts/check_model_report.py` | 모델 평가 리포트 기준 확인 |
| `scripts/check_offline_readiness.py` | 오프라인 배포 준비 상태 확인 |

검증 결과 예시:

```text
전체 테스트: 771 passed, 75 skipped
핵심 데모 경로 테스트: 138 passed
민감 기본값 검사: No sensitive defaults found.
Jetson/DeepStream 30분 안정성 관찰: healthy, RestartCount=0, dropped=0
```

관련 문서:

- `docs/OPERATIONS_RUNBOOK.md`
- `docs/PROJECT_REVIEW_2026-06.md`
- `docs/JETSON_EDGEX_FIELD_CHECKLIST.md`

## 5. 데이터 흐름

### AI 이벤트 흐름

```text
RTSP/웹캠/비디오
  -> AI Engine
  -> DetectionEvent
  -> MQTT publish
  -> EdgeX Adapter
  -> EdgeX event
  -> Kuiper Rule
  -> Action Layer
  -> 스피커/전광판/경광등/외부 API
```

### Public API 조회 흐름

```text
AI/Action/External 이벤트
  -> SQLite 또는 내부 상태 저장
  -> Public API
  -> Web UI / 외부 서버 / 대시보드
```

### 외형 분석 흐름

```text
YOLO person bbox
  -> crop
  -> HSV 또는 PP-Human attribute backend
  -> appearance_log 저장
  -> Public API 검색
```

## 6. 검증 결과

2026년 6월 리뷰 문서 기준으로 확인된 결과는 다음과 같다.

- 전체 테스트: `771 passed, 75 skipped`
- 핵심 데모 경로 테스트: `138 passed`
- 민감 기본값 검사 통과
- Public API, Alert API, Action Layer, Stream API, Demo UI 응답 확인
- EdgeX core-data 응답 확인
- Stream API 스냅샷 생성 확인
- Jetson/DeepStream 컨테이너 healthy 확인
- DeepStream 처리 로그에서 프레임과 이벤트 지속 증가 확인
- 30분 안정성 관찰에서 컨테이너 재시작 없음
- 헬멧 감지 대표 학습 모델 `helmet_model7_data_plus2` 기준 precision 0.95471, recall 0.94161, mAP50 0.97835 확인

실행 명령 예시:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/check_sensitive_defaults.py
.venv/bin/python scripts/smoke_test_deployment.py
.venv/bin/python scripts/smoke_test_data_flow.py
.venv/bin/python scripts/check_jetson_edgex_stack.py --host localhost --public-api-port 9000 --deepstream --check-appearance-status
```

## 7. 기술적으로 고민한 점

### 7.1 EdgeX와 SQLite 역할 분리

EdgeX는 이벤트 표준화와 외부 연동에 강점이 있고, SQLite는 엣지 장비의 로컬 저장과 장애 복구에 적합하다.
따라서 SQLite를 중앙 DB처럼 크게 키우기보다, 엣지 로컬 outbox, 운영 로그, 최근 검색용 저장소로 사용하는 구조를 권장했다.

### 7.2 Jetson에서 성능과 유지보수 균형

Jetson에서는 정확도만 높이는 것보다 FPS, 메모리, GPU 사용량, 배포 난이도가 중요하다.
그래서 PyTorch 모델만 사용하는 구조가 아니라 ONNX/TensorRT 변환을 고려했고, DeepStream 기반 처리 경로를 별도로 마련했다.

### 7.3 모델 교체 가능성

헬멧, 사람, 포즈, 외형 분석 모델은 향후 교체될 수 있다.
이를 위해 모델 경로를 환경변수로 분리하고, `models/model_manifest.json`에 모델별 목적과 기준을 기록하는 방식으로 정리했다.

### 7.4 이벤트 표준화와 하위 호환

기존 소비자가 `camera_id`, `type`, `confidence` 같은 평면 필드를 사용하고 있기 때문에 이를 바로 제거하지 않았다.
대신 신규 소비자를 위해 `schema_version`, `device`, `event`, `decoded`, `raw`, `event_id` 구조를 함께 제공했다.

## 8. 한계 및 향후 개선 사항

현재 남은 주요 과제는 다음과 같다.

1. 실제 현장 데이터 기준 모델 평가 리포트 작성
2. Jetson TensorRT 모델별 latency 측정
3. 카메라별 threshold 및 zone별 이벤트 정책 세분화
4. 운영 보안 강화
5. 실제 스피커/전광판/경광등 장비 연결 테스트
6. EdgeX/Action Layer 장애 상황에서 outbox 재전송 검증
7. 장기 통계용 중앙 DB 또는 리포트 시스템 분리

운영 전환 시 우선 잠가야 할 보안 항목:

- `PUBLIC_API_KEY` 필수화
- `INTERNAL_SERVICE_TOKEN` 필수화
- CORS 도메인 제한
- Stream API 접근 제한
- Grafana/MQTT/DB 비밀번호 강제
- 외부 노출 포트 최소화

## 9. 참고 문서

- `README.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/PROJECT_REVIEW_2026-06.md`
- `docs/DEVICE_SERVICE_ARCHITECTURE.md`
- `docs/EVENT_SCHEMA_STANDARD.md`
- `docs/EDGEX_SQLITE_DATA_ARCHITECTURE.md`
- `docs/MLOPS_MODEL_EVALUATION.md`
- `docs/PPHUMAN_ATTRIBUTE_INTEGRATION.md`
- `docs/JETSON_EDGEX_FIELD_CHECKLIST.md`
- `docs/OPERATIONS_RUNBOOK.md`

## 추가로 채워야 할 체크리스트

최종 제출 전에 아래 항목을 채우면 문서 완성도가 크게 올라갑니다.

| 항목 | 현재 상태 | 작성 방법 |
|---|---|---|
| 개발 기간 | TODO | 예: 2026.04 ~ 2026.06 |
| 담당 역할 | TODO | 예: AI 모델 연동, EdgeX 연동, Jetson 배포, 테스트 자동화 |
| 학습 데이터 수 | 작성 완료 | 실제 학습 데이터 7,035장, train 5,269 / test 1,766 |
| 데이터 출처 | 공간 마련 | Roboflow 링크, 다운로드 화면, 라벨링 기준 이미지 첨부 |
| 라벨링 증빙 | 공간 마련 | bbox 예시 이미지, class 정의, 변환 과정 첨부 |
| 학습 환경 | 작성 완료 | Jetson CUDA, YOLO, epoch 100, batch 4, imgsz 768 |
| 모델 성능 | 작성 완료 | 대표 모델 precision 0.95471, recall 0.94161, mAP50 0.97835 |
| Jetson 성능 | 일부 확인 | FPS, RAM, GPU 온도, 컨테이너 메모리 |
| 실제 장비 테스트 | TODO | 스피커/전광판/경광등 연결 결과 |
| 현장 테스트 결과 | TODO | 카메라 RTSP, 네트워크, 장애 복구 결과 |

## 문장 템플릿

### 업무 요약 문장

```text
본인은 CCTV AI 엣지 관제 시스템에서 영상 AI 이벤트 생성, EdgeX 기반 이벤트 연동,
Jetson/DeepStream 배포 구조, 운영 점검 자동화 업무를 수행했다.
```

### AI 모델 구현 문장

```text
YOLOv8 기반 헬멧/사람 감지 모델과 YOLOv8-pose 기반 낙상 감지 모델을 서비스 파이프라인에 통합했다.
모델 추론 결과는 bbox, confidence, class 정보를 기준으로 후처리되며,
zone 조건과 이벤트 필터를 거쳐 표준 DetectionEvent로 변환된다.
```

### EdgeX 구현 문장

```text
AI Engine에서 발생한 이벤트를 MQTT로 발행하고,
EdgeX Adapter에서 이를 구독하여 EdgeX Device/Profile/Resource 구조에 맞는 이벤트로 변환했다.
이를 통해 CCTV AI 이벤트를 EdgeX Core Data, Kuiper Rule Engine, Action Layer로 전달할 수 있는 구조를 구현했다.
```

### AI 학습/평가 문장

```text
모델 변경 시 운영 안정성을 확보하기 위해 모델별 artifact, 입력 크기, class, acceptance criteria를
`models/model_manifest.json`에 기록했다. 또한 `evaluate_detection.py`와 `check_model_report.py`를 통해
precision, recall, false positive, latency 기준으로 모델 교체 여부를 판단할 수 있는 평가 흐름을 마련했다.
```

### Jetson 배포 문장

```text
Jetson Orin 환경에서는 DeepStream, TensorRT, GStreamer 기반의 GPU 가속 처리를 고려하여
전용 Dockerfile과 Compose 구성을 작성했다. 또한 현장 배포 전 MQTT, EdgeX, Public API,
Action Layer, 출력 장비 연결 상태를 점검할 수 있는 스크립트를 구성했다.
```
