# CCTV Project — 실행 명령어 가이드

## 목차
1. [환경 설정](#1-환경-설정)
2. [AI 엔진 (메인 CCTV)](#2-ai-엔진-메인-cctv)
3. [액션 레이어](#3-액션-레이어)
4. [Alert API 서버](#4-alert-api-서버)
5. [EdgeX 어댑터](#5-edgex-어댑터)
6. [Kuiper 룰 배포](#6-kuiper-룰-배포)
7. [외부 MQTT 수신 (External Ingest)](#7-외부-mqtt-수신-external-ingest)
8. [AIoT TLV 파서 서버](#8-aiot-tlv-파서-서버)
9. [테스트](#9-테스트)
10. [Docker Compose](#10-docker-compose)
11. [모니터링 (옵션)](#11-모니터링-옵션)
12. [스모크 테스트 반복 실행](#12-스모크-테스트-반복-실행)

---

## 1. 환경 설정

### 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 의존성 설치

```bash
# AI 엔진 전체 (YOLO, OpenCV, torch 포함)
pip install -r requirements.txt

# 액션 레이어 전용 (torch 제외, 경량)
pip install -r requirements-action.txt

# 개발 도구 (pytest, black, mypy 포함)
pip install -r requirements-dev.txt

# AIoT TLV 파서 서버
pip install -r parser-python/requirements.txt
```

---

## 2. AI 엔진 (메인 CCTV)

진입점: `main.py`

### 기본 실행 (웹캠)

```bash
python main.py
```

### 웹캠 + 화면 표시

```bash
python main.py --display
```

### 비디오 파일 테스트

```bash
python main.py --video sample.mp4 --display
```

### 낙상 보조 검증 모델 shadow 모드

공공 falldata RF 모델은 MediaPipe feature 추출 환경과 sklearn 모델 실행 환경을 분리해서
사용합니다. 운영에 바로 `confirm`으로 넣기 전에 먼저 `shadow`로 metadata만 쌓아 확인하세요.

```bash
python3.10 -m venv .venv-falldata
.venv-falldata/bin/pip install numpy==1.26.1 scipy==1.11.3 scikit-learn==1.3.2 joblib==1.3.2

python3.10 -m venv .venv-mediapipe
.venv-mediapipe/bin/pip install opencv-python-headless mediapipe
```

준비 상태 점검:

```bash
python scripts/health/check_falldata_aux.py
```

샘플 비디오까지 포함한 end-to-end 점검:

```bash
python scripts/health/check_falldata_aux.py \
  --video 'external/OpenPAR/VTFPAR++/demo/video.mp4' \
  --max-frames 30
```

shadow 모드 실행:

```bash
FALLDATA_AUX_ENABLED=true \
FALLDATA_AUX_MODE=shadow \
FALLDATA_AUX_THRESHOLD=0.7 \
FALLDATA_AUX_FALL_CLASS_INDEX=0 \
python main.py --video sample.mp4 --display
```

`confirm` 모드는 보조모델이 낙상을 확인하지 못하면 기존 pose 낙상 이벤트를 버릴 수 있으므로,
현장 shadow 로그를 확인한 뒤에만 사용하세요.

### 다중 카메라 (cameras.json)

```bash
python main.py --cameras cameras.json
```

### CUDA 가속 + 다중 카메라

```bash
python main.py --cameras cameras.json --device cuda --display
```

### YOLO + 외형 속성 분석

HSV 기반 색상 분석은 별도 모델 없이 동작합니다. PP-Human 계열 ONNX 속성 모델을
붙일 때는 모델 파일과 라벨 맵 경로를 함께 지정합니다.

```bash
APPEARANCE_ENABLED=true \
APPEARANCE_BACKEND=pphuman \
APPEARANCE_MODEL_PATH=models/pphuman_attribute.onnx \
APPEARANCE_LABEL_MAP_PATH=config/appearance_pphuman_labels.example.json \
APPEARANCE_RUNTIME=onnxruntime \
python main.py --cameras cameras.json --device cuda
```

Jetson 환경에서 ONNX Runtime wheel이 맞지 않으면 Paddle 원본 모델을 직접 사용할 수 있습니다.

```bash
pip install -r requirements-appearance-paddle.txt

APPEARANCE_ENABLED=true \
APPEARANCE_BACKEND=pphuman \
APPEARANCE_MODEL_PATH=models/pphuman_attribute_src/PP-LCNet_x1_0_pedestrian_attribute_infer \
APPEARANCE_LABEL_MAP_PATH=config/appearance_pphuman_labels.example.json \
APPEARANCE_RUNTIME=paddle \
python main.py --cameras cameras.json --device cuda
```

카메라별 `detections`에는 `appearance`가 포함되어야 하며, 사람 bbox가 필요하므로
`person` 또는 `fall`/pose 감지가 함께 활성화됩니다.

### 외형 성별 임계값 점검

PP-Human 성별 속성은 `female_score` 하나로 나오므로, 애매한 구간은 `unknown`으로
두는 것이 운영 오탐을 줄이는 데 유리합니다. 저장된 crop 기준으로 최근 점수 분포와
임계값별 `male/female/unknown` 비율을 확인합니다.

```bash
python scripts/ops/evaluate_appearance_gender.py --limit 100
```

Jetson 컨테이너 안에서 실제 Paddle 런타임으로 확인하려면:

```bash
docker exec cctv-ai-engine python scripts/ops/evaluate_appearance_gender.py \
  --crop-dir /app/data/runtime/appearance_crops \
  --limit 100
```

현재 보수적 기본값은 아래와 같습니다.

```bash
APPEARANCE_GENDER_FEMALE_MIN_SCORE=0.75
APPEARANCE_GENDER_MALE_MAX_SCORE=0.25
APPEARANCE_GENDER_MIN_SAMPLES=3
```

### RAP/RAPv2 외형 학습 데이터 준비

RAP/RAPv2는 보행자 속성 인식(Pedestrian Attribute Recognition) 데이터셋입니다.
데이터셋 사용 조건과 라이선스를 먼저 확인한 뒤, 내려받은 이미지/annotation을
`data/external/rapv2` 같은 로컬 경로에 둡니다.

MAT annotation 구조를 먼저 점검합니다.

```bash
python scripts/datasets/prepare_rap_attribute_manifest.py \
  --mat data/external/rapv2/RAP_annotation.mat \
  --inspect-json data/processed/rapv2/inspect.json
```

annotation에서 이미지명/속성명/라벨 행렬이 자동 감지되면, 프로젝트용 학습 manifest로
변환합니다.

```bash
python scripts/datasets/prepare_rap_attribute_manifest.py \
  --mat data/external/rapv2/RAP_annotation.mat \
  --image-root data/external/rapv2/images \
  --output-csv data/processed/rapv2/appearance_manifest.csv
```

출력 CSV는 아래 공통 필드로 정리됩니다.

```text
image_path,gender,upper_color,lower_color,bag,hat,source_active_attributes
```

변환된 manifest를 학습/검증 리스트와 프로젝트용 label map으로 나눕니다.

```bash
python scripts/datasets/build_appearance_training_lists.py \
  --manifest data/processed/rapv2/appearance_manifest.csv \
  --output-dir data/processed/rapv2 \
  --val-ratio 0.2
```

생성 파일:

```text
data/processed/rapv2/train_list.txt
data/processed/rapv2/val_list.txt
data/processed/rapv2/appearance_label_map.json
data/processed/rapv2/summary.json
```

`train_list.txt`와 `val_list.txt`는 이미지 경로 뒤에 multi-label 0/1 vector가 붙는 형식입니다.

```text
data/external/rapv2/images/000001.jpg 0 1 0 0 ...
```

label map은 현재 PP-Human decoder와 맞도록 `gender=female` 단일 score,
`upper_color`, `lower_color`, `has_bag`, `has_hat` 구조로 생성됩니다.

RAP/RAPv2 annotation을 이미 CSV로 변환한 경우에는 CSV 입력도 사용할 수 있습니다.

```bash
python scripts/datasets/prepare_rap_attribute_manifest.py \
  --annotations-csv data/external/rapv2/attributes.csv \
  --image-root data/external/rapv2/images \
  --output-csv data/processed/rapv2/appearance_manifest.csv
```

주의:

- 공개 데이터셋만으로 바로 운영 성능을 보장할 수 없습니다.
- RAP/RAPv2로 기본 fine-tuning 후, `data/runtime/appearance_crops`에서 수집한 현장 crop을
  300~1,000장 정도 추가 라벨링해 한 번 더 fine-tuning하는 것을 권장합니다.
- 성별은 애매한 경우 `unknown`으로 남기는 정책을 유지합니다.

PaddleClas/PP-LCNet 계열로 학습할 때는 위 `train_list.txt`, `val_list.txt`를 dataset 입력으로
연결하고, export된 inference model 경로를 `APPEARANCE_MODEL_PATH`로 지정합니다.

```bash
APPEARANCE_MODEL_PATH=models/pphuman_attribute_rapv2_finetuned
APPEARANCE_LABEL_MAP_PATH=data/processed/rapv2/appearance_label_map.json
APPEARANCE_RUNTIME=paddle
```

### Rethinking_of_PAR 기준선 학습 준비

PyTorch 기반 Pedestrian Attribute Recognition 기준선으로
`valencebond/Rethinking_of_PAR`를 `external/Rethinking_of_PAR`에 받을 수 있습니다.
이 저장소는 dependency가 오래된 편이라 운영 `.venv`에 섞지 말고 별도 conda/venv에서 사용합니다.

```bash
git clone https://github.com/valencebond/Rethinking_of_PAR.git external/Rethinking_of_PAR
```

우리 manifest를 해당 저장소의 `dataset_all.pkl` 포맷으로 변환합니다.

```bash
python scripts/datasets/build_rethinking_par_dataset.py \
  --manifest data/processed/rapv2/appearance_manifest.csv \
  --image-root data/external/rapv2/images \
  --output-pkl external/Rethinking_of_PAR/data/RAP2/dataset_all.pkl \
  --val-ratio 0.2
```

학습 저장소에서 실행합니다.

```bash
cd external/Rethinking_of_PAR
CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/rapv2.yaml
```

주의:

- `Rethinking_of_PAR` 기본 README는 Python 3.7, PyTorch 1.x 계열을 전제로 합니다.
- 현재 프로젝트의 운영 환경과 분리된 학습 전용 환경에서 실행합니다.
- 학습이 끝난 `.pth`는 그대로 Jetson 운영에 넣기보다 ONNX export 또는 별도 추론 어댑터를 거쳐 연결합니다.

### OpenPAR 데이터셋 번들 확인

RAP/RAPv2 개별 다운로드가 어려우면 OpenPAR에서 안내하는 benchmark bundle을 우선 확인합니다.

```text
OpenPAR: https://github.com/Event-AHU/OpenPAR
PAR_public_benchmark_datasets: 10.3GB
지원 데이터셋: PETA, PA100K, RAPv1, RAPv2, WIDER, MSP60K
```

추천 순서:

1. OpenPAR README의 Dropbox 또는 BaiduDrive 링크로 benchmark bundle을 내려받습니다.
2. 압축을 풀어 `data/external/openpar` 아래에 배치합니다.
3. 우선 `PA100k`부터 검증합니다.

예상 구조:

```text
data/external/openpar/
  PA100k/
    data/
      000001.jpg
      ...
    annotation.mat
    dataset_all.pkl
```

다운로드 후 구조를 확인합니다.

```bash
python scripts/datasets/check_par_dataset_layout.py \
  --dataset-root external/OpenPAR \
  --dataset PA100K
```

`ready: True`가 나오면 학습용 pkl 또는 manifest 변환 단계로 넘어갑니다.

현재 PA100K 준비 상태 확인 예시:

```text
image_count: 100000
pkl_images: 100000
pkl_attributes: 26
pkl_first_image_exists: True
ready: True
```

`dataset_all.pkl` 내부 이미지 root가 다른 PC 경로를 가리키면 현재 이미지 폴더로 패치합니다.

```bash
python scripts/datasets/patch_par_dataset_root.py \
  --pkl external/OpenPAR/PA100k/dataset_all.pkl \
  --image-root external/OpenPAR/PA100k/data
```

`Rethinking_of_PAR` 기준선 학습에 같은 데이터를 연결하려면 pkl을 복사하고 이미지 폴더를 symlink로 연결합니다.

```bash
cp external/OpenPAR/PA100k/dataset_all.pkl \
  external/Rethinking_of_PAR/data/PA100k/dataset_all.pkl

ln -s ../../../OpenPAR/PA100k/data \
  external/Rethinking_of_PAR/data/PA100k/data
```

주의:

- 현재 운영 `.venv`는 학습 전용 환경이 아니므로 `Rethinking_of_PAR/train.py`를 바로 실행하면 `mmcv`, `yacs`, `tensorboard`, `visdom` 같은 학습 의존성이 없을 수 있습니다.
- 학습은 별도 conda/venv에서 진행하고, 운영 `.venv`에는 학습 의존성을 섞지 않습니다.
- PA100K 데이터 연결 확인 결과, `external/Rethinking_of_PAR` 기준으로 1-batch forward가 통과했습니다.

```text
images: (2, 3, 256, 192)
labels: (2, 26)
logits: (2, 26)
feature: (2, 2048, 8, 6)
```

현재 장비에서는 CUDA 초기화가 실패해 `torch.cuda.is_available()`가 `False`입니다. 전체 학습은 CUDA가 정상 동작하는 학습 환경에서 실행합니다.

Jetson에서 학습용 `.venv-par`를 사용할 때는 cuSPARSELt를 `.venv-par/lib`에 둔 상태로
`LD_LIBRARY_PATH`를 함께 지정합니다.

```bash
cd /media/sawwave/Learning11/CCTV-project/external/Rethinking_of_PAR

LD_LIBRARY_PATH=/media/sawwave/Learning11/CCTV-project/.venv-par/lib:${LD_LIBRARY_PATH:-} \
MPLCONFIGDIR=/tmp/matplotlib-cache \
/media/sawwave/Learning11/CCTV-project/.venv-par/bin/python train.py \
  --cfg ./configs/pedes_baseline/pa100k.yaml
```

현재 확인된 CUDA 스모크 결과:

```text
torch: 2.5.0a0+872d972e41.nv24.08
torch cuda: 12.6
cuda available: True
device: Orin
PA100K 1-batch CUDA forward: PASS
images: (2, 3, 256, 192)
labels: (2, 26)
logits: (2, 26)
```

### PA100K 학습 모델 export 및 연결

학습 완료 후 best checkpoint를 ONNX로 변환합니다. Jetson용 PyTorch 환경은
`libcusparseLt` 경로가 필요할 수 있으므로 `LD_LIBRARY_PATH`를 같이 지정합니다.

```bash
LD_LIBRARY_PATH=/media/sawwave/Learning11/CCTV-project/.venv-par/lib:${LD_LIBRARY_PATH:-} \
  .venv-par/bin/python scripts/convert/export_rethinking_par_onnx.py \
  --checkpoint external/Rethinking_of_PAR/exp_result/PA100k/resnet50.base.adam/img_model/ckpt_max_2026-06-19_11:19:51.pth \
  --output models/pa100k_resnet50_attr.onnx
```

현재 확인된 best checkpoint:

```text
epoch: 24
metric: 0.7650567363313108
input: [batch, 3, 256, 192]
output: [batch, 26]
```

PA100K 라벨맵은 아래 경로를 사용합니다.

```text
config/appearance_pa100k_labels.json
```

ONNXRuntime이 안정적으로 로드되는 환경에서는 아래 값으로 외형 백엔드를 PA100K 모델로
연결합니다.

```bash
APPEARANCE_ENABLED=true
APPEARANCE_BACKEND=pphuman
APPEARANCE_MODEL_PATH=models/pa100k_resnet50_attr.onnx
APPEARANCE_LABEL_MAP_PATH=config/appearance_pa100k_labels.json
APPEARANCE_RUNTIME=onnxruntime
APPEARANCE_SCORE_THRESHOLD=0.5
```

Jetson 운영 컨테이너에서는 PyPI `onnxruntime`가 native crash를 낼 수 있으므로,
DeepStream SGIE/TensorRT 경로를 우선 사용합니다. ONNX를 engine으로 변환합니다.

```bash
docker exec cctv-ai-engine /usr/src/tensorrt/bin/trtexec \
  --onnx=/app/models/pa100k_resnet50_attr.onnx \
  --saveEngine=/app/models/pa100k_resnet50_attr.engine \
  --fp16 \
  --builderOptimizationLevel=0 \
  --avgTiming=1 \
  --minShapes=images:1x3x256x192 \
  --optShapes=images:1x3x256x192 \
  --maxShapes=images:8x3x256x192 \
  --skipInference
```

DeepStream SGIE에서 PA100K engine을 쓰려면 아래 값을 지정합니다.

```bash
DS_PPHUMAN_INFER_CONFIG=config/deepstream/config_infer_pa100k.txt
APPEARANCE_LABEL_MAP_PATH=config/appearance_pa100k_labels.json
```

현재 primary YOLO는 raw tensor를 Python에서 직접 파싱하므로 DeepStream SGIE ROI 연결이
제한될 수 있습니다. 운영에서는 외형 worker가 TensorRT engine을 직접 실행하도록 아래
설정을 함께 사용합니다.

```bash
APPEARANCE_MODEL_PATH=models/pa100k_resnet50_attr.engine
APPEARANCE_RUNTIME=tensorrt
APPEARANCE_LABEL_MAP_PATH=config/appearance_pa100k_labels.json
```

주의:

- PA100K 모델은 성별, 연령대, 방향, 모자, 안경, 가방, 상/하의 형태 같은 26개 속성을 출력합니다.
- 상/하의 색상은 PA100K 라벨에 없으므로 기존 HSV 색상 추정과 함께 사용해야 합니다.
- 현재 Jetson 기본 Python 환경의 `onnxruntime`는 native crash가 날 수 있습니다. 이 경우 바로 운영에
  연결하지 말고 ONNXRuntime 패키지/컨테이너를 정리하거나 TensorRT 실행 어댑터를 별도로 추가합니다.

### 헬멧/낙상 감지 + MQTT 전송

```bash
python main.py \
  --cameras cameras.json \
  --device cuda \
  --confidence 0.5 \
  --pose-confidence 0.3 \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --mqtt-topic-prefix cctv/ai/events
```

### 위험 구역 탐지 + Zone API 활성화

```bash
python main.py \
  --cameras cameras.json \
  --zone-detection \
  --zones-config zones_config.json \
  --api-port 8765
```

### 데이터셋 수집 모드

```bash
python main.py \
  --cameras cameras.json \
  --collect-dataset \
  --dataset-dir ./collected_data
```

### 주요 인자 요약

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--cameras` | (없음) | 카메라 목록 JSON 파일 |
| `--video` | (없음) | 단일 비디오 파일 |
| `--device` | `cpu` | `cpu` 또는 `cuda` |
| `--confidence` | `0.5` | 헬멧 감지 임계값 |
| `--pose-confidence` | `0.3` | 사람 감지 임계값 |
| `--fps` | `30` | 목표 FPS |
| `--frame-skip` | `3` | AI 추론 간격 (N프레임마다) |
| `--display` | off | 화면 표시 |
| `--mqtt-broker` | `localhost` | MQTT 브로커 호스트 |
| `--mqtt-port` | `1883` | MQTT 브로커 포트 |
| `--api-port` | `0` | Zone API 포트 (0=비활성) |
| `--zone-detection` | off | 위험 구역 탐지 활성화 |
| `--no-debounce` | off | 이벤트 디바운싱 비활성화 |
| `--debounce` | `3.0` | 디바운싱 간격 (초) |

---

## 3. 액션 레이어

진입점: `runners/run_action_bridge.py`

스피커 / 전광판 / 경광등 조치 실행, 외부 플랫폼 HTTP 전송, SQLite 이벤트 저장.

### 이벤트 오탐 검수

Public API에서 이벤트를 `맞음 / 오탐 / 애매함`으로 라벨링할 수 있습니다.
검수 결과는 원본 이벤트 로그와 분리되어 `EVENT_REVIEW_DB`
기본값 `/app/data/runtime/event_reviews.db`에 저장됩니다.

```bash
curl -X POST "http://localhost:9000/api/v1/event-reviews" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "example-event-id",
    "status": "false_positive",
    "reviewer": "operator",
    "category": "head",
    "note": "화면 가장자리 작업자 오탐"
  }'
```

검수 누적 요약:

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "http://localhost:9000/api/v1/event-reviews/summary"
```

이벤트 조회 응답에는 운영용 `risk_score`도 포함됩니다. 값은 0~100이며 높을수록
우선 확인할 이벤트입니다. 모델 confidence, 이벤트 유형, severity, bbox 품질,
검수 결과를 함께 반영합니다.

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "http://localhost:9000/api/v1/events?limit=5"
```

### 기본 실행

```bash
python runners/run_action_bridge.py
```

### 환경 변수로 실행 (권장)

```bash
# .env 또는 shell export
export MQTT_BROKER=localhost
export MQTT_PORT=1883
export DB_PATH=/app/data/runtime/action_events.db

python runners/run_action_bridge.py
```

### CLI 인자로 실행

```bash
python runners/run_action_bridge.py \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --db-path ./action_events.db \
  --subscribe-topics "cctv/rules/intrusion/filtered,cctv/rules/intrusion/critical" \
  --alarm-topics "cctv/rules/intrusion/critical"
```

---

## 4. Alert API 서버

진입점: `runners/run_alert_api.py`

내부 HTTP Alert 수신 서버. 외부 플랫폼 → CCTV 시스템으로 알림 수신.

### 기본 실행

```bash
python runners/run_alert_api.py
```

### 포트 및 로그 경로 지정

```bash
python runners/run_alert_api.py \
  --host 0.0.0.0 \
  --port 8000 \
  --log-path ./alert_api_events.jsonl
```

### 헬스 체크 (서버 실행 후)

```bash
curl http://localhost:8000/health
```

### 알림 전송 테스트

```bash
curl -X POST http://localhost:8000/api/alerts \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam1", "event": "intrusion"}'
```

---

## 5. EdgeX 어댑터

진입점: `runners/run_edgex_adapter.py`

AI 엔진 MQTT 이벤트 → EdgeX Core Data/MessageBus 브릿지.

### 기본 실행

```bash
python runners/run_edgex_adapter.py
```

### EdgeX와 연동

```bash
python runners/run_edgex_adapter.py \
  --ai-mqtt-broker localhost \
  --ai-mqtt-port 1883 \
  --ai-topic-prefix cctv/ai/events \
  --edgex-metadata-url http://localhost:59881 \
  --edgex-data-url http://localhost:59880 \
  --edgex-mqtt-broker localhost \
  --edgex-mqtt-port 1883 \
  --service-name cctv-device-service
```

---

## 6. Kuiper 룰 배포

진입점: `runners/run_kuiper_rules.py`

eKuiper Rules Engine에 침입 감지 룰(스트림 + 규칙)을 배포.

### 기본 실행

```bash
python runners/run_kuiper_rules.py
```

### 브로커 및 Kuiper API 지정

```bash
python runners/run_kuiper_rules.py \
  --kuiper-api http://localhost:59720 \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --rules-file kuiper/rules/cctv_intrusion_rules.json
```

### 신뢰도 및 재시도 설정

```bash
python runners/run_kuiper_rules.py \
  --intrusion-confidence 0.7 \
  --critical-confidence 0.9 \
  --persist-hit-count 3 \
  --retry-count 5 \
  --retry-delay 5
```

---

## 7. 외부 MQTT 수신 (External Ingest)

진입점: `run_external_ingest.py`

외부 MQTT 브로커 구독 → 내부 이벤트 정규화 → (선택) 내부 MQTT 재발행.

### 기본 실행

```bash
python run_external_ingest.py \
  --mqtt-broker external-broker.example.com \
  --mqtt-port 1883 \
  --topic "sensors/#" \
  --topic "alerts/#"
```

### 인증 + 재발행

```bash
python run_external_ingest.py \
  --mqtt-broker external-broker.example.com \
  --mqtt-port 8883 \
  --mqtt-username myuser \
  --mqtt-password mypass \
  --topic "sensors/#" \
  --republish \
  --republish-broker localhost \
  --republish-port 1883 \
  --republish-topic-prefix cctv/external
```

### DB 저장 경로 지정

```bash
python run_external_ingest.py \
  --mqtt-broker localhost \
  --topic "test/#" \
  --db-path ./ingest_raw.db
```

---

## 8. AIoT TLV 파서 서버

진입점: `parser-python/main.py`

LoRaWAN TLV 센서 데이터 수신 → 파싱 → PostgreSQL 저장 + EdgeX 발행.

### 환경 변수 준비

```bash
# parser-python/.env 파일 생성
cp parser-python/.env.example parser-python/.env  # 없으면 직접 작성
```

`.env` 예시:
```ini
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PW=yourpassword
DB_NAME=aiot_sensor

NS_PARK_MQTT_HOST=ns.example.com
NS_PARK_MQTT_PORT=1883
NS_PARK_MQTT_ID=user

LAB_MQTT_HOST=lab-broker.example.com
LAB_MQTT_PORT=1883

EDGEX_MQTT_HOST=localhost
EDGEX_MQTT_PORT=1883

ROUTER=3500
NC_APPLICATION_IDS=app1,app2
NC_API_RUI=http://localhost:3000/api/v1/devices
```

### 서버 실행

```bash
cd parser-python
python main.py
```

### TLV 파서 단독 테스트

```bash
cd parser-python
pytest tests/test_tlv_parser.py -v
```

### 실시간 MQTT 수신 모니터링 (개발용)

```bash
cd parser-python
python live_receiver.py

# 특정 브로커만
python live_receiver.py --broker ns_park

# 다른 .env 경로 지정
python live_receiver.py --env ../.env
```

---

## 9. 테스트

### 전체 테스트 실행

```bash
python -m pytest tests/
```

### 상세 출력

```bash
python -m pytest tests/ -v
```

### 특정 파일만

```bash
python -m pytest tests/test_zone_detection.py -v
python -m pytest tests/test_ai_analysis.py -v
python -m pytest tests/test_action_bridge.py -v
```

### 커버리지 측정

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### TLV 파서 테스트 (별도 경로)

```bash
python -m pytest parser-python/tests/ -v
```

---

## 10. Docker Compose

### 전체 스택 시작

```bash
docker compose up -d
```

### 특정 서비스만 시작

```bash
# AI 엔진만
docker compose up -d cctv-ai-engine

# 액션 레이어만
docker compose up -d cctv-action-layer
```

### 로그 확인

```bash
docker compose logs -f cctv-ai-engine
docker compose logs -f cctv-action-layer
docker compose logs -f app-rules-engine
```

### 재시작

```bash
docker compose restart cctv-action-layer
```

### 서비스 재빌드 후 시작

```bash
docker compose up -d --build cctv-ai-engine

# Docker 빌드 전 정적 배포 준비 점검
python scripts/health/check_deployment_readiness.py

# 컨테이너 기동 후 API/Prometheus/Grafana 스모크 테스트
python scripts/smoke/smoke_test_deployment.py

# 컨테이너 기동 후 alert/sensor/action 데이터 플로우 스모크 테스트
python scripts/smoke/smoke_test_data_flow.py

# Dockerfile COPY 대상 파일 누락만 빠르게 확인
python scripts/health/check_dockerfile_sources.py

# helper 스크립트 사용
./docker-build.sh cctv-public-api cctv-action-layer

# Jetson compose 파일로 빌드
COMPOSE_FILE=docker-compose.jetson.yml ./docker-build.sh cctv-alert-api

# 빌드만 하고 시작하지 않기
START_AFTER_BUILD=0 ./docker-build.sh cctv-action-layer
```

### 전체 중지 및 제거

```bash
docker compose down
```

### Jetson Orin 전용 Compose

```bash
cp .env.jetson.example .env.jetson
docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d
```

Jetson 통합 스택은 compose project가 `edgex-jetson`으로 뜨므로, 기본
`docker compose ps`가 비어 보일 수 있습니다. Jetson 컨테이너를 확인하거나
재시작할 때는 반드시 `docker-compose.jetson.yml`을 지정합니다.

```bash
# Jetson 스택 서비스 확인
docker compose -f docker-compose.jetson.yml ps

# AI 엔진만 재시작
docker compose -f docker-compose.jetson.yml restart cctv-ai-engine

# 재시작 반영 확인
docker inspect cctv-ai-engine \
  --format '{{.Name}} {{.State.Status}} started={{.State.StartedAt}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'

# DeepStream/얼굴/외형 분석 로그 확인
docker logs --tail 120 cctv-ai-engine

# AI 엔진 내부 API health 확인
curl -fsS http://localhost:8765/health

# MJPEG Stream API health / 카메라 목록 확인
curl -fsS http://localhost:8769/health
curl -fsS http://localhost:8769/cameras
```

---

## 11. 모니터링 (옵션)

Prometheus / Grafana는 기본 스택과 분리된 `docker-compose.monitoring.yml`로 관리합니다.  
필요할 때만 켜고, 평상시엔 끄면 됩니다.

### 모니터링 스택 시작 (메인 스택과 함께)

```bash
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d prometheus grafana
```

### 모니터링만 단독으로 시작/중지

```bash
# 시작
docker compose -f docker-compose.monitoring.yml up -d

# 중지 (볼륨은 유지)
docker compose -f docker-compose.monitoring.yml down
```

### 접속

| 서비스      | URL                        |
|------------|----------------------------|
| Prometheus | http://localhost:9090       |
| Grafana    | http://localhost:3001       |

Grafana 초기 비밀번호: `.env`의 `GRAFANA_ADMIN_PASSWORD` (미설정 시 `admin`)

---

## 12. 스모크 테스트 반복 실행

이벤트/센서/액션 흐름 안정성 확인용 루프 스크립트:

```bash
# 60분간 30초 간격으로 반복 (기본값)
bash scripts/smoke/run_smoke_loop.sh 60 30

# 30분간 15초 간격
bash scripts/smoke/run_smoke_loop.sh 30 15
```

PASS/FAIL 누적 카운트와 실패 상세 내용을 실시간 출력합니다.  
전체 PASS면 exit 0, 하나라도 FAIL이면 exit 1을 반환합니다.
