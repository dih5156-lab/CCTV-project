# Model Evaluation Workflow

이 문서는 현재 CCTV 프로젝트에서 모델을 바꿀 때 최소한으로 확인할 MLOps 기준을 정리한다.

## 목적

- `models/model_manifest.json`에 모델 목적, artifact, 운영 기준을 기록한다.
- 고정 평가 데이터셋으로 precision, recall, false positive, latency를 산출한다.
- 평가 리포트를 `data/eval/reports/*.json`에 저장해 모델 교체 판단 근거로 사용한다.

## 모델 아티팩트 보관 정책

- 운영에 필요한 PyTorch 원본만 `models/*.pt`에 두고 `models/model_manifest.json`에 등록한다.
- 저장소 루트에 자동 다운로드되는 `yolo*.pt`는 캐시이므로 커밋하지 않는다.
- 장치별 생성물인 `*.onnx`, `*.engine`과 학습 중간 결과는 커밋하지 않는다.
- 단일 모델이 GitHub 일반 파일 제한에 가까워지면 Git LFS 또는 외부 모델 저장소를 먼저 도입한다.
- 모델을 교체할 때는 파일만 덮어쓰지 말고 manifest의 평가 결과와 체크섬을 함께 갱신한다.

## 평가 데이터 구조

```text
data/eval/helmet/
  images/
    frame_000001.jpg
  labels/
    frame_000001.txt
  classes.txt
```

`labels/*.txt`는 YOLO 형식이다.

```text
class_id x_center y_center width height
```

좌표는 0~1 사이의 normalized 값이어야 한다.

## 실행 예시

헬멧 모델 평가:

```bash
python scripts/ops/evaluate_detection.py \
  --model models/helmet_model_ver0.5.onnx \
  --dataset data/eval/helmet \
  --output data/eval/reports/helmet_model_ver0.5.json \
  --imgsz 320 \
  --conf 0.35 \
  --iou 0.5 \
  --warmup 1 \
  --target-classes helmet,head
```

사람 탐지 fallback 모델 평가:

```bash
python scripts/ops/evaluate_detection.py \
  --model models/yolov8n.onnx \
  --dataset data/eval/person \
  --output data/eval/reports/yolov8n.json \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.5 \
  --warmup 1 \
  --target-classes person
```

빠른 smoke test:

```bash
python scripts/ops/evaluate_detection.py \
  --model models/helmet_model_ver0.5.onnx \
  --dataset data/eval/helmet \
  --output data/eval/reports/smoke_helmet.json \
  --imgsz 320 \
  --limit 5
```

## 리포트 해석

리포트의 핵심 필드:

- `metrics.overall.precision`: 탐지한 것 중 실제 정답인 비율
- `metrics.overall.recall`: 실제 정답 중 모델이 찾은 비율
- `metrics.overall.fp`: 오탐 개수
- `metrics.overall.fn`: 미탐 개수
- `latency.avg_ms`: 이미지 1장 기준 평균 추론 시간
- `latency.p95_ms`: 느린 케이스까지 포함한 95% 지연 시간

## 기준 통과 확인

평가 리포트가 생성되면 manifest의 `acceptance_criteria`와 비교한다.

```bash
python scripts/health/check_model_report.py \
  --model-name helmet_model_ver0.5 \
  --report data/eval/reports/helmet_model_ver0.5.json
```

통과한 리포트를 manifest의 `latest_evaluation`에 반영하려면:

```bash
python scripts/health/check_model_report.py \
  --model-name helmet_model_ver0.5 \
  --report data/eval/reports/helmet_model_ver0.5.json \
  --update-manifest
```

## 운영 판단 기준

1차 기준은 `models/model_manifest.json`의 `acceptance_criteria`를 따른다.

- 헬멧 탐지는 recall을 precision보다 우선한다. 미착용을 놓치면 현장 안전 리스크가 크기 때문이다.
- 낙상/포즈 모델은 현장 오탐 비용이 있으므로 recall과 함께 false positive를 별도로 봐야 한다.
- Jetson 배포 전에는 ONNX 리포트와 TensorRT `.engine` 리포트를 분리해서 남긴다.

## falldata RF 모델 기준

`scripts/datasets/train_falldata_video_rf.py`로 만든 falldata 호환 RF 모델은
일반 탐지 모델과 별도로 평가한다.

필수 기준:

- `--cv-group-by scene_base`를 기본으로 사용한다.
- 같은 장면의 카메라 변형(`*_C1`, `*_C2` 등)이 train/test에 동시에 들어가면 안 된다.
- metrics JSON의 `holdout_split`, `holdout_errors`, `cross_validation.aggregate`를 함께 리뷰한다.
- `false_negative_count`가 0이 아니면 confirm/veto 운영 모드로 승격하지 않는다.
- 기본 승격 기준은 `false_positive_count=0`이다. 완화가 필요하면 현장 오탐 비용 기준으로 별도 승인한다.
- `scripts/health/check_falldata_model_report.py`를 통과해야 한다.
- 런타임 전환 전 `scripts/health/check_falldata_aux.py` smoke 결과를 저장한다.

운영 정책은 `docs/features/FALLDATA_INTEGRATION.md`의 `Runtime Policy Standard`를 따른다.

```bash
python scripts/health/check_falldata_model_report.py \
  --metrics-json models/experiments/falldata_sample_rf_metrics.json \
  --require-cross-validation
```

통과한 리포트만 manifest에 기록한다.

```bash
python scripts/health/check_falldata_model_report.py \
  --metrics-json models/experiments/falldata_sample_rf_metrics.json \
  --require-cross-validation \
  --update-manifest \
  --model-name falldata_sample_rf
```

## 아직 남은 일

- 고정 평가 데이터셋을 `data/eval/*` 아래에 준비한다.
- 현장 영상 기준 false positive per hour 평가 스크립트를 별도로 추가한다.
- CI에서 리포트 기준 미달 시 배포를 막는 게이트를 추가한다.
