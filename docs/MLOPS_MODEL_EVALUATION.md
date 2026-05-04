# Model Evaluation Workflow

이 문서는 현재 CCTV 프로젝트에서 모델을 바꿀 때 최소한으로 확인할 MLOps 기준을 정리한다.

## 목적

- `models/model_manifest.json`에 모델 목적, artifact, 운영 기준을 기록한다.
- 고정 평가 데이터셋으로 precision, recall, false positive, latency를 산출한다.
- 평가 리포트를 `reports/eval/*.json`에 저장해 모델 교체 판단 근거로 사용한다.

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
python scripts/evaluate_detection.py \
  --model models/helmet_model_ver0.5.onnx \
  --dataset data/eval/helmet \
  --output reports/eval/helmet_model_ver0.5.json \
  --imgsz 320 \
  --conf 0.35 \
  --iou 0.5 \
  --warmup 1 \
  --target-classes helmet,head
```

사람 탐지 fallback 모델 평가:

```bash
python scripts/evaluate_detection.py \
  --model models/yolov8n.onnx \
  --dataset data/eval/person \
  --output reports/eval/yolov8n.json \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.5 \
  --warmup 1 \
  --target-classes person
```

빠른 smoke test:

```bash
python scripts/evaluate_detection.py \
  --model models/helmet_model_ver0.5.onnx \
  --dataset data/eval/helmet \
  --output reports/eval/smoke_helmet.json \
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
python scripts/check_model_report.py \
  --model-name helmet_model_ver0.5 \
  --report reports/eval/helmet_model_ver0.5.json
```

통과한 리포트를 manifest의 `latest_evaluation`에 반영하려면:

```bash
python scripts/check_model_report.py \
  --model-name helmet_model_ver0.5 \
  --report reports/eval/helmet_model_ver0.5.json \
  --update-manifest
```

## 운영 판단 기준

1차 기준은 `models/model_manifest.json`의 `acceptance_criteria`를 따른다.

- 헬멧 탐지는 recall을 precision보다 우선한다. 미착용을 놓치면 현장 안전 리스크가 크기 때문이다.
- 낙상/포즈 모델은 현장 오탐 비용이 있으므로 recall과 함께 false positive를 별도로 봐야 한다.
- Jetson 배포 전에는 ONNX 리포트와 TensorRT `.engine` 리포트를 분리해서 남긴다.

## 아직 남은 일

- 고정 평가 데이터셋을 `data/eval/*` 아래에 준비한다.
- 현장 영상 기준 false positive per hour 평가 스크립트를 별도로 추가한다.
- CI에서 리포트 기준 미달 시 배포를 막는 게이트를 추가한다.
