# CCTV AIoT 안전관리 시스템

YOLO/YOLO-Pose 기반의 헬멧·사람·낙상 감지 시스템입니다. 일반 PC에서는 OpenCV 기반으로 기능을 확인하고, NVIDIA Jetson Orin에서는 DeepStream/TensorRT 파이프라인으로 운영합니다.

## 현재 상태

- 낙상 이벤트 타입은 `fall_detected`로 통합합니다.
- 스피커·전광판에는 방향과 관계없이 통합 낙상 문구를 출력합니다.
- DB/API에는 `fall_direction`, `fall_type`, `scene_cat_name`을 상세 메타데이터로 저장할 수 있습니다.
- 상세 방향을 판정하지 못하면 `fall_detail_status=unclassified`로 저장합니다.
- 낙상 방향 필터는 `/api/v1/events?fall_direction=front|side|back|unclassified`로 사용할 수 있습니다.
- 학습 파이프라인은 YOLO-Pose를 특징 추출기로 사용하고, 낙상/방향 분류기는 별도 Random Forest 모델로 학습합니다.

## 빠른 시작

### PC 개발 실행

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements/ai.txt
python main.py --help
```

카메라와 모델 경로는 `cameras.json`과 환경변수로 설정합니다. 비밀번호·RTSP 주소가 들어간 개인 설정 파일은 Git에 커밋하지 않습니다.

### Docker/EdgeX 실행

```bash
python scripts/ops/compose_stack.py up -d
python scripts/ops/compose_stack.py ps
python scripts/ops/compose_stack.py logs -f cctv-public-api
```

### Jetson 운영

```bash
python scripts/ops/compose_stack.py --jetson up -d
python scripts/ops/compose_stack.py --jetson ps
```

Jetson 학습 컨테이너는 `cctv-ai-engine`이며, GPU 상태는 다음으로 확인합니다.

```bash
docker exec cctv-ai-engine python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

## 주요 서비스와 포트

| 서비스 | 포트 | 역할 |
|---|---:|---|
| Public API | 9000 | 이벤트·카메라·외형 조회 API |
| Alert API | 8000 | 내부 이벤트 수신 |
| Action Layer | 8080 | 스피커·전광판·사이렌·이력 처리 |
| Stream API | 8769 | 카메라 미리보기 |
| MQTT | 1883 | AI 이벤트 전달 |
| Prometheus | 9090 | 메트릭 |

웹 시연 화면은 `web/public-demo.html`이며, 이벤트 조회에서 낙상 방향 필터를 제공합니다.

## 이벤트 흐름

```text
카메라
  → YOLO/YOLO-Pose
  → 낙상 판정
  → MQTT: cctv/ai/events/{camera_id}/fall_detected
  → Alert API / DB
  → Action Layer
  → 스피커·전광판·사이렌
```

출력 장치 문구는 `config/event_type_map.json`에서 관리합니다. 상세 방향 메타데이터는 출력 문구에 사용하지 않습니다.

## 학습과 검증

### 전체 낙상·방향 학습

데이터셋 라벨과 영상을 자동 매칭하고, 장면 그룹 단위로 train/validation을 나눕니다.

```bash
python scripts/run_fall_training_pipeline.py \
  --dataset-root "/path/to/Training" \
  --output-dir data/fall_eval/auto \
  --train \
  --train-direction \
  --decision-threshold 0.7
```

학습 산출물:

- `models/experiments/yolo_pose_fall_rf.pkl`: 낙상 분류기
- `models/experiments/fall_direction_rf.pkl`: 방향 보조 분류기
- `*_metrics.json`: 검증 지표
- `models/experiments/fall_model_comparison.json`: 기존/신규 모델 비교 결과

기존 모델보다 정밀도·재현율·F1이 낮아지면 신규 모델을 운영 후보로 선정하지 않습니다.

### 모델 비교만 실행

```bash
python scripts/compare_fall_models.py \
  --baseline-metrics models/experiments/yolo_pose_fall_cam2_continuous_200_80_640_metrics.json \
  --candidate-metrics models/experiments/yolo_pose_fall_rf_metrics.json \
  --baseline-model models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl \
  --candidate-model models/experiments/yolo_pose_fall_rf.pkl
```

## API 예시

```bash
curl "http://localhost:9000/api/v1/events?event_type=fall_detected&fall_direction=back"
```

지원 방향 값은 `front`, `side`, `back`, `unclassified` 및 한국어 `전면`, `측면`, `후면`입니다.

## 테스트

```bash
pytest -q
```

주요 테스트 범위:

- 이벤트 생성·정규화
- API 조회·방향 필터
- Action Layer 출력 문구
- 학습 manifest·모델 비교

## Git 관리 원칙

다음은 저장소에 커밋하지 않습니다.

- `.env`, `.env.jetson`, `cameras.json`, 얼굴 이미지
- `models/experiments/`의 모델·metrics 산출물
- `data/`의 런타임 DB·백업·학습 캐시
- 가상환경과 도구별 로컬 설정

상세 문서는 [docs/README.md](docs/README.md)에서 확인할 수 있습니다.

## 운영 참고 문서

- [운영 Runbook](docs/guides/OPERATIONS_RUNBOOK.md)
- [Jetson/EdgeX 현장 체크리스트](docs/guides/JETSON_EDGEX_FIELD_CHECKLIST.md)
- [프로젝트 구조](docs/modules/PROJECT_STRUCTURE.md)
- [배포 환경변수](docs/guides/DEPLOYMENT_ENVIRONMENT_VARIABLES.md)
