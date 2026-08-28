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
API 요청·응답 예시는 [API_QUICK_REFERENCE.md](docs/guides/API_QUICK_REFERENCE.md), MQTT·EdgeX·JSON 필드 계약은 [EVENT_DATA_CONTRACT.md](docs/guides/EVENT_DATA_CONTRACT.md)를 참고합니다.

### 이벤트 타입 요약

| 이벤트 | 의미 | 상세정보 저장 위치 |
|---|---|---|
| `person` | 사람 검출 | `object_id`, `bbox`, `confidence` |
| `helmet` / `head` | 헬멧 착용 / 미착용 | `metadata`, 외형 DB |
| `fall_detected` | 통합 낙상 이벤트 | `fall_direction`, `fall_type`, `keypoints` |
| `face_recognized` / `face_unknown` | 등록 / 미등록 얼굴 | `face_*` 메타데이터 |
| `danger_zone` / `intrusion` | 위험구역 침입 | `zone_id`, `zone_name`, `zone_event` |
| `zone_object` | 감시구역 객체 | `mode=object_watch` |
| `crowd_warning` | 인원 임계치 초과 | `person_count`, `threshold` |
| `appearance_match` | 외형 조건 일치 | 외형 DB의 색상·헬멧·가방 필드 |
| `unsafe_behavior` | 위험행동 | `reason`, `behavior`, `score` |

장치에는 통합 경보 문구를 출력하고, 방향·외형·구역 같은 세부값은 DB/API 조회용으로 보존합니다.

이벤트 계약 자동검증은 GPU 학습과 무관하게 실행할 수 있습니다.

```bash
python scripts/validate_event_contracts.py --samples
python scripts/validate_event_contracts.py --file /path/to/events.json
```

검증 실패는 종료 코드 `1`, 입력 파일 형식 오류는 `2`를 반환하며 결과는 JSON으로 출력됩니다.

검증된 샘플을 실제 MQTT에 보내지 않고 재생 계획만 확인하려면 다음을 실행합니다.

```bash
python scripts/smoke/replay_event_contract_samples.py
```

실제 브로커 발행은 명시적으로 `--publish`를 지정해야 하며, critical 이벤트는 기본적으로 제외됩니다.

```bash
python scripts/smoke/replay_event_contract_samples.py --publish
python scripts/smoke/replay_event_contract_samples.py --publish --include-critical
```

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

### 테스트 영상 재생 검증

먼저 방향별 테스트 영상을 자동으로 선정합니다.

```bash
python scripts/datasets/build_fall_test_manifest.py \
  --manifest data/fall_eval/auto/validation_manifest.jsonl \
  --output data/fall_eval/test_manifest.jsonl \
  --per-group 5
```

DeepStream 테스트 영상 재생과 TP/FN/FP/TN 집계는 기존 도구를 사용합니다.

```bash
python scripts/ops/evaluate_sample_deepstream_replay.py \
  --source-mode file \
  --max-videos 20 \
  --results-jsonl data/fall_eval/test_replay_results.jsonl \
  --results-csv data/fall_eval/test_replay_results.csv
```

재생 결과에 대해 운영 품질 기준을 확인합니다.

```bash
python scripts/quality_gate_fall_replay.py \
  --results-jsonl data/fall_eval/test_replay_results.jsonl \
  --min-precision 0.90 \
  --min-recall 0.80
```

모델 교체는 기본적으로 dry-run이며, 명시적으로 `--approve`를 지정해야 합니다.

```bash
python scripts/promote_fall_model.py \
  --comparison models/experiments/fall_model_comparison.json \
  --candidate-model models/experiments/yolo_pose_fall_rf.pkl \
  --target-model models/experiments/yolo_pose_fall_operational.pkl
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

## 업데이트 이력

### 2026-08-28
- 주간 자동 업데이트: 코드·설정·문서·테스트 변경사항 반영


### 2026-08-28 (8월 4주차, 20250824~20250828)

- 외형 멀티태스크 학습 산출물을 정리하고 MobileNetV3 crop-weighted 모델의 baseline/protected 멀티태스크 결과와 색상·비색상 F1을 기록했습니다.
- 외형 색상 검수 웹 화면을 재생성하고, 하의가 보이지 않는 샘플은 라벨링하지 않고 `unknown/exclude`로 보류하는 검수 기준을 정리했습니다.
- Jetson 외형 속성 경로를 HSV fallback에서 PA100K TensorRT(`pa100k_tensorrt`)로 전환해 성별·나이 결과가 실제 appearance DB에 적재되도록 수정했습니다.
- 성별 평가 스크립트의 PA100K 모델 기본 경로 오류를 수정하고, Jetson AI 엔진 재생성 후 TensorRT 모델 로드와 성별 출력(`male/female/unknown`)을 확인했습니다.
- 얼굴등록·성별 상태를 점검했습니다. 얼굴 등록은 현재 1명·1샘플이므로 추가 각도/조명 샘플 등록이 필요합니다.
- Jetson DeepStream 안정성 및 운영 스택을 재점검했습니다. 서비스 health 정상, 프레임 드롭 0, 처리량 약 30 FPS를 확인했습니다.
- 헬멧·낙상·외형 모델 평가 리포트를 정리했습니다. 낙상 pose 단일 프레임 Recall과 temporal 낙상 모델 Recall을 구분해 기록했습니다.
- 낙상 temporal 모델의 검증 결과(Recall 약 0.95)와 pose 검출 단계의 낮은 Recall을 구분하고, 미탐 원인을 평가 단계별로 정리했습니다.
- `models/head`, `models/fall`, `models/appearance`, `models/person`, `models/legacy` 중심으로 모델 파일을 역할별 관리하도록 정리했습니다.
- EdgeX·MQTT·AIoT 파서·Action Layer·전광판 device profile과 양방향 명령 흐름 문서를 보강했습니다.
- Grafana/Prometheus 운영 대시보드와 DeepStream 안정성 watch를 점검하고, 서비스 health·GPU·프레임 드롭 지표를 확인했습니다.
- 주간 Git 자동화 timer를 설치했습니다. 매주 금요일 17시에 코드·설정·문서·테스트를 자동 commit/push하며, 상세 작업 요약은 README에 직접 작성하는 방식입니다.
- 기본 `docker-compose.yml`은 Windows/일반 환경용으로 유지하고, Jetson 운영은 `docker-compose.jetson.yml`과 `.env.jetson`을 사용하도록 문서화했습니다.

### 2026-08-26

- helmet/head TensorRT 입력 크기(`320x320`)에 맞춰 DeepStream bbox 좌표 복원 로직을 수정했습니다. 라이브 화면에서 head 박스가 사람 머리 위치와 일치하는 것을 확인했습니다.
- 외형 색상 검수 라벨 `appearance_color_review_labels0825.json`을 반영한 재학습 파이프라인을 구성했습니다. 현재 batch size 32로 GPU 학습 중이며, 완료 후 성능 게이트를 확인합니다.
- 모델 정리 기준을 `models/head`, `models/fall`, `models/color`, `models/appearance`, `models/legacy`, `models/manifests`로 정의했습니다. 운영 중인 학습·서비스 경로 보호를 위해 실제 이동은 학습 완료 후 진행합니다.
- 모델 현황과 정리 계획을 [model_inventory_20260826.md](docs/operations/model_inventory_20260826.md)에 기록했습니다.
- GitHub Actions CI/CD workflow(`.github/workflows/ci.yml`)에서 테스트·Compose 검증·보안 검사·Docker 빌드를 수행하도록 유지합니다.

## Git 관리 원칙

다음은 저장소에 커밋하지 않습니다.

- `.env`, `.env.jetson`, `cameras.json`, 얼굴 이미지
- `models/experiments/`의 모델·metrics 산출물
- `data/`의 런타임 DB·백업·학습 캐시
- 가상환경과 도구별 로컬 설정

상세 문서는 [docs/README.md](docs/README.md)에서 확인할 수 있습니다.
외부 개발자에게 API를 설명할 때는 [API 빠른 참조](docs/guides/API_QUICK_REFERENCE.md)를 사용합니다.

## 운영 참고 문서

- [운영 Runbook](docs/guides/OPERATIONS_RUNBOOK.md)
- [Jetson/EdgeX 현장 체크리스트](docs/guides/JETSON_EDGEX_FIELD_CHECKLIST.md)
- [프로젝트 구조](docs/modules/PROJECT_STRUCTURE.md)
- [배포 환경변수](docs/guides/DEPLOYMENT_ENVIRONMENT_VARIABLES.md)
