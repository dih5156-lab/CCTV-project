---
name: cctv-model-evaluation
description: Evaluate and compare CCTV fall-detection models from existing reports and sample videos without changing training or runtime services.
---

# CCTV 모델 평가

## 목적

기존 평가 스크립트와 JSON 결과를 사용해 full-frame pose와 crop-pose 모델을 같은 조건으로 비교한다. 이 skill은 모델을 학습하거나 배포하지 않는다.

## 실행 전 확인

1. 현재 작업 디렉터리가 CCTV 프로젝트 루트인지 확인한다.
2. 학습 프로세스와 `cctv-ai-engine` 컨테이너를 중지·재시작하지 않는다.
3. 입력 영상, 모델 경로, 평가 설정을 결과에 기록할 수 있는지 확인한다.
4. API 키와 RTSP 비밀번호를 출력하지 않는다.
5. 모델 파일 이름에 `partial`, `smoke`, `probe`가 포함되어 있으면 운영 후보 비교를
   실행하지 않고 탐색 결과로만 표시한다. 최종 모델과 대응 metrics가 모두 존재하고,
   metrics 파일명이 `<model-stem>_metrics.json` 형식이며 `validation` 또는 `holdout`
   섹션이 있어야 한다.

   ```bash
   python scripts/ops/check_model_evaluation_readiness.py \
     --model <model.pt> --metrics <metrics.json>
   ```

## 권장 절차

1. 평가 대상 영상과 모델 파일을 읽기 전용으로 확인한다.
2. 실행 목적에 맞는 기존 평가 도구를 선택한다.

   ```bash
   python scripts/ops/evaluate_sample_deepstream_replay.py --help
   python scripts/ops/evaluate_detection.py --help
   ```

   `evaluate_detection.py`는 지정한 모델·데이터셋을 평가하는 독립 실행 도구다. 반면
   `evaluate_sample_deepstream_replay.py`는 파일/RTSP replay 과정에서 카메라 설정과
   AI Engine을 변경할 수 있으므로, 학습 중이거나 운영 장비에서는 실행하지 않는다.

3. 결과 JSON에 모델 버전, 실행 모드, 데이터셋·영상 목록, 측정 시각, operator를 기록한다.
4. 비교 결과의 최소 계약을 검증한다.

   ```bash
   python scripts/ops/validate_pose_crop_compare_report.py <report.json>
   ```

5. 낙상 분류 모델의 기존·후보 지표를 비교할 때는 결정 임계값을 모델별로 보존하는
   비교 도구를 사용한다.

   ```bash
   python scripts/compare_fall_models.py \
     --baseline-metrics <baseline.json> \
     --candidate-metrics <candidate.json> \
     --output <comparison.json>
   ```

6. precision, recall, F1, pose 생성률, 지연시간/FPS, GPU 메모리를 같은 조건에서 비교하고, 단일 지표만으로 promote하지 않는다.
7. 비교 JSON을 인수인계용 Markdown으로 변환한다.

   ```bash
   python scripts/ops/render_fall_model_comparison_markdown.py \
     <comparison.json> --output <comparison.md>
   ```

8. 결과와 실패 원인을 `docs/models/` 또는 `docs/testing/`에 저장한다.

## 결과 판정

- `promote`: 정확도와 지연시간 기준을 모두 만족하고 샘플·실시간 검증이 통과한 경우
- `rollback`: 치명적인 오탐·미탐 또는 운영 회귀가 확인된 경우
- `hold`: 측정값이 부족하거나 full-frame과 crop-pose 조건이 동일하지 않은 경우

최종 산출물이 없거나 partial/smoke 결과만 있는 경우에는 자동으로 `hold`로 기록한다.

판정 전에 반드시 사람이 원본 영상과 주요 오검출 사례를 확인한다. 결과 파일을 수정하는 자동화는 검증된 출력 경로에서만 수행한다.

## 금지 동작

- 학습 스크립트 실행, 중단, 재개
- Docker compose 재시작 또는 모델 파일 덮어쓰기
- 실시간 장치 제어 API 호출
- 인증정보·RTSP 자격증명을 로그에 남기기
