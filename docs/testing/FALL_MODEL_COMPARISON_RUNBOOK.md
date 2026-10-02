# 낙상 모델 비교 실행 런북

작성일: 2026-09-09  
상태: 학습 완료 후 실행

실제 Jetson full-frame/crop-pose 런타임 비교는
[`POSE_CROP_COMPARISON_CHECKLIST.md`](POSE_CROP_COMPARISON_CHECKLIST.md)를 함께
사용하고, 결과 JSON은 `scripts/ops/validate_pose_crop_compare_report.py`로
검증한다.

## 1. 목적

기존 낙상 모델과 2026-09-08부터 실행 중인 `same_split` 재학습 모델을 동일한
validation 영상과 실제 CCTV 테스트 영상에서 비교한다.

오픈데이터의 `N/FY/BY/SY`는 센서·특징 기반 방향 분류 모델 비교에 사용하며,
영상 기반 YOLO-Pose/RF/Temporal 모델 평가와 섞지 않는다.

## 2. 평가 순서

```text
1. 새 모델 파일과 checksum 확인
2. 동일 validation manifest로 기존 모델 실행
3. 동일 validation manifest로 새 모델 실행
4. 영상 단위 TP/FN/FP/TN과 pose 단계 결과 집계
5. 실제 CCTV 테스트 영상 replay
6. Jetson FPS·지연시간·GPU·메모리 측정
7. 운영 승격 또는 보류 결정
```

## 3. 비교 대상

### 영상 모델

- 기존 운영 또는 후보 YOLO-Pose/RF/Temporal 모델
- `/app/models/fall/yolo_pose_same_split_resume_20260908_rf.joblib`
- `/app/models/fall/fall_temporal_same_split_resume_20260908.pt`
- validation manifest: `/app/data/fall_eval/reconstructed_split_20260908_container/validation.jsonl`

실제 산출물 경로는 학습 완료 후 파일 존재 여부를 다시 확인한다.

현재 저장된 비교 기준 후보:

| 기준 | metrics 파일 | holdout 낙상 precision | recall |
|---|---|---:|---:|
| YOLO-Pose temporal | `models/experiments/fall_temporal_hybrid_full_seq60_candidate_metrics.json` | 0.9875 | 0.9542 |
| YOLO-Pose ExtraTrees | `models/experiments/yolo_pose_fall_extratrees_reviewed_hard498_depth16_t078_candidate_20260813_metrics.json` | 0.9963 | 0.8949 |

두 파일은 학습 조건이 다르므로 참고용 기준이다. 자동 승격 판정에는 새 학습과
동일한 split·threshold로 재평가한 기준 파일만 사용한다.

### 오픈데이터 방향 모델

```text
N  = not fall
FY = forward fall
BY = backward fall
SY = side fall
```

원본은 `falldata/낙상방향-003.tar` 내부 `SCH_FN/test.zip`이며 NPY/XLSX
특징 데이터다.

## 4. 필수 지표

| 단계 | 지표 |
|---|---|
| 사람 검출 | person precision, person recall |
| pose | pose 생성률, keypoint 수, 평균 confidence |
| 낙상 | fall precision, fall recall, F1, FP, FN |
| 운영 | event latency, FPS, frame drop, GPU memory, system memory |
| 안정성 | 중복 이벤트, 이벤트 누락, runtime error |

낙상 Recall은 영상 단위로 계산하고, pose가 아예 없는 낙상 영상은 별도
`pose_no_result`로 집계한다.

## 5. 판정 기준

- 새 모델은 기존 모델과 동일한 영상·threshold·집계 규칙으로 비교한다.
- 낙상 Recall 개선만으로 승격하지 않고 오탐률과 지연시간을 함께 확인한다.
- pose 생성률이 개선되지 않으면 RF·Temporal 성능 향상만으로 현장 문제 해결로
  판단하지 않는다.
- 새 모델은 shadow 또는 테스트 환경에서 먼저 실행하고, 결과 확인 전 운영
  모델과 장치 제어 경로를 교체하지 않는다.

## 6. 결과 기록 양식

```text
run_date:
model_version:
manifest:
threshold:
person_recall:
pose_generation_rate:
fall_precision:
fall_recall:
fall_f1:
false_positive:
false_negative:
event_latency_avg_ms:
event_latency_p95_ms:
fps:
gpu_memory_mb:
system_memory_mb:
frame_drop:
decision: promote | shadow | hold
notes:
```
