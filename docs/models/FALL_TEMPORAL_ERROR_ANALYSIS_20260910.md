# Temporal 낙상 모델 오류 분석

## 대상

- 모델: `models/fall/fall_temporal_same_split_resume_20260908.pt`
- 데이터셋: `fall_temporal_same_split_resume_20260908`
- 입력: YOLO-Pose 기반 temporal feature

## 동일 조건 비교 제한

기존 모델과 이 모델은 같은 cache를 그대로 공유할 수 없습니다. 기존 모델 checkpoint의 `summary_features`는 37개이고, 새 모델은 53개이며 sequence 전처리 설정도 다릅니다. 따라서 현재 지표 비교는 참고용입니다.

공정한 최종 비교를 위해서는 동일 manifest와 동일 holdout scene 분할을 유지한 채 각 모델이 학습될 당시의 전처리로 feature를 각각 재추출한 뒤, 두 모델을 각각 평가해야 합니다.

## 평가 결과

| 구분 | False Negative | False Positive |
|---|---:|---:|
| Validation | 59 | 21 |
| Holdout | 70 | 43 |

모델 리포트에서 `true=1`은 낙상, `true=0`은 비낙상으로 기록되어 있습니다.

## Holdout 오류 집중 장면

### 미탐

- `02587_H_D_SY`: 8건
- `02753_H_A_FY`: 5건
- `02764_H_A_FY`: 5건
- 그 외 장면은 대부분 1~2건

미탐 낙상 확률 평균은 약 0.23이며, 70건 중 16건만 0.40~0.50 경계 구간에 있습니다. 따라서 임계값을 낮추는 것만으로는 전체 미탐을 해결하기 어렵고, 특정 장면의 pose/feature 품질 확인이 필요합니다.

### 오탐

- `01476_O_F_N`: 8건
- `02360_H_A_N`: 6건
- `00932_O_E_N`: 5건
- `01842_Y_A_N`: 5건
- `02518_H_A_N`: 5건
- 그 외 장면은 1~2건

오탐 낙상 확률 평균은 약 0.79이며, 43건 중 9건만 0.50~0.60 경계 구간입니다. 일부 장면은 높은 확률로 반복 오탐하므로 장면 특화 hard-negative 보강이 필요합니다.

## 판단

새 Temporal 모델은 전체 지표는 높지만, holdout에서 특정 낙상 장면을 반복적으로 놓치고 특정 비낙상 장면을 낙상으로 오인합니다. 현재 결과만으로 운영 승격하지 않고, 오류 장면을 검토한 후 hard-case 재학습과 동일 holdout 재평가를 진행해야 합니다.

Validation threshold sweep에서는 threshold 0.35가 Recall 98.19%, False Negative 37건으로 가장 높은 Recall을 보였습니다. 대신 False Positive가 31건으로 늘어나므로, 현장 오탐 비용을 확인하기 전에는 threshold를 운영값으로 확정하지 않습니다.

## 다음 조치

1. 미탐 상위 3개 장면의 원본 pose/keypoint 생성률과 사람 박스 크기를 확인한다.
2. 오탐 상위 5개 장면을 hard-negative 목록으로 분류한다.
3. 가림·시점·낙상 방향·프레임 누락 원인을 라벨링한다.
4. 보강 데이터로 재학습한 뒤 동일한 holdout 분할에서 재평가한다.
5. 미탐과 오탐이 함께 감소할 때까지 운영 모델로 교체하지 않는다.

근거 파일: `data/fall_eval/fall_temporal_same_split_resume_20260908_metrics.json`
