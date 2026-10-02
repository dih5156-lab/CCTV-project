# 낙상 오류 영상 검수 및 재학습 준비 가이드

## 1. 목적

낙상 후보 모델이 놓친 영상과 낙상으로 잘못 판단한 영상을 사람이 확인하고,
확인된 결과만 다음 재학습 데이터에 반영한다. 검수 전에는 운영 모델과 기존
원본 라벨을 변경하지 않는다.

## 2. 검수 페이지 열기

미탐이 집중된 카메라부터 확인한다.

```text
data/fall_eval/error_review_priority_c1_c4_20260908/index.html
```

전체 미탐을 확인할 때는 다음 페이지를 사용한다.

```text
data/fall_eval/error_review_20260908/index.html
```

오탐 검수는 다음 페이지에서 진행한다.

```text
data/fall_eval/error_review_fp_20260908/index.html
```

브라우저에서 영상을 재생한 뒤 각 카드에 대해 다음 중 하나를 선택한다.

| 선택값 | 의미 | 재학습 반영 |
|---|---|---|
| `fall` | 실제 낙상 | 반영 후보 |
| `non_fall` | 실제 비낙상 | 반영 후보 |
| `ambiguous` | 영상만으로 판단하기 어려움 | 제외 |
| `exclude` | 영상 손상·중복 등 학습 부적합 | 제외 |

검수가 끝나면 반드시 `검수 JSON 다운로드` 버튼을 눌러 파일을 저장한다.
파일 기본 이름은 `fall_error_review_labels.json`이다. 브라우저를 닫으면
검수 내용이 사라질 수 있으므로 다운로드 파일을 별도 보관한다.

## 3. 검수 JSON 확인

다운로드한 파일이 검수 대상 manifest의 `review_id`와 일치하는지 먼저
확인한다. 임의로 review_id를 수정하거나 기존 평가 JSON에 덮어쓰지 않는다.

```bash
python scripts/ops/prepare_reviewed_fall_retraining.py \
  --train-manifest data/fall_eval/reconstructed_full_manifest_20260908.jsonl \
  --error-manifest data/fall_eval/error_review_priority_c1_c4_20260908/fall_error_review_manifest.jsonl \
  --labels /path/to/fall_error_review_labels.json \
  --output-dir data/fall_eval/reviewed_retrain_20260908 \
  --reviewed-weight 3
```

명령이 성공하면 다음 산출물이 생성된다.

- `train_manifest.jsonl`: 재학습 입력 manifest
- `reviewed_hard_cases.json`: 검수된 hard-case와 가중치
- `summary.json`: 반영·제외·라벨 수정 건수

validation 오류는 품질 확인용 피드백으로 보존되며, 기본적으로 학습 데이터에
직접 섞지 않는다. holdout 오류도 사람이 `fall` 또는 `non_fall`로 확인한
항목만 재학습에 반영한다.

## 4. 재학습 전 확인

다음 조건을 모두 확인한 뒤에만 재학습을 시작한다.

1. 검수 JSON의 모든 `review_id`가 error manifest에 존재한다.
2. `ambiguous`와 `exclude` 항목이 학습 manifest에 들어가지 않는다.
3. `summary.json`에서 반영·제외·라벨 수정 건수를 사람이 확인한다.
4. 기존 고정 test split은 학습 입력으로 사용하지 않는다.
5. 모델 평가 리포트에 dataset version, threshold, 교차검증 여부를 기록한다.

재학습 후에는 [모델 운영 승격 매트릭스](MODEL_READINESS_MATRIX.md)의 동일
게이트를 다시 실행한다. 미탐·오탐이 남아 있거나 교차검증이 비활성화된 경우
운영 모델로 승격하지 않는다.

## 5. 현재 검수 대상 요약

- 전체 manifest: 18,128건
- C1~C4 우선 미탐: 465건
- 전체 미탐: 603건
- 전체 오탐: 19건
- 영상 누락: 0건

현재 후보 모델은 holdout 미탐 285건, 오탐 9건이므로 검수 완료 전까지
`confirm` 운영 정책과 기존 모델을 유지한다.
