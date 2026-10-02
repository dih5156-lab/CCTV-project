# 낙상 모델 비교 결과

## 판정: `HOLD`

| 지표 | Baseline | Candidate | 변화 |
|---|---:|---:|---:|
| precision | 0.9895 | 0.9895 | +0.0000 |
| recall | 0.9691 | 0.9711 | +0.0020 |
| f1 | 0.9792 | 0.9802 | +0.0010 |

## 자동 검사

- precision 기준: `True`
- recall 기준: `True`
- F1 기준: `True`
- 데이터셋 호환성: `False`

- Baseline dataset: `fall_temporal_baseline_same_split_20260910`
- Candidate dataset: `fall_temporal_same_split_resume_20260908`

## 해석

두 모델은 동일한 `reconstructed_split_20260908_container` manifest와 동일한 `scene_base` Holdout 분할로 재평가했다. dataset_version 문자열은 산출물별 이름이 달라 자동 게이트가 보류했지만, 실제 평가 데이터와 분할 기준은 일치한다.

동일 조건 수치상 Candidate가 Validation Recall을 0.9691에서 0.9711로 높였고, Holdout Recall을 0.9650에서 0.9715로 높였다. Holdout False Negative도 86건에서 70건으로 감소했으며 False Positive는 42건에서 43건으로 1건 증가했다. 따라서 새 모델은 기존 baseline보다 유력하지만, 실제 CCTV shadow 테스트를 거친 뒤 운영 승격한다.
