# falldata RF 미탐 분석

## 대상

- 모델: `models/experiments/falldata_field_combined_rf_max120.pkl`
- 데이터셋: `field_combined_balanced_20260703`
- 평가 방식: `scene_base` 그룹 분리 holdout
- 품질 게이트 결과: **보류**

## 결과

- False Positive: 0건
- False Negative: 8건
- 미탐 대상은 모두 낙상 클래스(`true=0`)를 비낙상(`predicted=1`)으로 판정한 사례

| 장면 그룹 | 카메라 변형 | 미탐 수 | 낙상 확률 범위 |
|---|---|---:|---:|
| `00151_H_A_BY` | C1, C2, C5, C6 | 4 | 0.1507~0.4624 |
| `00003_H_A_FY` | C1, C2, C5, C7 | 4 | 0.1179~0.4681 |

## 해석

미탐이 두 장면 그룹에 집중되어 있어 일반적인 전체 데이터 편향보다 특정 장면·카메라 시점의 feature 추출 또는 동작 표현 문제가 우선 의심됩니다. C1·C2·C5·C6·C7에서 반복되므로 카메라별 시야, 사람 크기, 가림, 낙상 방향을 원본 샘플로 확인해야 합니다.

확률 0.53~0.56 수준의 사례는 결정 경계 조정으로 일부 회수할 수 있지만, 0.12~0.19 수준의 사례는 임계값만 낮춰 해결하기 어렵습니다. 임계값 조정은 False Positive 증가 여부를 함께 확인한 뒤에만 검토합니다.

## 다음 조치

1. 위 8개 장면의 원본 프레임과 feature 유효 프레임 수를 확인한다.
2. pose/keypoint 누락, 사람 크기, 가림, 낙상 방향별로 원인을 라벨링한다.
3. 원인이 feature 품질이면 해당 장면을 hard-case로 추가해 재학습한다.
4. 재학습 후 동일한 `scene_base` holdout으로 다시 평가한다.
5. False Negative가 0이 되지 않으면 운영 승격하지 않고 shadow 모드로 유지한다.

이 분석은 `scripts/health/check_falldata_model_report.py` 결과와
`models/experiments/falldata_field_combined_rf_max120_metrics.json`의
`holdout_errors.false_negatives`를 기준으로 작성했다.
