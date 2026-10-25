# TCN 앙상블·시간 안정화 적용안

현재 운영 RF는 그대로 두고, TCN을 shadow/비교 모델로 먼저 실행한다.

권장 후보는 다음과 같다.

- 균형형: `models/experiments/fall_temporal_hybrid_full_seq60_candidate.pt`
- 미탐 감소형: `models/experiments/fall_temporal_hybrid_hardcase_candidate.pt`

## 권장 런타임 설정

```dotenv
FALLDATA_AUX_TEMPORAL_COMPARE_MODEL_PATH=/app/models/experiments/fall_temporal_hybrid_full_seq60_candidate.pt
FALLDATA_AUX_TEMPORAL_SLIDING_WINDOW_SIZE=60
FALLDATA_AUX_TEMPORAL_SLIDING_WINDOW_STRIDE=20
FALLDATA_AUX_TEMPORAL_MIN_CONFIRMED_WINDOWS=2
```

2026-10-06 v0.111: shadow RF 비교 연결 I49는 로컬 수정·새 후보 검증을 마쳤다. 원 경보는 즉시 내부 큐로 전달하고 RF 결과는 기록만 한다. 선택 RF의 입력 출처 비호환(I37)은 남아 예측0이며, 모델 품질 합격·운영 반영은 아니다. [수정·검증](../planning/MODEL_INFERENCE_METRICS.md#shadow-rf-fix-v111).

`MIN_CONFIRMED_WINDOWS=2`는 한 번의 높은 점수만으로 이벤트를 확정하지 않고, 두 개의 연속 temporal window가 0.7 이상일 때만 확인한다. 운영 환경에 적용하기 전에는 `FALLDATA_AUX_MODE=shadow`로 RF와 TCN 결과를 함께 기록한다.

검증 결과(0.7 기준): 균형형 TCN은 정밀도 99.10%, 재현율 94.01%, 오탐 21건, 미탐 148건이다. hard-case TCN은 재현율 96.16%까지 높지만 오탐이 40건으로 증가하므로 미탐 우선 현장에서만 별도 비교한다.
