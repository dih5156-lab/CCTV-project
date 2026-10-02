# 낙상 모델 비교 기록 (2026-09-09)

## 비교 대상

| 구분 | 모델 | 데이터셋 버전 |
|---|---|---|
| Baseline | `models/experiments/fall_temporal_hybrid_120_val80.pt` | `fall_temporal_hybrid_120_val80_20260723` |
| Candidate | `models/experiments/fall_temporal_hybrid_all_cached.pt` | `fall_temporal_hybrid_all_cached_20260723` |

두 metrics 파일 모두 group shuffle 방식의 holdout을 사용했고 test group 목록은 동일하다. 다만 train class count와 dataset version이 다르므로, 이 결과는 동일 데이터셋의 엄밀한 A/B 검증이 아니라 참고 비교로 취급한다.

## Holdout 결과

| 지표 | Baseline | Candidate | 변화 |
|---|---:|---:|---:|
| Precision | 0.8571 | 0.8148 | -0.0423 |
| Recall | 0.7692 | 0.9167 | +0.1474 |
| F1 | 0.8108 | 0.8627 | +0.0519 |

## 판정

자동 판정은 **hold**다. 비교 도구는 이제 두 metrics에 `dataset_version`이 있고 값이
다르면 `dataset_compatible=false`로 처리해 자동 승격을 차단한다.

- Candidate는 recall과 F1이 좋아졌다.
- Precision이 baseline보다 낮아 `precision_not_lower` 조건을 통과하지 못했다.
- 실제 낙상 경보에서 오탐 증가가 허용되는지 현장 영상으로 확인해야 한다.
- 최종 학습 모델이 생성되면 동일한 데이터 분할과 동일한 runtime 조건으로 다시 비교한다.

비교 명령은 다음과 같다.

```bash
python scripts/compare_fall_models.py \
  --baseline-metrics models/experiments/fall_temporal_hybrid_120_val80_metrics.json \
  --candidate-metrics models/experiments/fall_temporal_hybrid_all_cached_metrics.json \
  --output /tmp/cctv_fall_model_comparison.json
```

이 기록은 모델 파일을 운영 환경에 배포하거나 decision threshold를 변경했다는 의미가 아니다.
