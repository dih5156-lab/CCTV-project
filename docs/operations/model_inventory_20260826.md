# 모델·데이터 운영 현황

최종 확인일: 2026-08-27

## 운영 또는 승인된 모델

| 영역 | 모델/경로 | 상태 | 비고 |
|---|---|---|---|
| 헬멧 | `models/head/helmet_model.engine` | 운영 | 320x320 TensorRT, precision 0.9014 / recall 0.9110 / 평균 21.2ms |
| 낙상 | `models/` 및 `data/training/`의 fall temporal 산출물 | shadow/검증 | 후보 모델은 운영 승격 전 검증 필요 |
| 외형·색상 | `models/appearance/` 및 `models/experiments/appearance_color_common6_reviewed0825/` | 학습 중 | `appearance_color_review_labels0825.json` 반영 모델 |

## 외형 모델 학습 데이터

- 원본 AI-Hub 데이터: `data/datasets/aihub_kreid/`
- 기본 manifest: `.worktrees/appearance-multitask-v1/data/training/aihub_kreid_multitask_v1/common6_masked/`
- 2026-08-25 검수 반영 manifest: `.worktrees/appearance-multitask-v1/data/training/aihub_kreid_multitask_v1/common6_reviewed0825/`
- 검수 라벨: `appearance_color_review_labels0825.json` (1,500건)

## 레거시 모델

운영 경로에 직접 연결하지 않고 `models/legacy/` 또는 별도 보관 경로에서만 유지한다. 운영 설정에서 참조되는 모델은 `models/model_manifest.json`과 각 서비스 설정에서 확인한다.

## 현재 디렉터리 구조

운영 설정과 manifest가 아래 기능별 경로를 직접 참조한다.

```text
models/
  head/       # helmet/head TensorRT·ONNX·PT
  fall/       # 낙상 temporal/RF 후보와 검증 산출물
  person/     # 사람 검출 fallback 모델
  appearance/ # 외형·색상 및 PP-Human/PA100K 모델
  legacy/     # 운영에서 참조하지 않는 과거 모델
  manifests/  # model_manifest 및 모델별 메타데이터
```

학습 산출물(`models/experiments/`, `models/training/`)은 학습 컨테이너와 공유되므로 별도로 보존한다.

## 학습 완료 후 확인할 항목

1. `appearance_color_common6_reviewed0825`의 `metrics.json`과 `color_gate.json` 확인
2. 기존 모델 대비 색상별 precision/recall/F1 비교
3. CCTV 대표 샘플에서 실제 오탐·미탐 확인
4. 게이트 통과 전에는 운영 모델로 교체하지 않음
