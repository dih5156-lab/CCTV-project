# 모델·데이터 운영 현황

최종 확인일: 2026-08-26

## 운영 또는 승인된 모델

| 영역 | 모델/경로 | 상태 | 비고 |
|---|---|---|---|
| 헬멧 | `models/helmet_model_ver0.5_320_fp16.engine` | 운영 후보 | 320x320 FP16 TensorRT, 재평가 완료 |
| 낙상 | `models/` 및 `data/training/`의 fall temporal 산출물 | shadow/검증 | 후보 모델은 운영 승격 전 검증 필요 |
| 외형·색상 | `models/experiments/appearance_color_common6_reviewed0825/` | 학습 중 | `appearance_color_review_labels0825.json` 반영 모델 |

## 외형 모델 학습 데이터

- 원본 AI-Hub 데이터: `data/datasets/aihub_kreid/`
- 기본 manifest: `.worktrees/appearance-multitask-v1/data/training/aihub_kreid_multitask_v1/common6_masked/`
- 2026-08-25 검수 반영 manifest: `.worktrees/appearance-multitask-v1/data/training/aihub_kreid_multitask_v1/common6_reviewed0825/`
- 검수 라벨: `appearance_color_review_labels0825.json` (1,500건)

## 레거시 모델

운영 경로에 직접 연결하지 않고 `models/legacy/` 또는 별도 보관 경로에서만 유지한다. 운영 설정에서 참조되는 모델은 `models/model_manifest.json`과 각 서비스 설정에서 확인한다.

## 정리 예정 디렉터리 구조

학습·운영 프로세스가 종료된 뒤 아래 구조로 이동한다. 기존 운영 경로는 심볼릭 링크로 유지해 설정 파일을 한 번에 바꾸지 않는다.

```text
models/
  head/       # helmet/head TensorRT·ONNX·PT
  fall/       # 낙상 temporal/RF 후보와 검증 산출물
  color/      # 외형·색상 후보와 TensorRT 산출물
  appearance/ # PP-Human/PA100K 외형 기본 모델
  legacy/     # 운영에서 참조하지 않는 과거 모델
  manifests/  # model_manifest 및 모델별 메타데이터
```

현재 학습 중에는 경로 이동·대량 삭제를 하지 않는다. 학습 완료 후 참조 경로를 점검하고, 파일 이동과 링크 생성, 로그 보존 기간(기본 14일)을 한 번에 적용한다.

## 학습 완료 후 확인할 항목

1. `appearance_color_common6_reviewed0825`의 `metrics.json`과 `color_gate.json` 확인
2. 기존 모델 대비 색상별 precision/recall/F1 비교
3. CCTV 대표 샘플에서 실제 오탐·미탐 확인
4. 게이트 통과 전에는 운영 모델로 교체하지 않음
