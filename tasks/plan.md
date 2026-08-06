# Implementation Plan: 배포 안정화 리팩토링 및 낙상 정확도 개선

## Overview

현재 운영 변경을 보존하면서 DeepStream 낙상 처리 경로를 작은 모듈로 정리하고, 재현 가능한 데이터셋·평가 기준을 만든 뒤 추가 학습과 Jetson 재생 검증까지 수행한다. 모델 교체는 기존 모델과 동일한 고정 검증셋에서 개선이 확인된 경우에만 진행한다.

## Current Baseline

- MediaPipe/Falldata RF 검증: 낙상 precision 0.655, recall 0.760, F1 0.704 (120개 검증 영상)
- YOLO-Pose RF 검증: 낙상 precision 1.000, recall 0.500, F1 0.667 (20개 검증 영상)
- 현장 수집 클립: 577개
- 현장 리뷰 레코드: 5,612개, 현재 모두 `unlabeled`
- 주요 위험: 장면 누수, 미라벨 현장 데이터, 학습/DeepStream 특징 불일치, 운영 임계값과 학습 임계값 분리

## Architecture Decisions

- 대규모 재작성 대신 낙상 처리·평가 경로만 단계적으로 분리한다.
- 학습/검증 분리는 영상 단위가 아니라 `scene_base` 그룹 단위로 유지한다.
- 정확도 목표는 accuracy보다 낙상 recall과 시간당 오탐을 우선한다.
- 후보 모델은 shadow/aux 경로에서 비교하고 운영 판정을 바로 교체하지 않는다.
- 기존 사용자 변경과 현재 운영 설정은 보존한다.

## Task List

### Phase 1: 기준선과 리팩토링 경계

- [ ] Task 1: 전체 테스트 및 운영 기준선 고정
  - 기존 테스트, lint, Compose 검증 결과를 저장한다.
  - 현재 모델/환경/데이터셋 체크섬과 평가 지표를 기록한다.
- [ ] Task 2: DeepStream 낙상 오케스트레이션 단순화
  - `DeepStreamProcessor`의 낙상 보조판정, shadow 기록, 이벤트 발행 책임을 기존 보조 모듈로 이동한다.
  - 공개 메서드와 이벤트 결과는 변경하지 않는다.
- [ ] Task 3: 학습·평가 공통 지표 정리
  - confusion matrix, precision/recall/F1, FP/FN 목록과 임계값 sweep 출력 형식을 통일한다.

### Checkpoint: 리팩토링

- [ ] 기존 테스트 수정 없이 통과
- [ ] Ruff 통과
- [ ] Jetson Compose config 통과
- [ ] 운영 AI 엔진 smoke test 통과

### Phase 2: 데이터 품질과 추가 학습

- [ ] Task 4: 현장 데이터 라벨링 가능 상태 정리
  - 중복/손상 클립과 영상-리뷰 연결 상태를 검사한다.
  - `fall`, `not_fall`, `ambiguous` 라벨 통계를 생성한다.
- [ ] Task 5: 고정 학습/검증 manifest 생성
  - 공개 데이터와 검수된 현장 데이터를 출처·장면별로 분리한다.
  - 동일 장면이 train/validation에 섞이지 않음을 자동 검사한다.
- [ ] Task 6: 후보 모델 추가 학습
  - Falldata RF와 runtime-aligned YOLO-Pose RF를 동일 데이터 분할로 학습한다.
  - 임계값 sweep과 feature-quality 제외 사유를 함께 저장한다.

### Checkpoint: 오프라인 품질

- [ ] 낙상 recall이 기존 0.76 이상
- [ ] 낙상 precision이 기존 0.655보다 악화되지 않음
- [ ] 그룹 분리 및 데이터 누수 검사 통과
- [ ] 모델·데이터·설정 체크섬 기록

### Phase 3: Jetson 운영 검증

- [ ] Task 7: DeepStream 고정 영상 재생 비교
  - 현 운영 모델과 후보 모델을 동일 fall/not-fall 영상에 재생한다.
  - TP/FN/FP/TN과 처리 시간을 비교한다.
- [ ] Task 8: shadow 배포 및 운영 임계값 조정
  - 후보를 보조판정으로 배포하고 운영 이벤트를 변경하지 않은 채 비교한다.
  - 충분한 검증 후에만 운영 모델/임계값 변경안을 만든다.

### Checkpoint: 배포 승인

- [ ] 고정 재생셋에서 기존보다 recall 개선
- [ ] 오탐 증가가 합의된 한도 이내
- [ ] Jetson 지연시간·메모리·GPU 사용량 허용 범위
- [ ] 롤백 가능한 이전 모델과 설정 보존

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 현장 데이터가 모두 미라벨 | 학습 불가/잘못된 정답 | 라벨링 전에는 공개 데이터 학습만 수행하고 현장 데이터는 검증에서 제외 |
| 장면 누수 | 지표 과대평가 | `scene_base` 그룹 분리 자동 검사 |
| recall 개선으로 오탐 급증 | 운영 알람 피로 | precision과 시간당 FP를 동시에 gate로 사용 |
| CPU 보조 모델 지연 | Jetson 프레임 드롭 | shadow 큐, 제한 FPS, latency 측정 유지 |
| 기존 미커밋 변경과 충돌 | 사용자 작업 손상 | 관련 파일만 최소 수정하고 매 단계 diff 확인 |

## Open Questions Requiring Approval

- 현장 577개 클립의 정답 라벨은 사람이 검수해야 하며 자동 추정 라벨을 학습 정답으로 사용하지 않는다.
- 1차 모델 승인 기준은 `fall recall >= 0.80`, `fall precision >= 0.70`을 제안한다.

