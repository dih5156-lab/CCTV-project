# 이번 주 실행 체크리스트

- [x] 1. 전체 테스트/lint/Compose 기준선 저장
- [x] 2. DeepStream 낙상 처리 경로 리팩토링
- [x] 3. 학습·평가 지표 출력 통일
- [x] 4. 현장 클립 무결성 및 라벨 연결 검사
- [ ] 5. 그룹 분리 train/validation manifest 생성
- [ ] 6. Falldata RF 후보 재학습 및 평가 (CUDA 호환 학습 환경 필요)
- [x] 7. YOLO-Pose RF POC 후보 재학습 및 평가
- [ ] 8. Jetson DeepStream 고정 영상 재생 비교
- [ ] 9. shadow 배포 및 임계값 튜닝
- [ ] 10. 배포/롤백 보고서 작성

## Proposed quality gate

- 낙상 recall: 0.80 이상
- 낙상 precision: 0.70 이상
- 데이터 누수 검사: 0건
- Jetson 처리 지연 및 프레임 드롭: 기존 대비 유의미한 악화 없음

## POC result — 2026-07-23

- Candidate: `yolo_pose_grouped_poc_20_val20`
- Group overlap: 0
- Best observed operating point: threshold 0.60, fall recall 0.875, precision 0.538
- Deployment decision: rejected; precision gate 0.70 not met
- Next requirement: label field clips and run the enlarged experiment in a CUDA-compatible environment

## Current best cached YOLO-Pose result — 2026-07-28

- Candidate: `yolo_pose_temporal_rf_balanced_120_val80`
- Best quality-gate point: threshold 0.50, fall recall 0.850, precision 0.872
- Validation errors: FP 5, FN 6 across 79 effective validation videos
- Train/validation scene-group overlap: 0
- Person selection now considers frame-to-frame center continuity; targeted tests pass
- Separate 320x320 FP16 auxiliary engine POC completed; the existing 640x640
  DeepStream primary engine was not changed
- Vertical-center feature candidates were rejected:
  - 120/80 at threshold 0.50: precision 0.842, recall 0.800 (FP 6, FN 8)
  - 240/80 at threshold 0.50: precision 0.861, recall 0.775 (FP 5, FN 9)
- Deployment decision: keep the current PT/37-feature model; do not switch to the
  auxiliary engine candidate yet
- Training-only group-OOF hard-case weighting candidate:
  - selected weight: 3.0 (weight 4.0 produced no further gain)
  - original validation 120: FP 1, FN 7 at threshold 0.50
  - independent audit 66, train/validation overlap 0: FP 1, FN 2
  - combined 186: precision 0.979, recall 0.913 (baseline 0.969 / 0.894)
- Current shared runtime threshold 0.70 makes the hard-weight candidate worse; do
  not deploy it with the shared threshold
- Added optional `FALLDATA_AUX_COMPARE_THRESHOLD` support so the YOLO-Pose RF can
  be shadow-tested at 0.50 without changing the primary verifier threshold
- Next requirement: shadow-test the weight-3 candidate with compare threshold 0.50
  and rebuild a production-optimized auxiliary engine
