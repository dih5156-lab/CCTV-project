# 이번 주 실행 체크리스트

- [x] 1. 전체 테스트/lint/Compose 기준선 저장
- [x] 2. DeepStream 낙상 처리 경로 리팩토링
- [x] 3. 학습·평가 지표 출력 통일
- [x] 4. 현장 클립 무결성 및 라벨 연결 검사
- [x] 5. 그룹 분리 train/validation/test manifest 생성
- [ ] 6. Falldata RF 후보 재학습 및 평가 (CUDA 호환 학습 환경 필요)
- [x] 7. YOLO-Pose RF POC 후보 재학습 및 평가
- [x] 8. Jetson DeepStream 고정 영상 재생 비교
- [x] 9. shadow 배포 및 임계값 튜닝 구조 검증
- [x] 10. 배포/롤백 보고서 작성

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

## 2026-09-08 진행 상태

- 그룹 단위 train/validation/test split을 생성했고 영상 누락은 0건이다.
- 낙상 후보 오류 603건과 오탐 19건의 브라우저 검수 페이지를 생성했다.
- temporal 후보는 Jetson shadow 경로에서 처리 안정성을 확인했지만, 정답
  라벨 부족으로 정확도 승격은 보류 중이다.
- temporal sliding window가 1개인 짧은 입력에서 확률이 높아도 prediction이
  0이 되던 후처리 문제를 수정하고 C2·C4 CUDA 영상으로 재검증했다.
- 동일 split temporal/RF 정식 재학습은 전체 feature cache가 없어 아직 수행하지 않았다.
- pose feature cache 생성기와 temporal 학습기의 margin 기본값을 120으로
  통일했고, 2건 smoke cache 및 CPU 2 epoch 입력 계약 검증을 완료했다.
- 정식 동일 split temporal/RF 비교용 전체 feature cache 생성은 아직 남아 있다.
- 호스트/Jetson 영상 루트 변환 옵션과 30건 GPU smoke 추출·temporal CUDA
  학습을 완료했다. 전체 cache는 장시간 분할 실행으로 진행해야 한다.
- offset/row-limit 기반 재개 방식을 추가하고, 두 개 chunk로 train·validation
  각 40건의 중복·누락 없는 처리를 검증했다.
- 정식 resume cache를 train·validation 각각 240건까지 확장했고, 두 영역의
  manifest/cache 대조에서 누락·추가 파일 0건을 확인했다.
- 정식 resume cache를 train·validation 각각 340건까지 확장했고, offset 240
  구간의 manifest/cache 대조에서도 누락·추가 파일 0건을 확인했다.
- 정식 resume cache를 train·validation 각각 440건까지 확장했고, offset 340
  구간의 manifest/cache 대조에서도 누락·추가 파일 0건을 확인했다.
- 정식 resume cache를 train·validation 각각 540건까지 확장했고, offset 440
  구간의 manifest/cache 대조에서도 누락·추가 파일 0건을 확인했다.
- 정식 resume cache를 train·validation 각각 640건까지 확장했고, offset 540
  구간의 manifest/cache 대조에서도 누락·추가 파일 0건을 확인했다.
- 정식 resume cache를 train·validation 각각 740건까지 확장했고, offset 640
  구간의 manifest/cache 대조에서도 누락·추가 파일 0건을 확인했다.
- 부분 cache의 단일 클래스 지표 오판을 막기 위해 temporal 학습기에 양쪽
  클래스 필수 검사를 추가하고 Jetson 실행 단계에서 차단을 확인했다.

## 2026-09-08 VS Code 종료 후 학습 작업 계속 진행

- [x] 기존 extractor PID 113293 생존 및 캐시 저장 진행 확인: train 740,
  validation 663건(관측 시점).
- [x] `data/fall_eval/continue_same_split_20260908.py` 문법 검사 및 컨테이너
  detached 실행(PID 116433). 기존 extractor가 종료되면 전체 train/validation을
  기존 옵션으로 추출하며, 이미 존재하는 캐시는 재사용한다.
- [x] 기존 구간이 train/validation 각 740건에서 종료된 후 후속 extractor
  PID 118460으로 자동 전환 확인. 로그에서 train 1~740 캐시 재사용 및
  741번째 영상 추출 시작 확인.
- [x] 전체 manifest 18,128건의 영상 존재·중복 ID·split 간 scene-group 겹침
  검사 통과. `data/fall_eval/same_split_manifest_audit_20260908.json`에 저장.
- [ ] 전체 캐시 생성 종료 코드와 manifest 대비 누락·중복·JSON 유효성 확인.
- [ ] 동일 split temporal/RF 학습 실행 및 검증 지표 비교.
- [x] 추출 완료를 감시하고 전체 캐시 수(12,688/2,720)를 확인한 뒤 RF,
  temporal 학습을 순차 실행하는 `data/fall_eval/post_extract_train_20260908.py`
  를 detached로 시작했다(PID 121206). 상태와 로그는 각각
  `same_split_post_train_20260908_status.json`, `same_split_post_train_20260908.log`에 남긴다.
- 학습 실행 시 두 모델 모두 `max-videos=0`, `validation-max-videos=0`,
  `max-frames=120`, `frame-stride=3`, `fall-window-margin-frames=120`,
  `random-state=42`, `min-pose-frames=3`을 명시한다. 두 학습기는 동일한
  `_group_holdout_indices(test_size=0.25)`를 사용한다. RF는
  `frames_with_pose`, temporal은 `frame_records` 길이로 제외하므로 전체
  캐시 완성 후 두 값 일치 및 최종 effective scene ID/holdout 일치를 확인한다.
  부분 캐시를 허용하는 temporal `--all-cached`는 정식 비교에 사용하지 않는다.
- [ ] 독립 test split 평가 및 최종 결과 기록. 운영 모델 승격은 별도 판단.

진행 상태: `data/fall_eval/same_split_continue_20260908_status.json`

후속 추출 로그: `data/fall_eval/same_split_continue_20260908.log`

편집기 종료에는 독립적이지만 컨테이너 재시작 시 자동 재실행되지는 않는다.
재개할 때는 상태 파일뿐 아니라 `/proc`의 실제 프로세스와 로그 증가를 확인한다.
