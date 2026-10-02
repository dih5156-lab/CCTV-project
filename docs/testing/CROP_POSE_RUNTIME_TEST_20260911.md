# DeepStream crop-Pose 런타임 검증 기록

검증일: 2026-09-11  
대상 영상: `data/fall_test/20260910/fall/20260910_141112_backward_fall.mp4`  
운영 학습 컨테이너: 변경하지 않음

## 확인한 내용

### ROI 등록 수정 후 실제 재생 완료

평가 세션을 종료까지 추적했고 종료 코드 0을 확인했다. 결과:
`data/fall_eval/crop_pose_roi_fixed_20260911.jsonl` 및 `.csv`.
입력은 `20260910_141112_backward_fall.mp4` (30초)다.
스크립트 판정은 FN, shadow 기록 0개, fall 이벤트 0개다.
보존한 `.runtime.log`에서 frames=320, crop_pose_results=0을 확인했다.
일반 이벤트 증가는 Pose 결과 수신의 증거가 아니다. **이 결과는 연결 진단용이며
모델 정확도 비교 및 승격 자료로 사용하지 않는다.**

따라서 터미널 실행 제한은 더 이상 평가의 차단 사유가 아니다. 다음 점검 대상은
SGIE tensor 수신 여부와 parser 필터링이며, Pose 출력 생성이 검증된 뒤 같은 영상으로
재평가한다. 평가 후 AI 엔진을 `.env.jetson`으로 재생성했다.

### 후속 코드 점검: Person ROI 등록 누락

Primary src probe가 passthrough로 구현돼 있어 raw Person tensor를 객체 메타로
등록하지 않는 누락을 확인했다. crop 모드에서 GIE 1 사람 박스를 tracker 앞에서
등록하도록 연결하고, 초기 object_id는 untracked 값으로 지정했다. 기존 full-frame
probe 동작은 유지한다. 따라서 앞선 엔진 로드/일반 이벤트 증가만으로 Pose SGIE
입력 및 낙상 판정이 검증됐다고 볼 수 없다.

수정 후 CPU 회귀 테스트: 134 passed, 2 skipped (종료 코드 0).
범위: DeepStream processor, pipeline builder, event factory, replay evaluator.
실제 영상의 Pose SGIE 출력과 낙상 TP/FN은 아직 미검증이다.

실행 도구는 `session_id`를 반환하며 `write_stdin`으로 완료까지 추적할 수 있다.
이전에 기록한 터미널 시간 제한 원인은 확인된 사실이 아니다. 이후 평가는
세션 ID와 종료 코드를 보존해 완료 여부를 판단한다.

임시 환경 파일에서 다음 구성을 적용해 `cctv-ai-engine`만 재생성했다.

| GIE | 모델 |
|---:|---|
| 1 | `models/person/yolov8n.engine` |
| 2 | `models/fall/yolov8n-pose.engine` |
| 3 | `models/head/helmet_model.engine` |
| 4 | `models/appearance/pa100k_resnet50_attr.engine` |

기동 로그에서 네 모델 모두 TensorRT 엔진과 전용 설정 파일이 정상적으로
로드되는 것을 확인했다. 재생 중 DeepStream 통계는 `frame_dropped=0`이었고,
약 16 ms 수준의 후처리 평균 시간이 기록됐다.

## 현재 판정 범위

### Native tracker 우회 및 좌표계 수정 후 결과

Primary tensor probe가 등록한 Person ROI를 `nvtracker`가 제거하는 현상을
재현했다. crop-Pose에서만 native tracker를 우회하고, 프로젝트에 이미 연결된
`SyntheticObjectIdAssigner`가 IoU 기반 ID를 부여하도록 변경했다. 기존
full-frame 경로의 nvtracker는 유지한다.

또한 평가 영상은 960x540인데 Compose가 `DS_STREAM_WIDTH/HEIGHT`를 컨테이너에
전달하지 않아 streammux가 기본 1920x1080 좌표계를 사용하고 있었다. 두 환경
변수를 Compose에 전달하고 평가 입력과 960x540으로 맞췄다. 운영 기본값은
1920x1080이다.

동일 영상 재생에서 단계별 카운터가
`probe/tensor/roi/person-object/pose-tensor/result = 544/544/366/364/364/327`
까지 증가했다. Pose 출력 최대 confidence도 좌표계 수정 전 0.03~0.06에서
수정 후 최대 0.56으로 상승했다. 따라서 Person ROI → Pose SGIE tensor → Pose
parser 연결은 실제 Jetson에서 검증됐다.

초기 평가 결과는 FN으로 표시됐지만, 실제 지정 경로가 아닌 이전 review-log를
읽은 평가기 오류였다. `--recreate-ai-engine`이 재시작을 자동 활성화하고, 영상별
재생성에서도 `FALL_SHADOW_REVIEW_LOG_PATH` override를 유지하도록 수정했다.

수정된 평가기로 동일 영상을 다시 실행한 결과는 TP 1, FN 0이며 최대 낙상 점수는
3.0이다. 결과는 `data/fall_eval/crop_pose_verified_v2_20260911.jsonl`에 저장했다.
near-miss도 함께 남으므로 전체 fall/not-fall 영상에 대한 Recall·오탐률 검증은
별도로 필요하다. 이 한 영상 결과만으로 모델 정확도를 확정하지 않는다.

### 2026-09-11 13:50 진단 결과 및 중단 지점

아래 이전 판정은 이 후속 결과로 갱신한다. `debugging-and-error-recovery`
스킬에 따라 실행 결과와 원인 가설을 분리했다.

- 결과 파일: `data/fall_eval/crop_pose_diagnostic_20260911.jsonl`.
- 대상: `20260910_141112_backward_fall`, 결과 FN, fall event 0,
  shadow record 0. Pose 입력이 정상임을 검증하지 못했으므로 모델 성능 비교에 사용하지 않는다.
- 보존 로그: `data/fall_eval/crop_pose_diagnostic_20260911.runtime.log`.
  마지막 통계는 frames=530, frame_dropped=0, crop_pose_results=0.
- 컨테이너에서 crop enabled=1, diagnostics=1, Pose GIE=2를 직접 확인했다.
  그런데 객체별 진단 로그가 없었다. 이는 객체 메타 전달 구간을 조사할 근거이며,
  tracker 또는 SGIE의 특정 결함을 확정하는 증거는 아니다.
- 재개 시 Primary 직후 등록 객체 수, tracker 통과 후 객체 수,
  Pose SGIE tensor 수를 차례로 측정한다. 엔진 로드 성공을 Pose 추론 성공으로
  간주하지 않는다. 헤드/헬멧 동시 실행 성능도 아직 미검증이다.
- 사용자 요청으로 추가 재현 실행을 멈추고 `.env.jetson`으로 AI 엔진을 복원한다.
  카메라 설정의 사전 상태는 `sample_eval` 파일 재생이다. 실제 CCTV 복귀를
  의미하지 않는다. 진단 코드 변경 후 CPU 회귀 테스트는 이번 진단 단계에서 재실행하지 않았다.

### Primary tensor probe 수정

NVIDIA DeepStream 설치 소스의 `deepstream-infer-tensor-meta-test`는 PGIE 바로
뒤의 queue src pad에서 tensor metadata를 읽는다. 같은 방식으로
`primary-tensor-queue`를 추가하고 probe 위치를 queue src pad로 옮겼다.
기존 구현처럼 primary-infer src pad에서 즉시 읽을 때 ROI가 등록되지 않던 문제를
대상으로 한 회귀 테스트를 추가했다. CPU 테스트에서 수정 전 5건 실패, 수정 후
5건 통과를 확인했다. 실제 Jetson 영상 재평가 결과는 아직 추가되지 않았다.

### 단계별 카운터로 확인한 tracker 입력 손실

queue 적용 후 동일 영상을 재평가하자 Primary probe와 ROI 등록은 정상화됐다.
Person interval 0 실행에서 `probe/tensor/roi/person-object/pose-tensor/result`는
`209/209/124/0/0/0`이었다. ROI 좌표와 confidence도 정상 범위였지만 tracker
뒤 객체가 0건이었다.

코드가 nvtracker의 `ll-config-file` 속성에 저수준 NvDCF YAML이 아니라
DeepStream 앱용 `[tracker]` 래퍼인 `config_tracker.txt`를 전달하고 있었다.
이를 `config_tracker_nvdcf.yml`로 수정했다. 실제 영상의 tracker/Pose 출력은
수정 후 재평가로 확인한다.

### 이전 실행 기록 (후속 결과와 구분)

이번 실행에서 `events_detected`가 증가하는 것까지는 확인했지만, 평가 스크립트가
사용하는 shadow 로그 경로와 컨테이너 환경의 로그 경로가 분리되어 영상별 TP/FN을
확정하지 못했다. 따라서 이번 기록은 **엔진 기동·파이프라인 연결 검증**으로만
취급하며, 모델 정확도 결과로 사용하지 않는다.

다음 실행에서는 평가 스크립트의 `--recreate-ai-engine` 옵션을 사용해 임시
Compose 환경을 실제 컨테이너에 반영하고, 명시한 `--review-log` 경로를
컨테이너 환경에도 자동 전달한다. 컨테이너의
`FALL_SHADOW_REVIEW_LOG_PATH`를 평가용 호스트 경로와 명시적으로 일치시킨 뒤
동일 영상을 재생해 Pose 생성률, 낙상 이벤트, FPS, frame drop을 저장한다.

검증 종료 후 운영 환경은 `DS_CROP_POSE_ENABLED=0`으로 복구했으며,
`cctv-ai-engine`은 healthy 상태다.
