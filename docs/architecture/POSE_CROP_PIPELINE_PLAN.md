# Person Detector 기반 Crop-Pose 전환 설계안

작성일: 2026-09-09  
상태: 학습 완료 후 POC

## 1. 현재 구조

```text
full-frame RTSP
→ pose_model.track(persist=True)
→ 사람 검출·추적·keypoint
→ 낙상 판정
```

현재는 pose 모델이 사람 검출까지 담당하므로 pose가 없는 프레임에서는 낙상
판정이 시작되지 않는다.

2026-09-11 기준으로 운영 기본 경로를 유지한 채 좌표 변환 공통 유틸리티와
opt-in crop 호출 POC, DeepStream Person Primary → Pose SGIE 토폴로지 및
사람 객체별 SGIE tensor parser를 추가했다. `POSE_CROP_MODE=0`이 기본값이며, 값을
`1/true/yes/on`으로 설정한 경우에만 person detector 선행 추론과 crop pose
호출을 사용한다.

- `src/core/_yolo_postprocess.py::expand_bbox_for_pose_crop`: 사람 bbox 여백 확장 및 프레임 clipping
- `src/core/_yolo_postprocess.py::extract_pose_crops`: 프레임에서 독립적인 pose crop과 원점 bbox 생성
- `src/core/_yolo_postprocess.py::restore_crop_keypoints`: crop keypoint를 원본 프레임 좌표로 복원
- `src/core/_yolo_postprocess.py::attach_pose_results_to_tracks`: crop 순서의 pose 결과를 person `track_id`와 결합
- `tests/test_yolo_postprocess_pose_crop.py`: 경계·잘못된 입력·offset 복원 검증
- `src/core/ai/analyzer.py::_run_pose_crop`: person track → crop pose predict →
  원본 좌표 복원 → `FallDetector` 연결 POC
- `src/core/deepstream_processor.py::_emit_crop_pose_events`: Person GIE 1의
  `obj_user_meta_list`에 저장된 Pose SGIE GIE 2 tensor를 ROI별로 복원하고
  부모 `object_id`를 낙상 이벤트에 연결

모델 로딩은 crop 모드에서 person 모델을 함께 준비하도록 확장했지만, 기본값
`POSE_CROP_MODE=0`에서는 기존 full-frame 추론을 그대로 사용한다. Jetson
실시간 성능과 실제 낙상 영상 검증 전에는 운영 모드로 전환하지 않는다.

## 2. 목표 구조

```text
full-frame RTSP
→ person_model.track(persist=True)
→ track_id + person bbox
→ bbox padding + crop
→ pose_model.predict(batch crops)
→ crop 좌표를 원본 프레임 좌표로 복원
→ track_id별 keypoint history
→ FallDetector
→ person/fall 중복 제거
→ 이벤트 발행
```

## 3. 인터페이스 원칙

FallDetector는 전체 YOLO 결과 인덱스가 아니라 사람 단위 입력을 받는다.

```python
fall_detector.detect_keypoints(
    keypoints=keypoints_xy_conf,
    bbox_width=person_bbox.width,
    bbox_height=person_bbox.height,
    bbox_y=person_bbox.y,
)
```

현재 코드에는 crop 경로가 직접 사용할 수 있는
`FallDetector.detect_keypoints(keypoints, bbox_width, bbox_height, bbox_y)`
메서드가 추가되어 있다. 기존 `detect(keypoints, idx, ...)` 하위 호환 API는
그대로 유지한다.

사람 검증과 어깨 위치 검증도 crop 입력용
`validate_person_keypoints()` 및 `validate_shoulder_keypoints()`로 제공한다.

track_id는 person detector tracker가 생성하며, pose 모델은 crop별 1회성
`predict()`만 수행한다.

## 4. 좌표 복원

crop 전 padding 좌표를 `(x1, y1)`로 기록한다. crop 내부 keypoint는 다음처럼
원본 좌표로 변환한다.

```text
original_x = crop_x + x1
original_y = crop_y + y1
```

resize 또는 letterbox가 적용되면 scale과 padding을 먼저 되돌린 뒤 offset을
더한다. 이 계산은 별도 단위 테스트로 검증한다.

## 5. 실패 처리

- 현재 POC는 pose가 없는 crop을 해당 프레임의 keypoint 없음으로 처리하고
  기존 full-frame 경로로 자동 전환하지 않는다.
- 이전 keypoint 재사용과 track별 temporal 상태 정리는 후속 단계에서 구현한다.
- temporal 상태가 추가되더라도 만료 시간과 track 종료 시 삭제를 함께 적용해
  오래된 keypoint로 낙상 이벤트를 만들지 않는다.

## 6. 성능 위험과 대응

- person 모델과 pose 모델 동시 로드로 GPU 메모리가 증가한다.
- 사람 수만큼 pose 추론이 늘어나므로 crop batch inference를 우선 검토한다.
- 기존 full-frame 경로를 삭제하지 않고 `POSE_CROP_MODE=0/1` feature flag로
  비교한다.
- `POSE_CROP_MODE` 기본값은 `0`이다. `1`이어도 analyzer에 `_run_pose_crop()`가
  없으면 자동으로 기존 full-frame 경로를 사용한다.
- Jetson에서 FPS, p95 지연시간, GPU memory, frame drop을 측정한 뒤 운영
  전환 여부를 결정한다.

## 7. 구현 순서

1. 현재 학습 완료와 모델 비교
2. `load_models()`의 person/pose 동시 로드 POC
3. person tracker → crop pose 연결 및 DeepStream SGIE tensor parser
4. 좌표 복원 및 track별 FallDetector 테스트
5. 녹화 영상 비교
6. shadow 실시간 테스트
7. 성능 기준 충족 시 운영 전환

## 8. Action Layer 경계

Action Layer는 Jetson 배포 단위로 각 보드에 존재한다. 다른 Jetson이 영상을
대신 추론하더라도 이벤트의 `source_site_id`를 유지하며, 다른 사이트의 물리
장치를 임의로 제어하지 않는다. 완전한 장치 failover가 필요하면 Action Layer를
Jetson 외부의 사이트 게이트웨이로 분리하는 별도 설계가 필요하다.

## 9. 비교 실행 절차

> **주의:** 학습 프로세스가 `cctv-ai-engine`에서 실행 중이면 아래 명령을
> 실행하지 않는다. `up -d cctv-ai-engine`은 해당 컨테이너를 재생성할 수 있어
> 학습을 중단시킬 수 있다. 학습 완료와 산출물 백업을 확인한 뒤 테스트용
> Jetson에서만 수행한다.

1. 기본 full-frame 결과를 먼저 저장한다. `POSE_CROP_MODE=0` 상태에서 동일한
   입력 영상과 `conf`, `imgsz`, frame stride를 사용해 결과 JSONL과 처리 시간을
   보관한다.
2. 검증용 환경에서만 crop 경로를 켠다.

   ```bash
   POSE_CROP_MODE=1 docker compose -f docker-compose.jetson.yml up -d cctv-ai-engine
   ```

3. 같은 영상을 다시 실행해 사람 검출 수, pose 생성률, 낙상 TP/FN, 평균 및
   p95 지연시간, FPS, GPU 메모리, frame drop을 비교한다. 결과에는 모델 파일,
   git revision, 환경변수와 측정 시각을 함께 기록한다.
4. 성능 또는 정확도 기준을 충족하지 못하면 즉시 기본 경로로 되돌린다.

   ```bash
   POSE_CROP_MODE=0 docker compose -f docker-compose.jetson.yml up -d cctv-ai-engine
   ```

위 명령은 해당 서비스만 재생성하므로, 실제 장비 검증 전에는 테스트용
Jetson과 녹화 영상으로 먼저 수행한다. 운영 전환 판단은 수치 비교표와
실시간 현장 낙상 확인을 모두 통과한 뒤 별도로 결정한다.
