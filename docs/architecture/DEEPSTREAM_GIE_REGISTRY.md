# DeepStream GIE ID 레지스트리

작성일: 2026-09-11
상태: crop-Pose 전환 기준표 (모델 변경 시 이 문서를 먼저 갱신)

GIE ID는 DeepStream 파이프라인에서 모델 결과를 구분하는 식별자다. 설정 파일,
Python tensor parser, `operate-on-gie-id`, 이벤트 metadata가 같은 ID를 사용해야
모델 결과가 다른 모델로 잘못 연결되지 않는다.

## 현재 운영 구성

| GIE ID | 역할 | 모델 | 처리 대상 | 설정 |
|---:|---|---|---|---|
| 1 | Primary Pose/Person | `models/fall/yolov8n-pose.engine` | 전체 프레임 | `config_infer_primary.txt` |
| 2 | Helmet/Head | `models/head/helmet_model.engine` | GIE 1 ROI | `config_infer_helmet.txt` |
| 3 | 외형 속성 | `models/appearance/pa100k_resnet50_attr.engine` | 사람 ROI | `config_infer_pa100k.txt` |

현재 Primary Pose 모델이 사람 bounding box와 keypoint를 동시에 생성한다. 따라서
별도 Person Primary GIE는 사용하지 않는다.

## crop-Pose 목표 구성

| GIE ID | 역할 | 적용 모델(고정 경로) | 입력/정밀도 | 처리 대상 | 설정 | 상태 |
|---:|---|---|---|---|---|
| 1 | Primary Person | `models/person/yolov8n.engine` | 640×640, FP16 | 전체 프레임 사람 검출 | `config_infer_person_primary.txt` | 엔진 로드 확인, 운영 기본값은 OFF |
| 2 | Pose SGIE | `models/fall/yolov8n-pose.engine` | 640×640, FP16 | GIE 1 사람 ROI keypoint 추정 | `config_infer_pose_sgie.txt` | ROI parser 반영, 실제 이벤트 검증 필요 |
| 3 | Helmet/Head SGIE | `models/head/helmet_model.engine` | 320×320, FP16 | GIE 1 사람 ROI 헬멧·head 검출 | `config_infer_helmet_sgie.txt` | 설정 반영, 결과 메타데이터 검증 필요 |
| 4 | Appearance SGIE | `models/appearance/pa100k_resnet50_attr.engine` | 192×256, FP16 | GIE 1 사람 ROI 외형 속성 | `config_infer_pa100k_sgie.txt` | 설정 반영, 결과 메타데이터 검증 필요 |

Pose와 Helmet은 같은 Person ROI를 공유한다. Person 모델은 프레임마다 한 번만
실행하며, Pose SGIE와 Helmet SGIE가 전체 프레임을 다시 탐색하지 않는다.

### Batch 크기 기준

Primary와 Secondary의 batch 의미가 다르다.

| 구분 | 기준 | 현재값 | 확장 기준 |
|---|---|---:|---|
| Person Primary | `nvstreammux`가 동시에 묶는 카메라 프레임 수 | 카메라 수 | 카메라 수와 동일 |
| Pose SGIE | 한 프레임 묶음에서 Pose로 보낼 Person ROI 수 | 1 | `카메라 수 × 카메라당 동시 사람 수 P95` |
| Helmet/Head SGIE | Helmet 모델로 보낼 Person ROI 수 | 1 | Pose와 같은 방식으로 산정하되 interval 적용량 고려 |
| Appearance SGIE | 외형 모델로 보낼 Person ROI 수 | 1 | reinfer interval 적용 후 동시 ROI 수 P95 |

현재 Person, Pose, Helmet 엔진의 TensorRT 입력 batch는 1로 고정되어 있다.
따라서 `DS_*_SGIE_BATCH_SIZE`도 1을 유지한다. 값을 4나 8로 올리려면 먼저
해당 최대 batch를 지원하는 ONNX/TensorRT 엔진을 다시 생성하고, 한 프레임에
나오는 사람 수 분포와 GPU 메모리, 지연시간을 함께 측정한다. 시작 권장값은
카메라 1대 기준 4이며, P95 동시 인원이 4명을 넘거나 여러 카메라를 묶을 때만
8 이상을 검토한다. batch를 크게 잡아도 입력이 모일 때까지 기다리거나 메모리만
늘어날 수 있으므로 최대 인원만 보고 과도하게 설정하지 않는다.

Person Primary 뒤에는 `primary-tensor-queue`를 두고 그 queue의 src pad에서
raw tensor를 Person 객체 메타로 등록한다. 이는 Primary src pad 시점보다 tensor
metadata가 확실히 준비된 경계이며, 등록된 객체가 tracker와 각 SGIE로 전달된다.
crop-Pose 모드의 Person Primary는 `DS_CROP_PERSON_INTERVAL=0`을 기본으로 한다.
현재 tracker는 신규 객체를 3프레임 동안 확인하고 중간 미검출 1프레임에도 조기
종료할 수 있으므로, interval 1로 Person을 격프레임 실행하면 track이 확정되기 전에
ROI가 사라질 수 있다. 기존 full-frame 경로의 `DS_PRIMARY_INTERVAL` 값은 유지한다.
낙상 SGIE 입력은 첫 검출 프레임부터 필요하므로 tracker의 `probationAge`는 0,
`earlyTerminationAge`는 3으로 설정한다. 이벤트 오탐 억제는 tracker가 객체 자체를
숨기는 방식보다 낙상 누적 판정과 이벤트 필터에서 처리한다.
crop-Pose 경로는 `config_tracker_iou.yml`을 사용한다. NvDCF는 영상 특징을 추가로
계산하며 현재 수동 ROI 경로에서 입력 제출 오류가 확인됐다. IOU tracker는 Person
bbox만으로 ID를 연결하므로 이 경로에 맞고 GPU 부하도 더 낮다. NvDCF 재도입은
가림·교차 상황에서 IOU ID 전환율을 측정한 뒤 별도 호환성 검증으로 결정한다.

### 모델 파일 및 역할

| 모델 | 파일 확인 | 형식/입력 | 역할 | GIE 여부 |
|---|---|---|---|
| `models/person/yolov8n.engine` | 존재 (약 8.7 MB) | TensorRT FP16, 640×640 | 전체 프레임 사람 검출 | GIE 1 |
| `models/fall/yolov8n-pose.engine` | 존재 (약 8.7 MB) | TensorRT FP16, 640×640 | 사람 ROI keypoint 추정 | GIE 2 |
| `models/head/helmet_model.engine` | 존재 (약 8.0 MB) | TensorRT FP16, 320×320 | 사람 ROI helmet/head 검출 | GIE 3 |
| `models/appearance/pa100k_resnet50_attr.engine` | 존재 (약 58 MB) | TensorRT FP16, 192×256 | 사람 ROI 외형 속성 | GIE 4 |

`fall_temporal_same_split_resume_20260908.pt`와
`yolo_pose_same_split_resume_20260908_rf.joblib`는 GIE가 아닌 Python 후처리
분류 모델이다. Pose keypoint에서 feature를 만든 뒤 낙상 점수를 계산하므로
GIE ID를 할당하지 않고 `FallDataAux`에서 관리한다.

## 코드 적용 규칙

- `DS_PERSON_GIE_ID=1`
- `DS_POSE_GIE_ID=2`
- `DS_HELMET_GIE_ID=3`
- `DS_PPHUMAN_GIE_ID=4`
- Pose/Helmet/Appearance SGIE의 `operate-on-gie-id=1`
- Tensor parser는 GIE ID별 task, label, threshold를 별도로 등록
- 기존 full-frame 경로는 별도 feature flag로 유지

## 전환 순서

1. Person Primary 설정과 Pose SGIE 설정을 추가한다.
2. Helmet SGIE의 `operate-on-gie-id`를 1로 고정한다.
3. 외형 SGIE를 GIE 4로 이동하고 기존 GIE 3 참조를 갱신한다.
4. Tensor parser와 이벤트 metadata의 GIE ID를 갱신한다.
5. 테스트 카메라에서 Person → Pose/Helmet/Appearance 연결을 확인한다.
6. FPS, frame drop, GPU 메모리, Pose 생성률을 기존 구성과 비교한다.

## 주의사항

GIE ID만 변경하고 parser 또는 SGIE 대상 ID를 변경하지 않으면 헬멧 결과가 Pose
결과로 해석되거나 외형 속성이 누락될 수 있다. 따라서 설정 파일과 코드 변경은
항상 이 레지스트리와 함께 검토한다.

## 변경 이력

| 날짜 | 변경 | 검증 상태 |
|---|---|---|
| 2026-09-11 | crop-Pose용 Person/ Pose/ Helmet/ Appearance 모델과 전용 SGIE 설정을 등록하고 GIE 1~4로 고정 | Person·Pose 엔진 기동 로그와 ROI parser 코드 확인. SGIE ROI 결과와 실제 낙상 이벤트는 후속 영상 검증 필요 |

새 모델을 추가하거나 교체할 때는 `적용 모델(고정 경로)`, 입력 크기·정밀도,
GIE ID, 설정 파일, 상태와 변경 이력을 함께 갱신한다. `.env.jetson`의
`DS_CROP_POSE_ENABLED`가 `0`인 동안 운영 파이프라인은 기존 full-frame 구성을
사용한다.

경로 변경 후에는 `python scripts/health/check_model_layout.py`를 실행해 네 개의
crop-Pose 설정 파일과 엔진 파일이 모두 존재하는지 확인한다.
