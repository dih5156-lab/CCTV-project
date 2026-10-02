# 실시간 낙상 테스트 데이터 수집 규칙

작성일: 2026-09-09  
목적: 실제 CCTV 테스트 영상을 모델 비교와 pose 미검출 원인 분석에 재사용한다.

## 1. 저장 위치

학습 데이터와 섞이지 않도록 테스트 영상은 다음 구조로 저장한다.

```text
data/fall_test/
└── YYYYMMDD/
    ├── fall/
    ├── notfall/
    └── manifest.csv
```

예시:

```text
data/fall_test/20260909/fall/
data/fall_test/20260909/notfall/
```

## 2. 파일명 규칙

```text
{label}_data_{YYYYMMDD}_{scenario}_{take}.mp4
```

- `label`: `fall` 또는 `notfall`
- `YYYYMMDD`: 실제 촬영일
- `scenario`: 낙상 방향·가림 상태 코드
- `take`: 같은 조건의 반복 번호(`01`, `02`, ...)

### 낙상 시나리오 코드

| 코드 | 의미 |
|---|---|
| `side` | 옆으로 넘어짐 |
| `back` | 뒤로 넘어짐 |
| `front` | 앞으로 넘어짐 |
| `upper_hidden` | 넘어질 때 상체가 보이지 않음 |
| `lower_hidden` | 넘어질 때 하체가 보이지 않음 |

예시:

```text
fall_data_20260909_side_01.mp4
fall_data_20260909_back_01.mp4
fall_data_20260909_front_01.mp4
fall_data_20260909_upper_hidden_01.mp4
fall_data_20260909_lower_hidden_01.mp4
```

정상 영상도 같은 장면과 움직임을 포함하도록 수집한다.

```text
notfall_data_20260909_walk_01.mp4
notfall_data_20260909_sit_01.mp4
notfall_data_20260909_lie_01.mp4
notfall_data_20260909_bend_01.mp4
notfall_data_20260909_upper_hidden_01.mp4
notfall_data_20260909_lower_hidden_01.mp4
```

`notfall`에는 실제 낙상과 혼동하기 쉬운 동작을 포함한다. 예: 앉기, 눕기,
일어나기, 물건 줍기, 화면 밖으로 이동하기.

## 3. manifest.csv 양식

영상 파일과 테스트 결과를 다음 한 줄에 하나씩 기록한다.

```csv
file,label,scenario,take,camera_id,model_version,person_detected,pose_detected,fall_event,event_latency_ms,notes
fall_data_20260909_side_01.mp4,fall,side,01,camera_1,,,,, ,
notfall_data_20260909_sit_01.mp4,notfall,sit,01,camera_1,,,,, ,
```

테스트 후 다음 필드는 반드시 기록한다.

- `person_detected`: 사람 검출 여부
- `pose_detected`: keypoint 생성 여부
- `fall_event`: 낙상 이벤트 발생 여부
- `event_latency_ms`: 낙상 시작부터 이벤트까지 지연시간
- `model_version`: 실제로 실행한 모델 파일 또는 버전
- `notes`: 가림, 조도, 거리, 프레임 드롭 등 특이사항

## 4. 수집 순서

1. 한 시나리오를 같은 카메라 위치에서 3회 이상 촬영한다.
2. 낙상 영상과 정상 영상을 같은 조도·거리·배경 조건으로 맞춘다.
3. 먼저 `fall` 영상을 수집하고, 같은 조건의 `notfall` 영상을 수집한다.
4. 모델 테스트 시 영상 원본은 수정하지 않고 결과 overlay 영상만 별도 저장한다.
5. 테스트가 끝나면 manifest와 pose diagnostics 로그를 함께 보관한다.

## 5. 판정 기준

영상 단위로 다음을 구분한다.

```text
TP: fall 영상에서 fall_event 발생
FN: fall 영상에서 fall_event 미발생
FP: notfall 영상에서 fall_event 발생
TN: notfall 영상에서 fall_event 미발생
```

추가로 FN은 다음 단계 중 어디에서 실패했는지 기록한다.

```text
사람 미검출 → pose 미생성 → pose 품질 부족 → temporal 판정 실패 → 이벤트 발행 실패
```

## 6. 모델 비교 원칙

- 같은 원본 영상과 같은 manifest로 기존 모델과 새 모델을 각각 실행한다.
- 파일명과 영상 내용은 모델 실행 사이에 변경하지 않는다.
- 오프라인 학습·검증 영상 결과와 실제 CCTV 결과를 같은 지표로 합치지 않는다.
- 실시간 테스트에서는 Recall뿐 아니라 FPS, 지연시간, GPU 메모리, 프레임 드롭을 함께 기록한다.

## 7. 기존 오픈데이터 활용

기존 `falldata/낙상방향-003.tar` 내부의 `SCH_FN/test.zip`에는 방향별 코드가
이미 포함되어 있다.

| 코드 | 의미 |
|---|---|
| `N` | 비낙상 |
| `FY` | 앞으로 넘어짐 |
| `BY` | 뒤로 넘어짐 |
| `SY` | 옆으로 넘어짐 |

이 테스트셋은 그룹명 아래에 `NPY`와 `XLSX` 파일이 들어 있는 센서·특징
데이터셋이며 MP4 영상셋이 아니다. 압축 목록 기준 그룹 수는 다음과 같다.

```text
N: 568개
FY: 776개
BY: 584개
SY: 344개
합계: 2,272개
```

따라서 이 데이터는 기존 방향 분류 모델과 센서/특징 기반 모델의 오프라인
비교에 사용한다. RTSP 입력, 사람 검출, pose keypoint 생성률은 이 데이터로
측정하지 않고 실제 CCTV 영상 또는 별도로 추출된 영상 manifest로 평가한다.

`C1`~`C8`은 같은 장면의 카메라 또는 촬영 변형을 나타내므로, 학습과 평가를
나눌 때 동일 장면 그룹이 서로 다른 split에 섞이지 않도록 한다. `H_A`, `O_E`
등의 추가 코드는 원본 데이터 정의를 확인하기 전까지 낙상 방향이나 가림
상태로 임의 해석하지 않는다.
