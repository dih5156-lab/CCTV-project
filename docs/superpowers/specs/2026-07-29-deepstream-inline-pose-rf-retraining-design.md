# DeepStream Inline Pose RF 재학습 설계

## 1. 목적

기존 YOLO Pose 오프라인 특징으로 학습한 Random Forest 모델은 실제
`DeepStream pose → 인라인 RF` 실행 경로에서 0.7 임계값 기준 낙상 재현율
5.3%를 기록했다. 정상 오탐은 없었지만 낙상 38개 중 36개를 놓쳤기 때문에
실사용할 수 없다.

이번 작업의 목적은 RF에 전달되기 직전의 실제 DeepStream 특징을 캡처하고,
그 특징으로 새 후보 RF 모델을 학습한 뒤 동일한 런타임 경로에서 다시
평가하는 것이다.

## 2. 범위와 성공 기준

### 포함 범위

- DeepStream 인라인 pose 특징의 선택적 sidecar 캡처
- 평가 영상의 scene ID와 정답 라벨 결합
- 동일 영상에서 생성된 여러 윈도우를 그룹으로 유지하는 데이터셋 생성
- 기존 모델을 보존한 신규 RF 후보 모델 학습
- Training 영상 낙상 40개 + 정상 40개로 특징 캡처 및 후보 학습
- Training과 겹치지 않는 Validation 영상 낙상 40개 + 정상 40개로
  0.7 임계값 고정 런타임 재평가
- 결과가 개선된 경우에만 200개 학습 + 80개 검증 이상으로 확대

### 제외 범위

- TensorRT 엔진 재생성
- DeepStream pose 엔진 교체
- 운영 알람의 `confirm` 전환
- 임계값 0.7 하향
- 별도 YOLO, MediaPipe, temporal 자식 추론 프로세스 재도입

### 1차 성공 기준

- 낙상 재현율 90% 이상
- 정상 오탐률 5% 이하
- `NO_RESULT` 비율 별도 보고
- 자식 추론 프로세스 0개
- 평가 종료 후 기존 카메라 설정 자동 복원
- 서비스 메모리와 health 상태에 명확한 회귀 없음

기준을 만족하지 못하면 모델은 후보 상태로 남기고 운영 모델을 교체하지
않는다.

## 3. 검토한 접근 방식

### A. 별도 sidecar JSONL 캡처 — 채택

RF 입력 특징을 별도 JSONL 파일에 저장한다. 리뷰 로그와 학습 데이터를
분리할 수 있고, 환경변수가 없으면 완전히 비활성화할 수 있다. 현재 특징은
`_summarize_frames`가 생성하는 작은 수치 벡터이므로 이미지나 전체
키포인트 프레임을 저장하는 것보다 I/O 부담이 작다.

### B. 기존 리뷰 로그에 특징 포함

구현은 단순하지만 운영 리뷰 로그 크기가 커지고 라벨 검토 데이터와 모델
학습 데이터가 섞인다. 장기 운영과 정리 작업이 어려워 채택하지 않는다.

### C. 별도 오프라인 DeepStream 추출기

데이터 생성 파이프라인은 분리되지만 GStreamer/DeepStream 파이프라인을
중복 구현해야 한다. 실제 서비스와 다른 전처리 경로가 다시 생길 수 있어
현재 단계에는 과도하다.

## 4. 구조

### 4.1 런타임 특징 캡처

캡처 지점은 `FallDataAuxVerifier._verify_inline_pose_rf`에서
`_summarize_frames` 호출 직후다. 이 시점의 `feature_names`와
`feature_vector`는 실제 classifier 입력 선택 이전의 canonical summary
특징이며, DeepStream 좌표·신뢰도·fall score 계산 결과를 그대로 반영한다.

새 환경변수:

- `FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH`
  - 미설정 또는 빈 값: 캡처 비활성화
  - 설정: 해당 JSONL 파일에 캡처 레코드 추가

기본 운영 설정에는 값을 넣지 않는다. 평가 명령에서만 임시로 활성화하고
평가 종료 후 원래 설정으로 복원한다.

sidecar 레코드 스키마:

```json
{
  "schema_version": 1,
  "captured_at": "UTC ISO-8601",
  "camera_id": "sample_eval",
  "runtime": "deepstream_pose_inline",
  "window_seconds": 3.0,
  "frames_with_pose": 67,
  "sampled_frames": 48,
  "feature_names": ["frames_seen"],
  "feature_vector": [48.0],
  "source_model": "models/experiments/current.pkl"
}
```

실제 `feature_names`와 `feature_vector`에는 전체 summary 특징이 들어간다.
벡터 길이와 이름 길이가 다르면 레코드를 저장하지 않고 오류 상태를
명시한다. 캡처 실패는 shadow 추론 결과를 막지 않는 fail-open 방식으로
처리한다.

### 4.2 영상 라벨 결합

DeepStream 런타임은 scene ID와 정답 라벨을 알지 못하므로 캡처 레코드에는
이를 넣지 않는다. 평가 스크립트가 영상별 sidecar offset을 기록하고,
해당 영상 재생 구간에서 추가된 캡처 레코드에 다음 값을 결합한다.

- `scene_id`
- `scene_group`
- `label`
- `is_fall`
- `camera`
- `video_path`

결합 결과는 별도 dataset JSONL로 저장한다. 유효한 캡처가 없는 영상은
`NO_RESULT` 목록에 남기되 학습 행으로 만들지 않는다.

### 4.3 윈도우 선택과 데이터 누출 방지

한 영상에서 여러 3초 윈도우가 생성될 수 있다. 모든 유효 윈도우는 사용할
수 있지만, 동일 `scene_group`의 윈도우는 반드시 같은 split에 배치한다.
행 단위 무작위 분할은 금지한다.

1차 Training 40+40 단계에서는 다음을 적용한다.

- 학습/검증 분리는 `scene_group` 기준
- 최종 런타임 평가는 별도 Validation 40+40만 사용
- Training과 Validation의 `scene_id` 및 `scene_group` 교집합이 있으면 평가 중단
- class weight는 `balanced`
- 고정 random seed 사용
- 입력 컬럼 순서는 sidecar의 `feature_names`로 검증
- NaN/Inf 또는 schema 불일치 행은 제외하고 개수 보고

### 4.4 후보 모델 학습

새 학습 스크립트는 sidecar 기반 dataset JSONL을 읽고 Random Forest를
학습한다. 기존 모델 파일은 덮어쓰지 않는다.

후보 bundle 필수 항목:

- `model`
- `feature_names`
- `training_config`
- `inference_config`
- `dataset_summary`
- `decision_threshold: 0.7`
- `feature_source: deepstream_pose_inline`
- 생성 시각과 입력 dataset 경로

학습 결과에는 holdout confusion matrix, precision, recall, false positive
count, false negative count와 scene ID별 예측을 저장한다.

### 4.5 재평가와 승격

후보 모델 경로를 평가 컨테이너의
`FALLDATA_AUX_COMPARE_MODEL_PATH`에만 임시 적용하고 `shadow` 모드에서
별도 Validation 40+40을 다시 재생한다.

평가 종료 시:

1. 기존 카메라 설정 복원
2. 기존 모델 경로 복원
3. 캡처 환경변수 비활성화
4. 컨테이너 health와 메모리 확인
5. 자식 추론 프로세스 0개 확인

성공 기준을 모두 만족하기 전에는 운영 모델 경로나 모드를 영구 변경하지
않는다.

## 5. 오류 처리

- 캡처 경로를 열 수 없음: 추론은 계속하고 로그 경고 및 캡처 실패 수 증가
- 특징 schema 불일치: 해당 행 제외, 모델 학습 중단 여부를 보고서에 명시
- 영상별 캡처 없음: `NO_RESULT`, 음성 판정으로 계산하지 않음
- Docker 재시작 실패: `finally`에서 카메라와 모델 설정 복원 재시도
- 학습 데이터가 한 클래스뿐임: 모델 생성 없이 명확한 오류 반환
- 그룹 분할 후 한 split에 클래스가 없음: 학습 중단
- 목표 성능 미달: 후보 모델 보존, 운영 설정 미변경

## 6. 테스트 전략

### 단위 테스트

- 캡처 비활성화 시 파일을 만들지 않음
- 캡처 활성화 시 feature name/vector가 정확히 기록됨
- 캡처 쓰기 실패가 추론 결과를 바꾸지 않음
- 평가기가 sidecar offset 이후 레코드만 scene 라벨과 결합
- `NO_RESULT` 영상은 학습 행에서 제외
- schema/벡터 길이 불일치 거부
- scene group 단위 split 보장
- 후보 bundle에 런타임 호환 필드 포함

### 통합 검증

- Training 영상 1개 정상 + 1개 낙상 스모크 캡처
- 생성 dataset으로 후보 RF 학습
- 후보 모델을 실제 `deepstream_pose_inline`에서 로드
- Training 40+40 캡처·학습
- 별도 Validation 40+40 재평가
- 설정 복원, health, 메모리, 자식 프로세스 확인

## 7. 영향 범위

- `src/core/ai/_falldata_aux.py`: 선택적 특징 캡처
- `scripts/ops/evaluate_sample_deepstream_replay.py`: sidecar와 scene 라벨 결합
- 신규 dataset 학습 스크립트: 캡처 dataset 기반 RF 학습
- 관련 단위 테스트

DeepStream TensorRT 엔진, 이벤트 규칙, API, DB schema, MQTT payload에는
변경이 없다. 캡처 환경변수가 비어 있으면 기존 운영 동작과 파일 출력은
동일하다.
