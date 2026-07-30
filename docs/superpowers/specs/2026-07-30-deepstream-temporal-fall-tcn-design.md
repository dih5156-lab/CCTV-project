# DeepStream 경량 시계열 낙상 TCN 재설계

## 1. 결론

DeepStream pose와 평가 기반은 유지하고, 3초 구간을 50개 요약값으로
압축하던 Random Forest 후보만 중단한다. 실제 DeepStream 런타임에서
생성된 48프레임 시퀀스를 입력으로 받는 작은 Conv1D TCN을 새 후보 모델로
학습한다.

운영 신뢰도 임계값은 0.7 이상으로 유지한다. 작은 검증 관문을 통과하기
전에는 전체 데이터 학습, TensorRT 변환, 운영 알람 연결을 진행하지 않는다.

## 2. 현재 확인된 문제

- 기존 인라인 RF는 정상 오탐이 적지만 낙상 재현율이 약 5.3%였다.
- 복도 hard-negative를 추가한 RF v2는 정상 8개 중 오탐을 0개로 줄였지만,
  낙상 8개 중 2개만 검출했다.
- 정상 복도와 낙상 복도의 summary 확률 구간이 겹쳐 class weight나
  임계값 조정만으로는 분리가 개선되지 않았다.
- 기존 `train_fall_temporal_tcn.py` 초안은 오프라인 YOLO pose 캐시를
  사용하며, 현재 DeepStream 입력과 정확히 같지 않다.
- 기존 TCN 캐시 선택 함수는 `_feature_path`의
  `fall_window_margin_frames` 인자를 전달하지 않아 관련 테스트가 실패한다.

따라서 프로젝트 전체를 처음부터 다시 만드는 것은 불필요하지만, 모델
입력과 학습 샘플 구성은 시계열 기준으로 다시 잡아야 한다.

## 3. 검토한 접근 방식

### A. summary RF 추가 튜닝

변경과 배포가 가장 작지만 이미 정상·낙상 특징이 겹치는 것이 확인됐다.
주 모델 후보로 계속 사용하지 않는다.

### B. 오프라인 YOLO pose TCN

기존 캐시를 재사용해 빠르게 학습할 수 있다. 그러나 학습 입력과 실제
DeepStream pose 입력 차이 때문에 이전과 같은 domain mismatch가 남는다.
구조와 학습 코드 검증용 보조 실험으로만 사용한다.

### C. DeepStream 48프레임 TCN — 채택

실제 런타임의 `fall_score`, bbox, keypoint 신뢰도, 자세 사유를 시간순으로
직접 사용한다. TensorRT가 지원하는 Conv1D, ReLU, mean/max pooling,
Linear 연산만 사용해 Jetson 배포 난이도를 낮춘다.

## 4. 범위

### 포함

- DeepStream sidecar schema에 선택된 48개 `frame_records` 추가
- 기존 manifest의 `fall_start_frame`, `fall_end_frame`, FPS를 사용한
  낙상 구간 정렬
- 정상, 낙상, 낙상 영상의 비낙상 구간 hard-negative 분리
- scene/group 누출 방지 split
- 경량 temporal-only TCN 학습 및 0.7 이상 임계값 탐색
- 작은 probe부터 단계적으로 실제 DeepStream replay 평가
- 정확도 관문 통과 후 ONNX export와 TensorRT 호환성 확인

### 제외

- pose TensorRT 엔진 교체
- 운영 `confirm` 모드 전환
- 임계값 0.7 미만 사용
- 원본 400GB 전체의 즉시 재처리
- RNN/LSTM/Transformer 또는 별도 상시 추론 프로세스
- API, DB schema, MQTT payload 변경

## 5. 데이터 흐름

1. manifest에서 Training과 Validation을 별도로 읽는다.
2. 낙상 영상은 주석 구간과 앞뒤 margin을 기준으로 짧은 replay clip을 만든다.
3. 정상 영상과 낙상 영상의 주석 밖 구간에서도 정상 clip을 만든다.
4. clip을 기존 DeepStream 파이프라인으로 재생한다.
5. `_verify_inline_pose_rf`가 선택한 동일한 48개 frame record를 sidecar에
   기록한다.
6. 평가기가 clip ID, scene ID, group, 라벨, 장소, 연령대, 낙상 방향을
   sidecar 레코드와 결합한다.
7. 동일 scene/group의 모든 clip은 반드시 같은 split에 둔다.
8. TCN은 `[batch, 48, feature_count]`를 입력받아 낙상 logit 하나를 출력한다.

낙상 양성 window는 주석 구간과 실제로 겹치는 clip만 사용한다. 낙상 영상
전체를 양성으로 취급하지 않는다. 주석이 없거나 FPS를 확인할 수 없는 낙상
영상은 양성 학습에서 제외하고 수량을 보고한다.

## 6. 입력과 모델

초기 입력은 현재 `FRAME_FEATURE_NAMES`와 동일한 다음 계열을 사용한다.

- fall score와 detection confidence
- bbox aspect와 area ratio
- visible keypoint 수와 평균 신뢰도
- torso horizontal, leg above head 등 9개 pose reason flag

시퀀스는 시간순으로 정렬하고 48개로 균일 샘플링한다. 부족한 앞부분은
0으로 padding하며 padding mask를 별도 입력으로 추가하지 않는다. 첫 모델은
기존 `FallTemporalTCN` 구조를 재사용하고 summary branch는 제거한다. 이는
summary RF의 장소 편향이 다시 모델에 직접 유입되는 것을 줄이기 위함이다.

체크포인트에는 다음 호환 정보를 반드시 저장한다.

- format version과 model type
- sequence length와 ordered feature names
- channels와 state dict
- decision threshold
- 데이터셋 버전과 split hash
- 학습/검증 지표

## 7. 학습과 평가 관문

임계값은 Validation 결과를 보고 임의로 낮추지 않는다. 0.70부터 0.95까지
탐색하되 성공 기준을 만족하는 임계값 중 가장 낮은 값을 후보로 선택한다.

단계별 관문은 다음과 같다.

1. Probe 8 낙상 + 8 정상: recall 75% 이상, 정상 FPR 10% 이하
2. Small 20 낙상 + 20 정상: recall 85% 이상, 정상 FPR 7% 이하
3. Candidate 40 낙상 + 40 정상: recall 90% 이상, 정상 FPR 5% 이하

각 단계는 Training 내부 group holdout이 아니라, Training과 겹치지 않는
Validation scene으로 최종 판단한다. room/corridor, 연령대, 낙상 방향별
결과와 `TP/TN/FP/FN/NO_RESULT`를 함께 보고한다. 앞 단계가 실패하면 다음
대규모 단계로 확대하지 않고 실패 scene과 라벨 정렬부터 점검한다.

## 8. 런타임과 TensorRT

정확도 검증 동안에는 `.pt` 후보를 오프라인 평가에만 사용한다. Candidate
관문 통과 후 다음 순서로 배포 호환성을 검증한다.

1. 고정 입력 `[1, 48, feature_count]` ONNX export
2. ONNX Runtime 또는 PyTorch와 출력 오차 비교
3. Jetson에서 TensorRT FP16 엔진 생성
4. 동일 입력에 대한 확률 오차와 지연시간 측정
5. 기존 프로세스 내부 shadow 추론으로 연결

TCN의 연산은 TensorRT 친화적으로 제한한다. 변환 실패 시 pose 엔진을
바꾸지 않고, 원인이 되는 TCN 연산만 교체한다. 정확도 기준을 만족하기
전에는 TensorRT 변환 작업 자체를 성능 개선으로 간주하지 않는다.

## 9. 오류 처리와 안전장치

- sidecar 미설정: 기존 동작 유지
- frame schema 불일치 또는 NaN/Inf: 해당 샘플 제외 및 수량 보고
- 주석/FPS 누락: 양성 샘플 생성 금지
- Training/Validation scene 중복: 학습 또는 평가 즉시 중단
- 한 split에 한 클래스만 존재: 학습 중단
- 정확도 관문 미달: 후보 보존, 운영 모델과 설정 미변경
- replay 종료 또는 실패: 카메라 설정, 모델 경로, capture 환경변수 복원

## 10. 구현 순서와 테스트

1. 현재 TCN의 캐시 인자 누락 테스트를 먼저 정상화한다.
2. frame record capture schema 테스트를 실패 상태로 추가한 뒤 구현한다.
3. 주석 구간 clip/label 정렬 단위 테스트를 추가한다.
4. group 누출, class 균형, feature order 검증 테스트를 추가한다.
5. temporal-only 학습과 checkpoint 계약 테스트를 추가한다.
6. 1 정상 + 1 낙상 replay capture 스모크 테스트를 수행한다.
7. Probe 8+8을 학습·평가하고 관문 결과를 저장한다.
8. 통과한 경우에만 20+20, 이후 40+40으로 확대한다.
9. Candidate 관문 통과 후에만 ONNX/TensorRT 검증을 시작한다.

## 11. 영향 범위

- `src/core/ai/_falldata_aux.py`: frame sequence sidecar 기록과 temporal
  후보 입력
- `src/core/ai/fall_temporal_model.py`: 기존 경량 temporal-only 모델 재사용
- `scripts/datasets/train_fall_temporal_tcn.py`: DeepStream capture dataset,
  group split, checkpoint와 threshold 선택
- `scripts/ops/evaluate_sample_deepstream_replay.py`: clip/annotation 결합과
  단계별 결과 보고
- 관련 단위 및 통합 테스트

기본 capture 환경변수가 비어 있고 후보 모델이 지정되지 않으면 현재 운영
동작에는 변화가 없다.
