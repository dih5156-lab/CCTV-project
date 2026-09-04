# 데이터 라벨링·재학습 운영

## 1. 데이터 수집 원칙

현장 영상은 최소 수집·접근 제한·보관 기간 준수를 기본으로 한다. 얼굴·사람 영상은 회사 개인정보 정책과 법무 검토를 따른다. 원본은 운영 DB와 분리하고, 분석용 사본에는 접근 권한과 만료일을 둔다.

## 2. 라벨 기준

| 영역 | 라벨 예시 | 주의 |
|---|---|---|
| 사람 | bbox, person | 작은 사람·가림도 별도 표시 |
| Pose | COCO 17 keypoint | 보이지 않는 관절 confidence 구분 |
| 낙상 | fall/non-fall, 시작·종료 | 앉기·눕기·쪼그리기 hard negative 포함 |
| 헬멧 | helmet/head | 미착용과 머리 검출을 혼동하지 않음 |
| 외형 | 상·하의 색상/형태/소지품 | 보이지 않는 속성은 unknown/exclude |

## 3. 데이터셋 분리

같은 장면의 카메라 변형이 train/test에 동시에 들어가지 않도록 `scene_base` 기준 group split을 사용한다. 정상·위험·조명·거리·가림·다중 인원 비율을 기록한다.

## 4. 재학습 절차

1. miss/오탐 영상을 수집하고 사람이 라벨링한다.
2. 기존 데이터와 중복·개인정보·잘못된 라벨을 검수한다.
3. 새 manifest와 split을 버전으로 저장한다.
4. 후보 모델을 `models/experiments/`에 저장한다.
5. 고정 평가셋과 현장 holdout을 모두 평가한다.
6. Precision/Recall/F1, FP/FN, latency, 낙상 false-positive-per-hour를 기록한다.
7. shadow로 운영 결과를 비교한다.
8. 기준 통과 후에만 ONNX/TensorRT 변환·승격한다.

## 5. 중단 기준

미탐이 증가하거나 현장 오탐이 안전 운영을 방해하면 즉시 confirm 승격을 중단한다. 데이터 leakage, label map 불일치, TensorRT 변환 손실이 발견되면 정확도 수치보다 원인을 먼저 해결한다.

