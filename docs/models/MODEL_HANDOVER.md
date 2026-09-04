# 적용 모델별 상세 인수인계서

최종 정리 기준: 2026-09-03

## 1. 문서 목적

이 문서는 모델을 처음 맡는 사람이 모델의 역할, 적용 위치, 검증 수치, 한계, 교체 절차를 이해하도록 작성했다. 수치는 반드시 평가 데이터셋·입력 크기·threshold·하드웨어와 함께 해석한다.

## 2. 기본 용어

| 용어 | 의미 |
|---|---|
| Precision | 모델이 정답이라고 한 것 중 실제 정답인 비율. 오탐이 적은지 판단 |
| Recall | 실제 정답 중 모델이 찾아낸 비율. 미탐이 적은지 판단 |
| F1 | Precision과 Recall의 균형값 |
| Accuracy | 전체 샘플 중 맞힌 비율. 클래스 불균형이 있으면 단독 사용 금지 |
| Confidence | 한 번의 예측 점수. 정확도와 다른 값 |
| Latency | 한 이미지·crop·프레임 처리 시간 |
| Shadow | 결과만 기록하고 운영 판정은 바꾸지 않는 비교 모드 |
| Candidate | 학습·검증 중인 후보 모델 |

`confidence=0.8`은 “이번 예측의 점수”이지 “정확도 80%”가 아니다. 현재 수치는 서로 다른 데이터셋과 조건에서 산출됐으므로 단순 순위 비교에 사용하지 않는다.

## 3. 전체 모델 파이프라인

```text
CCTV frame
  → YOLOv8n-Pose / YOLOv8m-Pose
  → bbox·keypoint 후처리
  → 낙상 규칙 및 temporal 보조 모델
  → person crop 기반 헬멧·외형·색상·얼굴 인식
  → 이벤트·검색·장치 제어
```

모델이 최종 이벤트를 단독 결정하지 않는다. 후처리 threshold, 시간 누적, cooldown, MQTT·Action Layer 상태까지 합쳐야 운영 결과가 된다.

## 4. History: 지금까지의 진행 과정

### 초기 단계

초기 낙상 보조는 MediaPipe feature 기반 RandomForest였다. 현재 `models/legacy/falldata_mediapipe/`의 `FNF_RF_SMOTE_CAM_1~8.pkl`은 역사적 재현·비교용이며 production이 직접 로드하지 않는다.

### 영상 분석 단계

PC는 OpenCV, Jetson은 DeepStream/TensorRT로 분리했다. YOLO-Pose를 이용해 사람 bbox와 17개 keypoint를 얻고, Jetson에서는 raw tensor → 좌표 복원 → 후처리 → OSD·이벤트 발행 순서를 연결했다.

### 낙상 보강 단계

낙상은 단일 프레임 판단에서 벗어나 bbox 비율, 머리 위치, 몸통 수평성, keypoint span, 앉기 억제, 시간 누적을 같이 평가하도록 보강했다. RF·TCN은 운영 전 shadow/confirm 비교를 위해 추가됐다.

### 헬멧·외형·얼굴 단계

헬멧은 320 TensorRT 엔진으로 운영 기준을 확보했다. 외형은 HSV/LAB 및 PA100K·PP-Human 경로를 유지하면서 AI-Hub 기반 multitask 후보를 검수 중이다. 얼굴은 InsightFace/ArcFace 우선, OpenCV Haar fallback 구조다.

## 5. 모델별 인수인계

### 5.1 YOLOv8n-Pose

| 항목 | 내용 |
|---|---|
| 파일 | `models/fall/yolov8n-pose.pt/.onnx/.engine` |
| 역할 | 사람 검출과 17개 keypoint 제공, 낙상 규칙 입력 |
| 입력 | 기본 320×320 |
| 출력 | person bbox, confidence, keypoint 좌표·confidence |
| 상태 | Jetson 기본 pose 후보 |
| 설정 | `POSE_MODEL_PATH` 및 DeepStream pose 설정 |

기준선은 Precision `0.8264`, Recall `0.3937`, 평균 `44.5ms`다. 61개 평가 이미지의 사람 검출 단계 수치다. Recall이 낮으므로 모든 사람을 안정적으로 찾는 모델로 설명하면 안 된다.

### 5.2 YOLOv8m-Pose

`models/fall/yolov8m-pose.pt/.onnx`에 있다. n보다 큰 정확도 비교 기준선이며 기록값은 Precision `0.8837`, Recall `0.4488`, 평균 `43.2ms`다. 설정상 640 모델이지만 기록 평가 설정은 imgsz 320이므로 재비교 시 동일 조건으로 다시 측정한다. 아직 운영 교체 모델로 확정하지 않는다.

### 5.3 YOLOv8n 사람 검출 fallback

`models/person/yolov8n.pt/.onnx/.engine`에 있다. 기록값은 Precision `0.8173`, Recall `0.6693`, 평균 `47.3ms`이며 61개 이미지 기준이다. keypoint가 없기 때문에 이 모델만으로 pose 낙상을 대체할 수 없다.

### 5.4 낙상 규칙 기반 detector

학습 모델이 아니라 `src/core/ai/_fall_detector.py`의 점수화 계층이다.

- 몸통이 수평인지
- bbox가 넓고 머리가 낮은지
- 수직 span이 낮은지
- 관절 confidence가 충분한지
- 앉기·쪼그리기·가림을 정상으로 억제할지
- 시간 누적 후 이벤트를 확정할지

규칙 threshold를 바꾸면 같은 YOLO 모델도 결과가 달라진다. 규칙 변경도 모델 교체와 같이 고정 영상 replay와 정상 영상 오탐 검증을 해야 한다.

### 5.5 헬멧 `helmet_model`

| 항목 | 내용 |
|---|---|
| 운영 파일 | `models/head/helmet_model.engine` |
| 원본 | `models/head/helmet_model.pt/.onnx` |
| 입력·클래스 | 320×320, `helmet`, `head` |
| 기록 수치 | Precision `0.9014`, Recall `0.9110`, 평균 `21.2ms` |
| 평가 조건 | 100개 이미지, `conf=0.15`, IoU `0.5` |
| 판단 | 현재 운영에 가장 가까운 승인 모델 |

manifest 기준은 Precision 0.85 이상, Recall 0.90 이상, latency 50ms 이하이며 기록상 만족한다. 다만 현장 조명·거리·가림을 대표하는지는 별도로 확인한다.

### 5.6 레거시 헬멧 `helmet_model_ver0.5`

`models/legacy/helmet_model_ver0.5.*`에 보관한다. Precision `0.9305`, Recall `0.8577`, 평균 `28.7ms`다. Precision은 높지만 운영 최소 Recall을 만족하지 못하므로 현재 운영 모델로 되돌리지 않는다.

### 5.7 Falldata RF 낙상 보조

현재 후보 예시는 `models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl`이다. 기록 실험 중 fall Precision `0.6727`, Recall `0.9487`인 결과가 있다. 미탐은 적지만 오탐 가능성이 높아 바로 confirm 차단 모델로 쓰지 않는다.

운영 정책은 먼저 `FALLDATA_AUX_MODE=shadow`로 비교하고, scene 단위 holdout·현장 라벨·false positive 기준을 확인하는 것이다. `models/legacy/falldata_mediapipe` 모델과 현재 YOLO-pose feature RF를 같은 모델로 취급하지 않는다.

### 5.8 TCN temporal 후보

- 균형형: `models/experiments/fall_temporal_hybrid_full_seq60_candidate.pt`
- 미탐 감소형: `models/experiments/fall_temporal_hybrid_hardcase_candidate.pt`

0.7 threshold 비교 기록:

| 후보 | Precision | Recall | 오탐 | 미탐 | 판단 |
|---|---:|---:|---:|---:|---|
| 균형형 | 99.10% | 94.01% | 21 | 148 | 우선 shadow |
| hard-case | 기록상 중심값은 Recall 96.16% | 96.16% | 40 | 미기록 | 미탐 우선 현장만 비교 |

권장 비교 설정은 60 frame window, stride 20, 연속 확인 window 2개다. 단, 기본 `.env.jetson.example`의 temporal compare path는 비어 있을 수 있으므로 값을 넣었다고 운영 적용된 것으로 간주하지 않는다.

### 5.9 색상 `appearance_color_yolov8n`

`models/appearance/appearance_color_yolov8n.pt/.onnx/.engine`이며 160×160 person crop을 입력으로 받는다. 클래스는 black, blue, brown, gray, green, orange, pink, purple, red, white, yellow다.

AI4C validation 기록은 Top-1 `0.8849`, accepted accuracy `0.9127`, 평균 `6.45ms`다. manifest 기준 Top-1 0.88 이상·latency 10ms 이하를 만족한다. Top-1 accuracy와 reviewed 후보의 macro F1은 다른 지표이므로 숫자를 합산하지 않는다.

### 5.10 PA100K / PP-Human 속성 모델

`pa100k_resnet50_attr.onnx`, `pphuman_attribute.onnx`는 성별·의복 형태·소지품 등 여러 attribute를 person crop에서 분류한다. Jetson은 PGIE person ROI → PP-Human TensorRT SGIE → appearance metadata 순서다.

PP-Human의 최신 평가값은 manifest에 없고 acceptance 기준만 `attribute accuracy 0.8 이상`, 평균 `35ms 이하`로 기록되어 있다. 따라서 현재 정확도를 숫자로 확정하지 않는다. 실행 선택은 `APPEARANCE_BACKEND`, `APPEARANCE_MODEL_PATH`, `APPEARANCE_LABEL_MAP_PATH`, `APPEARANCE_RUNTIME`으로 정한다.

### 5.11 reviewed multitask appearance 후보

후보는 `appearance_mobilenet_v3_crop_weighted/protected_multitask` 계열이다.

| 속성 | Macro F1 | 판단 |
|---|---:|---|
| 성별 | 0.8731 | 후보 수준 |
| 상의 형태 | 0.8704 | 후보 수준 |
| 하의 형태 | 0.6628 | 치마 보강 필요 |
| 상의 색상 | 0.6852 | 목표 0.70 미달 |
| 하의 색상 | 0.5261 | 운영 승격 불가 |
| 소지품 | 0.5147 | Recall 보강 필요 |

하의 brown/navy 및 누락 색상, 상의 blue/navy/black, 모자·가방·long_skirt를 먼저 보강한다. 이 후보를 현재 운영 모델로 무조건 교체하지 않는다.

### 5.12 얼굴 인식

`src/utils/face_recognition.py`는 InsightFace/ArcFace를 우선 사용하고, 불가하면 OpenCV Haar baseline으로 식별한다. 등록 데이터는 `known_faces.json`, `known_faces/`, Jetson runtime cache를 사용한다.

현재는 embedding shape·실행 여부 smoke 검사는 있지만 동일인/타인 기준 FAR·FRR·ROC 수치가 없다. 얼굴 confidence를 정확도로 표현하지 않는다. 운영 전 카메라별 얼굴 크기·각도·조명으로 FAR/FRR 평가가 필요하다.

## 6. 모델 신뢰도 요약

| 영역 | 확인된 수치 | 현재 신뢰도 판단 |
|---|---|---|
| 헬멧 TensorRT | P 0.9014 / R 0.9110 / 21.2ms | 고정 평가 기준 승인, 현장 대표성 추가 확인 |
| YOLOv8n-Pose | P 0.8264 / R 0.3937 / 44.5ms | 낙상 입력, Recall 개선 필요 |
| YOLOv8m-Pose | P 0.8837 / R 0.4488 / 43.2ms | 비교 기준선 |
| 사람 fallback | P 0.8173 / R 0.6693 / 47.3ms | pose 대체 아님 |
| 색상 YOLOv8n | Top-1 0.8849 / 6.45ms | 평가셋 기준 양호 |
| reviewed appearance | F1 0.5147~0.8731 | 하의 색상·소지품 보강 전 보류 |
| PP-Human | 최신 평가 없음 | 정확도 미확정 |
| 얼굴 | FAR/FRR 없음 | 운영 정확도 미확정 |
| 낙상 RF/TCN | 실험값만 존재 | shadow·현장 라벨 후 승격 |

## 7. 재학습·교체 절차

1. 기존과 동일한 validation/test split을 보존하고 scene group leakage를 막는다.
2. 정상·위험·가림·저조도·작은 사람·다중 인원 데이터를 포함한다.
3. 후보 artifact를 별도 경로에 저장한다.
4. 기존과 같은 입력 크기·confidence·IoU로 P/R/F1 또는 mAP, FP/FN, p95 latency를 측정한다.
5. 낙상은 영상 단위 Recall과 false-positive-per-hour를 추가 측정한다.
6. ONNX와 TensorRT를 각각 평가하고 변환 손실을 기록한다.
7. `models/model_manifest.json`에 모델 경로·체크섬·평가 리포트를 등록한다.
8. 운영 모델은 유지한 채 shadow로 후보를 비교한다.
9. 승인 기준과 현장 검수가 통과하면 `.env` 모델 경로를 변경하고 컨테이너를 재시작한다.
10. 배포 후 이벤트, latency, frame drop, 메모리, 장치 동작을 확인한다.
11. 문제 시 이전 artifact와 환경변수로 rollback하고 원인을 기록한다.

## 8. 앞으로의 권장 과제

### 낙상

동일한 fall/non-fall 고정 영상으로 규칙·RF·TCN을 재평가하고, 카메라별 false positive per hour와 miss 사례를 기록한다. DeepStream에서 shadow/confirm 초기화 연결과 이벤트 metadata를 실제 로그로 확인한다.

### Pose

작은 사람·가림·누운 자세·어두운 장면을 추가 수집한다. n/m을 동일 imgsz와 threshold로 비교하고, 사람 검출 Recall뿐 아니라 keypoint 누락률·temporal Recall도 측정한다.

### 외형

하의 brown/navy·누락 색상, 상의 blue/navy/black 혼동, 모자·가방·치마를 보강한다. 승격 기준은 상의 색상 Macro F1 0.70, 하의 0.65, 비색상 0.72, 성별 0.85 이상을 유지하는 것으로 기록되어 있다.

### 얼굴

등록자별 다양한 거리·각도·조명 pair를 만들고 FAR·FRR·ROC와 threshold별 운영 지점을 정한다.

## 9. 모델 문제와 파이프라인 문제 구분

| 증상 | 모델 외 우선 확인 |
|---|---|
| 결과 전혀 없음 | 모델 경로, TensorRT 호환성, GPU 초기화, camera flag |
| bbox는 있으나 이벤트 없음 | 후처리 threshold, temporal 조건, MQTT, cooldown |
| 이벤트는 있으나 장치 미동작 | Action policy, topic, IP·인증, 장치 cooldown |
| 색상만 이상함 | crop 좌표, 조명, label map, backend |
| Jetson이 점점 느려짐 | frame drop, queue, memory, engine context, 로그 저장 |

재학습 전에 원본 frame → crop → 모델 출력 → 후처리 → MQTT topic을 순서대로 저장해 고장 지점을 분리한다.

## 10. 교체 체크리스트

- [ ] 모델 목적·상태·artifact를 manifest에 기록했는가
- [ ] 입력 크기·label map·runtime이 일치하는가
- [ ] 고정 평가셋 P/R/F1 또는 mAP·FP/FN·latency를 기록했는가
- [ ] 실제 CCTV 샘플을 사람이 검수했는가
- [ ] ONNX/TensorRT 변환 후 재평가했는가
- [ ] shadow 기간과 승격 기준이 정해졌는가
- [ ] 이전 모델 rollback 방법이 남아 있는가
- [ ] `.env.example`, Compose, 운영 문서를 함께 갱신했는가

