# 낙상 오픈데이터 런타임 비교 (2026-09-14)

## 결론

오픈데이터의 장면 그룹 분리 test split에서 전방·측면·후방 낙상과 비낙상을
각 10개씩 선택해 총 40개 영상을 비교했다. 이 표본에서는 후보 모델이 낙상
30건을 모두 감지하고 비낙상 10건에서 오탐을 내지 않아 기준 모델보다 우수했다.

다만 이번 표본은 병원 환경의 C1 카메라 영상만 포함한다. 실제 현장 정제 영상
평가에서는 기준 모델이 후보 모델보다 우수했으므로, 이번 결과만으로 운영 모델을
교체하지 않는다. 현재 기준 모델을 유지하고 후보 모델은 C2~C8 및 현장 영상의
추가 shadow 비교 대상으로 둔다.

## 데이터 독립성 확인

평가 원본은 다음 scene-group 분할의 `test.jsonl`이다.

- train: 1,586개 장면 그룹
- validation: 340개 장면 그룹
- test: 340개 장면 그룹
- train-validation 중복: 0
- train-test 중복: 0
- validation-test 중복: 0

따라서 현재 재구성 split 안에서는 장면 그룹 누수가 없다. 다만 과거의 모든 모델
학습 이력까지 역추적해 동일 원천 장면이 한 번도 사용되지 않았음을 증명한 것은
아니다. 이 결과는 `split 기준 독립 test`로 표현하며 완전한 외부 독립 데이터
성능으로 과장하지 않는다.

## 평가 표본

| 구분 | 개수 |
| --- | ---: |
| 전방 낙상 | 10 |
| 측면 낙상 | 10 |
| 후방 낙상 | 10 |
| 비낙상 | 10 |
| 합계 | 40 |

- 카메라: C1 40개
- 장소: 병원 40개
- 성별: 남성 23개, 여성 17개
- 선택 방식: 방향별 scene group을 정렬한 뒤 선두 10개에서 카메라 1개 선택
- 영상 누락: 0개

## 실행 조건

- Pose 모델: `models/fall/yolov8n-pose.pt`
- 기준 모델: `models/fall/fall_temporal_baseline_same_split_20260910.pt`
- 후보 모델: `models/fall/fall_temporal_same_split_resume_20260908.pt`
- 장치: Jetson CUDA
- Pose 입력 크기: 640
- Pose confidence: 0.35
- Temporal threshold: 모델 checkpoint 값 0.5
- 실행 방식: 영상마다 baseline-first/candidate-first 순서를 교차
- 샘플링: 영상 전체에서 최대 30개 pose 프레임 균등 샘플링

## 전체 결과

| 지표 | 기준 모델 | 후보 모델 | 후보 - 기준 |
| --- | ---: | ---: | ---: |
| TP | 26 | 30 | +4 |
| FN | 4 | 0 | -4 |
| FP | 0 | 0 | 0 |
| TN | 10 | 10 | 0 |
| NO_RESULT | 0 | 0 | 0 |
| Precision | 100.00% | 100.00% | 0.00%p |
| Recall | 86.67% | 100.00% | +13.33%p |
| F1 | 92.86% | 100.00% | +7.14%p |
| 평균 실행시간/영상 | 26.984초 | 27.497초 | +0.513초 |

평균 실행시간은 라이브 AI 컨테이너와 GPU를 공유하면서 별도 프로세스로 순차
실행한 측정값이다. 실제 DeepStream 실시간 FPS 또는 이벤트 지연시간으로
해석하지 않는다.

## 방향별 Recall

| 방향 | 기준 모델 | 후보 모델 |
| --- | ---: | ---: |
| 전방 | 100.00% (10/10) | 100.00% (10/10) |
| 측면 | 80.00% (8/10) | 100.00% (10/10) |
| 후방 | 80.00% (8/10) | 100.00% (10/10) |
| 비낙상 specificity | 100.00% (10/10) | 100.00% (10/10) |

## 기준 모델 미탐

| scene_id | 방향·유형 | 기준 확률 | 후보 확률 | pose 프레임 |
| --- | --- | ---: | ---: | ---: |
| `00004_H_A_SY_C1` | 측면·미끄러짐 | 0.125564 | 0.673840 | 30 |
| `00008_H_A_SY_C1` | 측면·미끄러짐 | 0.106993 | 0.757744 | 30 |
| `00137_H_A_BY_C1` | 후방·중심을 잃음 | 0.417963 | 0.984240 | 30 |
| `00143_H_A_BY_C1` | 후방·걸려 넘어짐 | 0.216641 | 0.873593 | 30 |

네 사례 모두 pose 프레임이 30개 생성됐으므로 단순 pose 미생성보다 temporal
판별 차이일 가능성이 높다. 후보 모델은 네 사례를 모두 threshold 이상으로
판정했다.

## 현장 영상 결과와의 차이

같은 날 정리한 실제 현장 정제 영상 21개에서는 기준 모델이 Precision 100%,
Recall 71.43%, F1 83.33%였고 후보 모델은 Precision 80%, Recall 57.14%,
F1 66.67%였다. 오픈데이터에서는 후보가 우수하지만 현장 영상에서는 기준이
우수해 데이터 도메인에 따라 우열이 뒤집혔다.

따라서 현재 판정은 다음과 같다.

- 운영 기준 모델: 유지
- 후보 모델: shadow 비교 유지
- 오픈데이터 추가 평가: 진행
- 오픈데이터 결과만 근거로 한 재학습·자동 승격: 보류

## 다음 작업

1. 동일 test split에서 C2~C8 카메라를 포함하도록 표본을 확대한다.
2. 병원 외 환경과 보행 보조기·가림 장면을 별도 집계한다.
3. 영상 전체 균등 샘플링 의미를 유지하면서 pose 특징을 한 번만 추출해 두
   temporal 모델에 재사용하는 배치 평가 경로를 만든다.
4. 확대 오픈데이터와 실제 현장 holdout을 모두 통과한 모델만 운영 승격 후보로
   검토한다.

## C1~C8 확대 평가 진행 상태

2026-09-14에 방향별 10개 장면 그룹과 각 장면의 C1~C8 영상을 포함하는
확대 manifest를 생성하고 평가를 시작했다.

- 상태: 실행 중
- manifest: `data/fall_eval/open_data_test_20260914/balanced320_c1_c8.jsonl`
- 전체 영상: 320개
- 고유 장면 그룹: 40개
- 방향별: 전방 80, 측면 80, 후방 80, 비낙상 80
- 카메라별: C1~C8 각각 40개
- 영상 누락: 0개
- 중간 JSON: `data/fall_eval/open_data_test_20260914/balanced320_c1_c8_comparison.json`
- 중간 CSV: `data/fall_eval/open_data_test_20260914/balanced320_c1_c8_comparison.csv`
- 공유 pose 캐시: `data/fall_eval/open_data_test_20260914/shared_pose_cache/`

비교 도구는 첫 번째 temporal 모델이 전체 영상에서 추출한 pose 특징을 JSON으로
저장하고 두 번째 모델이 같은 특징을 재사용하도록 변경했다. 캐시 옵션을 사용하지
않는 기존 실행 동작은 유지된다. 실제 영상 2개에서 기존 방식과 새 방식의 판정,
낙상 확률, pose 프레임 수가 모두 일치함을 확인한 뒤 확대 평가에 적용했다.

진행률 확인:

```bash
rtk jq '{processed:(.videos|length),summary}' \
  data/fall_eval/open_data_test_20260914/balanced320_c1_c8_comparison.json
```

실행 프로세스 확인:

```bash
rtk docker exec cctv-ai-engine \
  pgrep -af compare_fall_temporal_runtime.py
```

## 산출물과 검증

- manifest: `data/fall_eval/open_data_test_20260914/balanced40.jsonl`
- JSON 결과: `data/fall_eval/open_data_test_20260914/balanced40_comparison.json`
- CSV 결과: `data/fall_eval/open_data_test_20260914/balanced40_comparison.csv`
- manifest 40행, CSV 헤더 포함 41행 확인
- 비교 도구 관련 테스트: 14개 통과
- 런타임 오류 및 `NO_RESULT`: 0건

SHA-256:

- manifest: `026a8845b38496b458427899077181eae81e1d36dbb63bedde496c0d56dd0b25`
- JSON: `8ae6bee23c40f0d612d6dd5e78093fbcee8cce5a45a0eaabe5b53be1ef05ed8b`
- CSV: `349b8811825d67496221539731177684bfe8492af00683e0ea269b0bf3e7a5ef`
