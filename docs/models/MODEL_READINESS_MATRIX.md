# 모델 운영 승격 매트릭스

## 1. 목적

이 문서는 `models/model_manifest.json`에 등록된 최신 평가값을 운영 기준과
비교해 모델을 승격할 수 있는지 판단한다. 수치는 저장소에 기록된 평가 결과를
기준으로 하며, 실제 현장 성능을 보증하는 값으로 해석하지 않는다.

판정은 정확도만 보지 않고 precision·recall·지연시간을 모두 확인한다.
하나라도 기준을 만족하지 못하거나 평가값이 없으면 운영 승격 불가로 둔다.

## 2. 현재 판정

| 모델 | 용도 | 평가 데이터 | Precision | Recall/Accuracy | 지연시간 | 판정 |
|---|---|---:|---:|---:|---:|---|
| `helmet_model` | 헬멧 탐지 | 100장 | 0.9014 | 0.9110 recall | 21.167ms | 승격 후보 |
| `helmet_model_ver0.5` | 헬멧 탐지 구버전 | 100장 | 0.9305 | 0.8577 recall | 28.674ms | 보류: recall 미달 |
| `yolov8n-pose` | 사람 자세·낙상 기반 | 61장 | 0.8264 | 0.3937 recall | 44.513ms | 보류: recall 미달 |
| `yolov8m-pose` | 정확도 비교 기준 | 61장 | 0.8837 | 0.4488 recall | 43.172ms | 보류: recall 미달 |
| `yolov8n` | 사람 탐지 fallback | 61장 | 0.8173 | 0.6693 recall | 47.259ms | 보류: recall 미달 |
| `appearance_color_yolov8n` | 의류 색상 분류 | AI4C validation | - | 0.8849 top-1 | 6.45ms | 기준 충족 후보 |
| `pphuman_attribute` | 사람 속성 분류 | 미평가 | - | 미평가 | 미평가 | 평가 필요 |

## 3. 기준 비교

### 헬멧 모델

`helmet_model`은 manifest 기준인 precision 0.85 이상, recall 0.90 이상,
평균 지연시간 50ms 이하를 모두 만족한다. 현재 기록만 놓고 보면 구버전
`helmet_model_ver0.5`보다 운영 후보로 적합하다.

단, 두 평가 모두 100장 기준이므로 카메라 각도·조도·안전모 종류가 현장과
다르면 현장 재평가가 필요하다.

### 자세·낙상 모델

두 pose 모델 모두 지연시간은 기준 이내지만 recall이 기준에 크게 미달한다.
낙상 감지는 놓치는 비용이 큰 업무이므로 precision만으로 승격하지 않는다.
현재 값으로는 운영 모델로 승격하지 않고, 고정 평가 세트 확대·threshold
재검토·현장 오탐/미탐 라벨링을 먼저 진행한다.

### 외관 모델

`appearance_color_yolov8n`은 manifest에 기록된 top-1 accuracy와 지연시간
기준을 충족한다. `pphuman_attribute`는 모델 파일과 연동 설정은 있으나
latest evaluation이 비어 있으므로 운영 승격 전 평가 리포트가 필요하다.

## 4. 승격 조건

1. 동일한 고정 평가 세트로 재현 가능한 평가 리포트를 생성한다.
2. 모델 manifest의 기준값과 실제 결과를 자동 비교한다.
3. 현장 라벨 세트에서 precision·recall·오탐률·미탐률을 추가 산출한다.
4. Jetson에서 평균·p95 지연시간, FPS, GPU 메모리, 장시간 프레임 드롭을 측정한다.
5. 모델 파일 checksum, 설정값, 평가 데이터 버전을 함께 기록한다.
6. 기준 미달 모델은 운영 경로에 연결하지 않고 shadow 또는 fallback으로 유지한다.

## 5. 다음 모델 QA 작업

- pose 낙상 모델의 61장 평가 세트를 낙상·비낙상·경계 사례로 확대
- 실제 영상에서 프레임 단위가 아닌 이벤트 단위 precision·recall 산출
- confidence threshold별 recall·오탐률 곡선 작성
- `pphuman_attribute` 평가 데이터와 속성별 accuracy 작성
- 승격 후보 모델의 Jetson TensorRT 장시간 부하 측정

## 6. 재현 가능한 데이터 분할

다음 재학습부터는 동일 `scene_group`이 서로 다른 split에 들어가지 않도록
그룹 단위로 분할한다.

```bash
python scripts/datasets/build_fall_test_manifest.py \
  --manifest data/fall_dataset/manifests/yolo_pose_train_balanced4000_plus_field_20260812.jsonl \
  --split-output-dir data/fall_dataset/manifests/reproducible_split_20260907
```

현재 생성 결과는 train 2,801행, validation 638행, test 613행이며 각 split의
영상 누락은 0건이다. 원본 manifest가 과거 학습에 사용된 자료이므로 이 split의
test 결과를 기존 모델의 독립 성능 증명으로 사용하지 않는다. 새 모델 학습부터
train만 사용하고 validation/test를 분리 보관한 뒤 평가한다.

## 7. 현재 Jetson 운영 점검 결과

현재 Jetson의 `FALLDATA_AUX_MODE`는 현장 검토 후 `confirm`으로 설정되어
있다. 따라서 운영 점검 시 기본값인 `shadow`를 그대로 기대하지 말고, 실제
승인된 정책을 명시한다.

```bash
python scripts/ops/check_fall_shadow_operation.py \
  --expected-mode confirm \
  --max-review-lines 500 \
  --log-since 10m
```

2026-09-07 재검증 결과:

- AI Engine container healthy, 재시작 0회, OOM 없음
- DeepStream 최근 통계 60건, frame dropped 0, failed 0
- 최근 confirm 기록 20건, 상태 `ok`, pending 0건
- Alert API health 정상
- 전체 점검 결과 PASS

`confirm` 전환은 모델 성능 승격 완료와 동일한 의미가 아니다. 실제 모델
승격은 고정 holdout과 현장 라벨 평가를 별도로 통과해야 한다.

## 8. 2026-09-07 낙상 후보 모델 게이트 결과

현재 비교 운영에 사용 중인 ExtraTrees 후보 모델의 원본 평가 리포트를
승격 게이트로 재검증했다.

| 항목 | 결과 | 판정 기준 | 판정 |
|---|---:|---:|---|
| 평가 행 수 | 14,740 | 재현 가능한 평가 리포트 | 통과 |
| holdout 낙상 precision | 0.9963 | 참고 지표 | 참고 |
| holdout 낙상 recall | 0.8949 | 0.90 이상 권장 | 미달 |
| holdout false negative | 285건 | 0건 | 실패 |
| holdout false positive | 9건 | 0건 | 실패 |
| 교차검증 | 비활성화 | 활성화 필수 | 실패 |

대상 리포트는
`yolo_pose_fall_extratrees_reviewed_hard498_depth16_t078_candidate_20260813_metrics.json`이며,
후보 모델은 `yolo_pose_fall_extratrees_reviewed_hard498_depth16_t078`이다.
구조·데이터 무결성 검사는 통과했지만, 낙상 285건을 놓치고 교차검증도
활성화되지 않아 운영 승격 게이트는 **실패**로 판정한다.

따라서 현재 Jetson의 `confirm` 정책과 `FALLDATA_AUX_COMPARE_VETO_ENABLED=false`
설정은 유지한다. 이 후보를 자동으로 `veto` 또는 물리 장치 제어 경로에
승격하지 않는다. 다음 학습에서는 재현 가능한 train/validation/test 분할을
사용하고, 교차검증·threshold별 recall/오탐률·이벤트 단위 성능을 함께 기록한
뒤 동일 게이트를 다시 실행한다.

## 9. 미탐·오탐 패턴과 재학습 우선순위

동일 리포트의 오류 목록을 추가 분석한 결과, 미탐은 특정 단일 장면에만
국한되지 않고 여러 카메라에 분산되어 있다. 다만 카메라 기준으로 `C2` 67건,
`C4` 62건, `C3` 48건, `C1` 43건이 발생해 네 카메라에 전체 미탐의 77%가
집중되어 있다. 장면 패턴으로는 `H_A_FY` 48건, `H_A_SY` 47건,
`O_E_FY` 32건, `H_A_BY` 29건 순으로 많다.

미탐 낙상 확률은 평균 0.6032, 범위 0.1637~0.7793이다. 현재 threshold
0.78 부근의 보수적 설정이 precision을 높이는 대신 낙상 recall을 희생하는
형태다. validation threshold sweep에서도 threshold 0.35는 recall 0.9891
대신 false positive 131건, threshold 0.75는 recall 0.8879와 false positive
16건을 보였다. 따라서 threshold만 조정해 false negative와 false positive를
동시에 0건으로 만드는 것은 현재 데이터와 특징으로는 현실적이지 않다.

재학습 우선순위는 다음과 같다.

1. `C2`, `C4`, `C3`, `C1`의 낙상·비낙상 원본 영상과 라벨을 우선 재검수한다.
2. `H_A_FY`, `H_A_SY`, `O_E_FY`, `H_A_BY` 장면군의 가림·자세 변화·짧은
   동작 구간을 별도 hard-case 세트로 관리한다.
3. threshold 후보를 validation에서 선택하되, frame 단위가 아니라 동일
   낙상 사건 단위로 중복 집계를 제거한다.
4. 새 학습은 `reproducible_split_20260907`의 train만 사용하고, validation과
   test는 학습 중 참조하지 않도록 보관한다.
5. 교차검증을 활성화하고, Jetson에서 p95 지연시간과 장시간 프레임 드롭을
   함께 측정한 뒤에만 운영 승격을 재검토한다.

## 10. 오류 영상 검수 산출물

평가 리포트의 `/app` 내부 경로를 현재 호스트의 실제 데이터 경로로
재매핑해 전체 manifest를 재생성했다.

- 전체 manifest: `data/fall_eval/reconstructed_full_manifest_20260908.jsonl`
- 전체 행: 18,128건
- 낙상: 13,584건
- 비낙상: 4,544건
- 영상 누락: 0건
- 미탐 검수 페이지: `data/fall_eval/error_review_20260908/index.html`
- 오탐 검수 페이지: `data/fall_eval/error_review_fp_20260908/index.html`
- 우선 검수 페이지(C1~C4): `data/fall_eval/error_review_priority_c1_c4_20260908/index.html`

검수 페이지에는 holdout·validation 오류가 합쳐져 있으며, 미탐 603건과
오탐 19건을 영상으로 확인할 수 있다. 검수자는 실제 정답을 `낙상`,
`비낙상`, `판단보류`, `학습제외` 중 하나로 선택하고 결과 JSON을 저장한다.
검수 결과를 반영하기 전까지 기존 평가 리포트의 자동 라벨을 수정하지 않는다.
우선 검수 페이지에는 C1~C4 미탐 465건이 포함되어 있으며, 이 페이지부터
검수하면 전체 미탐의 주요 집중 구간을 먼저 처리할 수 있다.

## 11. 대표 미탐 영상 시각 확인

미탐 페이지에서 C2·C4 대표 영상 3건의 낙상 전후 구간을 직접 확인했다.

| 장면 | 카메라 | 라벨 정보 | 확인 결과 | 모델 낙상 확률 |
|---|---:|---|---|---:|
| `00045_H_A_SY_C2` | C2 | 측면·중심을 잃고 넘어짐 | 서 있다가 매트 위로 내려앉는 흐름이 확인됨 | 0.6723 |
| `00083_H_A_SY_C4` | C4 | 측면·걸려 넘어짐·cane | 보행 보조기와 함께 균형을 잃고 넘어지는 흐름이 확인됨 | 0.5254 |
| `00139_H_A_SY_C2` | C2 | 측면·중심을 잃고 넘어짐 | 침대 가장자리에서 몸이 미끄러져 바닥으로 내려가는 흐름이 확인됨 | 0.7310 |

대표 3건은 정지 화면보다 전후 움직임에서 낙상 여부가 명확해지는 사례였다.
따라서 현재 미탐 문제는 단순 라벨 오류만으로 설명하기 어렵고, 자세 특징의
시간 변화·보행 보조기·침대 가장자리 가림을 함께 학습하거나 temporal 후처리로
보완해야 한다. 다만 3건의 확인만으로 전체 603건을 재라벨링하지 않으며,
나머지는 검수 페이지에서 개별 확인한다.

## 12. Temporal 후보의 실제 런타임 점검

2026-09-08 Jetson 컨테이너의 환경변수와 review log를 확인했다.

- temporal 모델: `fall_temporal_hybrid_full_seq60_candidate.pt`
- window: 60 frame
- stride: 20 frame
- 최소 확정 window: 2개
- near-miss temporal shadow: 활성화
- 누적 temporal 결과: 13,079건
- 처리 상태 `ok`: 13,079건
- temporal 확정: 1,036건
- runtime 오류: 확인되지 않음

현재 temporal 결과는 `sample_eval` 기반 near-miss 및 shadow 기록이며, 정답
라벨과 연결되어 있지 않다. 따라서 위 수치는 모델 정확도나 recall이 아니라
런타임 처리 안정성만 보여준다. 운영 승격 판단에는 검수 JSON과 사건 단위
정답 매핑이 추가로 필요하다.

또한 현재 `FALLDATA_AUX_MODE=confirm`이지만 temporal near-miss 결과의
자동 승격은 활성화하지 않은 상태로 확인했다. 즉 temporal 후보는 보조
비교·검수 경로에만 사용되고, 이 로그만으로 물리 장치 제어 모델로 승격하지
않는다.

누적 review log 전체 요약에서는 총 158,292건 중 `ok` 90,470건,
`insufficient_pose_records` 30,115건, `not_run` 33,504건, `error` 678건이
확인됐다. 최근 1시간 운영 점검은 오류 0건이었지만, 누적 기록의 오류·pose
부족 구간은 별도 원인 분석이 필요하다. 사람이 라벨을 입력한 기록은 52건
뿐이고 그중 `fall`은 1건이므로, 현재 human-label confusion 결과를 temporal
정확도 근거로 사용하지 않는다.

누적 `error` 678건을 표본 확인한 결과, 주요 원인은 MediaPipe 특징 추출
프로세스의 30초 timeout이었다. 이 오류는 2026-07-27의 과거 `sample_eval`
기록에 집중되어 있으며, 2026-09-07 최근 운영 점검에서는 오류 0건으로
확인됐다. 다만 과거 오류가 자동 재시도·유실 없이 복구되었는지는 별도
운영 검증이 필요하므로, temporal 승격 근거에는 누적 로그 전체를 사용하지
않는다.

## 13. 짧은 입력 window 판정 보정

대표 C2 영상 `00045_H_A_SY_C2`를 Jetson CUDA 환경에서 재현한 결과,
낙상 확률은 0.999966이었지만 평가 가능한 sliding window가 1개뿐인 상태에서
`min_confirmed_windows=2`를 적용해 기존 prediction이 0으로 나오는 문제가
확인됐다. 동일 입력을 최소 window 1개로 실행하면 prediction 1이 됐다.

이를 반영해 `scripts/inference/smoke_fall_temporal_model.py`가 실제 계산된
window 수보다 많은 확정 window를 요구하지 않도록 보정했다. 두 개 이상의
window가 계산되는 일반 입력에서는 기존의 연속 2개 조건을 유지하고, 한 개의
유효한 전체 window만 계산되는 짧은 입력에서는 한 개 조건으로 평가한다.

검증 결과:

- 회귀 테스트: 통과
- 실제 Jetson CUDA C2 영상: `prediction [1]`, `fall_probability 0.999966`
- 실제 Jetson CUDA C4 영상: `prediction [1]`, `fall_probability 0.999830`
- timeout: 발생하지 않음

추가로 수정 후 15분 런타임 점검에서 AI Engine healthy, 재시작 0회,
DeepStream frame dropped 0·failed 0, 최근 confirm 기록 20건 `ok`를 확인했다.

이 수정은 temporal 모델 자체의 성능을 높인 것이 아니라, 충분한 하나의
window가 있음에도 확정 개수 조건 때문에 발생하던 후처리 미탐을 줄이는
보정이다. 운영 승격 여부는 여전히 독립 라벨 평가를 통과해야 한다.

## 14. Temporal 후보와 RF 후보의 평가 비교

두 후보의 validation manifest 행 수는 3,424건으로 동일하지만, holdout
구성은 temporal 746건과 RF 3,768건으로 다르다. 따라서 아래 비교는
참고용이며 운영 승격 근거로 직접 사용하지 않는다.

| 후보 | 구간 | threshold | confusion matrix | 낙상 precision | 낙상 recall |
|---|---|---:|---|---:|---:|
| temporal hybrid | validation | 0.70 | `[[2324, 148], [21, 931]]` | 0.9910 | 0.9401 |
| temporal hybrid | holdout | 0.70 | `[[396, 19], [5, 326]]` | 0.9875 | 0.9542 |
| ExtraTrees RF | validation | 0.78 | `[[2154, 318], [10, 942]]` | 0.9954 | 0.8714 |
| ExtraTrees RF | holdout | 0.78 | `[[2427, 285], [9, 1047]]` | 0.9963 | 0.8949 |

동일 validation 행 수 기준으로는 temporal 후보가 RF 후보보다 recall이
높아 미탐 감소 방향의 유력 후보로 보인다. 하지만 학습 데이터·holdout
분할·평가 방식이 완전히 동일하지 않고 교차검증도 확인되지 않았으므로,
현재 설정을 temporal 운영 승격으로 변경하지 않는다. 다음 비교에서는 동일
scene-group split, 동일 사건 단위 라벨, 동일 threshold sweep, 교차검증을
적용한 뒤 승격 게이트를 실행한다.

현재 정식 temporal 재학습용 feature cache는 아직 생성하지 않았고, 기존 temporal
metrics가 참조한 원본 manifest도 보존되어 있지 않다. 다만 원본 데이터를
재구성한 manifest와 pose 모델을 이용한 2건 smoke cache 생성·temporal 입력
검증은 완료했다. 이 smoke 검증에서 pose cache 생성기와 temporal 학습기의
`fall_window_margin_frames` 기본값이 달라 캐시를 찾지 못하는 문제가 확인되어,
학습기 기본값을 120으로 통일했다. smoke 학습은 CPU에서 2 epoch 실행되었으며
입력 계약과 모델 생성만 확인한 것이므로 성능 근거로 사용하지 않는다.

동일 조건 비교를 진행하려면 정식 scene-group split별 pose feature cache를
생성한 뒤 temporal과 RF를 같은 train/validation/test 입력으로 재평가해야 한다.

2026-09-08에 재구성 manifest를 scene-group 기준으로 train 12,688건,
validation 2,720건, test 2,720건으로 분할했다. 호스트 manifest의 절대경로를
Jetson 컨테이너 경로로 변환하는 `--source-video-root`와
`--target-video-root` 옵션을 추가했고, train 20건·validation 10건의 GPU
smoke 추출 및 temporal CUDA 학습을 완료했다. 현재 캐시 생성 속도 기준으로
전체 추출은 수십 시간이 걸릴 수 있으므로, 정식 실행은 기존 캐시를 재사용하는
방식으로 분할 실행한다.

정식 chunk 실행 시에는 다음 원칙을 따른다.

```bash
# 예: train과 validation을 각각 20행씩 처리
python scripts/datasets/train_yolo_pose_fall_rf.py \
  --manifest data/fall_eval/reconstructed_split_20260908_container/train.jsonl \
  --validation-manifest data/fall_eval/reconstructed_split_20260908_container/validation.jsonl \
  --feature-cache data/fall_eval/yolo_pose_same_split_resume_train_cache_20260908 \
  --validation-feature-cache data/fall_eval/yolo_pose_same_split_resume_validation_cache_20260908 \
  --max-videos 0 --validation-max-videos 0 \
  --train-row-offset 0 --validation-row-offset 0 --row-limit 20 \
  --extract-only

# 다음 chunk는 offset을 20씩 증가
python scripts/datasets/train_yolo_pose_fall_rf.py \
  --manifest data/fall_eval/reconstructed_split_20260908_container/train.jsonl \
  --validation-manifest data/fall_eval/reconstructed_split_20260908_container/validation.jsonl \
  --feature-cache data/fall_eval/yolo_pose_same_split_resume_train_cache_20260908 \
  --validation-feature-cache data/fall_eval/yolo_pose_same_split_resume_validation_cache_20260908 \
  --pose-model models/fall/yolov8n-pose.pt \
  --max-videos 0 --validation-max-videos 0 \
  --train-row-offset 20 --validation-row-offset 20 --row-limit 20 \
  --max-frames 120 --frame-stride 3 --fall-window-margin-frames 120 \
  --imgsz 640 --prediction-batch-size 8 --extract-only
```

`max-videos`는 기존 클래스 균형 선택 로직이므로 정식 chunk에서는 0으로 두고,
`row-limit`과 offset으로만 범위를 제어해야 한다. 이 방식으로 40건 train 및
40건 validation을 중복·누락 없이 검증했다.

부분 cache만으로 temporal 학습을 실행하면 한 클래스만 남을 수 있다. 이를
방지하기 위해 학습기는 train·validation 중 하나라도 fall/non_fall 두 클래스가
모두 없으면 즉시 중단한다. 따라서 부분 cache로 생성된 단일 클래스 지표는
성능 자료로 사용하지 않으며, 양쪽 클래스가 확보된 cache에서만 평가한다.

현재 정식 resume cache는 train·validation 각각 manifest 선두 740건까지
생성되었고, 두 영역 모두 누락·추가 파일이 0건이다. 이는 전체 학습 성능이
아니라 추출 진행률과 데이터 무결성 증거로만 관리한다.

관련 기준 문서:

- [모델 인수인계서](MODEL_HANDOVER.md)
- [모델 운영 현황](../operations/model_inventory_20260826.md)
- [Edge AIoT 플랫폼 성숙도 평가](../architecture/EDGE_PLATFORM_MATURITY_ASSESSMENT.md)
