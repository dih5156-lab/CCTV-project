# 학습 모델 분석 및 개선 우선순위

분석일: 2026-09-17 · 대상: 현재 저장소의 모델·학습/평가 코드·저장된 결과·실행 컨테이너 설정

## 1. 결론

**낙상 Candidate는 공개 데이터 미탐 감소에 유망하지만, 지금 바로 운영 승격할 근거는 부족하다. 모델 크기를 늘리기보다 평가/입력 계약과 현장 데이터를 먼저 정리해야 한다.**

헬멧은 기록된 제한적 평가 기준을 충족한다. 외형 모델은 속성별 차이가 크며, 특히 하의 색상·모자·치마를 완성된 기능으로 보기는 어렵다. 모델 파일 존재, ONNX 변환 성공, 평가 게이트 통과, 현장 운영 합격은 서로 다른 판단이다.

이번 작업은 저장된 결과의 재집계와 코드·설정 감사다. 재학습, 새 영상 추론, TensorRT 빌드, 운영 모델 교체는 하지 않았다. 모든 원천 데이터로 학습을 재현한 독립 인증 보고서도 아니다. `ai-engineer`, `specialized-model-qa` 기준으로 데이터 독립성·평가 분모·입력 일치·운영 적용 여부를 분리했다.

## 2. 모델별 현재 판단

| 대상 | 확인한 근거 | 판단 |
|---|---|---|
| 낙상 Baseline temporal | 동일 split validation recall 96.91%; 공개 영상 320건 recall 91.67% | 비교 기준 유지. 현장 전체 합격을 뜻하지 않음 |
| 낙상 Candidate temporal | validation recall 97.11%; 공개 영상 recall 97.50%, FP 증가 | shadow 비교 유지, 바로 승격하지 않음 |
| 동일 split 낙상 RF | validation P/R 98.73%/94.90%, threshold 0.6 | 비교용 기준. temporal보다 이 평가의 recall이 낮음 |
| 현재 설정의 cam2 RF | 80건 validation P/R 97.30%/90.00%, threshold 0.7 | 동일 split RF와 다른 모델. 제한된 자료의 보조 비교 모델 |
| YOLO pose/person | 61장 사람 bbox 평가에서 n-pose recall 39.37%, m-pose 44.88%, person 66.93% | pose/사람 검출 병목 점검 필요. 이 수치를 낙상 recall로 쓰면 안 됨 |
| `helmet_model` | 100장, helmet P/R 90.14%/91.10%, 평균 21.167ms | 기록된 기준 충족. 미착용/head와 현장 ROI 평가는 별도 필요 |
| `appearance_color_yolov8n` | manifest top-1 88.49%, accepted accuracy 91.27%, 평균 6.45ms | 기록상 후보 수준. 원본 평가·선택 비율·현장 색상별 검증 필요 |
| PA100K / PP-Human | 파일·설정 존재; PP-Human latest evaluation 없음 | 실제 활성 backend별 라벨 평가가 부족 |
| 외형 통합 protected 모델 | 상의 색상 F1 0.6852, 하의 0.5261, 소지품 0.5147 | 전체 기능 일괄 승격 불가. 부족 클래스 우선 보강 |
| RAPv2 reviewed ONNX 후보 | 변환 parity 통과, metadata의 학습 단계는 `color_baseline` | 변환 성공이지 모든 속성 품질 합격은 아님 |

위 지표는 서로 다른 데이터·과제이므로 모델 간 단일 순위로 합치지 않는다. confidence 점수도 보정 검증 없이 실제 정답 확률로 해석하지 않는다.

## 3. 운영 설정과 문서의 차이

실행 중인 `cctv-ai-engine`에서 모델 관련 환경변수와 파일 존재만 읽었다.

- `FALLDATA_AUX_MODE=shadow`
- `FALLDATA_AUX_NEAR_MISS_PROMOTE_ENABLED=false`
- `FALLDATA_AUX_COMPARE_VETO_ENABLED=false`
- temporal 비교 경로: `models/fall/fall_temporal_same_split_resume_20260908.pt`
- RF 비교 경로: `models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl`
- `DS_FALL_SCORE_THRESHOLD=3.0`, `DS_CROP_POSE_ENABLED=0`
- `APPEARANCE_BACKEND=hsv`, PA100K SGIE 설정 경로도 별도로 존재

`src/core/deepstream_processor.py:_init_event_filters`는 자세 기반 `FallDetector`를 생성한다. 따라서 문서의 “Baseline 유지”를 **Baseline temporal이 모든 실시간 알람을 최종 판정 중**이라는 의미로 확대해서는 안 된다. 현재 설정은 규칙/자세 기반 주 경로와 보조 shadow 비교를 구분해야 한다. 설정 확인만으로 최근 shadow 결과가 정상 누적 중인지까지 입증한 것은 아니다.

색상 경로 경고에 대해서는 현재 `/app/models/appearance/appearance_color_yolov8n.engine`이 존재함을 확인했다. 예전의 파일 누락 상태가 지금도 그대로라고 볼 수 없다. 다만 모델 로딩·최근 추론 성공까지 이번에 확인하지는 않았다.

`APPEARANCE_MODEL_PATH=models/pa100k_resnet50_attr.engine`는 컨테이너에 없는 옛 경로다. 실제 파일 및 `config_infer_pa100k.txt`는 `models/appearance/pa100k_resnet50_attr.engine`를 사용한다. HSV/SGIE 경로와 구분해 정리해야 하며, 이 환경변수만으로 현재 PA100K 전체가 고장났다고 단정할 수 없다.

## 4. 낙상 학습 구조

### 모델 자체는 경량이며 Baseline/Candidate 구조가 같다

CPU에서 두 체크포인트를 `weights_only=True`로 읽어 확인했다.

- 구조: `FallTemporalHybrid`
- 입력: 최근 30개 pose 기록 × 15개 특징
- 추가 입력: 53개 영상 요약 특징
- temporal channels: 32
- 파라미터 수: 각각 **6,401개**
- 판정 threshold: 각각 0.5
- 학습 설정: AdamW, learning rate 0.001, weight decay 0.0001, batch 16, seed 42

`src/core/ai/fall_temporal_model.py`의 temporal 분기는 작은 Conv1D 두 층과 평균/최대 pooling을 사용하고, 요약 특징 분기와 합쳐 분류한다. RGB 영상 자체를 대형 신경망으로 학습하는 구조가 아니다. `fall_score`, bbox 비율, keypoint 품질, 자세 규칙에 해당하는 특징에 의존하므로 앞단 pose와 규칙 변경이 모델 입력에도 영향을 준다.

**해석:** 이 작은 분류기를 더 크게 만드는 것이 우선 해법이라는 근거는 없다. pose 추출, 영상 구간 선택, 사람 선택, 별도 프로세스 실행 비용을 먼저 분리 측정해야 한다. 저장된 영상별 실행시간은 pose·프로세스 시작·GPU 공유가 포함되어 실시간 FPS나 6,401개 파라미터 모델 단독 지연시간이 아니다.

### 학습량과 early stopping

두 모델 모두 23 epoch까지 실행됐고 holdout loss 최저점은 epoch 8이다.

| 모델 | epoch 8 train/holdout loss | epoch 23 train/holdout loss |
|---|---|---|
| Baseline | 0.03580 / 0.07087 | 0.01894 / 0.09047 |
| Candidate | 0.03619 / 0.07139 | 0.02098 / 0.09002 |

후반부에 train loss만 더 낮아지는 패턴이다. 학습 코드는 최저 holdout loss의 체크포인트를 복원한다. 따라서 더 오래 학습하면 개선된다고 볼 수 없으며, 현재 best checkpoint 선택은 유지할 만하다.

## 5. 데이터 분할과 평가 독립성

현재 manifest를 직접 읽어 재집계했다.

| split | 영상 행 | 장면 그룹 | 낙상 / 비낙상 |
|---|---:|---:|---:|
| train | 12,688 | 1,586 | 9,504 / 3,184 |
| validation | 2,720 | 340 | 2,040 / 680 |
| test | 2,720 | 340 | 2,040 / 680 |

세 split 사이 장면 그룹 교집합은 각각 0이다. 같은 사건의 다른 카메라가 다른 split으로 넘어가는 기본 누수 방지는 되어 있다. 사람 ID·촬영 세션 단위 독립성과 모든 과거 모델의 학습 이력까지 검증한 것은 아니다.

중요한 제한:

1. `train_fall_temporal_tcn.py`는 train 안에서 25% 그룹을 다시 holdout으로 나누고 그 loss로 checkpoint를 선택한다. **이 holdout은 모델 선택용 검증셋이지 미사용 최종 시험셋이 아니다.** 외부 validation에도 threshold sweep을 수행하므로 최종 평가는 별도 test/현장 신규 사건으로 해야 한다.
2. “동일 split”이어도 유효 표본은 완전히 같지 않다. Baseline train 12,686건, Candidate 12,688건이며 validation 낙상 support도 2,039/2,040으로 다르다. Baseline은 pose 1~2개인 3개 영상(train 2, validation 1)을 제외했다. 비교에는 공통 유효 입력과 전체 입력의 실패율을 모두 제시해야 한다.
3. 공개 평가 320개 영상은 **40개 사건 × 8개 카메라**다. 320개의 독립 사건으로 계산한 신뢰구간/유의성 주장을 피하고 장면 그룹 기준으로 비교해야 한다.
4. test는 반복 비교 후 재학습 설계에 쓰기 시작하면 더 이상 완전한 미사용 시험셋이 아니다. 개선용 hard-case와 최종 승인용 새 촬영 세트를 분리해야 한다.

## 6. 낙상 실제 평가 결과 재검산

저장된 각 영상의 `is_fall`과 `prediction`으로 TP/FN/FP/TN을 다시 계산했다. 개별 결과 및 summary와 모두 일치했다. **새 추론을 돌린 결과는 아니다.**

### 공개 데이터 C1~C8 320개 영상

| 모델 | TP | FN | FP | TN | Precision | Recall | 정상 영상 오탐 비율 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 220 | 20 | 2 | 78 | 99.10% | 91.67% | 2/80 = 2.50% |
| Candidate | 234 | 6 | 7 | 73 | 97.10% | 97.50% | 7/80 = 8.75% |

Candidate가 미탐을 14건 줄인 것은 의미가 있다. 반대로 오탐 5건 증가는 안전한 상시 운영에서 무시할 수 없다. Candidate의 오류 13건 중 8건이 C2에 집중됐다. 이 비율은 고른 평가 클립의 결과이며 FP/day로 환산할 수 없다.

### 현장 원본 23개 영상

| 모델 | TP | FN | FP | TN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 10 | 6 | 0 | 7 | 100.00% | 62.50% |
| Candidate | 9 | 7 | 2 | 5 | 81.82% | 56.25% |

이 원본 23건에는 기존 검수에서 “낙상 전환 없음”으로 제외한 2건이 포함된다. 정제 21건 문서의 71.43%/57.14% recall과 혼용하면 안 된다. 어느 쪽도 큰 현장 일반화 성능을 확정할 표본은 아니다. 다만 공개 데이터만으로 Candidate 승격을 결정할 수 없다는 방향은 일치한다.

### “영상 30초부터 보면 좋아질 것”에 대한 판단

9/16 저장 결과에서 왼쪽 30초 이후는 Baseline/Candidate 모두 검출했다. 오른쪽은 Baseline 0.998, Candidate 0.343으로 Candidate만 threshold 0.5 미만이다. 20~30초 낙상 전환 구간은 두 모델 모두 검출했고, 30~40초 사후 자세에서는 Candidate가 약했다.

따라서 초반 배회에 의한 희석은 일부 있지만 유일한 원인은 아니다. 모든 영상을 일률적으로 30초 잘라내기보다 **사건 시작·전환·사후 자세와 전체 연속 영상 평가를 분리**해야 한다. 사건 검출과 누워 있는 사람 지속 감지도 요구사항을 나눠야 한다.

## 7. 우선 수정해야 할 모델 파이프라인 문제

### High — 특징 캐시가 추출 조건을 충분히 식별하지 않음

`scripts/datasets/train_yolo_pose_fall_rf.py:_feature_path`는 scene ID, max_frames, stride, margin만 파일명에 넣는다. `_ensure_features`는 파일이 있으면 내부 metadata 대조 없이 재사용한다. pose 모델 hash, 입력 크기, confidence, 원본 영상/라벨 변경, 특징 코드 버전이 key에 없다.

**영향:** 같은 디렉터리를 재사용하면 조건을 바꿔도 옛 특징으로 학습/평가할 수 있다. 현재 모든 캐시가 오염됐다는 판정은 아니다.

**수정/완료 기준:** 모델·전처리·특징 스키마·원본/라벨 구간의 fingerprint를 저장하고 재사용 전에 검증한다. 조건 변경 시 캐시 재사용 거부를 회귀 테스트로 고정한다. 기존 캐시는 검증·백업 전에 삭제하지 않는다.

### High — 학습/평가/실시간의 시간축과 pose 부족 처리 차이

학습 특징은 라벨 낙상 구간과 margin을 사용한다. 영상 smoke 평가에서는 전체 영상의 균등 샘플을 사용하며, encoder는 정렬된 기록 중 최근 30개를 가져오고 부족한 부분을 0으로 채운다. 시간 간격 자체는 이 temporal 입력에 없다. 학습 30개와 긴 영상의 30개는 같은 시간 길이를 뜻하지 않는다.

학습은 최소 pose 수 미달을 제외하지만 `smoke_fall_temporal_model.py`는 같은 품질 거부 조건을 적용하지 않고 점수를 출력할 수 있다. 실제 타임라인 문서에는 pose 1개인데 Candidate 점수가 0.998인 구간이 있다. 큰 점수와 충분한 근거를 혼동하면 안 된다.

**수정/완료 기준:** 고정 시간 길이·샘플링 간격·최소 관측률·최대 결측 구간·인물/카메라 ID를 계약화한다. 부족 입력은 `insufficient_evidence`로 별도 집계하고 분모에서 조용히 제외하지 않는다. 필요 시 시간 간격/결측 mask를 추가하되 특징 버전 변경과 재학습을 함께 수행한다. 실시간 sliding-window와 동일한 평가 경로를 제공한다.

### High — 범용 모델 합격 검사가 일부 정확도 기준을 무시

`scripts/health/check_model_report.py`의 검사 구현은 precision/recall/평균 지연시간만 처리한다. manifest의 `min_top1_accuracy`, `min_attribute_accuracy`는 검사하지 않는다. 빠진 latency를 0으로 취급한다.

모의 입력 `{}`에 색상 모델 기준 `{min_top1_accuracy: 0.88, max_avg_latency_ms: 10}`을 적용하자 latency 0ms 검사만 생성되고 전체 통과가 됐다. 이 도구의 PASS는 색상/속성 모델의 품질 통과 근거로 사용할 수 없다.

**수정/완료 기준:** 과제별 필수 지표 스키마, 알 수 없는 기준 거부, 값 누락/NaN 거부, 표본 수·class coverage·모델/데이터 hash 검증을 추가한다. 실제 불완전 결과로 운영 승격됐다고 확인한 것은 아니다.

### Medium — 재현 가능한 모델 등록·승격 계약 부족

중앙 `model_manifest.json`의 모델 목록에는 temporal Baseline/Candidate, 실제 비교 RF, PA100K, 외형 통합 후보가 완전한 독립 항목으로 정리되어 있지 않다. 일부는 deployment profile에 경로만 있다. 색상 PT hash는 현재 파일과 일치했지만 다른 핵심 평가 결과에는 모델/데이터 hash 결합이 부족하다.

`scripts/promote_fall_model.py`의 불리언 승인과 지정한 candidate 파일을 강하게 결합하고, 원자적 교체·이전 모델 복구·특징 스키마 검사를 추가해야 한다. `dataset_version`은 산출물 이름 대신 데이터·split hash로 비교한다. 단순히 현재 HOLD 검사를 우회해서는 안 된다.

## 8. 헬멧·사람 검출 모델 분석

`helmet_model_trt_320_conf015.json`을 확인했다. helmet 기준 TP 256, FP 28, FN 25이며 평가 conf=0.15다. 대상 클래스가 `helmet` 하나이므로 `head` 검출 및 “미착용 경보” 전체의 recall 91.10%를 증명하지 않는다.

운영은 사람 ROI 안의 SGIE 및 후처리 조건을 사용하므로 평가와 같은 입력/threshold인지 확인해야 한다. full-frame 탐지, crop 탐지, 추적 후 경보를 각각 검증한다. 헬멧은 즉시 재학습보다 원거리·야간·가림·미착용 head를 포함한 현장 평가를 먼저 권한다.

pose/person의 manifest 61장 수치는 `data/eval/person`의 bbox precision/recall이다. keypoint 품질이나 사건 단위 낙상 성적이 아니다. manifest의 입력 크기와 평가 설정에도 차이가 있어 현재 640 입력의 운영 성능으로 그대로 인용하면 안 된다. 사람 누락률→pose 확보율→낙상 판정→경보 전달을 단계별로 측정한다.

## 9. 외형·색상 학습 모델 분석

### protected_multitask 실제 학습 결과

관련 산출물은 메인 `models/experiments`가 아니라 `.worktrees/appearance-multitask-v1/models/experiments/appearance_mobilenet_v3_crop_weighted/protected_multitask/metrics.json`에 있다. 문서만 보고 파일이 없다고 삭제/실패 판정하면 안 된다.

| 속성 | Macro F1 | 주요 제한 |
|---|---:|---|
| 성별 외형 분류 | 0.8731 | female recall 79.49%, male 95.97%로 편차; 확정 정보로 취급하지 않기 |
| 상의 형태 | 0.8704 | sleeveless 평가 표본 없음 |
| 하의 형태 | 0.6628 | long_skirt recall 11.69%; short_skirt/dress 표본 없음 |
| 상의 색상 | 0.6852 | blue recall 36.92%, navy/black 혼동 점검 |
| 하의 색상 | 0.5261 | brown F1 0.0336, navy 0.2160 |
| 소지품 | 0.5147 | hat recall 8.67%, backpack 41.31%, bag 46.59% |

하의 red/green/yellow/purple/orange는 해당 검증셋에 0건이다. `best_color_gate.passed=false`이며 부족 클래스가 명시돼 있다. Macro F1은 관측된 유효 클래스 기준으로 해석하고, 지원하려는 모든 클래스에 대한 성능으로 확대하지 않는다.

**다음 학습:** 보이지 않는 하의/소지품에 강제 음성/색상 라벨을 붙이지 말고 unknown/mask를 유지한다. brown/navy 및 모자·치마의 실제 CCTV crop을 먼저 보강하고, 사람/촬영 세션 단위로 분할한다. 색상 학습으로 기존 성별·의류 성능이 악화되지 않는지도 head별로 확인한다. 민감한 외형 추정은 불확실성을 표시하고 단독 식별·자동 불이익 판단에 사용하지 않는다.

### 내보낸 RAPv2 후보와 기존 통합 후보는 구분해야 함

`models/appearance/candidates/appearance_multitask_v1_rapv2_reviewed/onnx_export_report.json`은 parity 통과, 최대 차이 약 0.00000978이다. 그러나 metadata의 model_version은 `appearance_multitask_v1.color_baseline`이다. 마지막 학습 로그의 상/하의 색상 F1은 약 0.6944/0.5967, 성별 0.4880, 소지품 0.2289다. 마지막 로그를 내보낸 best checkpoint의 정확한 성적으로 단정하지는 않는다.

`color_gate`의 상태도 `baseline_reference`다. 즉 `deployable=true`와 `parity_passed=true`는 변환 가능성/수치 일치의 근거이지 통합 기능 승격 기준이 아니다. 8/28 protected 모델의 좋은 비색상 성적을 이 파일의 성적으로 가져오면 안 된다.

기존 색상 YOLO의 AI4C top-1 88.49%와 위 다중 head macro F1도 서로 다른 지표/데이터다. accepted accuracy 91.27%에는 얼마나 많은 입력을 거절했는지 coverage가 필요하다. PT→ONNX→실제 Jetson 엔진의 동일 crop 결과와 클래스별 정확도·unknown 비율을 함께 평가한다.

## 10. 재학습·검증 우선순위

1. **평가 신뢰성 복구:** 캐시 fingerprint, 부족 pose 결과, 모델별 합격 검사, 모델/데이터/환경 등록을 먼저 고친다. 운영 모델은 자동 교체하지 않는다.
2. **낙상 입력 통일:** 학습과 같은 pose 해상도·confidence, 동일 시간창·인물/카메라 기준으로 두 모델을 비교한다. 공개 320건의 C2 오류와 현장 뒤/옆 낙상·사후 자세를 우선 검수한다.
3. **현장 데이터 보강:** 기존 hard-positive는 개선용으로 편입하되, 정상 착석·굽힘·눕기·물건 줍기·가림을 포함한 hard-negative와 별도 미사용 촬영 세트를 확보한다. 이미 평가에 사용한 영상을 새 독립 test라고 부르지 않는다.
4. **최소 재학습 1회:** 현재 경량 구조와 early stopping을 유지한 통제 실험부터 한다. 같은 검증셋에서 threshold를 선택하고 최종 test에는 다시 맞추지 않는다. 미탐·오탐·감지 지연·입력 부족률을 함께 비교한다.
5. **실시간 shadow/UAT:** 동일 스트림의 baseline/candidate 판정을 비교하되 장비 출력은 분리한다. 정상 카메라 운영시간을 분모로 오경보/카메라·시간과 사건별 recall, p95 지연, FPS·메모리·drop을 측정한다. “2~3일 실행” 자체가 합격 조건은 아니다.
6. **외형 개선:** 운영 검색에 필요한 속성부터 범위를 확정하고 하의 색상·모자·치마 부족 데이터를 보강한다. 헬멧은 별도 현장 ROI/미착용 검증 후 재학습 필요성을 결정한다.

합격 기준 수치는 안전 요구와 오경보 허용량을 먼저 정해야 한다. 현재 97.5% recall 한 항목만으로 합격이라 하거나 모델 자체가 무가치하다고 결론내리는 것 모두 근거가 부족하다.

## 11. 검증·산출물·한계

- 모델 관련 기존 테스트: **50 passed** (6.18초).
- 이번 재집계 스크립트 Ruff: **통과**.
- split 그룹 중복: 0; 공개/현장 저장 결과의 row 판정 및 summary 재집계: 모두 일치.
- manifest 등록 artifact 경로: 확인 대상 모두 존재. 색상 PT hash 일치.
- temporal checkpoint 메타데이터와 파라미터 수: CPU 안전 로드로 확인.
- 새 모델 정확도 측정, 학습 복제, calibration/SHAP, 사람 ID 누수 전체 조사, TensorRT 성능 측정 및 현장 UAT: 미실시. 관련 지표가 없다는 것과 모델이 그 기준에 실패했다는 것은 구분한다.

추가 파일은 이 보고서와 읽기 전용 재집계 스크립트뿐이다. 운영 소스·모델·환경변수는 변경하지 않았다.

재집계 실행:

```bash
rtk proxy .venv/bin/python docs/reports/model_audit_20260917.py
rtk proxy .venv/bin/ruff check docs/reports/model_audit_20260917.py
rtk proxy .venv/bin/python -m pytest tests/test_fall_temporal_model.py tests/test_train_fall_temporal_tcn.py tests/test_train_yolo_pose_fall_rf.py tests/test_compare_fall_models.py tests/test_compare_fall_temporal_runtime.py tests/test_check_model_report.py -q -o addopts='--tb=short'
```

주요 증거 파일:

- `data/fall_eval/fall_temporal_{baseline_same_split_20260910,same_split_resume_20260908}_metrics.json`
- `data/fall_eval/yolo_pose_same_split_resume_20260908_rf_metrics.json`
- `data/fall_eval/reconstructed_split_20260908_container/{train,validation,test}.jsonl`
- `data/fall_eval/open_data_test_20260914/balanced320_c1_c8_comparison.json`
- `data/fall_eval/yesterday_20260910_temporal_runtime_comparison_imgsz640.json`
- `docs/models/FALL_RECORDED_VIDEO_TIMELINE_ANALYSIS_20260916.md` 및 해당 원본 JSON
- `data/eval/reports/helmet_model_trt_320_conf015.json`, `models/model_manifest.json`
- `.worktrees/appearance-multitask-v1/models/experiments/appearance_mobilenet_v3_crop_weighted/protected_multitask/metrics.json`
- `models/appearance/candidates/appearance_multitask_v1_rapv2_reviewed/onnx_export_report.json`
