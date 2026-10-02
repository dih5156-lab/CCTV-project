# 낙상 오류 정제 및 다음 실행 계획 (2026-09-14)

## 오늘 완료한 판단

- 금요일 베이스라인 모델을 유지한다.
- 평가 pose 입력을 학습 조건과 같은 `640/0.35`로 수정했다.
- 원본 23개 전체 비교와 FN 구간 클립 재평가를 완료했다.
- 재촬영 2개, genuine hard-positive 2개, 이벤트 구간 평가 대상 2개로 분리했다.
- 가림 낙상 3개는 640 입력에서 모두 TP로 회복돼 재촬영·재학습 대상에서 뺐다.
- 정상 보행 7개는 베이스라인 FP가 0이므로 기존 hard-negative 추가 계획을
  철회했다.

## 오류 분류

| 장면 | 640 원본 | 구간 결과 | 최종 조치 |
| --- | --- | --- | --- |
| `140519_backward_fall` | FN | FN | positive hard-case 재학습 |
| `141112_backward_fall` | FN | TP | 이벤트 중심 window 평가 |
| `141230_left_fall` | FN | 낙상 전환 없음 | 제외 후 재촬영 |
| `141722_right_fall` | FN | 낙상 전환 없음 | 제외 후 재촬영 |
| `141753_right_fall` | FN | FN | positive hard-case 재학습 |
| `141827_right_fall` | FN | TP | 이벤트 중심 window 평가 |

상세 수치와 검수 상태는
`data/fall_eval/yesterday_20260910_baseline_error_triage.jsonl`에 기록했다.

## 생성한 학습 준비 자료

- 시각 근거: `data/fall_eval/error_review_20260914/`
- 정제 구간 영상 4개: `data/fall_eval/retraining_clips_20260914/`
- 구간 manifest: `data/fall_eval/retraining_clips_manifest_20260914.jsonl`
- 구간 모델 결과: `data/fall_eval/retraining_clips_comparison_20260914.json`

정제 영상은 원본을 보존하고 별도 H.264 파일로 만들었다. 현 단계에서 실제
재학습에 바로 넣을 대상은 `140519_backward`와 `141753_right` 두 개다.

## 다음 실행 순서

1. `141230_left`, `141722_right`를 다시 촬영한다.
2. 전체 낙상 영상에 이벤트 시작·종료 구간을 붙여 event-window 평가 manifest를
   만든다. 긴 원본 전체의 균등 샘플 결과와 분리한다.
3. `140519_backward`, `141753_right`를 기존 데이터에 hard-positive로 추가해
   재학습 1회를 수행한다.
4. 동일 event-window 세트와 별도 미사용 영상에서 baseline 대비 FN/FP를
   비교한다.

## 재촬영 기준

- 현재 카메라 원본 해상도 유지
- 평가·feature 추출 `imgsz=640`, confidence `0.35` 고정
- 낙상 전 정상 자세 3~5초
- 낙상 전환이 프레임에서 완전히 보이도록 촬영
- 바닥 자세 3~5초
- 파일명에 방향과 take 번호 기록
- 넘어짐이 화면 경계나 가구에 완전히 가려지지 않도록 구성

## 통과 기준

- 정제 event-window 세트에서 baseline 대비 FN/FP 비열화 없음
- 기존 holdout에서 baseline 대비 FN/FP 비열화 없음
- `NO_RESULT=0`
- Jetson 실시간 실행에서 frame drop 증가 없음

현재 운영 모델, threshold, crop-pose feature flag는 변경하지 않았다.
