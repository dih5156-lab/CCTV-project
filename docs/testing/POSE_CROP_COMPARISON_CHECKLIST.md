# Full-frame / Crop-pose 비교 체크리스트

이 문서는 동일한 녹화 영상에서 기존 full-frame 경로와 `POSE_CROP_MODE=1`
경로를 비교할 때 사용한다. 학습 산출물과 무관한 런타임 검증 문서이며,
운영 기본값은 계속 `POSE_CROP_MODE=0`으로 둔다.

학습이 `cctv-ai-engine` 컨테이너에서 진행 중이면 Compose 재생성 명령을
실행하지 않는다. 학습 완료와 모델 백업 후 별도 테스트 장치에서 검증한다.

## 실행 전 고정값

- 영상 파일, 해상도, 재생 속도, frame stride
- pose/person 모델 파일과 버전
- `conf`, `iou`, `imgsz`, Jetson 전원 모드
- 테스트 시각, 장치 ID, git revision

## 기록할 값

| 구분 | 기록 항목 | 산출 방법 |
| --- | --- | --- |
| 검출 | person 검출 수, track 수 | 이벤트 JSONL 집계 |
| pose | pose 생성률, keypoint 누락 프레임 수 | person 이벤트의 keypoints 유무 |
| 낙상 | TP, FP, FN, precision, recall | 영상별 정답 라벨과 대조 |
| 지연 | 평균·p95 frame latency | 프레임 타임스탬프 차이 |
| 처리량 | 평균 FPS, frame drop 수 | DeepStream/runtime 지표 |
| 자원 | GPU 메모리, CPU, RAM | Jetson 모니터링 로그 |

## 판정 순서

1. 두 경로가 같은 프레임 범위를 처리했는지 확인한다.
2. pose 생성률과 낙상 FN을 먼저 비교한다. 사람 검출이 늘어도 FN이 증가하면
   운영 전환하지 않는다.
3. p95 지연시간과 frame drop이 현장 허용 기준 이내인지 확인한다.
4. 5개 낙상 유형(옆·뒤·앞·상체 가림·하체 가림)을 각각 분리해 결과를 남긴다.
5. 기준 미달이면 `POSE_CROP_MODE=0`으로 되돌리고 비교 결과에 실패 원인을
   기록한다.

## 결과 파일 규칙

```text
data/fall_eval/reports/pose_crop_compare_<YYYYMMDD>_<version>.json
```

JSON에는 `mode`, `model_paths`, `runtime_config`, `metrics`, `video_cases`,
`decision`, `operator`, `measured_at` 필드를 포함한다. 원본 영상은 결과 JSON과
같은 case ID를 사용해 추적 가능하게 관리한다.

작성 후에는 다음 명령으로 형식을 확인한다.

```bash
python scripts/ops/validate_pose_crop_compare_report.py \
  data/fall_eval/reports/pose_crop_compare_<YYYYMMDD>_<version>.json
```
