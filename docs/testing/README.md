# 테스트 문서 색인

낙상 모델과 crop-pose 검증 문서는 다음 순서로 사용한다.

1. [`FALL_TEST_DATA_COLLECTION.md`](FALL_TEST_DATA_COLLECTION.md) — 낙상·비낙상
   영상과 5개 낙상 유형의 수집 규칙
2. [`FALL_MODEL_COMPARISON_RUNBOOK.md`](FALL_MODEL_COMPARISON_RUNBOOK.md) — 모델
   평가와 결과 보관 절차
3. [`POSE_CROP_COMPARISON_CHECKLIST.md`](POSE_CROP_COMPARISON_CHECKLIST.md) —
   full-frame/crop-pose 런타임 비교 항목과 판정 기준
4. [`pose_crop_compare_report.template.json`](pose_crop_compare_report.template.json)
   — 비교 결과 JSON 시작 템플릿

결과 JSON은 다음 도구로 형식을 확인한다.

```bash
python scripts/ops/validate_pose_crop_compare_report.py <report.json>
```

