# 상·하의 색상 검수 운영 가이드

## 결론

검수 작업은 다음 순서로 진행합니다.

```text
검수 대상 manifest 생성
→ 로컬 HTML 생성 및 브라우저 검수
→ 검수 JSON 다운로드
→ DB 변경 dry-run 확인
→ DB 백업 및 변경 적용
→ 재학습 준비용 라벨 데이터 내보내기
```

색상을 지정해도 모델이 자동으로 학습되거나 Jetson에 자동 배포되지는 않습니다.

## 1. 검수 대상 생성

프로젝트 루트에서 실행합니다.

```bash
rtk test .venv/bin/python scripts/ops/build_appearance_color_review_manifest.py \
  --db data/runtime/appearances.db \
  --output data/runtime/appearance_color_review_manifest.json \
  --limit 200
```

이 manifest에는 기존 DB 값과 HSV/LAB/model 후보가 함께 저장됩니다.

## 2. 로컬 검수 HTML 생성

```bash
rtk test .venv/bin/python scripts/ops/build_appearance_review_html.py \
  --manifest data/runtime/appearance_color_review_manifest.json \
  --output data/runtime/appearance_color_review.html
```

`data/runtime/appearance_color_review.html`을 Chrome 같은 브라우저로 엽니다. 별도 웹서버나 인터넷 연결은 필요하지 않습니다.

각 사진에서 다음 두 값을 독립적으로 선택할 수 있습니다.

- 상의 정답
- 하의 정답

`변경 안 함`은 해당 값을 수정하지 않는다는 뜻입니다. `exclude`는 DB 색상값을 변경하지 않고 해당 필드를 학습 데이터에서 제외합니다.

검수가 끝나면 브라우저를 닫기 전에 **검수 라벨 JSON 다운로드**를 누릅니다. 기본 파일명은 `appearance_color_review_labels.json`입니다.

## 3. DB 변경 미리보기

다운로드한 JSON 경로를 `--labels`에 지정합니다.

```bash
rtk test .venv/bin/python scripts/ops/apply_appearance_color_review_labels.py \
  --db data/runtime/appearances.db \
  --labels /path/to/appearance_color_review_labels.json
```

`--apply`를 생략하면 dry-run으로 실행되어 DB를 수정하거나 백업 파일을 만들지 않습니다. 출력의 `changes`에서 ID별 기존값과 변경값을 확인합니다.

잘못된 색상, 중복 ID 또는 DB에 없는 ID가 하나라도 있으면 전체 작업이 중단됩니다.

## 4. DB에 실제 적용

dry-run 결과를 확인한 후에만 `--apply`를 추가합니다.

```bash
rtk test .venv/bin/python scripts/ops/apply_appearance_color_review_labels.py \
  --db data/runtime/appearances.db \
  --labels /path/to/appearance_color_review_labels.json \
  --apply
```

실제 변경이 있으면 먼저 DB 옆에 다음 형태의 백업을 생성합니다.

```text
appearances.db.YYYYMMDD_HHMMSS_microseconds.bak
```

출력의 `backup` 경로를 확인합니다. 모든 DB 변경은 하나의 SQLite 트랜잭션으로 적용됩니다.

## 5. 재학습 준비용 라벨 내보내기

```bash
rtk test .venv/bin/python scripts/ops/export_appearance_color_review_labels.py \
  --manifest data/runtime/appearance_color_review_manifest.json \
  --labels /path/to/appearance_color_review_labels.json \
  --output-dir data/training/appearance_color_reviews
```

다음 파일이 생성됩니다.

- `reviewed_appearance_colors.csv`: 사람이 실제로 선택한 상의·하의 라벨
- `reviewed_appearance_colors.json`: 원본 후보값과 사람 정답을 함께 보존한 감사 자료
- `summary.json`: 색상별 수량, 부분 검수, 제외, 누락 crop 및 학습기 미지원 색상 통계

미선택 필드는 빈 값으로 유지됩니다. 기존 DB 값을 임의로 정답으로 채우지 않습니다. 상의와 하의가 모두 미선택인 행과 crop 파일이 사라진 행은 학습 CSV에서 제외됩니다.

## 재학습 시 주의사항

현재 multi-label 학습 목록은 다음 9개 색상을 지원합니다.

```text
black, white, gray, red, blue, green, yellow, brown, purple
```

런타임 판정은 여기에 `orange`, `pink`도 사용합니다. 두 색상을 선택한 라벨은 유실하지 않고 `summary.json`의 `multilabel_unsupported_fields`에 집계하지만, 기존 multi-label 모델에 바로 넣으면 안 됩니다.

재학습은 다음 조건을 확인한 뒤 별도로 진행합니다.

1. 색상별 검수 데이터 수량과 불균형 확인
2. 부분 라벨을 사용할지, 상·하의가 모두 검수된 행만 사용할지 결정
3. 기존 검증 세트로 정확도와 오탐 회귀 확인
4. 모델 출력 구조가 바뀌면 TensorRT 및 DeepStream 설정도 함께 갱신

검수 JSON이나 DB 변경만으로 현재 실행 중인 모델의 추론 결과는 바뀌지 않습니다.
