# 낙상 영상 수집 및 라벨링

## 결론

낙상 학습·평가 영상은 런타임 DB나 일반 로그와 섞지 않고
`data/fall_dataset/` 아래에서 관리합니다. 영상과 라벨은 개인정보가 포함될 수
있으므로 Git에는 저장하지 않습니다.

## 디렉터리 구조

```text
data/fall_dataset/
├── clips/
│   ├── pending/             # 아직 라벨링하지 않은 영상
│   └── labeled/
│       ├── fall/            # 낙상 확인 영상
│       └── non_fall/        # 앉기·줍기 등 비낙상 영상
├── annotations/
│   └── review.jsonl         # 영상별 라벨과 메타데이터
└── manifests/
    └── field_combined_manifest.jsonl
```

## 1. 저장소 초기화

```bash
python scripts/datasets/collect_fall_dataset.py --init
```

## 2. 영상 추가

현장에서 복사한 MP4 파일을 라벨 대기 목록에 추가합니다.

```bash
python scripts/datasets/collect_fall_dataset.py \
  --video /path/to/camera_1_20260708.mp4 \
  --camera camera_1 \
  --source field_manual \
  --note "의자에서 일어나다가 발생한 후보"
```

스크립트는 영상을 `clips/pending/`으로 복사하고 SHA-256 값으로 중복 수집을
차단하며 `annotations/review.jsonl`에 메타데이터를 추가합니다. 원본 파일은
삭제하지 않습니다.

OpenCV 분석 경로에서 shadow 저장을 사용할 때는 다음 경로를 사용합니다.

```env
FALL_SHADOW_REVIEW_LOG_PATH=/app/data/fall_dataset/annotations/review.jsonl
FALL_SHADOW_SAVE_CLIPS=true
FALL_SHADOW_CLIP_DIR=/app/data/fall_dataset/clips/pending
```

현재 DeepStream 경로는 shadow 저장 메서드는 있지만 운영 초기화 연결이 완전히
검증되지 않았습니다. Jetson에서 자동 저장된다고 가정하지 말고 우선 위 수집
스크립트 또는 실제 파일 생성 여부로 확인해야 합니다.

## 3. 라벨링

GUI가 사용 가능한 PC에서 실행합니다.

```bash
python scripts/ops/label_fall_shadow_clips.py --camera camera_1
```

- `F`: 낙상 (`clips/labeled/fall/`로 이동)
- `N`: 비낙상 (`clips/labeled/non_fall/`로 이동)
- `S`: 판단 보류
- `Space`: 일시 정지
- `R`: 처음부터 재생
- `Q`: 저장 후 종료

GUI를 열기 전에 대상만 확인하려면 다음 명령을 사용합니다.

```bash
python scripts/ops/label_fall_shadow_clips.py --camera camera_1 --list
```

브라우저에서 검수하려면 HTML 페이지를 생성합니다. 다음 예시는 최근 후보 중
같은 카메라에서 30초 이내에 반복된 영상을 줄이고 최대 50건을 선택합니다.

```bash
python scripts/ops/label_fall_shadow_clips.py \
  --include-sample-eval \
  --dedupe-seconds 30 \
  --limit 50 \
  --export-html data/fall_dataset/review/fall_review.html
```

HTML 파일을 브라우저로 열고 `낙상`, `비낙상`, `보류`를 선택한 뒤
`검수 JSON 다운로드` 버튼으로 `fall_review_labels.json`을 저장합니다. 선택값은
해당 브라우저에 임시 저장되지만, 다른 PC나 브라우저에는 공유되지 않습니다.

원본 클립은 브라우저 호환성이 낮을 수 있으므로 HTML 생성 시
`review/browser_clips/` 아래에 H.264 검수용 복사본을 만듭니다. 다음처럼 로컬
HTTP 서버를 실행한 뒤 `http://127.0.0.1:8091/fall_review.html`로 접속합니다.

```bash
python -m http.server 8091 \
  --bind 0.0.0.0 \
  --directory data/fall_dataset/review
```

다운로드한 라벨을 반영할 때는 다음 명령을 실행합니다.

```bash
python scripts/ops/label_fall_shadow_clips.py \
  --import-labels /path/to/fall_review_labels.json
```

반영 전에 전체 라벨과 클립 경로를 검증하며, `review.jsonl` 백업을 생성합니다.
`fall`과 `non_fall` 영상만 각 labeled 폴더로 이동하고 `보류` 영상은 pending
폴더에 유지합니다.

## 4. 학습·평가 manifest 생성

```bash
python scripts/datasets/build_field_fall_manifest.py \
  --base-manifest data/fall_eval/sample_manifest.jsonl
```

결과는 기본적으로
`data/fall_dataset/manifests/field_combined_manifest.jsonl`에 생성됩니다.
카메라와 촬영일이 같은 영상은 같은 장면 그룹으로 취급하여 train/test 누수를
줄입니다.

## 운영 주의사항

- 얼굴과 차량번호가 포함될 수 있으므로 접근 권한과 보존 기간을 설정합니다.
- 라벨이 애매한 영상은 억지로 분류하지 말고 `S`로 보류합니다.
- 원본 영상을 프레임으로 쪼개 train/test에 무작위 분배하면 안 됩니다.
- 모델 변경 전 실제 낙상 미탐과 비낙상 오탐을 영상 단위로 검토합니다.
