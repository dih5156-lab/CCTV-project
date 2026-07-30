# DeepStream Temporal Fall TCN Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 실제 DeepStream pose 48프레임 시퀀스로 경량 TCN을 학습하고, 분리된 Validation 8 낙상 + 8 정상 probe에서 임계값 0.7 이상, recall 75% 이상, 정상 FPR 10% 이하인지 판정한다.

**Architecture:** 기존 인라인 RF 경로가 선택한 frame record를 schema v2 sidecar로 보존한다. 신규 trainer는 이 JSONL만 입력받아 temporal-only Conv1D 모델을 학습하며, 기존 RF와 운영 설정은 변경하지 않는다. Probe 관문을 통과한 경우에만 후속 20+20 계획을 작성한다.

**Tech Stack:** Python 3.10, NumPy, PyTorch, pytest, DeepStream/GStreamer replay, JSONL

## Global Constraints

- 운영 신뢰도 임계값은 0.70 미만으로 낮추지 않는다.
- 입력 크기는 `[batch, 48, len(FRAME_FEATURE_NAMES)]`로 고정한다.
- 첫 후보는 `FallTemporalTCN`만 사용하고 `FallTemporalHybrid` summary branch는 사용하지 않는다.
- Training과 Validation의 `scene_id` 및 `group_id`가 하나라도 겹치면 중단한다.
- pose TensorRT 엔진, API, DB schema, MQTT payload, 운영 `confirm` 설정은 변경하지 않는다.
- 새 라이브러리를 추가하지 않는다.
- 모든 shell 명령은 `rtk`로 시작한다.
- 기존 dirty worktree의 관련 없는 변경은 수정하거나 커밋하지 않는다.

---

## File Structure

- `src/core/ai/_falldata_aux.py`: 실제 런타임에서 선택된 48개 frame record를 sidecar schema v2로 기록한다.
- `src/core/ai/fall_temporal_model.py`: 기존 고정 길이 encoder와 경량 temporal-only TCN을 그대로 공유한다.
- `scripts/datasets/train_fall_temporal_tcn.py`: 기존 오프라인 캐시 TCN 경로의 인자 계약만 정상화한다.
- `scripts/datasets/train_deepstream_pose_fall_tcn.py`: DeepStream capture JSONL 검증, group split, 학습, 임계값 선택, checkpoint 및 metrics 생성을 담당한다.
- `scripts/ops/evaluate_sample_deepstream_replay.py`: schema v2 frame record를 manifest 메타데이터와 결합한다.
- `tests/test_train_fall_temporal_tcn.py`: 기존 캐시 경로 회귀 테스트를 담당한다.
- `tests/test_falldata_aux.py`: sidecar schema v2 캡처를 검증한다.
- `tests/test_evaluate_sample_deepstream_replay.py`: schema v2 라벨 결합과 잘못된 sequence 거부를 검증한다.
- `tests/test_train_deepstream_pose_fall_tcn.py`: 신규 dataset/trainer의 순수 로직과 checkpoint 계약을 검증한다.
- `data/fall_eval/`: probe capture, labeled dataset, 모델 성능 결과만 저장하며 소스 커밋에는 포함하지 않는다.

### Task 1: 기존 TCN 캐시 경로 계약 정상화

**Files:**
- Modify: `scripts/datasets/train_fall_temporal_tcn.py`
- Modify: `tests/test_train_fall_temporal_tcn.py`

**Interfaces:**
- Consumes: `_feature_path(feature_cache, row, max_frames, frame_stride, fall_window_margin_frames) -> Path`
- Produces: `_select_cached_rows(..., fall_window_margin_frames: int) -> list[dict[str, Any]]`

- [ ] **Step 1: 현재 실패를 그대로 재현한다**

Run:

```bash
rtk pytest tests/test_train_fall_temporal_tcn.py::test_select_cached_rows_keeps_only_rows_with_matching_feature_files -q
```

Expected: `_feature_path()`의 `fall_window_margin_frames` 누락으로 FAIL.

- [ ] **Step 2: 명시적인 margin 계약을 테스트에 반영한다**

테스트 캐시 파일을 다음 이름으로 만들고 호출 인자에 `fall_window_margin_frames=60`을 추가한다.

```python
(tmp_path / "scene_b_labeled_window_max30_stride6_margin60.json").write_text(
    "{}",
    encoding="utf-8",
)

selected = train_fall_temporal_tcn._select_cached_rows(
    rows,
    feature_cache=tmp_path,
    max_frames=30,
    frame_stride=6,
    fall_window_margin_frames=60,
)
```

- [ ] **Step 3: 최소 구현으로 모든 `_feature_path` 호출을 일치시킨다**

`--fall-window-margin-frames` 기본값을 60으로 추가하고
`_load_cached_dataset`, `_select_cached_rows`, `main`의 train/validation 호출에
동일 값을 전달한다.

```python
def _select_cached_rows(
    rows: list[dict[str, Any]],
    *,
    feature_cache: Path,
    max_frames: int,
    frame_stride: int,
    fall_window_margin_frames: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _feature_path(
            feature_cache,
            row,
            max_frames,
            frame_stride,
            fall_window_margin_frames,
        ).exists()
    ]
```

- [ ] **Step 4: 관련 테스트를 검증한다**

Run:

```bash
rtk pytest tests/test_train_fall_temporal_tcn.py tests/test_train_yolo_pose_fall_rf.py -q
```

Expected: 두 파일의 테스트가 모두 PASS.

- [ ] **Step 5: 이 작업 파일만 커밋한다**

```bash
rtk git add scripts/datasets/train_fall_temporal_tcn.py tests/test_train_fall_temporal_tcn.py
rtk git commit -m "fix: align temporal cache window contract"
```

### Task 2: DeepStream frame sequence sidecar schema v2

**Files:**
- Modify: `src/core/ai/_falldata_aux.py`
- Modify: `tests/test_falldata_aux.py`

**Interfaces:**
- Consumes: `_summarize_frames(...)[frame_records]`
- Produces: JSONL schema v2 with `frame_records: list[dict[str, Any]]`

- [ ] **Step 1: 48개 frame record가 기록되는 실패 테스트를 추가한다**

기존 `test_inline_feature_capture_writes_exact_summary_vector`를 확장해 다음을
검증한다.

```python
capture = json.loads(capture_path.read_text(encoding="utf-8"))
assert capture["schema_version"] == 2
assert capture["sampled_frames"] == 48
assert len(capture["frame_records"]) == 48
assert capture["frame_records"][0]["timestamp"] <= capture["frame_records"][-1]["timestamp"]
assert capture["frame_feature_names"] == list(FRAME_FEATURE_NAMES)
```

- [ ] **Step 2: 테스트가 schema v1 또는 누락 키로 실패하는지 확인한다**

Run:

```bash
rtk pytest tests/test_falldata_aux.py::test_inline_feature_capture_writes_exact_summary_vector -q
```

Expected: `schema_version`, `frame_records`, 또는 `frame_feature_names` assertion으로 FAIL.

- [ ] **Step 3: sidecar writer에 JSON-safe frame record를 추가한다**

원본 deque를 바꾸지 않고 숫자와 문자열 목록만 새 dict로 복사한다.

```python
record = {
    "schema_version": 2,
    "frame_feature_names": list(FRAME_FEATURE_NAMES),
    "frame_records": [dict(frame_record) for frame_record in summary["frame_records"]],
    # 기존 summary 필드는 유지
}
```

직렬화 실패는 기존과 동일하게 fail-open 처리하며 추론 결과를 변경하지 않는다.

- [ ] **Step 4: 캡처와 기존 인라인 RF 테스트를 검증한다**

Run:

```bash
rtk pytest tests/test_falldata_aux.py -q
```

Expected: 전체 PASS.

- [ ] **Step 5: 이 작업 파일만 커밋한다**

```bash
rtk git add src/core/ai/_falldata_aux.py tests/test_falldata_aux.py
rtk git commit -m "feat: capture DeepStream temporal pose sequences"
```

### Task 3: schema v2 라벨 결합과 sequence 검증

**Files:**
- Modify: `scripts/ops/evaluate_sample_deepstream_replay.py`
- Modify: `tests/test_evaluate_sample_deepstream_replay.py`

**Interfaces:**
- Consumes: schema v1 summary record 또는 schema v2 temporal record
- Produces: labeled schema v2 row with `scene_id`, `group_id`, `label`, `is_fall`, `split_source`, `frame_records`

- [ ] **Step 1: 유효한 temporal capture 결합 테스트를 추가한다**

```python
capture = {
    "schema_version": 2,
    "runtime": "deepstream_pose_inline",
    "feature_names": ["frames_seen"],
    "feature_vector": [48.0],
    "frame_feature_names": list(FRAME_FEATURE_NAMES),
    "frame_records": [{"timestamp": float(index)} for index in range(48)],
}
labeled, errors = replay._label_feature_capture_records([capture], manifest_row)
assert errors == []
assert labeled[0]["frame_records"] == capture["frame_records"]
assert labeled[0]["group_id"] == manifest_row["scene_group"]
```

- [ ] **Step 2: 비정상 temporal capture 거부 테스트를 추가한다**

`frame_records`가 list가 아니거나 비어 있는 schema v2 레코드는 결과 행에서
제외되고 `record 0: invalid frame_records` 오류를 반환해야 한다.

- [ ] **Step 3: 실패 상태를 확인한다**

Run:

```bash
rtk pytest tests/test_evaluate_sample_deepstream_replay.py -q
```

Expected: 신규 temporal validation 테스트가 FAIL.

- [ ] **Step 4: 기존 schema v1 호환성을 유지하며 v2만 추가 검증한다**

```python
if int(record.get("schema_version") or 1) >= 2:
    frame_records = record.get("frame_records")
    if not isinstance(frame_records, list) or not frame_records:
        errors.append(f"record {index}: invalid frame_records")
        continue
```

라벨 결합 시 manifest의 `fall_start_frame`, `fall_end_frame`,
`scene_position`, `scene_location`, `age_group`, `fall_direction`도 보존한다.

- [ ] **Step 5: evaluator 테스트를 검증하고 커밋한다**

Run:

```bash
rtk pytest tests/test_evaluate_sample_deepstream_replay.py -q
```

Expected: 전체 PASS.

```bash
rtk git add scripts/ops/evaluate_sample_deepstream_replay.py tests/test_evaluate_sample_deepstream_replay.py
rtk git commit -m "feat: label DeepStream temporal captures"
```

### Task 4: DeepStream temporal dataset loader와 누출 방지

**Files:**
- Create: `scripts/datasets/train_deepstream_pose_fall_tcn.py`
- Create: `tests/test_train_deepstream_pose_fall_tcn.py`

**Interfaces:**
- Produces: `TemporalCaptureDataset(sequences, labels, scene_ids, group_ids, metadata)`
- Produces: `load_temporal_capture_datasets(paths: Sequence[Path], sequence_length: int = 48) -> TemporalCaptureDataset`
- Produces: `assert_validation_disjoint(training, validation) -> None`

- [ ] **Step 1: loader 실패 테스트를 작성한다**

두 개의 schema v2 JSONL 행으로 `[2, 48, feature_count]` float32 배열이
생성되고 label/group 순서가 유지되는지 검증한다. feature name 순서 불일치,
NaN, 빈 sequence, 단일 클래스는 각각 `ValueError`가 나야 한다.

- [ ] **Step 2: 테스트가 모듈 부재로 실패하는지 확인한다**

Run:

```bash
rtk pytest tests/test_train_deepstream_pose_fall_tcn.py -q
```

Expected: 신규 trainer 모듈이 없어 collection 또는 import FAIL.

- [ ] **Step 3: strict loader를 최소 구현한다**

```python
@dataclass(frozen=True)
class TemporalCaptureDataset:
    sequences: np.ndarray
    labels: np.ndarray
    scene_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]
```

각 `frame_records`는 `encode_frame_sequence(..., sequence_length=48)`로
변환한다. `frame_feature_names`는 정확히 `FRAME_FEATURE_NAMES` 순서여야 한다.

- [ ] **Step 4: Training/Validation 중복 테스트와 구현을 추가한다**

`scene_ids` 또는 `group_ids` 교집합이 있으면 교집합 값을 포함한
`ValueError`를 발생시킨다.

- [ ] **Step 5: loader 테스트를 검증하고 커밋한다**

Run:

```bash
rtk pytest tests/test_train_deepstream_pose_fall_tcn.py -q
```

Expected: 전체 PASS.

```bash
rtk git add scripts/datasets/train_deepstream_pose_fall_tcn.py tests/test_train_deepstream_pose_fall_tcn.py
rtk git commit -m "feat: load DeepStream temporal fall datasets"
```

### Task 5: temporal-only 학습, 임계값 선택, checkpoint 계약

**Files:**
- Modify: `scripts/datasets/train_deepstream_pose_fall_tcn.py`
- Modify: `tests/test_train_deepstream_pose_fall_tcn.py`

**Interfaces:**
- Produces: `select_threshold(labels, probabilities, minimum_threshold=0.70) -> dict[str, Any]`
- Produces: `train_candidate(training, validation, config) -> tuple[dict[str, Any], dict[str, Any]]`
- Checkpoint model type: `deepstream_pose_temporal_tcn`

- [ ] **Step 1: 임계값 선택 실패 테스트를 작성한다**

0.70~0.95 후보 중 recall 75%, FPR 10% 기준을 만족하는 가장 낮은 값을
선택하고, 만족하는 값이 없으면 `passed=False`와 모든 sweep 결과를
반환하는지 검증한다.

- [ ] **Step 2: checkpoint 계약 실패 테스트를 작성한다**

작은 synthetic dataset을 2 epoch 학습해 다음 필드가 존재하는지 검증한다.

```python
assert checkpoint["format_version"] == 2
assert checkpoint["model_type"] == "deepstream_pose_temporal_tcn"
assert checkpoint["sequence_length"] == 48
assert checkpoint["frame_feature_names"] == list(FRAME_FEATURE_NAMES)
assert checkpoint["decision_threshold"] >= 0.70
assert "split_hash" in checkpoint
```

- [ ] **Step 3: 신규 테스트의 실패를 확인한다**

Run:

```bash
rtk pytest tests/test_train_deepstream_pose_fall_tcn.py -q
```

Expected: `select_threshold` 또는 `train_candidate` 부재로 FAIL.

- [ ] **Step 4: 기존 `FallTemporalTCN`을 사용한 최소 trainer를 구현한다**

- seed 고정
- group 기준 Training 내부 holdout
- `BCEWithLogitsLoss(pos_weight=negative/positive)`
- AdamW와 early stopping
- Validation은 학습 및 early stopping에 사용하지 않음
- 최종 metrics에 TP/TN/FP/FN, precision, recall, FPR, scene별 확률 저장

- [ ] **Step 5: CLI와 overwrite 보호를 추가한다**

```bash
rtk proxy python scripts/datasets/train_deepstream_pose_fall_tcn.py --help
```

CLI는 `--train-dataset`, `--validation-dataset`, `--output-model`,
`--metrics-json`, `--epochs`, `--device`를 제공한다. 기존 output이 있으면
`--overwrite` 없이는 중단한다.

- [ ] **Step 6: trainer 테스트를 검증하고 커밋한다**

Run:

```bash
rtk pytest tests/test_train_deepstream_pose_fall_tcn.py tests/test_fall_temporal_model.py -q
```

Expected: 전체 PASS.

```bash
rtk git add scripts/datasets/train_deepstream_pose_fall_tcn.py tests/test_train_deepstream_pose_fall_tcn.py
rtk git commit -m "feat: train DeepStream temporal fall TCN"
```

### Task 6: 전체 정적 검증과 1+1 capture smoke

**Files:**
- Modify only if a failing related test demonstrates a defect in Tasks 1-5.
- Create runtime artifacts under `data/fall_eval/`.

**Interfaces:**
- Consumes: evaluator capture CLI and schema v2 trainer
- Produces: one normal and one fall labeled temporal record set

- [ ] **Step 1: 관련 테스트 묶음을 실행한다**

Run:

```bash
rtk pytest tests/test_train_fall_temporal_tcn.py tests/test_falldata_aux.py tests/test_evaluate_sample_deepstream_replay.py tests/test_train_deepstream_pose_fall_tcn.py tests/test_fall_temporal_model.py -q
```

Expected: 0 failures.

- [ ] **Step 2: 전체 테스트를 실행한다**

Run:

```bash
rtk pytest tests/ -q
```

Expected: 0 failures. 관련 없는 기존 실패가 있으면 테스트명과 원인을 분리해
보고하고, Tasks 1-5의 관련 테스트가 통과한 증거를 유지한다.

- [ ] **Step 3: 1 정상 + 1 낙상 DeepStream replay capture를 수행한다**

기존 evaluator의 manifest/경로를 그대로 사용하고 결과를 다음 파일로
분리한다.

```bash
rtk proxy python scripts/ops/evaluate_sample_deepstream_replay.py \
  --label normal \
  --limit 1 \
  --feature-capture-log data/fall_eval/temporal_probe_smoke_capture.jsonl \
  --feature-dataset-jsonl data/fall_eval/temporal_probe_smoke_dataset.jsonl

rtk proxy python scripts/ops/evaluate_sample_deepstream_replay.py \
  --label fall \
  --limit 1 \
  --feature-capture-log data/fall_eval/temporal_probe_smoke_capture.jsonl \
  --feature-dataset-jsonl data/fall_eval/temporal_probe_smoke_dataset.jsonl
```

- [ ] **Step 4: smoke dataset 계약을 확인한다**

각 클래스가 한 행 이상이고 모든 schema v2 행에 48개 frame record와
Training/Validation metadata가 존재하는지 trainer loader로 확인한다.
`NO_RESULT`이면 정상 음성으로 계산하지 않고 capture trigger 경로를
진단한다.

- [ ] **Step 5: 코드 변경이 생긴 경우에만 해당 파일을 커밋한다**

Runtime JSONL과 모델 artifact는 커밋하지 않는다.

### Task 7: Validation 8+8 probe 학습과 관문 판정

**Files:**
- Create runtime artifacts:
  - `data/fall_eval/temporal_probe_train.jsonl`
  - `data/fall_eval/temporal_probe_validation.jsonl`
  - `models/falldata/deepstream_pose_temporal_tcn_probe.pt`
  - `data/fall_eval/deepstream_pose_temporal_tcn_probe_metrics.json`
- Source files are unchanged unless a test first demonstrates a defect.

**Interfaces:**
- Consumes: disjoint Training/Validation schema v2 JSONL
- Produces: probe checkpoint and metrics with `passed` gate

- [ ] **Step 1: Training capture를 장소와 라벨 기준으로 구성한다**

Training manifest에서 room/corridor가 섞이도록 낙상과 정상 scene을
수집한다. 동일 scene의 여러 window는 같은 `group_id`를 유지한다.

- [ ] **Step 2: 별도 Validation 8+8 capture를 생성한다**

Training과 `scene_id`, `group_id`가 겹치지 않는 낙상 8개와 정상 8개를
사용한다. evaluator의 `NO_RESULT`는 dataset 행 수에 포함하지 않는다.

- [ ] **Step 3: probe 후보를 학습한다**

Run:

```bash
rtk proxy python scripts/datasets/train_deepstream_pose_fall_tcn.py \
  --train-dataset data/fall_eval/temporal_probe_train.jsonl \
  --validation-dataset data/fall_eval/temporal_probe_validation.jsonl \
  --output-model models/falldata/deepstream_pose_temporal_tcn_probe.pt \
  --metrics-json data/fall_eval/deepstream_pose_temporal_tcn_probe_metrics.json \
  --epochs 100 \
  --device auto
```

- [ ] **Step 4: 고정 관문을 판정한다**

PASS 조건은 모두 만족해야 한다.

- 선택 임계값 `>= 0.70`
- 낙상 recall `>= 0.75`
- 정상 FPR `<= 0.10`
- Validation fall support `>= 8`
- Validation normal support `>= 8`
- Training/Validation scene 및 group 교집합 0

- [ ] **Step 5: 결과에 따라 다음 행동을 하나만 선택한다**

- PASS: 20+20 확대 설계를 작성한다.
- FAIL: FP/FN scene, 장소, frame sequence, 라벨 정렬을 분석하고 전체 데이터
  학습은 시작하지 않는다.

Runtime artifact는 운영 모델 경로에 복사하지 않으며 운영 compose 설정도
변경하지 않는다.
