# DeepStream Inline Pose RF Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan.

**Goal:** DeepStream에서 실제 생성되는 pose 요약 특징으로 RF 후보 모델을 다시 학습하고, 독립 Validation 40+40에서 임계값 0.7 기준 낙상 재현율 90% 이상·비낙상 오탐률 5% 이하인지 검증한다.

**Architecture:** 운영 추론 경로의 `_summarize_frames()` 직후 특징 벡터를 선택적으로 별도 JSONL에 기록한다. 재생 평가기는 각 캡처에 manifest의 label/group 메타데이터를 결합하고, 전용 학습기는 그룹 단위 분할로 후보 RF bundle을 생성한다. 기존 모델과 운영 설정은 덮어쓰지 않으며 후보 모델은 shadow 모드에서만 독립 Validation 데이터로 평가한다.

**Tech Stack:** Python 3.10, pytest, scikit-learn RandomForest, joblib, DeepStream, Docker Compose, JSONL

## Global Constraints

- 기존 알람 판정·발행 동작은 변경하지 않는다.
- 특징 캡처는 `FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH`가 비어 있으면 완전히 비활성화한다.
- 캡처 파일 쓰기 실패는 추론을 중단하지 않는 fail-open 동작으로 처리한다.
- 학습용 Training 40+40과 최종 평가용 Validation 40+40의 `group_id`/`scene_id` 중복을 허용하지 않는다.
- 후보 모델의 운영 임계값은 `0.7`로 고정한다.
- 기존 모델 파일을 덮어쓰지 않는다.
- 최종 평가 전까지 후보 모델을 publish/veto 경로에 연결하지 않는다.
- Jetson에서 별도 pose 추론 프로세스를 만들지 않고 기존 DeepStream pose 결과만 사용한다.
- 각 구현 단계는 실패 테스트 작성 → 최소 구현 → 관련 테스트 통과 순서로 진행한다.

---

## Task 1: DeepStream inline 특징 캡처 설정과 writer 추가

**Files:**

- Modify: `src/core/ai/_falldata_aux.py`
- Modify: `tests/test_falldata_aux.py`

### Step 1: 환경 설정 파싱 실패 테스트 작성

`tests/test_falldata_aux.py`에 다음 검증을 추가한다.

```python
def test_config_reads_optional_inline_feature_capture_path(monkeypatch, tmp_path):
    capture_path = tmp_path / "inline-features.jsonl"
    monkeypatch.setenv(
        "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH",
        str(capture_path),
    )

    config = FallDataAuxConfig.from_env()

    assert config.inline_feature_capture_path == capture_path


def test_config_disables_inline_feature_capture_when_env_is_blank(monkeypatch):
    monkeypatch.setenv("FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH", "  ")

    config = FallDataAuxConfig.from_env()

    assert config.inline_feature_capture_path is None
```

### Step 2: 설정 테스트 실패 확인

Run:

```bash
rtk pytest tests/test_falldata_aux.py -k inline_feature_capture_path
```

Expected: `FallDataAuxConfig`에 필드가 없어 실패.

### Step 3: 선택적 경로 설정 구현

`FallDataAuxConfig`에 다음 필드를 추가한다.

```python
inline_feature_capture_path: Path | None = None
```

`from_env()`에서 공백 제거 후 값이 있을 때만 `Path`로 만든다.

```python
capture_path_text = os.getenv(
    "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH",
    "",
).strip()
inline_feature_capture_path = (
    Path(capture_path_text) if capture_path_text else None
)
```

### Step 4: writer 동작 실패 테스트 작성

같은 테스트 파일에 다음 세 경우를 추가한다.

```python
def test_inline_feature_capture_writes_exact_summary_vector(tmp_path):
    capture_path = tmp_path / "inline-features.jsonl"
    verifier = _build_verifier(
        inline_feature_capture_path=capture_path,
    )
    summary = {
        "frames_seen": 12,
        "frames_with_pose": 10,
        "feature_names": ["torso_angle_mean", "hip_speed_max"],
        "feature_vector": [41.5, 0.82],
        "reason_counts": {},
        "frame_records": [],
    }

    status = verifier._write_inline_feature_capture(
        "camera-1",
        summary,
        window_seconds=3.0,
    )

    record = json.loads(capture_path.read_text(encoding="utf-8"))
    assert status == "written"
    assert record["schema_version"] == 1
    assert record["runtime"] == "deepstream_pose_inline"
    assert record["camera_id"] == "camera-1"
    assert record["feature_names"] == summary["feature_names"]
    assert record["feature_vector"] == summary["feature_vector"]


def test_inline_feature_capture_is_noop_when_disabled():
    verifier = _build_verifier(inline_feature_capture_path=None)

    status = verifier._write_inline_feature_capture(
        "camera-1",
        _valid_summary(),
        window_seconds=3.0,
    )

    assert status is None


def test_inline_feature_capture_failure_is_fail_open(tmp_path):
    verifier = _build_verifier(
        inline_feature_capture_path=tmp_path,
    )

    status = verifier._write_inline_feature_capture(
        "camera-1",
        _valid_summary(),
        window_seconds=3.0,
    )

    assert status == "error"
```

### Step 5: JSONL writer 최소 구현

`FallDataAuxVerifier`에 append 충돌을 막을 lock과 writer를 추가한다.

```python
self._inline_feature_capture_lock = threading.Lock()
```

```python
def _write_inline_feature_capture(
    self,
    camera_name: str,
    summary: dict[str, Any],
    *,
    window_seconds: float,
) -> str | None:
    path = self.config.inline_feature_capture_path
    if path is None:
        return None

    record = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "deepstream_pose_inline",
        "camera_id": camera_name,
        "window_seconds": float(window_seconds),
        "frames_seen": int(summary["frames_seen"]),
        "frames_with_pose": int(summary["frames_with_pose"]),
        "sampled_frames": len(summary.get("frame_records", [])),
        "feature_names": list(summary["feature_names"]),
        "feature_vector": list(summary["feature_vector"]),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._inline_feature_capture_lock:
            with path.open("a", encoding="utf-8") as fp:
                fp.write(line)
        return "written"
    except OSError:
        logger.exception("DeepStream inline pose feature capture failed")
        return "error"
```

### Step 6: 실제 inline 요약 지점에 writer 연결

`_verify_inline_pose_rf()`에서 아래 순서를 보장한다.

```python
summary = _summarize_frames(selected_records, max_frames)
capture_status = self._write_inline_feature_capture(
    camera_name,
    summary,
    window_seconds=self.config.candidate_window_seconds,
)
```

캡처는 RF bundle의 `feature_names` 선택 전 수행한다. 그래야 현재 모델과 특징 목록이 다르더라도 DeepStream 원본 요약 특징이 보존된다. 캡처가 활성화된 경우에만 진단 결과에 `feature_capture_status`를 포함한다.

### Step 7: 관련 테스트 실행

Run:

```bash
rtk pytest tests/test_falldata_aux.py
```

Expected: 전체 통과.

### Step 8: 커밋

```bash
rtk git add src/core/ai/_falldata_aux.py tests/test_falldata_aux.py
rtk git commit -m "feat: capture DeepStream inline pose features"
```

---

## Task 2: 재생 평가기에 캡처 라벨링과 안전한 컨테이너 재생성 추가

**Files:**

- Modify: `scripts/ops/evaluate_sample_deepstream_replay.py`
- Modify: `tests/test_evaluate_sample_deepstream_replay.py`

### Step 1: 캡처 레코드 검증·라벨링 실패 테스트 작성

```python
def test_label_feature_capture_records_adds_manifest_metadata():
    capture = {
        "schema_version": 1,
        "runtime": "deepstream_pose_inline",
        "feature_names": ["a", "b"],
        "feature_vector": [1.0, 2.0],
    }
    manifest_row = {
        "video_path": "/dataset/scene-001.mp4",
        "scene_id": "scene-001",
        "group_id": "subject-001",
        "is_fall": True,
    }

    labeled, errors = _label_feature_capture_records(
        [capture],
        manifest_row,
    )

    assert errors == []
    assert labeled[0]["label"] == 1
    assert labeled[0]["is_fall"] is True
    assert labeled[0]["scene_id"] == "scene-001"
    assert labeled[0]["group_id"] == "subject-001"
    assert labeled[0]["video_path"] == manifest_row["video_path"]


@pytest.mark.parametrize(
    "capture",
    [
        {"schema_version": 2, "runtime": "deepstream_pose_inline",
         "feature_names": ["a"], "feature_vector": [1.0]},
        {"schema_version": 1, "runtime": "offline",
         "feature_names": ["a"], "feature_vector": [1.0]},
        {"schema_version": 1, "runtime": "deepstream_pose_inline",
         "feature_names": ["a", "b"], "feature_vector": [1.0]},
    ],
)
def test_label_feature_capture_records_rejects_invalid_records(capture):
    labeled, errors = _label_feature_capture_records(
        [capture],
        _manifest_row(is_fall=False),
    )

    assert labeled == []
    assert errors
```

### Step 2: 실패 확인

Run:

```bash
rtk pytest tests/test_evaluate_sample_deepstream_replay.py \
  -k "label_feature_capture_records"
```

Expected: helper가 없어 실패.

### Step 3: 캡처 라벨링 helper 구현

```python
def _label_feature_capture_records(
    records: list[dict[str, Any]],
    manifest_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    labeled: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        names = record.get("feature_names")
        vector = record.get("feature_vector")
        if record.get("schema_version") != 1:
            errors.append(f"record {index}: unsupported schema_version")
            continue
        if record.get("runtime") != "deepstream_pose_inline":
            errors.append(f"record {index}: unexpected runtime")
            continue
        if not isinstance(names, list) or not isinstance(vector, list):
            errors.append(f"record {index}: features must be lists")
            continue
        if len(names) != len(vector):
            errors.append(f"record {index}: feature length mismatch")
            continue

        output = dict(record)
        output.update(
            {
                "label": 1 if bool(manifest_row["is_fall"]) else 0,
                "is_fall": bool(manifest_row["is_fall"]),
                "scene_id": str(manifest_row["scene_id"]),
                "group_id": str(manifest_row["group_id"]),
                "video_path": str(manifest_row["video_path"]),
            }
        )
        labeled.append(output)
    return labeled, errors
```

### Step 4: 환경 override를 쓰는 컨테이너 재생성 테스트 작성

기존 `_restart_ai_engine()`의 단순 restart 동작은 유지하고, 최초 설정 적용·최종 복구에만 다음 helper를 사용한다.

```python
def test_recreate_ai_engine_passes_scoped_environment(
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    _recreate_ai_engine(
        tmp_path / "docker-compose.jetson.yml",
        tmp_path / ".env",
        environment_overrides={
            "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH":
                "/app/data/fall_eval/capture.jsonl",
            "FALLDATA_AUX_COMPARE_MODEL_PATH":
                "/app/models/candidate.joblib",
        },
    )

    command, kwargs = calls[0]
    assert command[-4:] == [
        "up", "-d", "--force-recreate", "cctv-ai-engine"
    ]
    assert kwargs["env"]["FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH"].endswith(
        "capture.jsonl"
    )
```

### Step 5: 안전한 recreate helper 구현

```python
def _recreate_ai_engine(
    compose_file: Path,
    compose_env_file: Path | None = None,
    *,
    environment_overrides: dict[str, str] | None = None,
) -> None:
    command = ["docker", "compose"]
    if compose_env_file is not None:
        command.extend(["--env-file", str(compose_env_file)])
    command.extend(
        ["-f", str(compose_file), "up", "-d", "--force-recreate",
         "cctv-ai-engine"]
    )
    environment = os.environ.copy()
    environment.update(environment_overrides or {})
    subprocess.run(command, check=True, env=environment)
```

### Step 6: CLI와 per-video 캡처 수집 구현

`main()`에 다음 선택 인자를 추가한다.

```python
parser.add_argument("--feature-capture-log", type=Path)
parser.add_argument("--feature-dataset-jsonl", type=Path)
parser.add_argument("--runtime-compare-model-path", type=Path)
```

규칙:

- `--feature-capture-log`와 `--feature-dataset-jsonl`은 함께 지정한다.
- 두 경로는 프로젝트 root 내부여야 하며 container 경로 `/app/<relative-path>`로 변환한다.
- 실행 시작 전에 기존 capture log의 byte offset을 저장한다.
- 각 영상 재생 뒤 `_read_new_jsonl_records(path, offset)`으로 새 레코드만 읽는다.
- 새 레코드에 현재 manifest row를 결합해 `--feature-dataset-jsonl`에 쓴다.
- 영상당 캡처 레코드가 0개이면 결과를 `NO_RESULT`로 기록하고 이유를 남긴다.
- invalid 캡처는 조용히 버리지 않고 결과의 `feature_capture_errors`에 남긴다.
- `--runtime-compare-model-path`는 프로젝트 내부 후보 모델만 허용하고 container 경로로 변환한다.
- 최초 한 번 `_recreate_ai_engine()`에 `environment_overrides`를 전달해 capture/model 설정을 적용한다.
- 영상 사이에는 기존 `_restart_ai_engine()`을 사용한다.
- `finally`에서 환경 override 없이 `_recreate_ai_engine()`을 호출하여 원래 compose/env 설정으로 복구한다.
- 기존 `--restore-camera-config` 복구도 같은 `finally`에서 유지한다.

### Step 7: 복구 동작 테스트 작성

평가 도중 replay가 예외를 발생시키는 테스트를 만들고 다음 호출 순서를 확인한다.

```python
assert recreate_calls[0]["environment_overrides"] == {
    "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH":
        "/app/data/fall_eval/capture.jsonl",
}
assert recreate_calls[-1]["environment_overrides"] is None
assert original_camera_config_was_restored
```

### Step 8: 평가기 전체 테스트 실행

Run:

```bash
rtk pytest tests/test_evaluate_sample_deepstream_replay.py
```

Expected: 전체 통과.

### Step 9: 커밋

```bash
rtk git add scripts/ops/evaluate_sample_deepstream_replay.py \
  tests/test_evaluate_sample_deepstream_replay.py
rtk git commit -m "feat: label DeepStream replay feature captures"
```

---

## Task 3: DeepStream 캡처 전용 RF 학습기 추가

**Files:**

- Create: `scripts/datasets/train_deepstream_pose_fall_rf.py`
- Create: `tests/test_train_deepstream_pose_fall_rf.py`

### Step 1: dataset loader 실패 테스트 작성

```python
def test_load_capture_datasets_combines_files_with_stable_features(tmp_path):
    normal_path = _write_capture_dataset(
        tmp_path / "normal.jsonl",
        label=0,
        group_id="normal-group",
        feature_names=["a", "b"],
        vectors=[[0.1, 0.2], [0.2, 0.3]],
    )
    fall_path = _write_capture_dataset(
        tmp_path / "fall.jsonl",
        label=1,
        group_id="fall-group",
        feature_names=["a", "b"],
        vectors=[[0.8, 0.9], [0.9, 1.0]],
    )

    dataset = load_capture_datasets([normal_path, fall_path])

    assert dataset.feature_names == ["a", "b"]
    assert dataset.x.shape == (4, 2)
    assert dataset.y.tolist() == [0, 0, 1, 1]
    assert set(dataset.groups.tolist()) == {
        "normal-group", "fall-group"
    }
```

다음 거부 테스트도 추가한다.

- schema version 불일치
- `runtime != deepstream_pose_inline`
- 파일 간 `feature_names` 순서 불일치
- vector 길이 불일치
- NaN/Inf 포함
- 한 클래스만 존재
- `group_id` 또는 `scene_id` 누락

### Step 2: loader 테스트 실패 확인

Run:

```bash
rtk pytest tests/test_train_deepstream_pose_fall_rf.py \
  -k "load_capture"
```

Expected: 신규 module이 없어 실패.

### Step 3: 엄격한 dataset loader 구현

명시적 구조체를 사용한다.

```python
@dataclass(frozen=True)
class CaptureDataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    scene_ids: tuple[str, ...]
    feature_names: list[str]
    source_paths: tuple[Path, ...]
```

`load_capture_datasets(paths: Sequence[Path]) -> CaptureDataset`는 입력 파일을
순서대로 읽고, 첫 레코드의 `feature_names`를 기준 schema로 고정한 뒤 모든
레코드를 검증해 `np.ndarray`로 변환한다. 빈 파일이나 빈 입력 목록도
`ValueError`로 거부한다.

모든 숫자는 `float`로 변환한 뒤 `np.isfinite()`로 검사한다. 동일 group이 여러 행에 반복되는 것은 허용하지만 동일 group에서 label이 섞이면 실패시킨다.

### Step 4: 데이터 누수 방지 테스트 작성

```python
def test_assert_validation_disjoint_rejects_group_overlap():
    with pytest.raises(ValueError, match="group overlap"):
        assert_validation_disjoint(
            training_groups={"subject-001"},
            training_scene_ids={"scene-001"},
            validation_rows=[
                {
                    "group_id": "subject-001",
                    "scene_id": "scene-900",
                }
            ],
        )


def test_assert_validation_disjoint_rejects_scene_overlap():
    with pytest.raises(ValueError, match="scene overlap"):
        assert_validation_disjoint(
            training_groups={"subject-001"},
            training_scene_ids={"scene-001"},
            validation_rows=[
                {
                    "group_id": "subject-900",
                    "scene_id": "scene-001",
                }
            ],
        )
```

### Step 5: 그룹 분할·RF bundle 생성 테스트 작성

```python
def test_train_candidate_uses_group_split_and_threshold_point_seven():
    dataset = _balanced_grouped_dataset()

    bundle, metrics = train_candidate(
        dataset,
        random_state=42,
        validation_fraction=0.25,
    )

    assert bundle["schema_version"] == 1
    assert bundle["feature_source"] == "deepstream_pose_inline"
    assert bundle["feature_names"] == dataset.feature_names
    assert bundle["decision_threshold"] == 0.7
    assert bundle["inference_config"]["max_frames"] == 48
    assert bundle["inference_config"]["candidate_window_seconds"] == 3.0
    assert set(metrics["train_groups"]).isdisjoint(
        metrics["holdout_groups"]
    )
```

### Step 6: 학습기 최소 구현

다음 고정값으로 시작한다.

```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=random_state,
    n_jobs=-1,
)
```

분할은 `GroupShuffleSplit`을 사용한다. 여러 캡처가 같은 영상/사람에서 나와도 train과 holdout에 동시에 들어가지 않게 `group_id`를 기준으로 나눈다.

bundle 형식:

```python
bundle = {
    "schema_version": 1,
    "model": model,
    "feature_source": "deepstream_pose_inline",
    "feature_names": dataset.feature_names,
    "decision_threshold": 0.7,
    "training_config": {
        "random_state": random_state,
        "validation_fraction": validation_fraction,
        "class_weight": "balanced",
    },
    "inference_config": {
        "max_frames": 48,
        "candidate_window_seconds": 3.0,
    },
    "dataset_summary": {
        "rows": int(dataset.x.shape[0]),
        "groups": len(set(dataset.groups.tolist())),
        "source_paths": [str(path) for path in dataset.source_paths],
    },
}
```

metrics JSON에는 threshold 0.7 기준 confusion matrix, precision, recall, FPR, ROC-AUC, train/holdout group 목록을 저장한다.

### Step 7: CLI 구현

```text
--dataset PATH              반복 가능, 최소 1개
--validation-manifest PATH  독립 Validation과 group/scene 중복 검사
--output-model PATH         기존 파일이면 실패
--output-metrics PATH       기존 파일이면 실패
--random-state INT          기본 42
--validation-fraction FLOAT 기본 0.25
```

`--output-model` 또는 `--output-metrics`가 이미 존재하면 `--overwrite` 없이는 실패하게 한다. 운영 모델 경로를 실수로 덮어쓰지 않는 것이 목적이다.

### Step 8: 학습기 테스트 실행

Run:

```bash
rtk pytest tests/test_train_deepstream_pose_fall_rf.py
```

Expected: 전체 통과.

### Step 9: 커밋

```bash
rtk git add scripts/datasets/train_deepstream_pose_fall_rf.py \
  tests/test_train_deepstream_pose_fall_rf.py
rtk git commit -m "feat: train RF from DeepStream pose captures"
```

---

## Task 4: Jetson Compose 환경 전달과 정적 검증 추가

**Files:**

- Modify: `docker-compose.jetson.yml`
- Modify: `scripts/health/check_compose_runtime_assumptions.py`
- Modify: `tests/test_check_compose_runtime_assumptions.py`

### Step 1: Compose wiring 실패 테스트 작성

`test_falldata_aux_wiring_requires_fail_open_and_jetson_paths`의 fixture/검증을 확장해 다음 환경 전달이 없으면 실패하게 한다.

```yaml
FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH: ${FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH:-}
```

테스트는 capture 기본값이 빈 문자열인 것도 확인한다. 기본 실행에서 파일 I/O가 생기면 안 된다.

### Step 2: 실패 확인

Run:

```bash
rtk pytest tests/test_check_compose_runtime_assumptions.py \
  -k falldata_aux_wiring
```

Expected: 신규 환경 변수가 없어 실패.

### Step 3: Compose와 health check 구현

`cctv-ai-engine.environment`의 기존 `FALLDATA_AUX_INLINE_POSE_RF` 주변에 다음 한 줄을 추가한다.

```yaml
FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH: ${FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH:-}
```

`check_falldata_aux_wiring()`은 변수의 존재와 빈 기본값을 검사한다. 경로 자체는 실험 실행 때 평가기가 project-relative 경로로 제한한다.

### Step 4: 정적 검증 실행

Run:

```bash
rtk pytest tests/test_check_compose_runtime_assumptions.py \
  -k falldata_aux_wiring
rtk proxy .venv/bin/python -m py_compile \
  scripts/ops/evaluate_sample_deepstream_replay.py \
  scripts/datasets/train_deepstream_pose_fall_rf.py
rtk git diff --check
```

Expected: 모두 통과.

### Step 5: 커밋

```bash
rtk git add docker-compose.jetson.yml \
  scripts/health/check_compose_runtime_assumptions.py \
  tests/test_check_compose_runtime_assumptions.py
rtk git commit -m "chore: wire optional DeepStream feature capture"
```

---

## Task 5: 전체 회귀 테스트와 1+1 캡처 smoke 검증

**Files:**

- No production file changes expected
- Generate:
  - `data/fall_eval/deepstream_capture_smoke_notfall.jsonl`
  - `data/fall_eval/deepstream_capture_smoke_fall.jsonl`
  - `data/fall_eval/deepstream_dataset_smoke_notfall.jsonl`
  - `data/fall_eval/deepstream_dataset_smoke_fall.jsonl`

### Step 1: 관련 전체 테스트 실행

```bash
rtk pytest tests/test_falldata_aux.py \
  tests/test_evaluate_sample_deepstream_replay.py \
  tests/test_train_deepstream_pose_fall_rf.py \
  tests/test_check_compose_runtime_assumptions.py
```

Expected: 전체 통과.

### Step 2: 비낙상 Training 1개 캡처

```bash
rtk proxy .venv/bin/python \
  scripts/ops/evaluate_sample_deepstream_replay.py \
  --manifest data/fall_eval/open_fall_train_cam2_container_manifest.jsonl \
  --label notfall \
  --max-videos 1 \
  --feature-capture-log \
    data/fall_eval/deepstream_capture_smoke_notfall.jsonl \
  --feature-dataset-jsonl \
    data/fall_eval/deepstream_dataset_smoke_notfall.jsonl \
  --results-jsonl \
    data/fall_eval/deepstream_capture_smoke_notfall_results.jsonl \
  --results-csv \
    data/fall_eval/deepstream_capture_smoke_notfall_results.csv
```

### Step 3: 낙상 Training 1개 캡처

```bash
rtk proxy .venv/bin/python \
  scripts/ops/evaluate_sample_deepstream_replay.py \
  --manifest data/fall_eval/open_fall_train_cam2_container_manifest.jsonl \
  --label fall \
  --max-videos 1 \
  --feature-capture-log \
    data/fall_eval/deepstream_capture_smoke_fall.jsonl \
  --feature-dataset-jsonl \
    data/fall_eval/deepstream_dataset_smoke_fall.jsonl \
  --results-jsonl \
    data/fall_eval/deepstream_capture_smoke_fall_results.jsonl \
  --results-csv \
    data/fall_eval/deepstream_capture_smoke_fall_results.csv
```

### Step 4: smoke 산출물 검증

두 labeled dataset 모두에 최소 1개 유효 레코드가 있어야 한다. 다음을 확인한다.

- `runtime == "deepstream_pose_inline"`
- `schema_version == 1`
- `len(feature_names) == len(feature_vector)`
- 모든 feature 값이 finite
- notfall의 `label == 0`
- fall의 `label == 1`
- 두 파일의 `feature_names` 순서가 동일

### Step 5: 운영 복구 확인

```bash
rtk docker compose --env-file .env \
  -f docker-compose.jetson.yml ps cctv-ai-engine
rtk docker inspect cctv-ai-engine
rtk ps -ef
```

확인 기준:

- `cctv-ai-engine`가 healthy
- container 환경에 smoke capture 경로가 남아 있지 않음
- 기존 shadow 설정이 유지됨
- 별도 YOLO/pose Python 추론 child process가 없음

---

## Task 6: Training 40+40 캡처와 후보 RF 학습

**Files:**

- Generate:
  - `data/fall_eval/deepstream_capture_train_notfall40_20260729.jsonl`
  - `data/fall_eval/deepstream_capture_train_fall40_20260729.jsonl`
  - `data/fall_eval/deepstream_dataset_train_notfall40_20260729.jsonl`
  - `data/fall_eval/deepstream_dataset_train_fall40_20260729.jsonl`
  - `models/falldata/deepstream_pose_rf_candidate_20260729.joblib`
  - `data/fall_eval/deepstream_pose_rf_candidate_train_metrics_20260729.json`

### Step 1: Training 비낙상 40개 캡처

Task 5의 비낙상 명령에서 `--max-videos 40`과 Training 산출물 경로를 사용한다.

### Step 2: Training 낙상 40개 캡처

Task 5의 낙상 명령에서 `--max-videos 40`과 Training 산출물 경로를 사용한다.

### Step 3: 캡처 품질 gate

학습 전에 다음 조건을 만족하지 않으면 중단한다.

- 각 label에서 유효 영상 36개 이상(90%)
- 두 label 모두 feature schema 동일
- group/scene 메타데이터 누락 0개
- non-finite feature 0개
- Training과 Validation manifest의 group/scene 중복 0개

유효 캡처가 36개 미만이면 모델을 억지로 학습하지 않고, `NO_RESULT` 영상의 pose event/RTSP/replay 타이밍부터 조사한다.

### Step 4: 후보 RF 학습

```bash
rtk proxy .venv/bin/python \
  scripts/datasets/train_deepstream_pose_fall_rf.py \
  --dataset \
    data/fall_eval/deepstream_dataset_train_notfall40_20260729.jsonl \
  --dataset \
    data/fall_eval/deepstream_dataset_train_fall40_20260729.jsonl \
  --validation-manifest \
    data/fall_eval/open_fall_val_cam2_container_manifest.jsonl \
  --output-model \
    models/falldata/deepstream_pose_rf_candidate_20260729.joblib \
  --output-metrics \
    data/fall_eval/deepstream_pose_rf_candidate_train_metrics_20260729.json
```

### Step 5: 학습 산출물 검증

다음을 확인한다.

- bundle `feature_source == deepstream_pose_inline`
- `decision_threshold == 0.7`
- `max_frames == 48`
- Training 내부 holdout group 중복 없음
- class별 표본 수와 group 수가 metrics에 기록됨
- 기존 운영/비교 모델 파일의 hash와 수정 시간이 바뀌지 않음

Training 내부 holdout 지표는 모델 선택 참고값일 뿐, 실사용 가능 판정에는 사용하지 않는다.

---

## Task 7: 독립 Validation 40+40 shadow 평가와 수용 판정

**Files:**

- Generate:
  - `data/fall_eval/deepstream_candidate_val_notfall40_20260729.jsonl`
  - `data/fall_eval/deepstream_candidate_val_fall40_20260729.jsonl`
  - `data/fall_eval/deepstream_candidate_validation_summary_20260729.json`

### Step 1: 후보 모델 경로 호환성 사전 점검

후보 bundle을 현재 `_verify_inline_pose_rf()`가 읽을 수 있는지 단위 로딩으로 확인한다. `feature_names`, `model.predict_proba`, threshold 타입이 기존 계약과 일치해야 한다.

### Step 2: Validation 비낙상 40개 평가

```bash
rtk proxy .venv/bin/python \
  scripts/ops/evaluate_sample_deepstream_replay.py \
  --manifest data/fall_eval/open_fall_val_cam2_container_manifest.jsonl \
  --label notfall \
  --max-videos 40 \
  --score-source inline_pose_rf \
  --runtime-compare-model-path \
    models/falldata/deepstream_pose_rf_candidate_20260729.joblib \
  --results-jsonl \
    data/fall_eval/deepstream_candidate_val_notfall40_20260729.jsonl \
  --results-csv \
    data/fall_eval/deepstream_candidate_val_notfall40_20260729.csv
```

### Step 3: Validation 낙상 40개 평가

```bash
rtk proxy .venv/bin/python \
  scripts/ops/evaluate_sample_deepstream_replay.py \
  --manifest data/fall_eval/open_fall_val_cam2_container_manifest.jsonl \
  --label fall \
  --max-videos 40 \
  --score-source inline_pose_rf \
  --runtime-compare-model-path \
    models/falldata/deepstream_pose_rf_candidate_20260729.joblib \
  --results-jsonl \
    data/fall_eval/deepstream_candidate_val_fall40_20260729.jsonl \
  --results-csv \
    data/fall_eval/deepstream_candidate_val_fall40_20260729.csv
```

### Step 4: 수용 지표 계산

`NO_RESULT`는 성능에서 제외해 유리하게 만들지 않고 실패 표본으로 함께 보고한다.

```text
recall = TP / (TP + FN + fall_no_result)
FPR    = FP / (TN + FP)
coverage = evaluated / requested
```

`NO_RESULT`가 FPR을 인위적으로 낮추지 않도록 FPR과 coverage를 반드시 함께
판정한다. 낙상 `NO_RESULT`는 보수적으로 FN과 동일하게 recall 분모에 넣는다.

수용 조건:

- 임계값 0.7
- 낙상 recall ≥ 90%
- 비낙상 FPR ≤ 5%
- 각 class coverage ≥ 95%
- 추론 예외 0건
- 별도 pose 추론 child process 0개

### Step 5: 결과 분기

모든 조건 통과:

- 후보 모델은 “shadow 검증 통과”로만 표시한다.
- 운영 publish/veto 연결은 별도의 승인 작업으로 남긴다.

하나라도 실패:

- 기존 모델과 운영 설정을 그대로 유지한다.
- confusion matrix와 실패 scene 목록을 저장한다.
- threshold를 0.7 아래로 낮춰 통과 처리하지 않는다.
- FN이 많으면 특징 분리도/pose 품질을, FP가 많으면 앉기·눕기 hard negative를 우선 분석한다.

### Step 6: 최종 운영 복구 확인

```bash
rtk docker compose --env-file .env \
  -f docker-compose.jetson.yml ps cctv-ai-engine
rtk docker inspect cctv-ai-engine
rtk ps -ef
rtk git status --short
```

최종 보고에는 다음을 구분해 기록한다.

- 검증된 사실: test 결과, TP/TN/FP/FN/NO_RESULT, recall/FPR/coverage, runtime 상태
- 아직 검증되지 않은 사항: 실제 CCTV 장시간 운용 성능, 다른 카메라 각도/환경 일반화
- 변경 파일과 생성 산출물
- 기존 모델 미변경 여부
