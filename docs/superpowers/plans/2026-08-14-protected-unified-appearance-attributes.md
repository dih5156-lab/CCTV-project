# Protected Unified Appearance Attributes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 색상 성능을 보호하면서 성별, 상·하의 색상과 종류, 가방·백팩·모자·마스크를 분류하고 추적 ID당 대표 이미지 1장을 검색 가능한 기록으로 남기는 1차 통합 외형 모델을 구축한다.

**Architecture:** AI-Hub K-ReID 고해상도 PNG/XML에서 사람 ID 분리 및 작업별 unknown mask가 포함된 manifest를 만든다. ResNet50 공유 백본 위에 그룹별 독립 헤드를 두고 11색 체크포인트의 백본·색상 행을 이식한 뒤 보호 학습과 공동 미세조정을 수행한다. 운영에서는 기존 검출·추적 결과를 0.5초 간격으로 분석하고, 색상 기준을 통과한 후보만 ONNX/TensorRT shadow 경로를 거쳐 승격한다.

**Tech Stack:** Python 3.10, PyTorch/torchvision, NumPy 1.x training environment, OpenCV/Pillow, ONNX, TensorRT/DeepStream, FastAPI, SQLite, pytest

## Global Constraints

- 운영 PA100K TensorRT, HSV/LAB, 기존 색상 엔진은 후보 승격 전까지 변경하지 않는다.
- 색상은 `black, white, gray, red, blue, green, yellow, brown, purple, navy, orange` 11종으로 고정한다.
- 1차 소지품은 `bag, backpack, hat, mask`만 포함한다.
- 작은 물건, Re-ID, 낙상 파이프라인은 수정하지 않는다.
- `defined_*_color=false`와 미노출 부위는 음성 라벨이 아니라 loss mask 0으로 처리한다.
- train/validation/test는 person ID가 겹치지 않아야 한다.
- 실시간 화면에는 추가 bbox와 속성 텍스트를 표시하지 않는다.
- 추적 ID당 최종 대표 이미지는 정확히 1장만 남긴다.
- 통합 색상 macro F1이 동일 조건 전용 색상 모델보다 0.02 초과 하락하거나 개별 색상 F1이 0.05 초과 하락하면 통합 색상 승격을 금지한다.
- 검증 표본 30개 미만 클래스는 통과가 아니라 평가 보류로 처리한다.
- 모든 shell 명령은 프로젝트 규칙에 따라 `rtk`를 접두어로 사용한다.
- 새 외부 라이브러리를 추가하지 않는다.

---

## Milestone A: 재현 가능한 데이터와 보호형 통합 모델

### Task 1: 통합 속성 스키마와 라벨 인코딩

**Files:**
- Create: `scripts/datasets/appearance_multitask_schema.py`
- Create: `tests/test_appearance_multitask_schema.py`

**Interfaces:**
- Produces: `HEAD_SPECS: tuple[HeadSpec, ...]`
- Produces: `encode_annotation(row: Mapping[str, object]) -> EncodedTargets`
- Produces: `build_label_map() -> dict[str, object]`
- `EncodedTargets.values`와 `EncodedTargets.masks`는 head 이름별 `np.ndarray`를 반환한다.

- [ ] **Step 1: 실패하는 스키마 테스트 작성**

```python
def test_encode_annotation_masks_undefined_lower_color():
    encoded = encode_annotation({
        "gender": "female",
        "upper_color": "orange",
        "upper_color_defined": True,
        "lower_color": "black",
        "lower_color_defined": False,
        "upper_clothes": "long_sleeve",
        "lower_clothes": "long_pants",
        "items": ["bag", "mask"],
    })
    assert encoded.values["upper_color"].item() == UPPER_COLORS.index("orange")
    assert encoded.masks["upper_color"].item() == 1.0
    assert encoded.masks["lower_color"].item() == 0.0
    assert encoded.values["items"].tolist() == [1.0, 0.0, 0.0, 1.0]
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_appearance_multitask_schema.py -q`

Expected: `ModuleNotFoundError: scripts.datasets.appearance_multitask_schema`

- [ ] **Step 3: 최소 스키마 구현**

```python
@dataclass(frozen=True)
class HeadSpec:
    name: str
    classes: tuple[str, ...]
    multilabel: bool = False


@dataclass(frozen=True)
class EncodedTargets:
    values: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]


UPPER_COLORS = ("black", "white", "gray", "red", "blue", "green", "yellow", "brown", "purple", "navy", "orange")
LOWER_COLORS = UPPER_COLORS
UPPER_CLOTHES = ("long_sleeve", "short_sleeve", "sleeveless")
LOWER_CLOTHES = ("long_pants", "short_pants", "long_skirt", "short_skirt", "dress")
ITEMS = ("bag", "backpack", "hat", "mask")
```

`gender`, `upper_color`, `lower_color`, `upper_clothes`, `lower_clothes`는 정수 class index와 scalar mask를 만들고 `items`는 길이 4의 float vector와 길이 4의 mask를 만든다. `build_label_map()`은 ONNX flat output의 index, field, value, threshold와 head slice를 모두 기록한다.

- [ ] **Step 4: GREEN 확인**

Run: `rtk pytest tests/test_appearance_multitask_schema.py -q`

Expected: PASS

- [ ] **Step 5: 커밋**

```bash
rtk git add scripts/datasets/appearance_multitask_schema.py tests/test_appearance_multitask_schema.py
rtk git commit -m "feat: define appearance multitask schema"
```

### Task 2: AI-Hub 고해상도 multitask manifest 생성

**Files:**
- Create: `scripts/datasets/prepare_aihub_kreid_multitask.py`
- Create: `tests/test_prepare_aihub_kreid_multitask.py`
- Reuse: `scripts/datasets/appearance_multitask_schema.py`

**Interfaces:**
- Consumes: `encode_annotation()`와 스키마 클래스 목록
- Produces: `parse_annotation(xml_bytes: bytes, image_member: str) -> dict[str, object] | None`
- Produces: `select_person_disjoint_rows(rows, split_ratios, seed) -> dict[str, list[dict]]`
- Produces: `prepare_dataset(zip_paths, output_dir, max_rows, seed) -> dict[str, object]`
- Outputs: `manifest.jsonl`, `split_report.json`, `label_distribution.json`, `images/{train,val,test}/`

- [ ] **Step 1: XML·분할·희귀색 보존 실패 테스트 작성**

```python
def test_parse_annotation_preserves_item_types_and_color_mask():
    row = parse_annotation(XML_WITH_BAG_BACKPACK_MASK, "H1/sample.png")
    assert row["items"] == ["bag", "backpack", "mask"]
    assert row["upper_color"] == "orange"
    assert row["upper_color_defined"] is True
    assert row["lower_color_defined"] is False


def test_person_ids_do_not_cross_splits():
    splits = select_person_disjoint_rows(_rows_for_people(12), (0.7, 0.15, 0.15), 42)
    person_sets = [{row["person_id"] for row in rows} for rows in splits.values()]
    assert person_sets[0].isdisjoint(person_sets[1])
    assert person_sets[0].isdisjoint(person_sets[2])
    assert person_sets[1].isdisjoint(person_sets[2])
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_prepare_aihub_kreid_multitask.py -q`

Expected: FAIL because parser and splitter do not exist.

- [ ] **Step 3: ZIP 스트리밍 구현**

전체 ZIP을 풀지 않는다. XML을 순회해 `person_id/session/camera/frame`, 작업별 값과 defined flag를 읽고, 같은 person/session/camera에서 시간적으로 인접한 프레임은 간격 샘플링한다. 희귀 조합 대표를 먼저 선택하고 남은 슬롯을 균등하게 채운다. 선택된 PNG만 ZIP에서 추출한다.

CLI는 다음과 같이 고정한다.

```bash
rtk proxy .venv/bin/python -m scripts.datasets.prepare_aihub_kreid_multitask \
  --dataset-root data/datasets/aihub_kreid/015.한국인재식별이미지/01.데이터 \
  --output-dir data/training/aihub_kreid_multitask_v1 \
  --max-rows 100000 --seed 42 \
  --train-ratio 0.70 --validation-ratio 0.15 --test-ratio 0.15
```

- [ ] **Step 4: 단위 테스트와 소형 ZIP 통합 테스트 통과**

Run: `rtk pytest tests/test_prepare_aihub_kreid_multitask.py -q`

Expected: PASS and fixture output contains disjoint train/val/test person IDs.

- [ ] **Step 5: 커밋**

```bash
rtk git add scripts/datasets/prepare_aihub_kreid_multitask.py tests/test_prepare_aihub_kreid_multitask.py
rtk git commit -m "feat: prepare person-disjoint appearance data"
```

### Task 3: 공유 백본과 독립 multi-head 모델

**Files:**
- Create: `scripts/train/appearance_multitask_model.py`
- Create: `tests/test_appearance_multitask_model.py`

**Interfaces:**
- Produces: `AppearanceMultiTaskModel(backbone: nn.Module, feature_dim: int)`
- Produces: `forward_heads(images: Tensor) -> dict[str, Tensor]`
- Produces: `forward(images: Tensor) -> Tensor` as flat probability vector in label-map order
- Produces: `masked_multitask_loss(outputs, targets, masks, loss_weights) -> tuple[Tensor, dict[str, Tensor]]`

- [ ] **Step 1: 출력 차원과 mask 실패 테스트 작성**

```python
def test_model_emits_expected_head_shapes():
    model = AppearanceMultiTaskModel(_tiny_backbone(), feature_dim=16)
    heads = model.forward_heads(torch.zeros(2, 3, 64, 32))
    assert heads["gender"].shape == (2, 2)
    assert heads["upper_color"].shape == (2, 11)
    assert heads["lower_color"].shape == (2, 11)
    assert heads["upper_clothes"].shape == (2, 3)
    assert heads["lower_clothes"].shape == (2, 5)
    assert heads["items"].shape == (2, 4)


def test_masked_loss_ignores_unknown_lower_color():
    total, parts = masked_multitask_loss(outputs, targets, masks, LOSS_WEIGHTS)
    total.backward()
    assert parts["lower_color"].item() == 0.0
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_appearance_multitask_model.py -q`

Expected: FAIL because model module does not exist.

- [ ] **Step 3: 모델과 loss 구현**

`forward_heads()`는 logits를 반환한다. 단일 선택 head는 masked cross entropy, items는 element-wise masked BCEWithLogitsLoss를 사용한다. `forward()`는 단일 선택 head에 softmax, items에 sigmoid를 적용해 `build_label_map()` 순서로 concatenate한다.

```python
class AppearanceMultiTaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            spec.name: nn.Linear(feature_dim, len(spec.classes))
            for spec in HEAD_SPECS
        })
```

- [ ] **Step 4: GREEN 확인**

Run: `rtk pytest tests/test_appearance_multitask_model.py -q`

Expected: PASS with finite loss and gradients.

- [ ] **Step 5: 커밋**

```bash
rtk git add scripts/train/appearance_multitask_model.py tests/test_appearance_multitask_model.py
rtk git commit -m "feat: add protected appearance multi-head model"
```

### Task 4: 11색 체크포인트 이식과 보호 학습 단계

**Files:**
- Create: `scripts/train/appearance_transfer.py`
- Create: `tests/test_appearance_transfer.py`
- Modify: `scripts/train/appearance_multitask_model.py`

**Interfaces:**
- Produces: `load_native11_weights(model, checkpoint_path, legacy_attr_names) -> TransferReport`
- Produces: `configure_training_phase(model, phase: Literal["protected", "joint"], new_head_lr: float) -> list[dict]`
- `TransferReport` records copied backbone keys, copied color rows, missing keys and rejected shape mismatches.

- [ ] **Step 1: 가중치 행 이식과 freeze 실패 테스트 작성**

```python
def test_native11_color_rows_are_copied_by_attribute_name(tmp_path):
    report = load_native11_weights(model, checkpoint_path, legacy_names)
    assert report.copied_color_rows == 22
    assert torch.equal(model.heads["upper_color"].weight[9], legacy_weight[19])


def test_protected_phase_freezes_backbone_and_color_heads():
    groups = configure_training_phase(model, "protected", 1e-3)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not any(p.requires_grad for p in model.heads["upper_color"].parameters())
    assert {group["name"] for group in groups} == {"new_heads"}
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_appearance_transfer.py -q`

Expected: FAIL because transfer functions do not exist.

- [ ] **Step 3: 이름 기반 이식과 optimizer group 구현**

checkpoint key prefix를 정규화하고 backbone shape가 같은 key만 복사한다. legacy flat classifier는 `upper_<color>`와 `lower_<color>` 이름으로 새 head row에 복사한다. joint phase optimizer groups는 `new_heads: 1.0 * lr`, `color_heads: 0.1 * lr`, `backbone: 0.1 * lr`로 만든다.

- [ ] **Step 4: GREEN 확인**

Run: `rtk pytest tests/test_appearance_transfer.py tests/test_appearance_multitask_model.py -q`

Expected: PASS and report explicitly lists all 22 color rows.

- [ ] **Step 5: 커밋**

```bash
rtk git add scripts/train/appearance_transfer.py scripts/train/appearance_multitask_model.py tests/test_appearance_transfer.py
rtk git commit -m "feat: protect native color weights during transfer"
```

### Task 5: 학습·평가·색상 승격 gate

**Files:**
- Create: `scripts/train/train_appearance_multitask.py`
- Create: `scripts/ops/evaluate_appearance_multitask.py`
- Create: `tests/test_evaluate_appearance_multitask.py`
- Create: `config/appearance_multitask_v1.json`

**Interfaces:**
- Produces checkpoint with `format_version`, `state_dict`, `head_specs`, `label_map`, `model_version`, `dataset_hash`, `phase_history`.
- Training CLI accepts `--training-mode color_baseline|protected_multitask`; both modes consume the same manifest, split IDs, augmentation seed and total epoch count.
- Produces `evaluate_predictions(y_true, y_prob, masks, label_map) -> dict[str, object]`.
- Produces `compare_color_candidate(candidate_metrics, baseline_metrics) -> dict` with `passed`, macro/individual deltas, insufficient classes.

- [ ] **Step 1: gate 실패 테스트 작성**

```python
def test_color_gate_rejects_macro_f1_drop_over_two_points():
    result = compare_color_candidate(
        {"upper_color": {"macro_f1": 0.77, "classes": {}}},
        {"upper_color": {"macro_f1": 0.80, "classes": {}}},
    )
    assert result["passed"] is False
    assert result["reasons"] == ["upper_color macro_f1 delta -0.0300 < -0.0200"]


def test_class_with_fewer_than_30_examples_is_pending():
    metrics = evaluate_predictions(y_true, y_prob, masks, label_map)
    assert metrics["lower_color"]["classes"]["orange"]["status"] == "pending"
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_evaluate_appearance_multitask.py -q`

Expected: FAIL because evaluator does not exist.

- [ ] **Step 3: trainer와 evaluator 구현**

trainer는 seed 42, protected phase와 joint phase를 별도 epoch 범위로 실행하고 매 epoch 작업별 metrics를 저장한다. `color_baseline` 모드는 같은 공유 백본과 색상 head만 학습하며 non-color loss를 0으로 두고, `protected_multitask`와 동일한 manifest·person split·augmentation seed·총 epoch 수를 사용한다. 최고 checkpoint는 전체 평균이 아니라 `color gate passed`, 그다음 non-color macro F1 순서로 선택한다. config 기본값은 다음과 같다.

```json
{
  "seed": 42,
  "input_height": 256,
  "input_width": 192,
  "batch_size": 64,
  "protected_epochs": 3,
  "joint_epochs": 7,
  "new_head_lr": 0.0003,
  "color_and_backbone_lr_scale": 0.1,
  "minimum_evaluation_examples": 30
}
```

- [ ] **Step 4: evaluator 테스트와 CPU mini-epoch 통과**

Run: `rtk pytest tests/test_evaluate_appearance_multitask.py tests/test_appearance_multitask_model.py tests/test_appearance_transfer.py -q`

Run: `rtk proxy .venv/bin/python -m scripts.train.train_appearance_multitask --training-mode protected_multitask --config config/appearance_multitask_v1.json --manifest tests/fixtures/appearance_multitask/manifest.jsonl --output-dir /tmp/appearance_multitask_smoke --device cpu --max-batches 2`

Expected: exit 0, finite losses, checkpoint and metrics JSON created.

- [ ] **Step 5: 커밋**

```bash
rtk git add scripts/train/train_appearance_multitask.py scripts/ops/evaluate_appearance_multitask.py tests/test_evaluate_appearance_multitask.py config/appearance_multitask_v1.json
rtk git commit -m "feat: train and gate unified appearance candidate"
```

### Task 6: 실제 데이터 생성과 GPU 학습 실행

**Files:**
- Create: `scripts/train/run_appearance_multitask_v1.sh`
- Create: `tests/scripts/test_run_appearance_multitask_v1.sh`
- Create: `data/training/aihub_kreid_multitask_v1/` (gitignored generated artifacts)
- Create: `models/experiments/appearance_multitask_v1/` (gitignored candidate artifacts)

**Interfaces:**
- Consumes Tasks 1-5 CLI.
- Produces dataset reports, baseline metrics, protected candidate checkpoint and per-class comparison report.

- [ ] **Step 1: wrapper syntax test 추가**

Create `tests/scripts/test_run_appearance_multitask_v1.sh` using the repository shell-script test pattern:

```bash
bash -n scripts/train/run_appearance_multitask_v1.sh
```

- [ ] **Step 2: wrapper 작성**

wrapper는 `.training_env/numpy1`을 `PYTHONPATH` 앞에 두고 데이터 준비가 완료됐을 때만 학습한다. 먼저 `--training-mode color_baseline`, 다음으로 `--training-mode protected_multitask`를 동일 config와 manifest로 실행하고 두 validation/test 결과를 gate CLI에 전달한다. API 키, 토큰, 경로 비밀값은 기록하지 않는다. 로그는 `data/training/aihub_kreid_multitask_v1/logs/`에 남긴다.

- [ ] **Step 3: 100k 이하 데이터 생성**

Run: Task 2의 CLI.

Expected: no person overlap, missing images 0, 모든 head의 분포와 pending classes가 보고됨.

- [ ] **Step 4: 11색 기준선과 보호형 후보 학습**

Run: `rtk proxy scripts/train/run_appearance_multitask_v1.sh`

Expected: color baseline과 protected/joint phases가 동일 split·seed·총 epoch 조건으로 완료되고, 각 best checkpoint와 metrics JSON이 생성된다. 장시간 실행은 user systemd unit으로 시작하고 unit 이름과 로그 경로를 기록한다.

- [ ] **Step 5: 색상 gate 결과 확인**

Run: `rtk json models/experiments/appearance_multitask_v1/color_gate.json`

Expected: `passed=true`이면 Milestone B의 통합 경로로 진행한다. `passed=false`이면 모델을 삭제하지 않고 색상 전용+속성 모델 2경로를 runtime candidate로 기록한다.

- [ ] **Step 6: wrapper만 커밋**

```bash
rtk git add scripts/train/run_appearance_multitask_v1.sh tests/scripts/test_run_appearance_multitask_v1.sh
rtk git commit -m "ops: add appearance multitask training runner"
```

## Milestone B: 런타임 누적·대표 이미지·검색

### Task 7: grouped label map decoder 호환

**Files:**
- Modify: `src/core/ai/_attribute_backends.py:48-115`
- Test: `tests/test_attribute_backends.py`
- Create: `config/appearance_multitask_v1_labels.json`

**Interfaces:**
- Consumes flat probability output and Task 1 label map.
- Produces attributes including `gender`, `upper_color`, `lower_color`, `upper_clothes`, `lower_clothes`, `has_bag`, `has_backpack`, `has_hat`, `has_mask`, and `attribute_scores`.

- [ ] **Step 1: decoder 실패 테스트 작성**

```python
def test_decoder_handles_grouped_clothes_and_items():
    attrs = decode_pphuman_scores(SCORES, MULTITASK_LABEL_MAP)
    assert attrs["upper_clothes"] == "long_sleeve"
    assert attrs["lower_clothes"] == "long_pants"
    assert attrs["has_bag"] is True
    assert attrs["has_backpack"] is False
    assert attrs["has_hat"] is True
    assert attrs["has_mask"] is True
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_attribute_backends.py::test_decoder_handles_grouped_clothes_and_items -q`

Expected: FAIL until grouped metadata and expected fields are decoded.

- [ ] **Step 3: 최소 decoder 확장**

기존 `labels` 배열 호환을 유지한다. `head_slices`는 길이 검증에 사용하고, 알 수 없는 head는 무시한다. threshold 미달 단일 선택 head는 필드를 만들지 않아 downstream에서 unknown으로 처리한다.

- [ ] **Step 4: 회귀 테스트 통과**

Run: `rtk pytest tests/test_attribute_backends.py tests/test_appearance_analyzer.py -q`

Expected: 기존 PA100K tests와 새 grouped tests 모두 PASS.

- [ ] **Step 5: 커밋**

```bash
rtk git add src/core/ai/_attribute_backends.py tests/test_attribute_backends.py config/appearance_multitask_v1_labels.json
rtk git commit -m "feat: decode unified appearance attributes"
```

### Task 8: 추적 속성 누적과 대표 이미지 1장

**Files:**
- Create: `src/core/ai/_appearance_track_summary.py`
- Create: `tests/test_appearance_track_summary.py`
- Modify: `src/core/ai/_appearance_pipeline.py:29-82,396-488,531-578,681-742`

**Interfaces:**
- Produces: `TrackAppearanceSummary.observe(timestamp, bbox, crop, attributes) -> ObservationDecision`
- Produces: `TrackAppearanceSummary.finalize() -> FinalAppearanceRecord`
- Produces: `RepresentativeImageStore.consider(track_key, crop, quality) -> Path | None`
- `ObservationDecision.should_infer` enforces 0.5-second interval and 48x96 minimum crop.

- [ ] **Step 1: 간격·투표·대표 1장 실패 테스트 작성**

```python
def test_track_summary_throttles_and_keeps_one_best_image(tmp_path):
    summary = TrackAppearanceSummary("cam1", 7, min_interval_sec=0.5)
    assert summary.should_infer(10.0, (0, 0, 100, 200)) is True
    assert summary.should_infer(10.2, (0, 0, 100, 200)) is False
    store = RepresentativeImageStore(tmp_path)
    first = store.consider(("cam1", 7), blurry_crop, quality=0.2)
    second = store.consider(("cam1", 7), sharp_crop, quality=0.9)
    assert second == first
    assert len(list(tmp_path.glob("*.jpg"))) == 1
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_appearance_track_summary.py -q`

Expected: FAIL because summary module does not exist.

- [ ] **Step 3: 독립 summary 모듈 구현**

quality는 visibility, crop area, Laplacian variance, clipping penalty, mean attribute confidence를 정규화해 합산한다. 더 좋은 후보는 동일 destination에 원자적 임시 파일 교체로 덮어쓴다. categorical은 confidence-weighted vote, boolean items는 평균 probability와 threshold로 finalize한다.

- [ ] **Step 4: pipeline에 최소 연결**

`AppearancePipeline`은 summary 객체에 관찰을 전달하고 최종 payload를 기존 log insert 경로로 전달한다. overlay 렌더링 코드는 추가하지 않는다. 기존 smoothing API는 유지하고 새 필드만 summary에 확장한다.

- [ ] **Step 5: 테스트 통과**

Run: `rtk pytest tests/test_appearance_track_summary.py tests/test_appearance_pipeline.py -q`

Expected: one image per track, interval throttle, unknown handling, legacy pipeline tests all PASS.

- [ ] **Step 6: 커밋**

```bash
rtk git add src/core/ai/_appearance_track_summary.py src/core/ai/_appearance_pipeline.py tests/test_appearance_track_summary.py tests/test_appearance_pipeline.py
rtk git commit -m "feat: summarize tracked appearance with one image"
```

### Task 9: SQLite 속성 필드와 검색 API 확장

**Files:**
- Modify: `src/services/appearance_log.py:29-62,85-114,126-334`
- Modify: `src/api/v1/search.py:100-249`
- Test: `tests/test_appearance_log.py`
- Create: `tests/test_search_api.py`

**Interfaces:**
- `AppearanceLog.insert()` accepts `upper_clothes`, `lower_clothes`, `has_bag`, existing `has_backpack`, `has_hat`, `has_mask`, `model_version`, `label_map_version`, `first_seen`, `last_seen`.
- `AppearanceLog.search()` filters `upper_clothes`, `lower_clothes`, `has_bag`, `has_backpack`, `has_hat`, `has_mask` without breaking existing arguments.
- `AppearanceRecord` exposes new fields and existing `crop_url`.

- [ ] **Step 1: additive migration과 검색 실패 테스트 작성**

```python
def test_schema_migration_adds_multitask_columns_without_losing_rows(tmp_path):
    log = _legacy_log_with_one_row(tmp_path)
    reopened = AppearanceLog(log.db_path)
    reopened.insert(**MULTITASK_RECORD)
    rows = reopened.search(
        upper_clothes="long_sleeve",
        has_bag=True,
        has_backpack=False,
        has_hat=True,
        has_mask=True,
    )
    assert len(rows) == 1
    assert reopened.count() == 2
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_appearance_log.py::test_schema_migration_adds_multitask_columns_without_losing_rows -q`

Expected: FAIL with unexpected insert/search keyword or missing column.

- [ ] **Step 3: additive SQLite migration 구현**

`_ensure_columns()`에만 `ALTER TABLE ... ADD COLUMN`을 추가한다. 기존 컬럼 이름을 바꾸거나 데이터를 재작성하지 않는다. boolean은 SQLite INTEGER, confidence와 version 세부값은 기존 `attribute_metadata` JSON에도 보존한다.

- [ ] **Step 4: API query와 response 확장**

`upper_clothes`, `lower_clothes`, `has_bag`, `has_backpack`, `has_hat`, `has_mask` query parameters를 추가하고 `_to_record()`에 새 필드를 매핑한다. 자연어 alias는 `긴팔/반팔/민소매/긴바지/반바지/치마/원피스/가방/백팩/모자/마스크`를 명시적으로 매핑한다.

- [ ] **Step 5: 회귀 테스트 통과**

Run: `rtk pytest tests/test_appearance_log.py tests/test_search_api.py tests/test_appearance_pipeline.py -q`

Expected: migration, old filters, new combined filters and crop URL tests all PASS.

- [ ] **Step 6: 커밋**

```bash
rtk git add src/services/appearance_log.py src/api/v1/search.py tests/test_appearance_log.py tests/test_search_api.py
rtk git commit -m "feat: search clothing and core belongings"
```

## Milestone C: 변환·shadow·조건부 승격

### Task 10: ONNX 내보내기와 PyTorch parity

**Files:**
- Create: `scripts/convert/export_appearance_multitask_onnx.py`
- Create: `tests/test_export_appearance_multitask_onnx.py`

**Interfaces:**
- Consumes Task 5 checkpoint.
- Produces fixed-shape `[N, 36]` probability output and label map.
- Produces `onnx_export_report.json` with max absolute difference and model hashes.

- [ ] **Step 1: export wrapper 실패 테스트 작성**

```python
def test_exported_model_matches_pytorch(tmp_path):
    report = export_and_compare(checkpoint, output_path, sample_batch)
    assert report["output_shape"] == [2, 36]
    assert report["max_abs_diff"] <= 1e-4
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_export_appearance_multitask_onnx.py -q`

Expected: FAIL because exporter does not exist.

- [ ] **Step 3: exporter 구현**

입력 이름은 `images`, 출력 이름은 `attributes`, dynamic axis는 batch만 허용한다. preprocess mean/std와 input 256x192를 export report와 label map에 기록한다.

- [ ] **Step 4: parity test 통과**

Run: `rtk pytest tests/test_export_appearance_multitask_onnx.py -q`

Expected: PASS; onnxruntime가 환경에 없으면 실제 export CLI smoke가 명확한 dependency error로 종료되고 unit test는 export graph metadata를 검증한다.

- [ ] **Step 5: 커밋**

```bash
rtk git add scripts/convert/export_appearance_multitask_onnx.py tests/test_export_appearance_multitask_onnx.py
rtk git commit -m "feat: export unified appearance model to onnx"
```

### Task 11: TensorRT candidate wiring과 health check

**Files:**
- Create: `scripts/convert/build_appearance_multitask_tensorrt.sh`
- Create: `scripts/ops/compare_appearance_multitask_tensorrt.py`
- Create: `tests/test_compare_appearance_multitask_tensorrt.py`
- Create: `tests/scripts/test_build_appearance_multitask_tensorrt.sh`
- Create: `config/deepstream/config_infer_appearance_multitask_v1.txt`
- Modify: `.env.jetson.example`
- Modify: `docker-compose.jetson.yml`
- Modify: `scripts/health/check_compose_runtime_assumptions.py`
- Test: `tests/test_check_compose_runtime_assumptions.py`

**Interfaces:**
- TensorRT builder consumes the fixed-shape ONNX model and creates a versioned FP16 engine plus build log; it never overwrites the current PA100K or color engines.
- Backend comparison produces per-head argmax agreement and maximum absolute probability difference for the same sample batch.
- Candidate-only env: `APPEARANCE_MULTITASK_SHADOW_ENABLED`, `APPEARANCE_MULTITASK_MODEL_PATH`, `APPEARANCE_MULTITASK_LABEL_MAP_PATH`, `APPEARANCE_MULTITASK_INTERVAL_SEC`.
- Default `APPEARANCE_MULTITASK_SHADOW_ENABLED=0`; existing production paths remain defaults.

- [ ] **Step 1: builder·parity·wiring failure test 작성**

```python
def test_multitask_shadow_wiring_is_disabled_by_default_and_complete():
    result = check_appearance_multitask_shadow_wiring()
    assert result["passed"] is True
    assert result["detail"].endswith("shadow disabled by default")


def test_tensorrt_comparison_rejects_head_disagreement():
    report = compare_backend_outputs(ONNX_OUTPUTS, TENSORRT_OUTPUTS_WITH_DRIFT, LABEL_MAP)
    assert report["passed"] is False
    assert "lower_color" in report["failed_heads"]
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_compare_appearance_multitask_tensorrt.py tests/test_check_compose_runtime_assumptions.py -q`

Expected: FAIL because backend comparator and shadow wiring check do not exist.

- [ ] **Step 3: versioned TensorRT engine build와 parity 구현**

builder는 Jetson 컨테이너의 `trtexec`를 사용해 `images:1x3x256x192` FP16 engine을 새 버전 경로에 만든다. 출력 파일이 이미 있으면 명시적 `--force` 없이는 덮어쓰지 않는다. 동일한 고정 sample batch에 대해 ONNX와 TensorRT 출력을 비교하며 각 단일 선택 head argmax 일치율 99% 이상, 전체 probability 최대 절대 오차 0.01 이하를 요구한다.

Run: `rtk proxy bash tests/scripts/test_build_appearance_multitask_tensorrt.sh`

Run: `rtk pytest tests/test_compare_appearance_multitask_tensorrt.py -q`

Expected: shell syntax PASS and comparator accepts matching fixtures but rejects drift fixtures.

- [ ] **Step 4: candidate-only 설정 구현**

운영 `APPEARANCE_MODEL_PATH`, PA100K engine, 색상 engine 기본값을 바꾸지 않는다. 새 engine과 label map은 별도 volume path로 mount하고 shadow flag가 1일 때만 초기화한다. secret은 추가하지 않는다.

- [ ] **Step 5: 설정 검증**

Run: `rtk pytest tests/test_check_compose_runtime_assumptions.py -q`

Run: `rtk proxy .venv/bin/python scripts/health/check_compose_runtime_assumptions.py`

Expected: PASS with current production and candidate shadow paths both reported.

- [ ] **Step 6: 커밋**

```bash
rtk git add scripts/convert/build_appearance_multitask_tensorrt.sh scripts/ops/compare_appearance_multitask_tensorrt.py tests/test_compare_appearance_multitask_tensorrt.py tests/scripts/test_build_appearance_multitask_tensorrt.sh config/deepstream/config_infer_appearance_multitask_v1.txt .env.jetson.example docker-compose.jetson.yml scripts/health/check_compose_runtime_assumptions.py tests/test_check_compose_runtime_assumptions.py
rtk git commit -m "ops: wire appearance multitask shadow candidate"
```

### Task 12: shadow 비교와 승격 보고서

**Files:**
- Create: `scripts/ops/shadow_compare_appearance_multitask.py`
- Create: `tests/test_shadow_compare_appearance_multitask.py`
- Create: `scripts/ops/check_appearance_multitask_promotion.py`
- Create: `tests/test_check_appearance_multitask_promotion.py`

**Interfaces:**
- Shadow row records baseline/candidate attributes, latency, disagreement, camera, track and representative crop.
- Promotion checker consumes offline color gate, 30-minute runtime stats and shadow disagreement summary.
- Produces `promotion_report.json` with `passed`, `blocked_reasons`, `fallback_mode`.

- [ ] **Step 1: promotion gate failure tests 작성**

```python
def test_promotion_blocks_color_regression_and_frame_drop():
    report = build_promotion_report(
        color_gate={"passed": False, "reasons": ["upper_blue f1 delta -0.08"]},
        runtime={"frame_drop_delta_percentage_points": 1.3},
        shadow={"rows": 5000},
    )
    assert report["passed"] is False
    assert report["fallback_mode"] == "dedicated_color_plus_multitask_attributes"
```

- [ ] **Step 2: RED 확인**

Run: `rtk pytest tests/test_shadow_compare_appearance_multitask.py tests/test_check_appearance_multitask_promotion.py -q`

Expected: FAIL because scripts do not exist.

- [ ] **Step 3: shadow logger 구현**

운영 결과를 변경하지 않고 candidate result와 latency를 JSONL에 추가한다. crop은 Task 8의 대표 이미지 경로를 참조하고 복사본을 만들지 않는다. PII나 얼굴 embedding을 새 로그에 넣지 않는다.

- [ ] **Step 4: promotion checker 구현**

다음 중 하나라도 발생하면 `passed=false`: color gate fail, p95가 기존 두 경로 합보다 큼, frame drop +1%p 이상, load error, shadow rows 5000 미만. 색상만 실패하면 fallback은 2모델, runtime 안정성도 실패하면 기존 운영 모델 유지다.

- [ ] **Step 5: 테스트 통과**

Run: `rtk pytest tests/test_shadow_compare_appearance_multitask.py tests/test_check_appearance_multitask_promotion.py -q`

Expected: PASS for unified promotion, 2-model fallback and full rollback fixtures.

- [ ] **Step 6: 커밋**

```bash
rtk git add scripts/ops/shadow_compare_appearance_multitask.py scripts/ops/check_appearance_multitask_promotion.py tests/test_shadow_compare_appearance_multitask.py tests/test_check_appearance_multitask_promotion.py
rtk git commit -m "ops: gate unified appearance model promotion"
```

### Task 13: 전체 검증과 운영 인계

**Files:**
- Modify: `docs/guides/EVENT_DATA_CONTRACT.md`
- Create: `docs/guides/APPEARANCE_MULTITASK_OPERATION.md`
- Modify: `tests/test_appearance_pipeline.py`
- Modify: `tests/test_search_api.py`

**Interfaces:**
- Documents new stored fields, unknown semantics, one-image policy, shadow enable/disable and rollback commands.

- [ ] **Step 1: 전체 관련 테스트 실행**

Run:

```bash
rtk pytest \
  tests/test_appearance_multitask_schema.py \
  tests/test_prepare_aihub_kreid_multitask.py \
  tests/test_appearance_multitask_model.py \
  tests/test_appearance_transfer.py \
  tests/test_evaluate_appearance_multitask.py \
  tests/test_attribute_backends.py \
  tests/test_appearance_track_summary.py \
  tests/test_appearance_pipeline.py \
  tests/test_appearance_log.py \
  tests/test_search_api.py \
  tests/test_export_appearance_multitask_onnx.py \
  tests/test_compare_appearance_multitask_tensorrt.py \
  tests/test_check_compose_runtime_assumptions.py \
  tests/test_shadow_compare_appearance_multitask.py \
  tests/test_check_appearance_multitask_promotion.py
```

Expected: all collected tests PASS, none skipped due to application logic.

Run: `rtk proxy bash tests/scripts/test_run_appearance_multitask_v1.sh`

Expected: PASS and the runner has valid shell syntax and required baseline/candidate invocations.

Run: `rtk proxy bash tests/scripts/test_build_appearance_multitask_tensorrt.sh`

Expected: PASS and the TensorRT builder has valid shell syntax and versioned output guards.

- [ ] **Step 2: API·DB·fallback smoke 실행**

Run: `rtk proxy .venv/bin/python scripts/smoke/smoke_test_data_flow.py`

Expected: existing event/appearance flow passes and legacy fields remain readable.

- [ ] **Step 3: Jetson 30분 shadow 실행**

기존 운영 모델을 유지한 상태로 candidate shadow flag만 활성화한다. 시작 전/후 GPU 메모리, CPU, p50/p95 latency, frame drop, error count를 수집한다. 결과는 `promotion_report.json`에 연결한다.

- [ ] **Step 4: 승격 경계 확인**

`promotion_report.json`이 통과해도 운영 기본값을 자동 변경하지 않는다. 단일 카메라 canary 활성화 명령과 즉시 rollback 명령까지만 준비하고, 실제 canary 전환은 사용자 승인 후 별도 운영 작업으로 실행한다. gate가 실패하면 보고서의 `fallback_mode`에 따라 2모델 후보 또는 기존 운영 유지로 종료한다.

- [ ] **Step 5: 문서 작성**

운영 문서에 다음 명령을 실제 unit/container 이름으로 기록한다: shadow enable, status, log tail, disable, 기존 engine rollback. 새 검색 필드와 대표 이미지 1장 정책을 데이터 계약에 추가한다.

- [ ] **Step 6: 문서와 최종 테스트 커밋**

```bash
rtk git add docs/guides/EVENT_DATA_CONTRACT.md docs/guides/APPEARANCE_MULTITASK_OPERATION.md tests/test_appearance_pipeline.py tests/test_search_api.py
rtk git commit -m "docs: hand off protected appearance model operations"
```

## Final Acceptance Checklist

- [ ] 생성 데이터의 train/validation/test person ID 교집합이 모두 0이다.
- [ ] 모든 head의 표본 수, mask 수, 클래스별 metrics가 JSON으로 남는다.
- [ ] 11색 전용 기준선과 통합 후보가 같은 holdout에서 비교된다.
- [ ] 색상 gate 실패 시 2모델 fallback이 자동 선택되고 운영 모델은 바뀌지 않는다.
- [ ] 추적 ID당 대표 이미지가 정확히 1장이다.
- [ ] 실시간 화면에 새 bbox 또는 속성 overlay가 없다.
- [ ] 검색 API가 성별·색상·의류·핵심 소지품 조합과 대표 이미지를 반환한다.
- [ ] PyTorch/ONNX 출력 차이가 1e-4 이하이다.
- [ ] ONNX/TensorRT 단일 선택 head argmax 일치율이 99% 이상이고 probability 최대 절대 오차가 0.01 이하이다.
- [ ] 30분 shadow에서 frame drop 증가가 1%p 미만이다.
- [ ] 최종 promotion report가 unified, 2-model fallback 또는 existing-production 중 하나를 명시한다.
