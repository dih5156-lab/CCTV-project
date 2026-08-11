# Appearance Color Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 로컬 HTML에서 상의·하의 색상을 독립적으로 검수하고, 검증된 JSON을 dry-run 후 DB에 반영하며 재학습 준비용 라벨 데이터로 내보낸다.

**Architecture:** 기존 manifest 생성기는 유지하고 HTML 생성기만 양쪽 필드를 렌더링하도록 확장한다. DB 변경과 학습 데이터 내보내기는 각각 독립된 CLI 스크립트로 분리하며, 두 스크립트 모두 같은 JSON 계약을 엄격히 검증한다. 모델 재학습과 배포는 이 계획에서 실행하지 않는다.

**Tech Stack:** Python 3, 표준 라이브러리(`argparse`, `csv`, `json`, `sqlite3`, `pathlib`), pytest, 브라우저 내장 JavaScript

## Global Constraints

- 기존 사용자 변경 파일을 덮어쓰거나 되돌리지 않는다.
- 새 외부 라이브러리를 추가하지 않는다.
- DB 적용은 기본 dry-run이고 `--apply`가 있을 때만 수행한다.
- DB 적용 전 SQLite backup API로 일관된 백업을 만든다.
- 허용 색상은 `black`, `blue`, `brown`, `gray`, `green`, `orange`, `pink`, `purple`, `red`, `white`, `yellow`이다.
- `null`과 `exclude`는 DB를 변경하지 않는다.
- 미검수 필드를 기존 DB 값으로 자동 라벨링하지 않는다.
- 자동 재학습과 자동 Jetson/DeepStream 배포는 구현하지 않는다.

---

### Task 1: 상·하의 검수 HTML 및 JSON 계약

**Files:**
- Modify: `scripts/ops/build_appearance_review_html.py`
- Create: `tests/test_build_appearance_review_html.py`

**Interfaces:**
- Consumes: `manifest_path: Path`의 `items[].stored`, `items[].candidates`, `items[].hard_fields`
- Produces: `build_document(payload: dict) -> str`, 스키마 버전 1의 `appearance_color_review_labels.json`

- [ ] **Step 1: 두 필드 렌더링을 요구하는 실패 테스트 작성**

```python
from scripts.ops.build_appearance_review_html import build_document


def test_build_document_renders_upper_and_lower_review_fields():
    payload = {
        "items": [{
            "id": 323388,
            "crop_path": "/tmp/person.jpg",
            "stored": {"upper_color": "black", "lower_color": "blue"},
            "candidates": {
                "upper_color": {"hsv_color": "black", "lab_color": "gray", "model_color": "white"},
                "lower_color": {"hsv_color": "blue", "lab_color": "blue", "model_color": "black"},
            },
        }]
    }

    document = build_document(payload)

    assert "data-id='323388' data-field='upper_color'" in document
    assert "data-id='323388' data-field='lower_color'" in document
    assert "상의 정답" in document
    assert "하의 정답" in document
    assert "appearance_color_review_labels.json" in document
    assert "schema_version:1" in document
```

- [ ] **Step 2: 테스트가 기존 하의 전용 구현 때문에 실패하는지 확인**

Run: `rtk pytest tests/test_build_appearance_review_html.py -q`

Expected: FAIL because `build_document` 또는 상의 드롭다운이 없다.

- [ ] **Step 3: 문서 생성 함수를 분리하고 두 드롭다운을 최소 구현**

```python
COLOR_OPTIONS = (
    "black", "blue", "brown", "gray", "green", "orange",
    "pink", "purple", "red", "white", "yellow",
)
REVIEW_FIELDS = ("upper_color", "lower_color")
DOWNLOAD_SCRIPT = """
function downloadLabels() {
  const byId = new Map();
  for (const select of document.querySelectorAll("select[data-id][data-field]")) {
    const id = Number(select.dataset.id);
    const item = byId.get(id) || {id, upper_color: null, lower_color: null};
    item[select.dataset.field] = select.value || null;
    byId.set(id, item);
  }
  const payload = {schema_version:1, items:[...byId.values()]};
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], {type:"application/json"})
  );
  anchor.download = "appearance_color_review_labels.json";
  anchor.click();
}
"""


def _select(item_id: int, field: str) -> str:
    options = "<option value=''>변경 안 함</option>"
    options += "".join(f"<option>{color}</option>" for color in COLOR_OPTIONS)
    options += "<option>exclude</option>"
    return f"<select data-id='{item_id}' data-field='{field}'>{options}</select>"


def build_document(payload: dict) -> str:
    rows = []
    for item in payload.get("items", []):
        crop_uri = "file://" + str(Path(item["crop_path"]).resolve())
        stored = item.get("stored", {})
        candidates = item.get("candidates", {})
        cells = [f"<td>{item['id']}</td>", f"<td><img src='{html.escape(crop_uri)}'></td>"]
        for field in REVIEW_FIELDS:
            candidate = candidates.get(field, {})
            cells.extend([
                f"<td>{html.escape(str(stored.get(field)))}</td>",
                f"<td>{html.escape(str(candidate.get('hsv_color')))}</td>",
                f"<td>{html.escape(str(candidate.get('lab_color')))}</td>",
                f"<td>{html.escape(str(candidate.get('model_color')))}</td>",
                f"<td>{_select(int(item['id']), field)}</td>",
            ])
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(rows)
    headers = (
        "<th>ID</th><th>crop</th>"
        "<th>상의 DB</th><th>상의 HSV</th><th>상의 LAB</th><th>상의 model</th><th>상의 정답</th>"
        "<th>하의 DB</th><th>하의 HSV</th><th>하의 LAB</th><th>하의 model</th><th>하의 정답</th>"
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>Appearance color review</title>"
        f"<h1>상의·하의 색상 검수 ({len(rows)}건)</h1>"
        "<p>브라우저를 닫기 전에 반드시 검수 JSON을 다운로드하세요.</p>"
        "<button onclick='downloadLabels()'>검수 라벨 JSON 다운로드</button>"
        f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"
        f"<script>{DOWNLOAD_SCRIPT}</script>"
    )
```

다운로드 JavaScript는 각 행을 다음 구조로 묶는다.

```javascript
const byId = new Map();
for (const select of document.querySelectorAll("select[data-id][data-field]")) {
  const id = Number(select.dataset.id);
  const item = byId.get(id) || {id, upper_color: null, lower_color: null};
  item[select.dataset.field] = select.value || null;
  byId.set(id, item);
}
const payload = {schema_version: 1, items: [...byId.values()]};
```

- [ ] **Step 4: HTML 단위 테스트 통과 확인**

Run: `rtk pytest tests/test_build_appearance_review_html.py -q`

Expected: PASS.

- [ ] **Step 5: 기존 CLI로 임시 HTML 생성 확인**

Run: `rtk test python scripts/ops/build_appearance_review_html.py --manifest data/runtime/appearance_color_review_manifest.json --output /tmp/appearance-color-review.html`

Expected: manifest가 존재하면 exit 0과 `wrote /tmp/appearance-color-review.html`; manifest가 없으면 테스트 fixture 기반 단위 테스트 결과만 기록하고 운영 파일을 만들지 않는다.

- [ ] **Step 6: 작업 단위 커밋**

```bash
rtk git add scripts/ops/build_appearance_review_html.py tests/test_build_appearance_review_html.py
rtk git commit -m "feat: review upper and lower appearance colors"
```

### Task 2: 검수 JSON 기반 DB dry-run 및 안전 적용

**Files:**
- Create: `scripts/ops/apply_appearance_color_review_labels.py`
- Create: `tests/test_apply_appearance_color_review_labels.py`

**Interfaces:**
- Produces: `validate_labels(payload: dict) -> list[dict]`
- Produces: `plan_updates(db_path: Path, items: list[dict]) -> list[dict]`
- Produces: `apply_review_labels(db_path: Path, labels_path: Path, apply: bool = False, backup_path: Path | None = None) -> dict`
- CLI: `--db PATH --labels PATH [--apply] [--backup PATH]`

- [ ] **Step 1: 부분 수정과 dry-run 무변경을 요구하는 실패 테스트 작성**

```python
def test_dry_run_plans_partial_update_without_changing_database(tmp_path):
    db_path = create_appearance_db(tmp_path, upper="black", lower="blue")
    labels_path = write_labels(tmp_path, {
        "schema_version": 1,
        "items": [{"id": 1, "upper_color": "white", "lower_color": None}],
    })

    summary = apply_review_labels(db_path, labels_path)

    assert summary["mode"] == "dry-run"
    assert summary["updates"] == 1
    assert read_colors(db_path, 1) == ("black", "blue")
```

- [ ] **Step 2: 잘못된 라벨·중복 ID·없는 ID가 전체 거부되는 실패 테스트 추가**

```python
@pytest.mark.parametrize("items", [
    [{"id": 1, "upper_color": "cyan", "lower_color": None}],
    [{"id": 1, "upper_color": "red", "lower_color": None},
     {"id": 1, "upper_color": None, "lower_color": "black"}],
    [{"id": 999, "upper_color": "red", "lower_color": None}],
])
def test_invalid_review_input_changes_nothing(tmp_path, items):
    db_path = create_appearance_db(tmp_path, upper="black", lower="blue")
    labels_path = write_labels(
        tmp_path, {"schema_version": 1, "items": items}
    )
    with pytest.raises(ValueError):
        apply_review_labels(db_path, labels_path, apply=True)
    assert read_colors(db_path, 1) == ("black", "blue")
    with pytest.raises(ValueError):
        apply_review_labels(db_path, labels_path, apply=True)
    assert read_colors(db_path, 1) == ("black", "blue")
```

- [ ] **Step 3: 실패 원인이 기능 부재인지 확인**

Run: `rtk pytest tests/test_apply_appearance_color_review_labels.py -q`

Expected: FAIL because 적용 모듈이 아직 없다.

- [ ] **Step 4: 엄격한 입력 검증과 변경 계획 구현**

```python
ALLOWED_COLORS = frozenset({
    "black", "blue", "brown", "gray", "green", "orange",
    "pink", "purple", "red", "white", "yellow",
})
SKIP_VALUES = {None, "exclude"}


def validate_labels(payload: dict) -> list[dict]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("items"), list):
        raise ValueError("unsupported review label schema")
    validated = []
    seen_ids = set()
    for raw_item in payload["items"]:
        item_id = raw_item.get("id")
        if not isinstance(item_id, int) or isinstance(item_id, bool) or item_id <= 0:
            raise ValueError(f"invalid id: {item_id!r}")
        if item_id in seen_ids:
            raise ValueError(f"duplicate id: {item_id}")
        seen_ids.add(item_id)
        item = {"id": item_id}
        for field in ("upper_color", "lower_color"):
            value = raw_item.get(field)
            if value not in ALLOWED_COLORS and value not in SKIP_VALUES:
                raise ValueError(f"invalid {field}: {value!r}")
            item[field] = value
        validated.append(item)
    return validated


def plan_updates(db_path: Path, items: list[dict]) -> list[dict]:
    ids = [item["id"] for item in items]
    placeholders = ",".join("?" for _ in ids)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT id, upper_color, lower_color FROM appearance_log WHERE id IN ({placeholders})",
            ids,
        ).fetchall() if ids else []
    current = {int(row["id"]): dict(row) for row in rows}
    missing = sorted(set(ids) - set(current))
    if missing:
        raise ValueError(f"appearance ids not found: {missing}")
    updates = []
    for item in items:
        before = current[item["id"]]
        after = {}
        for field in ("upper_color", "lower_color"):
            value = item[field]
            if value not in SKIP_VALUES and value != before[field]:
                after[field] = value
        if after:
            updates.append({"id": item["id"], "before": before, "after": after})
    return updates
```

- [ ] **Step 5: 실제 적용·백업·트랜잭션 테스트 작성**

```python
def test_apply_creates_backup_and_updates_both_fields(tmp_path):
    db_path = create_appearance_db(tmp_path, upper="black", lower="blue")
    labels_path = write_labels(tmp_path, {
        "schema_version": 1,
        "items": [{"id": 1, "upper_color": "white", "lower_color": "gray"}],
    })
    summary = apply_review_labels(
        db_path, labels_path, apply=True, backup_path=tmp_path / "before.db"
    )
    assert summary["mode"] == "applied"
    assert read_colors(db_path, 1) == ("white", "gray")
    assert read_colors(tmp_path / "before.db", 1) == ("black", "blue")
```

- [ ] **Step 6: SQLite backup API와 단일 트랜잭션 적용 구현**

```python
def _backup_database(source_path: Path, destination_path: Path) -> None:
    if destination_path.exists():
        raise FileExistsError(destination_path)
    with sqlite3.connect(source_path) as source, sqlite3.connect(destination_path) as destination:
        source.backup(destination)


def apply_review_labels(
    db_path: Path,
    labels_path: Path,
    apply: bool = False,
    backup_path: Path | None = None,
) -> dict:
    items = validate_labels(json.loads(labels_path.read_text(encoding="utf-8")))
    updates = plan_updates(db_path, items)
    if not apply:
        return {"mode": "dry-run", "updates": len(updates), "changes": updates}
    resolved_backup_path = backup_path or db_path.with_name(
        f"{db_path.name}.{datetime.now():%Y%m%d_%H%M%S}.bak"
    )
    _backup_database(db_path, resolved_backup_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for update in updates:
            assignments = ", ".join(f"{field} = ?" for field in update["after"])
            values = [*update["after"].values(), update["id"]]
            connection.execute(
                f"UPDATE appearance_log SET {assignments} WHERE id = ?", values
            )
        connection.commit()
    return {
        "mode": "applied",
        "updates": len(updates),
        "changes": updates,
        "backup": str(resolved_backup_path),
    }
```

- [ ] **Step 7: 적용 스크립트 테스트 통과 확인**

Run: `rtk pytest tests/test_apply_appearance_color_review_labels.py -q`

Expected: PASS.

- [ ] **Step 8: 작업 단위 커밋**

```bash
rtk git add scripts/ops/apply_appearance_color_review_labels.py tests/test_apply_appearance_color_review_labels.py
rtk git commit -m "feat: safely apply reviewed appearance colors"
```

### Task 3: 재학습 준비용 검수 데이터 내보내기

**Files:**
- Create: `scripts/ops/export_appearance_color_review_labels.py`
- Create: `tests/test_export_appearance_color_review_labels.py`

**Interfaces:**
- Consumes: 원본 review manifest와 스키마 버전 1 labels JSON
- Produces: `export_reviewed_labels(manifest_path: Path, labels_path: Path, output_dir: Path) -> dict`
- Outputs: `reviewed_appearance_colors.csv`, `reviewed_appearance_colors.json`, `summary.json`

- [ ] **Step 1: 미검수값을 채우지 않는 내보내기 실패 테스트 작성**

```python
def test_export_preserves_only_human_reviewed_fields(tmp_path):
    manifest_path = write_manifest(tmp_path, stored_upper="black", stored_lower="blue")
    labels_path = write_labels(tmp_path, {
        "schema_version": 1,
        "items": [{"id": 1, "upper_color": "white", "lower_color": None}],
    })

    summary = export_reviewed_labels(manifest_path, labels_path, tmp_path / "out")

    row = read_single_csv_row(tmp_path / "out/reviewed_appearance_colors.csv")
    assert row["upper_color"] == "white"
    assert row["lower_color"] == ""
    assert row["upper_reviewed"] == "true"
    assert row["lower_reviewed"] == "false"
    assert summary["partial_reviews"] == 1
```

- [ ] **Step 2: exclude·누락 crop·미지원 multi-label 색상 집계 실패 테스트 추가**

```python
def test_export_reports_excluded_missing_and_training_unsupported_rows(tmp_path):
    existing_crop = tmp_path / "person.jpg"
    existing_crop.write_bytes(b"crop")
    manifest_path = write_manifest_items(tmp_path, [
        {"id": 1, "crop_path": str(existing_crop)},
        {"id": 2, "crop_path": str(tmp_path / "missing.jpg")},
    ])
    labels_path = write_labels(tmp_path, {
        "schema_version": 1,
        "items": [
            {"id": 1, "upper_color": "pink", "lower_color": "exclude"},
            {"id": 2, "upper_color": "black", "lower_color": "blue"},
        ],
    })
    summary = export_reviewed_labels(
        manifest_path, labels_path, tmp_path / "out"
    )
    assert summary["excluded_fields"] == 1
    assert summary["missing_crops"] == 1
    assert summary["multilabel_unsupported_fields"] == 1
    assert summary["excluded_fields"] == 1
    assert summary["missing_crops"] == 1
    assert summary["multilabel_unsupported_fields"] == 1
```

- [ ] **Step 3: 기존 단일 라벨 importer에서 실패하는지 확인**

Run: `rtk pytest tests/test_export_appearance_color_review_labels.py -q`

Expected: FAIL because 새 JSON 계약과 CSV/감사용 JSON 출력이 없다.

- [ ] **Step 4: 검수 라벨 원본 내보내기 최소 구현**

```python
try:
    from .apply_appearance_color_review_labels import validate_labels
except ImportError:
    from apply_appearance_color_review_labels import validate_labels


MULTILABEL_COLORS = frozenset({
    "black", "white", "gray", "red", "blue",
    "green", "yellow", "brown", "purple",
})


def _write_review_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image_path", "appearance_log_id", "upper_color", "lower_color",
        "upper_reviewed", "lower_reviewed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_reviewed_labels(manifest_path: Path, labels_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = validate_labels(json.loads(labels_path.read_text(encoding="utf-8")))
    manifest_by_id = {int(item["id"]): item for item in manifest.get("items", [])}
    rows, audited = [], []
    summary = {
        "rows": 0,
        "partial_reviews": 0,
        "excluded_fields": 0,
        "missing_crops": 0,
        "multilabel_unsupported_fields": 0,
    }
    for label in labels:
        source = manifest_by_id.get(label["id"])
        if source is None:
            raise ValueError(f"manifest id not found: {label['id']}")
        crop_path = Path(source.get("crop_path", ""))
        if not crop_path.exists():
            summary["missing_crops"] += 1
        reviewed = {}
        for field in ("upper_color", "lower_color"):
            value = label[field]
            if value == "exclude":
                summary["excluded_fields"] += 1
                reviewed[field] = ""
            elif value is None:
                reviewed[field] = ""
            else:
                reviewed[field] = value
                if value not in MULTILABEL_COLORS:
                    summary["multilabel_unsupported_fields"] += 1
        upper_reviewed = bool(reviewed["upper_color"])
        lower_reviewed = bool(reviewed["lower_color"])
        if upper_reviewed != lower_reviewed:
            summary["partial_reviews"] += 1
        rows.append({
            "image_path": str(crop_path),
            "appearance_log_id": label["id"],
            "upper_color": reviewed["upper_color"],
            "lower_color": reviewed["lower_color"],
            "upper_reviewed": str(upper_reviewed).lower(),
            "lower_reviewed": str(lower_reviewed).lower(),
        })
        audited.append({**source, "human_review": label})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_review_csv(output_dir / "reviewed_appearance_colors.csv", rows)
    (output_dir / "reviewed_appearance_colors.json").write_text(
        json.dumps({"schema_version": 1, "items": audited}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["rows"] = len(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
```

- [ ] **Step 5: 내보내기 테스트 통과 확인**

Run: `rtk pytest tests/test_export_appearance_color_review_labels.py -q`

Expected: PASS.

- [ ] **Step 6: 작업 단위 커밋**

```bash
rtk git add scripts/ops/export_appearance_color_review_labels.py tests/test_export_appearance_color_review_labels.py
rtk git commit -m "feat: export reviewed appearance training labels"
```

### Task 4: 운영 사용법과 회귀 검증

**Files:**
- Create: `docs/guides/APPEARANCE_COLOR_REVIEW.md`
- Modify: `docs/superpowers/plans/2026-08-11-appearance-color-review.md` (체크박스 상태만 갱신)

**Interfaces:**
- Documents: manifest 생성 → HTML 생성 → JSON 다운로드 → dry-run → apply → 학습 라벨 export

- [ ] **Step 1: 실행 가능한 운영 명령 문서화**

```bash
rtk test python scripts/ops/build_appearance_color_review_manifest.py \
  --db data/runtime/appearances.db \
  --output data/runtime/appearance_color_review_manifest.json \
  --limit 200

rtk test python scripts/ops/build_appearance_review_html.py \
  --manifest data/runtime/appearance_color_review_manifest.json \
  --output data/runtime/appearance_color_review.html

rtk test python scripts/ops/apply_appearance_color_review_labels.py \
  --db data/runtime/appearances.db \
  --labels /path/to/appearance_color_review_labels.json

rtk test python scripts/ops/apply_appearance_color_review_labels.py \
  --db data/runtime/appearances.db \
  --labels /path/to/appearance_color_review_labels.json \
  --apply

rtk test python scripts/ops/export_appearance_color_review_labels.py \
  --manifest data/runtime/appearance_color_review_manifest.json \
  --labels /path/to/appearance_color_review_labels.json \
  --output-dir data/training/appearance_color_reviews
```

문서에는 dry-run과 백업 확인, 브라우저 종료 전 JSON 다운로드, 라벨 누적만으로 모델이 자동 학습되지 않는다는 점을 명시한다.

- [ ] **Step 2: 관련 테스트 전체 실행**

Run: `rtk pytest tests/test_build_appearance_review_html.py tests/test_apply_appearance_color_review_labels.py tests/test_export_appearance_color_review_labels.py tests/test_audit_appearance_colors.py tests/test_appearance_log.py -q`

Expected: 모든 테스트 PASS.

- [ ] **Step 3: Python 문법 검증**

Run: `rtk test python -m compileall -q scripts/ops/build_appearance_review_html.py scripts/ops/apply_appearance_color_review_labels.py scripts/ops/export_appearance_color_review_labels.py`

Expected: exit 0.

- [ ] **Step 4: 변경 범위와 사용자 작업 보존 확인**

Run: `rtk git status --short`

Expected: 이번 계획 파일과 구현 대상 외의 기존 수정 파일은 내용이 변경되지 않았다.

- [ ] **Step 5: 문서 및 최종 작업 커밋**

```bash
rtk git add docs/guides/APPEARANCE_COLOR_REVIEW.md docs/superpowers/plans/2026-08-11-appearance-color-review.md
rtk git commit -m "docs: explain appearance color review workflow"
```
