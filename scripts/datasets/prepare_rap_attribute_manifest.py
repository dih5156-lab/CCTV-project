#!/usr/bin/env python3
"""Convert RAP/RAPv2 pedestrian attribute annotations into a compact manifest.

The script intentionally does not download RAP/RAPv2. Those datasets usually
require following the dataset owner's access and license terms first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import pickle
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
COLORS = (
    "black",
    "white",
    "gray",
    "red",
    "green",
    "blue",
    "navy",
    "silver",
    "yellow",
    "brown",
    "purple",
    "pink",
    "orange",
    "mixture",
    "other",
)
OUTPUT_FIELDS = (
    "image_path",
    "split",
    "source_split",
    "source_index",
    "group_id",
    "gender",
    "upper_color",
    "upper_color_labels",
    "lower_color",
    "lower_color_labels",
    "bag",
    "hat",
    "source_active_attributes",
)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _looks_like_image_name(value: str) -> bool:
    return value.lower().endswith(IMAGE_EXTENSIONS)


def _active_attributes(row: dict[str, str]) -> list[str]:
    active: list[str] = []
    for key, value in row.items():
        if key in {"image_path", "file_name", "filename", "name", "attributes"}:
            continue
        normalized_value = str(value).strip().lower()
        if normalized_value in {"1", "true", "yes", "y"}:
            active.append(key)
    if row.get("attributes"):
        active.extend(
            item.strip()
            for item in re.split(r"[;|,]", row["attributes"])
            if item.strip()
        )
    return active


def _choose_gender(active: Iterable[str]) -> str:
    normalized = [_normalize_name(item) for item in active]
    if any("female" in item or "femal" in item or "woman" in item for item in normalized):
        return "female"
    if any(
        ("male" in item or "man" in item)
        and "female" not in item
        and "woman" not in item
        for item in normalized
    ):
        return "male"
    return "unknown"


def _color_labels(active: Iterable[str], *, part: str) -> list[str]:
    normalized = [_normalize_name(item) for item in active]
    part_tokens = {
        "upper": ("upper", "up", "ub", "top", "shirt", "coat", "jacket", "torso"),
        "lower": ("lower", "lb", "bottom", "pants", "trousers", "skirt", "dress"),
    }[part]
    return [
        color
        for color in COLORS
        if any(color in item and any(token in item for token in part_tokens) for item in normalized)
    ]


def _choose_color(active: Iterable[str], *, part: str) -> str:
    matched = _color_labels(active, part=part)
    if "mixture" in matched:
        return "mixture"
    concrete = [color for color in matched if color != "other"]
    if len(concrete) == 1:
        return concrete[0]
    if len(concrete) > 1:
        return "mixture"
    if "other" in matched:
        return "other"
    return "unknown"


def _choose_yes_no_unknown(active: Iterable[str], keywords: tuple[str, ...]) -> str:
    normalized = [_normalize_name(item) for item in active]
    if any(any(keyword in item for keyword in keywords) for item in normalized):
        return "yes"
    return "unknown"


def canonicalize_row(row: dict[str, str], *, image_root: str = "") -> dict[str, str]:
    image_name = (
        row.get("image_path")
        or row.get("file_name")
        or row.get("filename")
        or row.get("name")
        or ""
    )
    active = _active_attributes(row)
    upper_colors = _color_labels(active, part="upper")
    lower_colors = _color_labels(active, part="lower")
    image_path = str(Path(image_root) / image_name) if image_root else image_name
    return {
        "image_path": image_path,
        "split": row.get("split", ""),
        "source_split": row.get("source_split", ""),
        "source_index": row.get("source_index", ""),
        "group_id": row.get("group_id", ""),
        "gender": _choose_gender(active),
        "upper_color": _choose_color(active, part="upper"),
        "upper_color_labels": ";".join(upper_colors),
        "lower_color": _choose_color(active, part="lower"),
        "lower_color_labels": ";".join(lower_colors),
        "bag": _choose_yes_no_unknown(active, ("bag", "backpack", "handbag")),
        "hat": _choose_yes_no_unknown(active, ("hat", "cap")),
        "source_active_attributes": ";".join(active),
    }


def _group_key_from_image_name(image_name: str) -> str:
    """Return a stable track-level key so adjacent frames stay in one split."""
    normalized = Path(image_name).name
    return re.sub(r"-frame\d+(?:-line\d+)?(?=\.[^.]+$)", "", normalized)


def _group_split(group_id: str, *, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "val"
    return "test"


def _source_split_map(partition: Any, *, image_count: int) -> dict[int, str]:
    result: dict[int, str] = {}
    if not hasattr(partition, "items"):
        return result
    for split in ("train", "val", "test"):
        values = partition.get(split, [])
        for raw_index in values:
            index = int(raw_index)
            if 0 <= index < image_count:
                result.setdefault(index, split)
    return result


def _pkl_payload_to_rows(
    payload: dict[str, Any],
    *,
    image_root: str,
    split_mode: str,
    split_seed: str,
) -> list[dict[str, str]]:
    image_names = [str(value) for value in payload.get("image_name", [])]
    attr_names = [str(value) for value in payload.get("attr_name", [])]
    labels = payload.get("label")
    if not image_names or not attr_names or labels is None:
        raise SystemExit("RAPv2 PKL에 image_name, attr_name 또는 label이 없습니다.")
    if tuple(getattr(labels, "shape", ())) != (len(image_names), len(attr_names)):
        raise SystemExit(
            "RAPv2 PKL 라벨 크기가 맞지 않습니다: "
            f"images={len(image_names)} attrs={len(attr_names)} "
            f"label_shape={getattr(labels, 'shape', None)}"
        )

    source_splits = _source_split_map(payload.get("partition", {}), image_count=len(image_names))
    female_indexes = {
        index
        for index, name in enumerate(attr_names)
        if _normalize_name(name) in {"female", "femal"}
    }
    rows: list[dict[str, str]] = []
    for index, (image_name, label_row) in enumerate(zip(image_names, labels)):
        active = [attr for attr, value in zip(attr_names, label_row) if int(value) > 0]
        group_id = _group_key_from_image_name(image_name)
        source_split = source_splits.get(index, "unassigned")
        if split_mode == "group-hash":
            split = _group_split(group_id, seed=split_seed)
        elif split_mode == "source":
            split = source_split
        else:
            split = "all"
        row = canonicalize_row(
            {
                "image_path": image_name,
                "attributes": ";".join(active),
                "split": split,
                "source_split": source_split,
                "source_index": str(index),
                "group_id": group_id,
            },
            image_root=image_root,
        )
        if female_indexes:
            row["gender"] = (
                "female" if any(int(label_row[index]) > 0 for index in female_indexes) else "male"
            )
        rows.append(row)
    return rows


def _load_rap_pickle(path: Path) -> dict[str, Any]:
    """Load the known RAPv2 pickle while blocking arbitrary pickle globals."""
    try:
        import numpy as np  # type: ignore
        from easydict import EasyDict  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("RAPv2 PKL 변환에는 numpy와 easydict가 필요합니다.") from exc

    allowed = {
        ("easydict", "EasyDict"): EasyDict,
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            key = (module, name)
            if key not in allowed:
                raise pickle.UnpicklingError(f"허용되지 않은 PKL 타입: {module}.{name}")
            return allowed[key]

    with path.open("rb") as handle:
        payload = RestrictedUnpickler(handle, encoding="latin1").load()
    if not isinstance(payload, dict):
        raise SystemExit(f"RAPv2 PKL 최상위 타입이 dict가 아닙니다: {type(payload).__name__}")
    return payload


def _password_candidates(path: Path) -> list[bytes]:
    return [line.strip().encode() for line in path.read_text().splitlines() if line.strip()]


def _find_zip_password(archive: zipfile.ZipFile, password_file: Path) -> bytes | None:
    encrypted = next((item for item in archive.infolist() if item.flag_bits & 1 and not item.is_dir()), None)
    if encrypted is None:
        return None
    for candidate in _password_candidates(password_file):
        try:
            with archive.open(encrypted, pwd=candidate) as handle:
                handle.read(1)
            return candidate
        except RuntimeError:
            continue
    raise SystemExit(f"압축 암호가 맞지 않습니다: {password_file}")


def _extract_dataset_zip(zip_path: Path, output_root: Path, password_file: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    output_root_resolved = output_root.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        password = _find_zip_password(archive, password_file)
        for member in archive.infolist():
            destination = (output_root / member.filename).resolve()
            if output_root_resolved not in destination.parents and destination != output_root_resolved:
                raise SystemExit(f"안전하지 않은 ZIP 경로를 발견했습니다: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.exists() and destination.stat().st_size == member.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, pwd=password) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    return output_root / "RAP_dataset"


def _build_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "row_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "source_split_counts": dict(Counter(row["source_split"] for row in rows)),
        "gender_counts": dict(Counter(row["gender"] for row in rows)),
    }
    for field in ("upper_color", "lower_color"):
        stats[f"{field}_counts"] = dict(Counter(row[field] for row in rows))
        labels_field = f"{field}_labels"
        stats[f"{field}_multilabel_counts"] = dict(
            Counter(
                color
                for row in rows
                for color in row[labels_field].split(";")
                if color
            )
        )
        stats[f"{field}_by_split"] = {
            split: dict(Counter(row[field] for row in rows if row["split"] == split))
            for split in ("train", "val", "test")
        }
    return stats


def _write_review_html(path: Path, rows: list[dict[str, str]], *, per_color: int) -> int:
    selected: list[tuple[str, str, dict[str, str]]] = []
    for color in COLORS:
        candidates_by_field: dict[str, list[dict[str, str]]] = {}
        for field in ("upper_color", "lower_color"):
            candidates = [row for row in rows if row[field] == color]
            candidates.sort(key=lambda row: hashlib.sha256(row["image_path"].encode()).digest())
            candidates_by_field[field] = candidates[:per_color]
        for sample_index in range(per_color):
            for field in ("upper_color", "lower_color"):
                candidates = candidates_by_field[field]
                if sample_index < len(candidates):
                    selected.append((field, color, candidates[sample_index]))

    options = "".join(f"<option>{html.escape(color)}</option>" for color in COLORS)
    table_rows: list[str] = []
    for item_id, (field, color, row) in enumerate(selected):
        image_path = Path(row["image_path"]).resolve()
        image_src = Path(os.path.relpath(image_path, path.parent.resolve())).as_posix()
        table_rows.append(
            "<tr>"
            f"<td>{item_id}</td><td>{html.escape(field)}</td><td>{html.escape(color)}</td>"
            f"<td><img src='{html.escape(image_src)}' loading='lazy'></td>"
            f"<td><select data-id='{item_id}' data-path='{html.escape(row['image_path'])}' "
            f"data-field='{html.escape(field)}' data-current='{html.escape(color)}'>"
            "<option value=''>정상</option>"
            f"{options}<option>exclude</option></select></td></tr>"
        )
    script = """
function downloadLabels() {
  const items = [...document.querySelectorAll('select[data-id]')].map(select => ({
    id: Number(select.dataset.id), image_path: select.dataset.path,
    field: select.dataset.field, current_label: select.dataset.current,
    review_label: select.value || null
  }));
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([JSON.stringify({schema_version:1, items}, null, 2)], {type:'application/json'}));
  link.download = 'rapv2_color_review_labels.json'; link.click();
}
"""
    selected_counts = Counter(field for field, _, _ in selected)
    document = f"""<!doctype html><meta charset='utf-8'><title>RAPv2 color review</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee}}table{{border-collapse:collapse}}td,th{{border:1px solid #555;padding:6px}}img{{max-width:240px;max-height:260px}}button{{margin:12px;padding:8px}}</style>
<h1>RAPv2 상·하의 색상 표본 검수 ({len(selected)}건)</h1>
<p>상의 {selected_counts['upper_color']}건 / 하의 {selected_counts['lower_color']}건</p>
<p>라벨이 맞으면 정상, 틀리면 올바른 색상, 사용할 수 없으면 exclude를 선택하세요.</p>
<button onclick='downloadLabels()'>검수 라벨 JSON 다운로드</button>
<table><thead><tr><th>ID</th><th>영역</th><th>현재 라벨</th><th>이미지</th><th>검수</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<script>{script}</script>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return len(selected)


def _read_attribute_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise SystemExit(
            "RAP .mat 변환에는 scipy가 필요합니다. "
            "설치 후 다시 실행하세요: pip install scipy"
        ) from exc
    return loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _iter_values(value: Any, *, depth: int = 0) -> Iterable[Any]:
    if depth > 8:
        return
    yield value
    if hasattr(value, "_fieldnames"):
        for field in value._fieldnames:
            yield from _iter_values(getattr(value, field), depth=depth + 1)
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_values(child, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_values(child, depth=depth + 1)


def _to_string_list(value: Any) -> list[str]:
    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover
        np = None  # type: ignore

    if np is not None and isinstance(value, np.ndarray):
        flattened = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        flattened = list(value)
    else:
        flattened = [value]

    result: list[str] = []
    for item in flattened:
        if isinstance(item, bytes):
            result.append(item.decode("utf-8", errors="ignore"))
        elif isinstance(item, str):
            result.append(item)
        elif hasattr(item, "tolist"):
            nested = _to_string_list(item.tolist())
            result.extend(nested)
    return [item.strip() for item in result if item.strip()]


def _find_image_names(mat_payload: dict[str, Any]) -> list[str]:
    best: list[str] = []
    for value in _iter_values(mat_payload):
        strings = _to_string_list(value)
        image_names = [item for item in strings if _looks_like_image_name(item)]
        if len(image_names) > len(best):
            best = image_names
    return best


def _find_attribute_names(mat_payload: dict[str, Any]) -> list[str]:
    best: list[str] = []
    for value in _iter_values(mat_payload):
        strings = _to_string_list(value)
        if not strings or any(_looks_like_image_name(item) for item in strings):
            continue
        normalized = [_normalize_name(item) for item in strings]
        hits = sum(
            any(token in item for token in ("gender", "female", "male", "upper", "lower", "bag", "hat"))
            for item in normalized
        )
        if hits >= 2 and len(strings) > len(best):
            best = strings
    return best


def _find_label_matrix(mat_payload: dict[str, Any], *, rows: int, cols: int) -> list[list[int]]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("RAP .mat 변환에는 numpy가 필요합니다.") from exc

    candidates: list[Any] = []
    for value in _iter_values(mat_payload):
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if value.shape == (rows, cols):
                candidates.append(value)
            elif value.shape == (cols, rows):
                candidates.append(value.T)
    if not candidates:
        raise SystemExit(
            f"라벨 행렬을 찾지 못했습니다. expected shape=({rows}, {cols}). "
            "먼저 --inspect-json으로 annotation 구조를 확인하세요."
        )
    matrix = candidates[0].astype(int)
    return matrix.tolist()


def _mat_to_rows(path: Path) -> list[dict[str, str]]:
    payload = _load_mat(path)
    image_names = _find_image_names(payload)
    attr_names = _find_attribute_names(payload)
    if not image_names or not attr_names:
        raise SystemExit(
            "RAP annotation에서 이미지명 또는 속성명을 찾지 못했습니다. "
            "--inspect-json으로 구조를 확인한 뒤 CSV 방식으로 변환하세요."
        )
    labels = _find_label_matrix(payload, rows=len(image_names), cols=len(attr_names))
    rows: list[dict[str, str]] = []
    for image_name, label_row in zip(image_names, labels):
        row = {"image_path": image_name}
        for attr_name, value in zip(attr_names, label_row):
            row[attr_name] = str(int(value))
        rows.append(row)
    return rows


def _inspect_mat(path: Path) -> dict[str, Any]:
    payload = _load_mat(path)
    image_names = _find_image_names(payload)
    attr_names = _find_attribute_names(payload)
    return {
        "mat_file": str(path),
        "image_count_guess": len(image_names),
        "attribute_count_guess": len(attr_names),
        "image_name_sample": image_names[:5],
        "attribute_name_sample": attr_names[:20],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare RAP/RAPv2 attribute manifest for CCTV appearance fine-tuning.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--annotations-csv", type=Path, help="Normalized RAP attribute CSV.")
    input_group.add_argument("--mat", type=Path, help="RAP/RAPv2 MATLAB annotation file.")
    input_group.add_argument("--pkl", type=Path, help="RAPv2 EasyDict/NumPy annotation pickle.")
    parser.add_argument("--image-root", default="", help="Prefix to prepend to image paths.")
    parser.add_argument("--output-csv", type=Path, help="Output canonical manifest CSV.")
    parser.add_argument("--inspect-json", type=Path, help="Write MAT structure summary JSON and exit.")
    parser.add_argument("--dataset-zip", type=Path, help="Optional encrypted RAPv2 image ZIP.")
    parser.add_argument("--password-file", type=Path, help="Password candidate file for --dataset-zip.")
    parser.add_argument("--extract-root", type=Path, help="Project-local extraction destination.")
    parser.add_argument(
        "--split-mode",
        choices=("group-hash", "source", "none"),
        default="group-hash",
        help="Use track-group hashing for full RAPv2, source PKL partitions, or no split.",
    )
    parser.add_argument("--split-seed", default="cctv-rapv2-v1-20")
    parser.add_argument("--stats-json", type=Path, help="Write class/split distribution JSON.")
    parser.add_argument("--review-html", type=Path, help="Write a color sample review page.")
    parser.add_argument("--review-per-color", type=int, default=20)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.inspect_json:
        if not args.mat:
            raise SystemExit("--inspect-json은 --mat과 함께 사용하세요.")
        summary = _inspect_mat(args.mat)
        args.inspect_json.parent.mkdir(parents=True, exist_ok=True)
        args.inspect_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.inspect_json}")
        return 0

    if not args.output_csv:
        raise SystemExit("--output-csv가 필요합니다.")

    image_root = args.image_root
    if args.dataset_zip:
        if not args.password_file or not args.extract_root:
            raise SystemExit("--dataset-zip에는 --password-file과 --extract-root가 필요합니다.")
        extracted_images = _extract_dataset_zip(args.dataset_zip, args.extract_root, args.password_file)
        if not image_root:
            image_root = str(extracted_images)

    if args.pkl:
        manifest = _pkl_payload_to_rows(
            _load_rap_pickle(args.pkl),
            image_root=image_root,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
        )
    else:
        source_rows = (
            _read_attribute_csv(args.annotations_csv)
            if args.annotations_csv
            else _mat_to_rows(args.mat)
        )
        manifest = [canonicalize_row(row, image_root=image_root) for row in source_rows]
    _write_manifest(args.output_csv, manifest)
    print(f"wrote {args.output_csv} rows={len(manifest)}")
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(
            json.dumps(_build_stats(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.stats_json}")
    if args.review_html:
        review_count = _write_review_html(
            args.review_html,
            manifest,
            per_color=max(1, args.review_per_color),
        )
        print(f"wrote {args.review_html} rows={review_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
