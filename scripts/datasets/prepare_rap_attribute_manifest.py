#!/usr/bin/env python3
"""Convert RAP/RAPv2 pedestrian attribute annotations into a compact manifest.

The script intentionally does not download RAP/RAPv2. Those datasets usually
require following the dataset owner's access and license terms first.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
COLORS = ("black", "white", "gray", "red", "blue", "green", "yellow", "brown", "purple")
OUTPUT_FIELDS = (
    "image_path",
    "gender",
    "upper_color",
    "lower_color",
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
    if any("female" in item or "woman" in item for item in normalized):
        return "female"
    if any(
        ("male" in item or "man" in item)
        and "female" not in item
        and "woman" not in item
        for item in normalized
    ):
        return "male"
    return "unknown"


def _choose_color(active: Iterable[str], *, part: str) -> str:
    normalized = [_normalize_name(item) for item in active]
    part_tokens = {
        "upper": ("upper", "top", "shirt", "coat", "jacket", "torso"),
        "lower": ("lower", "bottom", "pants", "trousers", "skirt", "dress"),
    }[part]
    for color in COLORS:
        for item in normalized:
            if color in item and any(token in item for token in part_tokens):
                return color
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
    image_path = str(Path(image_root) / image_name) if image_root else image_name
    return {
        "image_path": image_path,
        "gender": _choose_gender(active),
        "upper_color": _choose_color(active, part="upper"),
        "lower_color": _choose_color(active, part="lower"),
        "bag": _choose_yes_no_unknown(active, ("bag", "backpack", "handbag")),
        "hat": _choose_yes_no_unknown(active, ("hat", "cap")),
        "source_active_attributes": ";".join(active),
    }


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
    parser.add_argument("--image-root", default="", help="Prefix to prepend to image paths.")
    parser.add_argument("--output-csv", type=Path, help="Output canonical manifest CSV.")
    parser.add_argument("--inspect-json", type=Path, help="Write MAT structure summary JSON and exit.")
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

    source_rows = _read_attribute_csv(args.annotations_csv) if args.annotations_csv else _mat_to_rows(args.mat)
    manifest = [canonicalize_row(row, image_root=args.image_root) for row in source_rows]
    _write_manifest(args.output_csv, manifest)
    print(f"wrote {args.output_csv} rows={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
