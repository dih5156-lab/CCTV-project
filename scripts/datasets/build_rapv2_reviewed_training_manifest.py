#!/usr/bin/env python3
"""Merge quality-gated RAPv2 color labels into an existing JSONL manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SUPPORTED_COLORS = {
    "black",
    "white",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "purple",
    "navy",
    "orange",
}
COLOR_FIELDS = ("upper_color", "lower_color")


def build_quality_report(items: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        field = str(item.get("field", ""))
        current = str(item.get("current_label", ""))
        if field in COLOR_FIELDS and current:
            grouped[f"{field}.{current}"].append(item)

    report: dict[str, dict[str, Any]] = {}
    for key, values in sorted(grouped.items()):
        preserved = sum(
            item.get("review_label") is None
            or item.get("review_label") == item.get("current_label")
            for item in values
        )
        excluded = sum(item.get("review_label") == "exclude" for item in values)
        report[key] = {
            "reviewed": len(values),
            "preserved": preserved,
            "changed": len(values) - preserved - excluded,
            "excluded": excluded,
            "keep_rate": preserved / len(values),
        }
    return report


def _resolved_review_label(item: Mapping[str, Any]) -> str | None:
    review_label = item.get("review_label")
    if review_label is None:
        return str(item.get("current_label", "")).strip().lower() or None
    value = str(review_label).strip().lower()
    return None if value == "exclude" else value


def convert_rap_row(
    source: Mapping[str, str],
    *,
    reviewed_items: Mapping[tuple[str, str], Mapping[str, Any]],
    quality_report: Mapping[str, Mapping[str, Any]],
    supported_colors: set[str],
    minimum_keep_rate: float,
    container_image_root: str,
) -> dict[str, Any]:
    source_path = str(source.get("image_path", ""))
    row: dict[str, Any] = {
        "source": "rapv2_reviewed",
        "source_index": str(source.get("source_index", "")),
        "person_id": f"rapv2:{source.get('group_id', '')}",
        "split": str(source.get("split", "")),
        "image_path": str(Path(container_image_root) / Path(source_path).name),
        "gender": "",
        "gender_defined": False,
        "upper_clothes": "",
        "upper_clothes_defined": False,
        "lower_clothes": "",
        "lower_clothes_defined": False,
        "items": [],
        "items_defined": False,
        "human_reviewed": False,
    }
    for field in COLOR_FIELDS:
        current = str(source.get(field, "")).strip().lower()
        review_item = reviewed_items.get((source_path, field))
        if review_item is not None:
            resolved = _resolved_review_label(review_item)
            row["human_reviewed"] = True
            is_defined = resolved in supported_colors
            row[field] = resolved if is_defined else ""
            row[f"{field}_defined"] = is_defined
            continue
        quality = quality_report.get(f"{field}.{current}", {})
        keep_rate = float(quality.get("keep_rate", 0.0))
        is_defined = current in supported_colors and keep_rate >= minimum_keep_rate
        row[field] = current if is_defined else ""
        row[f"{field}_defined"] = is_defined
    return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sanitize_base_rows(
    rows: list[dict[str, Any]], supported_colors: set[str]
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for field in COLOR_FIELDS:
            if row.get(f"{field}_defined") is True and row.get(field) not in supported_colors:
                row[field] = ""
                row[f"{field}_defined"] = False
        sanitized.append(row)
    return sanitized


def _select_quality_rows(
    rows: list[dict[str, Any]], *, max_rows_per_label: int
) -> list[dict[str, Any]]:
    forced = [row for row in rows if row["human_reviewed"]]
    forced_ids = {str(row["source_index"]) for row in forced}
    selected_ids = set(forced_ids)
    for field in COLOR_FIELDS:
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row[f"{field}_defined"] and str(row["source_index"]) not in forced_ids:
                by_label[str(row[field])].append(row)
        for candidates in by_label.values():
            candidates.sort(
                key=lambda row: hashlib.sha256(
                    f"{field}:{row['source_index']}".encode("utf-8")
                ).digest()
            )
            selected_ids.update(
                str(row["source_index"]) for row in candidates[:max_rows_per_label]
            )
    return [row for row in rows if str(row["source_index"]) in selected_ids]


def _distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in COLOR_FIELDS:
        values = Counter(
            str(row[field])
            for row in rows
            if row.get(f"{field}_defined") is True
        )
        result[field] = {"defined_rows": sum(values.values()), "values": dict(sorted(values.items()))}
    return result


def build_combined_manifest(
    *,
    review_json: Path,
    rap_manifest_csv: Path,
    base_manifest_jsonl: Path,
    output_manifest_jsonl: Path,
    report_json: Path,
    minimum_keep_rate: float,
    max_rows_per_label: int,
    reviewed_train_repeat: int,
    container_image_root: str,
) -> dict[str, Any]:
    review_document = json.loads(review_json.read_text(encoding="utf-8"))
    review_items = review_document.get("items", [])
    if not isinstance(review_items, list) or not review_items:
        raise ValueError("review JSON must contain non-empty items")
    keyed_reviews = {
        (str(item["image_path"]), str(item["field"])): item for item in review_items
    }
    if len(keyed_reviews) != len(review_items):
        raise ValueError("review JSON contains duplicate image/field entries")
    quality_report = build_quality_report(review_items)
    converted = [
        convert_rap_row(
            row,
            reviewed_items=keyed_reviews,
            quality_report=quality_report,
            supported_colors=SUPPORTED_COLORS,
            minimum_keep_rate=minimum_keep_rate,
            container_image_root=container_image_root,
        )
        for row in _read_csv(rap_manifest_csv)
    ]
    usable = [
        row
        for row in converted
        if any(row[f"{field}_defined"] for field in COLOR_FIELDS)
    ]
    selected = _select_quality_rows(usable, max_rows_per_label=max_rows_per_label)
    repeated: list[dict[str, Any]] = []
    for row in selected:
        repeated.append(row)
        if row["human_reviewed"] and row["split"] == "train":
            for repeat_index in range(1, reviewed_train_repeat):
                duplicate = dict(row)
                duplicate["review_repeat_index"] = repeat_index
                repeated.append(duplicate)

    base_rows = _sanitize_base_rows(_read_jsonl(base_manifest_jsonl), SUPPORTED_COLORS)
    combined = [*base_rows, *repeated]
    output_manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_jsonl.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "review_json": str(review_json),
        "base_manifest": str(base_manifest_jsonl),
        "base_rows": len(base_rows),
        "rap_rows_usable": len(usable),
        "rap_rows_selected": len(selected),
        "human_reviewed_rows": sum(row["human_reviewed"] for row in selected),
        "reviewed_train_repeat": reviewed_train_repeat,
        "combined_rows": len(combined),
        "minimum_keep_rate": minimum_keep_rate,
        "max_rows_per_label": max_rows_per_label,
        "quality_report": quality_report,
        "rap_distribution": _distribution(selected),
        "combined_distribution": _distribution(combined),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--rap-manifest", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--minimum-keep-rate", type=float, default=0.60)
    parser.add_argument("--max-rows-per-label", type=int, default=3000)
    parser.add_argument("--reviewed-train-repeat", type=int, default=5)
    parser.add_argument(
        "--container-image-root",
        default="/app/data/datasets/rapv2/RAP_dataset",
    )
    args = parser.parse_args()
    if not 0.0 <= args.minimum_keep_rate <= 1.0:
        raise SystemExit("--minimum-keep-rate must be between 0 and 1")
    if args.max_rows_per_label < 1 or args.reviewed_train_repeat < 1:
        raise SystemExit("row limits and repeat count must be positive")
    report = build_combined_manifest(
        review_json=args.review_json,
        rap_manifest_csv=args.rap_manifest,
        base_manifest_jsonl=args.base_manifest,
        output_manifest_jsonl=args.output_manifest,
        report_json=args.report_json,
        minimum_keep_rate=args.minimum_keep_rate,
        max_rows_per_label=args.max_rows_per_label,
        reviewed_train_repeat=args.reviewed_train_repeat,
        container_image_root=args.container_image_root,
    )
    print(json.dumps({key: report[key] for key in ("base_rows", "rap_rows_selected", "combined_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
