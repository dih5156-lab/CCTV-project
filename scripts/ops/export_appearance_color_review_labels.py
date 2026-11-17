#!/usr/bin/env python3
"""Export reviewed appearance colors as auditable training-source labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .apply_appearance_color_review_labels import validate_labels
except ImportError:
    from apply_appearance_color_review_labels import validate_labels

COLOR_FIELDS = ("upper_color", "lower_color")
MULTILABEL_COLORS = frozenset(
    {
        "black",
        "white",
        "gray",
        "red",
        "blue",
        "green",
        "yellow",
        "brown",
        "purple",
    }
)
CSV_FIELDS = (
    "image_path",
    "appearance_log_id",
    "upper_color",
    "lower_color",
    "upper_reviewed",
    "lower_reviewed",
)


def _manifest_items_by_id(manifest: dict[str, Any]) -> dict[int, dict]:
    items_by_id: dict[int, dict] = {}
    for item in manifest.get("items", []):
        item_id = int(item["id"])
        if item_id in items_by_id:
            raise ValueError(f"duplicate manifest id: {item_id}")
        items_by_id[item_id] = item
    return items_by_id


def _reviewed_value(value: str | None) -> str:
    if value is None or value == "exclude":
        return ""
    return value


def _write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_reviewed_labels(
    manifest_path: Path,
    labels_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = validate_labels(labels_payload)
    manifest_by_id = _manifest_items_by_id(manifest)

    rows: list[dict[str, Any]] = []
    audited: list[dict[str, Any]] = []
    upper_colors: Counter[str] = Counter()
    lower_colors: Counter[str] = Counter()
    reviewed_items = 0
    unreviewed_items = 0
    partial_reviews = 0
    excluded_fields = 0
    missing_crops = 0
    unsupported_fields = 0

    for label in labels:
        source = manifest_by_id.get(label["id"])
        if source is None:
            raise ValueError(f"manifest id not found: {label['id']}")

        selected_values = [label[field] for field in COLOR_FIELDS]
        has_review = any(value is not None for value in selected_values)
        if has_review:
            reviewed_items += 1
        else:
            unreviewed_items += 1

        crop_path = Path(source.get("crop_path", ""))
        if not crop_path.is_file():
            missing_crops += 1
            audited.append(
                {
                    **source,
                    "human_review": label,
                    "export_status": "missing_crop",
                }
            )
            continue

        reviewed = {
            field: _reviewed_value(label[field]) for field in COLOR_FIELDS
        }
        for field in COLOR_FIELDS:
            value = label[field]
            if value == "exclude":
                excluded_fields += 1
            elif value is not None and value not in MULTILABEL_COLORS:
                unsupported_fields += 1

        upper_reviewed = bool(reviewed["upper_color"])
        lower_reviewed = bool(reviewed["lower_color"])
        if upper_reviewed != lower_reviewed:
            partial_reviews += 1

        if upper_reviewed:
            upper_colors[reviewed["upper_color"]] += 1
        if lower_reviewed:
            lower_colors[reviewed["lower_color"]] += 1

        export_status = "exported"
        if not upper_reviewed and not lower_reviewed:
            export_status = "excluded" if has_review else "unreviewed"
        else:
            rows.append(
                {
                    "image_path": str(crop_path),
                    "appearance_log_id": label["id"],
                    "upper_color": reviewed["upper_color"],
                    "lower_color": reviewed["lower_color"],
                    "upper_reviewed": str(upper_reviewed).lower(),
                    "lower_reviewed": str(lower_reviewed).lower(),
                }
            )

        audited.append(
            {
                **source,
                "human_review": label,
                "export_status": export_status,
            }
        )

    summary: dict[str, Any] = {
        "manifest": str(manifest_path),
        "labels": str(labels_path),
        "reviewed_items": reviewed_items,
        "unreviewed_items": unreviewed_items,
        "exported_rows": len(rows),
        "partial_reviews": partial_reviews,
        "excluded_fields": excluded_fields,
        "missing_crops": missing_crops,
        "multilabel_unsupported_fields": unsupported_fields,
        "upper_colors": dict(sorted(upper_colors.items())),
        "lower_colors": dict(sorted(lower_colors.items())),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_review_csv(output_dir / "reviewed_appearance_colors.csv", rows)
    (output_dir / "reviewed_appearance_colors.json").write_text(
        json.dumps(
            {"schema_version": 1, "items": audited},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = export_reviewed_labels(
        manifest_path=args.manifest,
        labels_path=args.labels,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
