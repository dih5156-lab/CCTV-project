#!/usr/bin/env python3
"""Prepare retraining inputs from manually reviewed fall-model errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data/fall_eval/auto/train_manifest.jsonl"
DEFAULT_ERROR_MANIFEST = (
    PROJECT_ROOT
    / "data/fall_error_review/full_14702/fall_error_review_manifest.jsonl"
)
DEFAULT_LABELS = PROJECT_ROOT / "fall_error_review_labels.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/fall_eval/reviewed_retrain_20260812"
ALLOWED_LABELS = {"fall", "non_fall", "ambiguous", "exclude"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_pointer:
        return [json.loads(line) for line in file_pointer if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_pointer:
        for row in rows:
            file_pointer.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_review_inputs(
    *,
    train_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    labels_payload: dict[str, Any],
    additional_train_rows: list[dict[str, Any]],
    additional_error_rows: list[dict[str, Any]],
    additional_labels_payload: dict[str, Any],
) -> dict[str, Any]:
    merged_train_rows = [*train_rows, *additional_train_rows]
    merged_error_rows = [*error_rows, *additional_error_rows]
    merged_label_items = [
        *(labels_payload.get("items") or []),
        *(additional_labels_payload.get("items") or []),
    ]
    scene_ids = [str(row.get("scene_id") or "") for row in merged_train_rows]
    review_ids = [str(row.get("review_id") or "") for row in merged_error_rows]
    label_review_ids = [str(item.get("review_id") or "") for item in merged_label_items]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("duplicate scene_id while merging review inputs")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("duplicate review_id while merging error rows")
    if len(label_review_ids) != len(set(label_review_ids)):
        raise ValueError("duplicate review_id while merging label items")
    return {
        "train_rows": merged_train_rows,
        "error_rows": merged_error_rows,
        "labels_payload": {"schema_version": 1, "items": merged_label_items},
    }


def prepare_retraining_inputs(
    *,
    train_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    labels_payload: dict[str, Any],
    reviewed_weight: float,
) -> dict[str, Any]:
    if reviewed_weight < 1.0:
        raise ValueError("reviewed_weight must be at least 1.0")
    if labels_payload.get("schema_version") != 1 or not isinstance(
        labels_payload.get("items"), list
    ):
        raise ValueError("labels must use schema_version=1 with items")

    errors_by_review_id = {str(row["review_id"]): row for row in error_rows}
    labels_by_review_id: dict[str, str] = {}
    for item in labels_payload["items"]:
        review_id = str(item.get("review_id") or "")
        label = str(item.get("label") or "")
        if review_id not in errors_by_review_id:
            raise ValueError(f"unknown review_id: {review_id}")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"invalid review label: {label}")
        if review_id in labels_by_review_id:
            raise ValueError(f"duplicate review_id: {review_id}")
        labels_by_review_id[review_id] = label

    holdout_labels: dict[str, str] = {}
    validation_feedback_preserved = 0
    for review_id, label in labels_by_review_id.items():
        error = errors_by_review_id[review_id]
        if error["evaluation_split"] == "validation":
            validation_feedback_preserved += 1
            continue
        holdout_labels[str(error["scene_id"])] = label

    prepared_rows: list[dict[str, Any]] = []
    reviewed_items: list[dict[str, Any]] = []
    excluded_from_train = 0
    corrected_labels = 0
    train_scene_ids = {str(row.get("scene_id") or "") for row in train_rows}
    unknown_train_scenes = sorted(set(holdout_labels) - train_scene_ids)
    if unknown_train_scenes:
        raise ValueError(f"reviewed scene missing from train manifest: {unknown_train_scenes[0]}")

    for source_row in train_rows:
        row = dict(source_row)
        scene_id = str(row.get("scene_id") or "")
        reviewed_label = holdout_labels.get(scene_id)
        if reviewed_label in {"exclude", "ambiguous"}:
            excluded_from_train += 1
            continue
        if reviewed_label in {"fall", "non_fall"}:
            original_label = "fall" if bool(row.get("is_fall")) else "non_fall"
            if reviewed_label != original_label:
                corrected_labels += 1
                row["is_fall"] = reviewed_label == "fall"
                row["label"] = "fall" if reviewed_label == "fall" else "not_fall"
            reviewed_items.append(
                {"scene_id": scene_id, "weight": float(reviewed_weight)}
            )
        prepared_rows.append(row)

    summary = {
        "source_train_rows": len(train_rows),
        "prepared_train_rows": len(prepared_rows),
        "reviewed_training_hard_cases": len(reviewed_items),
        "excluded_from_train": excluded_from_train,
        "corrected_labels": corrected_labels,
        "validation_feedback_preserved": validation_feedback_preserved,
        "unreviewed_errors": len(error_rows) - len(labels_by_review_id),
    }
    return {
        "train_rows": prepared_rows,
        "reviewed_hard_cases": {"schema_version": 1, "items": reviewed_items},
        "summary": summary,
    }


def append_reviewed_training_cases(
    *,
    train_rows: list[dict[str, Any]],
    reviewed_hard_cases: dict[str, Any],
    additional_error_rows: list[dict[str, Any]],
    additional_labels_payload: dict[str, Any],
    reviewed_weight: float,
    additional_host_data_root: str | None = None,
    additional_container_data_root: str | None = None,
) -> dict[str, Any]:
    if reviewed_weight < 1.0:
        raise ValueError("reviewed_weight must be at least 1.0")
    if reviewed_hard_cases.get("schema_version") != 1 or not isinstance(
        reviewed_hard_cases.get("items"), list
    ):
        raise ValueError("reviewed hard cases must use schema_version=1 with items")
    if additional_labels_payload.get("schema_version") != 1 or not isinstance(
        additional_labels_payload.get("items"), list
    ):
        raise ValueError("additional labels must use schema_version=1 with items")

    errors_by_review_id = {
        str(row.get("review_id") or ""): row for row in additional_error_rows
    }
    if len(errors_by_review_id) != len(additional_error_rows) or "" in errors_by_review_id:
        raise ValueError("duplicate or missing review_id in additional errors")

    labels_by_review_id: dict[str, str] = {}
    for item in additional_labels_payload["items"]:
        review_id = str(item.get("review_id") or "")
        label = str(item.get("label") or "")
        if review_id not in errors_by_review_id:
            raise ValueError(f"unknown review_id: {review_id}")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"invalid review label: {label}")
        if review_id in labels_by_review_id:
            raise ValueError(f"duplicate review_id: {review_id}")
        labels_by_review_id[review_id] = label

    missing_labels = sorted(set(errors_by_review_id) - set(labels_by_review_id))
    if missing_labels:
        raise ValueError(f"unreviewed additional error: {missing_labels[0]}")

    prepared_rows = [dict(row) for row in train_rows]
    hard_case_items = [dict(item) for item in reviewed_hard_cases["items"]]
    existing_scene_ids = {str(row.get("scene_id") or "") for row in prepared_rows}
    hard_case_scene_ids = {str(item.get("scene_id") or "") for item in hard_case_items}
    if len(existing_scene_ids) != len(prepared_rows) or "" in existing_scene_ids:
        raise ValueError("duplicate or missing scene_id in train rows")
    if len(hard_case_scene_ids) != len(hard_case_items) or "" in hard_case_scene_ids:
        raise ValueError("duplicate or missing scene_id in reviewed hard cases")

    additional_training_rows = 0
    excluded_or_ambiguous = 0
    corrected_labels = 0
    for review_id, label in labels_by_review_id.items():
        if label in {"exclude", "ambiguous"}:
            excluded_or_ambiguous += 1
            continue
        row = dict(errors_by_review_id[review_id])
        scene_id = str(row.get("scene_id") or "")
        if not scene_id or scene_id in existing_scene_ids:
            raise ValueError(f"duplicate or missing additional scene_id: {scene_id}")
        original_label = "fall" if bool(row.get("is_fall")) else "non_fall"
        if label != original_label:
            corrected_labels += 1
        row["is_fall"] = label == "fall"
        row["label"] = "fall" if label == "fall" else "not_fall"
        if additional_host_data_root and additional_container_data_root:
            video_path = Path(str(row.get("video_path") or ""))
            try:
                relative_video_path = video_path.relative_to(additional_host_data_root)
            except ValueError:
                pass
            else:
                row["video_path"] = str(
                    Path(additional_container_data_root) / relative_video_path
                )
        prepared_rows.append(row)
        hard_case_items.append({"scene_id": scene_id, "weight": float(reviewed_weight)})
        existing_scene_ids.add(scene_id)
        hard_case_scene_ids.add(scene_id)
        additional_training_rows += 1

    return {
        "train_rows": prepared_rows,
        "reviewed_hard_cases": {"schema_version": 1, "items": hard_case_items},
        "summary": {
            "source_train_rows": len(train_rows),
            "prepared_train_rows": len(prepared_rows),
            "source_reviewed_hard_cases": len(reviewed_hard_cases["items"]),
            "reviewed_training_hard_cases": len(hard_case_items),
            "additional_review_items": len(labels_by_review_id),
            "additional_training_rows": additional_training_rows,
            "additional_excluded_or_ambiguous": excluded_or_ambiguous,
            "additional_corrected_labels": corrected_labels,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--error-manifest", type=Path, default=DEFAULT_ERROR_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reviewed-weight", type=float, default=3.0)
    parser.add_argument(
        "--base-reviewed-hard-cases",
        type=Path,
        help="Append reviewed errors to an already prepared training manifest.",
    )
    parser.add_argument("--additional-host-data-root")
    parser.add_argument("--additional-container-data-root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = read_jsonl(args.train_manifest)
    error_rows = read_jsonl(args.error_manifest)
    labels_payload = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.base_reviewed_hard_cases:
        prepared = append_reviewed_training_cases(
            train_rows=train_rows,
            reviewed_hard_cases=json.loads(
                args.base_reviewed_hard_cases.read_text(encoding="utf-8")
            ),
            additional_error_rows=error_rows,
            additional_labels_payload=labels_payload,
            reviewed_weight=args.reviewed_weight,
            additional_host_data_root=args.additional_host_data_root,
            additional_container_data_root=args.additional_container_data_root,
        )
    else:
        prepared = prepare_retraining_inputs(
            train_rows=train_rows,
            error_rows=error_rows,
            labels_payload=labels_payload,
            reviewed_weight=args.reviewed_weight,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = args.output_dir / "train_manifest.jsonl"
    hard_cases = args.output_dir / "reviewed_hard_cases.json"
    summary_path = args.output_dir / "summary.json"
    write_jsonl(train_manifest, prepared["train_rows"])
    hard_cases.write_text(
        json.dumps(prepared["reviewed_hard_cases"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        **prepared["summary"],
        "source_train_manifest": str(args.train_manifest),
        "source_error_manifest": str(args.error_manifest),
        "source_labels": str(args.labels),
        "base_reviewed_hard_cases": (
            str(args.base_reviewed_hard_cases)
            if args.base_reviewed_hard_cases
            else None
        ),
        "train_manifest": str(train_manifest),
        "reviewed_hard_cases": str(hard_cases),
        "reviewed_weight": args.reviewed_weight,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
