"""상의·하의 검수 결과를 재학습 준비 데이터로 내보내는 테스트."""

from __future__ import annotations

import csv
import importlib
import json

import pytest


def _export_module():
    return importlib.import_module(
        "scripts.ops.export_appearance_color_review_labels"
    )


def _write_json(path, payload: dict):
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _manifest_item(item_id: int, crop_path: str) -> dict:
    return {
        "id": item_id,
        "crop_path": crop_path,
        "stored": {"upper_color": "black", "lower_color": "blue"},
        "candidates": {
            "upper_color": {
                "hsv_color": "black",
                "lab_color": "gray",
                "model_color": "white",
            },
            "lower_color": {
                "hsv_color": "blue",
                "lab_color": "blue",
                "model_color": "black",
            },
        },
    }


def _read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_preserves_only_human_reviewed_fields(tmp_path):
    crop_path = tmp_path / "person.jpg"
    crop_path.write_bytes(b"crop")
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        {"items": [_manifest_item(1, str(crop_path))]},
    )
    labels_path = _write_json(
        tmp_path / "labels.json",
        {
            "schema_version": 1,
            "items": [
                {"id": 1, "upper_color": "white", "lower_color": None}
            ],
        },
    )
    output_dir = tmp_path / "out"

    summary = _export_module().export_reviewed_labels(
        manifest_path,
        labels_path,
        output_dir,
    )

    rows = _read_csv_rows(output_dir / "reviewed_appearance_colors.csv")
    assert rows == [
        {
            "image_path": str(crop_path),
            "appearance_log_id": "1",
            "upper_color": "white",
            "lower_color": "",
            "upper_reviewed": "true",
            "lower_reviewed": "false",
        }
    ]
    assert summary["reviewed_items"] == 1
    assert summary["exported_rows"] == 1
    assert summary["partial_reviews"] == 1
    assert summary["upper_colors"]["white"] == 1
    assert summary["lower_colors"] == {}
    assert json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    ) == summary


def test_export_reports_excluded_missing_and_training_unsupported_rows(
    tmp_path,
):
    existing_crop = tmp_path / "person.jpg"
    existing_crop.write_bytes(b"crop")
    missing_crop = tmp_path / "missing.jpg"
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        {
            "items": [
                _manifest_item(1, str(existing_crop)),
                _manifest_item(2, str(missing_crop)),
            ]
        },
    )
    labels_path = _write_json(
        tmp_path / "labels.json",
        {
            "schema_version": 1,
            "items": [
                {"id": 1, "upper_color": "pink", "lower_color": "exclude"},
                {"id": 2, "upper_color": "black", "lower_color": "blue"},
            ],
        },
    )
    output_dir = tmp_path / "out"

    summary = _export_module().export_reviewed_labels(
        manifest_path,
        labels_path,
        output_dir,
    )

    assert summary["reviewed_items"] == 2
    assert summary["exported_rows"] == 1
    assert summary["excluded_fields"] == 1
    assert summary["missing_crops"] == 1
    assert summary["multilabel_unsupported_fields"] == 1
    rows = _read_csv_rows(output_dir / "reviewed_appearance_colors.csv")
    assert len(rows) == 1
    assert rows[0]["upper_color"] == "pink"
    assert rows[0]["lower_color"] == ""
    audit = json.loads(
        (output_dir / "reviewed_appearance_colors.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["items"][0]["export_status"] == "exported"
    assert audit["items"][1]["export_status"] == "missing_crop"


def test_export_rejects_label_id_missing_from_manifest(tmp_path):
    manifest_path = _write_json(tmp_path / "manifest.json", {"items": []})
    labels_path = _write_json(
        tmp_path / "labels.json",
        {
            "schema_version": 1,
            "items": [
                {"id": 99, "upper_color": "black", "lower_color": "blue"}
            ],
        },
    )

    with pytest.raises(ValueError, match="manifest id not found"):
        _export_module().export_reviewed_labels(
            manifest_path,
            labels_path,
            tmp_path / "out",
        )


def test_export_skips_fully_unreviewed_item(tmp_path):
    crop_path = tmp_path / "person.jpg"
    crop_path.write_bytes(b"crop")
    manifest_path = _write_json(
        tmp_path / "manifest.json",
        {"items": [_manifest_item(1, str(crop_path))]},
    )
    labels_path = _write_json(
        tmp_path / "labels.json",
        {
            "schema_version": 1,
            "items": [
                {"id": 1, "upper_color": None, "lower_color": None}
            ],
        },
    )
    output_dir = tmp_path / "out"

    summary = _export_module().export_reviewed_labels(
        manifest_path,
        labels_path,
        output_dir,
    )

    assert summary["reviewed_items"] == 0
    assert summary["unreviewed_items"] == 1
    assert summary["exported_rows"] == 0
    assert _read_csv_rows(
        output_dir / "reviewed_appearance_colors.csv"
    ) == []
