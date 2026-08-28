"""상의·하의 색상 검수 HTML 생성 테스트."""

from __future__ import annotations

import json

from scripts.ops import build_appearance_review_html as review_html


def _payload(crop_path: str) -> dict:
    return {
        "items": [
            {
                "id": 323388,
                "crop_path": crop_path,
                "stored": {"upper_color": "black", "lower_color": "blue"},
                "candidates": {
                    "upper_color": {
                        "hsv_color": "black",
                        "lab_color": "gray",
                        "model_color": "white",
                        "model_confidence": 0.91,
                    },
                    "lower_color": {
                        "hsv_color": "blue",
                        "lab_color": "blue",
                        "model_color": "black",
                        "model_confidence": 0.73,
                    },
                },
            }
        ]
    }


def test_build_document_renders_upper_and_lower_review_fields(tmp_path):
    crop_path = tmp_path / "person.jpg"
    crop_path.write_bytes(b"crop")

    document = review_html.build_document(_payload(str(crop_path)))

    assert "data-id='323388' data-field='upper_color'" in document
    assert "data-id='323388' data-field='lower_color'" in document
    assert "상의 정답" in document
    assert "하의 정답" in document
    assert "상의 DB" in document
    assert "하의 DB" in document
    assert "<option>pink</option>" in document
    assert "<option>navy</option>" in document
    assert "<option>yellow</option>" in document
    assert "<option>exclude</option>" in document
    assert "appearance_color_review_labels.json" in document
    assert "schema_version:1" in document
    assert "upper_color:null" in document
    assert "lower_color:null" in document


def test_build_writes_document_from_manifest(tmp_path):
    crop_path = tmp_path / "person.jpg"
    crop_path.write_bytes(b"crop")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_payload(str(crop_path)), ensure_ascii=False),
        encoding="utf-8",
    )
    output_path = tmp_path / "review.html"

    review_html.build(manifest_path, output_path)

    document = output_path.read_text(encoding="utf-8")
    assert "상의·하의 색상 검수 (1건)" in document
    assert "src='person.jpg'" in document
    assert "file://" not in document
