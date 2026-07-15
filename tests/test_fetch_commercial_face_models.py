import hashlib
import json
from pathlib import Path

import pytest

from scripts.models.fetch_commercial_face_models import (
    load_manifest,
    verify_file,
)


def _write_manifest(path: Path, *, model_url: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "yunet": {
                        "revision": "a" * 40,
                        "license": "MIT",
                        "artifacts": {
                            "model": {
                                "filename": "yunet.onnx",
                                "url": model_url,
                                "sha256": "1" * 64,
                            },
                            "license": {
                                "filename": "yunet.LICENSE",
                                "url": "https://huggingface.co/opencv/yunet/resolve/"
                                + "a" * 40
                                + "/LICENSE",
                                "sha256": "2" * 64,
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_manifest_accepts_revision_pinned_artifacts(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    revision = "a" * 40
    _write_manifest(
        manifest_path,
        model_url=f"https://huggingface.co/opencv/yunet/resolve/{revision}/yunet.onnx",
    )

    manifest = load_manifest(manifest_path)

    assert manifest["yunet"]["revision"] == revision


def test_load_manifest_rejects_unpinned_url(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        model_url="https://huggingface.co/opencv/yunet/resolve/main/yunet.onnx",
    )

    with pytest.raises(ValueError, match="pinned revision"):
        load_manifest(manifest_path)


def test_load_manifest_requires_model_and_license_artifacts(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        model_url="https://huggingface.co/opencv/yunet/resolve/"
        + "a" * 40
        + "/yunet.onnx",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["models"]["yunet"]["artifacts"]["license"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="license"):
        load_manifest(manifest_path)


def test_verify_file_accepts_matching_sha256(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"verified model")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    verify_file(artifact, expected)


def test_verify_file_rejects_hash_mismatch(tmp_path):
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(artifact, "0" * 64)


def test_load_manifest_rejects_unknown_top_level_field(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        model_url="https://huggingface.co/opencv/yunet/resolve/"
        + "a" * 40
        + "/yunet.onnx",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown manifest fields"):
        load_manifest(manifest_path)
