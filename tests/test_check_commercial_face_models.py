import hashlib
import json
from pathlib import Path

from scripts.health.check_commercial_face_models import check_commercial_face_models


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(path: Path, model_data: bytes, license_data: bytes, *, filename="yunet.onnx"):
    revision = "a" * 40
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "yunet": {
                        "revision": revision,
                        "license": "MIT",
                        "engine_filename": "yunet_fp16.engine",
                        "artifacts": {
                            "model": {
                                "filename": filename,
                                "url": f"https://huggingface.co/opencv/yunet/resolve/{revision}/{filename}",
                                "sha256": _sha(model_data),
                            },
                            "license": {
                                "filename": "yunet.LICENSE",
                                "url": f"https://huggingface.co/opencv/yunet/resolve/{revision}/LICENSE",
                                "sha256": _sha(license_data),
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_readiness_passes_for_verified_artifacts_and_engine(tmp_path):
    model_data, license_data = b"onnx", b"MIT"
    manifest = tmp_path / "manifest.json"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _manifest(manifest, model_data, license_data)
    (model_dir / "yunet.onnx").write_bytes(model_data)
    (model_dir / "yunet.LICENSE").write_bytes(license_data)
    (model_dir / "yunet_fp16.engine").write_bytes(b"engine")

    result = check_commercial_face_models(manifest, model_dir)

    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])


def test_readiness_fails_for_missing_or_changed_artifacts(tmp_path):
    manifest = tmp_path / "manifest.json"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _manifest(manifest, b"onnx", b"MIT")
    (model_dir / "yunet.onnx").write_bytes(b"changed")
    (model_dir / "yunet.LICENSE").write_bytes(b"MIT")

    result = check_commercial_face_models(manifest, model_dir)

    assert result["passed"] is False
    failed_names = {item["name"] for item in result["checks"] if not item["passed"]}
    assert "yunet model" in failed_names
    assert "yunet engine" in failed_names


def test_readiness_rejects_insightface_pretrained_artifact(tmp_path):
    manifest = tmp_path / "manifest.json"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _manifest(manifest, b"onnx", b"MIT", filename="insightface_w600k.onnx")

    result = check_commercial_face_models(manifest, model_dir, require_engines=False)

    assert result["passed"] is False
    assert "InsightFace" in result["checks"][0]["detail"]
