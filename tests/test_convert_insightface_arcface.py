from pathlib import Path

import pytest

from scripts.convert.convert_insightface_arcface_to_engine import (
    build_trtexec_command,
    validate_arcface_artifact,
)


def test_build_trtexec_command_uses_fixed_arcface_shape():
    command = build_trtexec_command(
        Path("w600k_r50.onnx"),
        Path("w600k_r50_fp16.engine"),
        Path("trtexec"),
    )

    assert "--minShapes=input.1:1x3x112x112" in command
    assert "--optShapes=input.1:1x3x112x112" in command
    assert "--maxShapes=input.1:1x3x112x112" in command
    assert "--fp16" in command
    assert "--skipInference" in command


def test_validate_arcface_artifact_rejects_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="ArcFace ONNX model not found"):
        validate_arcface_artifact(tmp_path / "missing.onnx")
