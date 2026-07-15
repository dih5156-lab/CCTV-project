from pathlib import Path

import pytest

from scripts.convert.convert_commercial_face_models_to_engine import (
    MODEL_SPECS,
    build_trtexec_command,
    validate_onnx_artifact,
)


@pytest.mark.parametrize(
    ("model_name", "shape"),
    [("yunet", "input:1x3x640x640"), ("sface", "data:1x3x112x112")],
)
def test_build_command_uses_fixed_fp16_profile(model_name, shape):
    command = build_trtexec_command(
        model_name=model_name,
        onnx_path=Path(f"{model_name}.onnx"),
        engine_path=Path(f"{model_name}.engine"),
        trtexec=Path("/usr/src/tensorrt/bin/trtexec"),
    )

    assert "--fp16" in command
    assert f"--minShapes={shape}" in command
    assert f"--optShapes={shape}" in command
    assert f"--maxShapes={shape}" in command
    assert f"--onnx={model_name}.onnx" in command
    assert f"--saveEngine={model_name}.engine" in command


def test_build_command_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_trtexec_command(
            "arcface", Path("model.onnx"), Path("model.engine"), Path("trtexec")
        )


def test_validate_onnx_artifact_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        validate_onnx_artifact("yunet", tmp_path / "missing.onnx")


def test_model_specs_keep_sface_initializers_out_of_runtime_profile():
    assert MODEL_SPECS["sface"].input_name == "data"
    assert MODEL_SPECS["sface"].shape == (1, 3, 112, 112)
