"""Convert pinned OpenCV YuNet and SFace ONNX models to TensorRT engines."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    input_name: str
    shape: tuple[int, ...]
    default_onnx: Path
    default_engine: Path


MODEL_SPECS = {
    "yunet": ModelSpec(
        input_name="input",
        shape=(1, 3, 640, 640),
        default_onnx=Path("models/commercial_face/face_detection_yunet_2023mar.onnx"),
        default_engine=Path("models/commercial_face/yunet_fp16.engine"),
    ),
    "sface": ModelSpec(
        input_name="data",
        shape=(1, 3, 112, 112),
        default_onnx=Path("models/commercial_face/face_recognition_sface_2021dec.onnx"),
        default_engine=Path("models/commercial_face/sface_fp16.engine"),
    ),
}


def validate_onnx_artifact(model_name: str, onnx_path: Path) -> None:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model: {model_name}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    runtime_inputs = [
        tensor for tensor in model.graph.input if tensor.name not in initializer_names
    ]
    expected = MODEL_SPECS[model_name]
    if [tensor.name for tensor in runtime_inputs] != [expected.input_name]:
        raise ValueError(
            f"unexpected {model_name} runtime inputs: "
            f"{[tensor.name for tensor in runtime_inputs]}"
        )
    dimensions = tuple(
        dimension.dim_value
        for dimension in runtime_inputs[0].type.tensor_type.shape.dim
    )
    if dimensions != expected.shape:
        raise ValueError(
            f"unexpected {model_name} input shape: {dimensions}, expected {expected.shape}"
        )


def build_trtexec_command(
    model_name: str,
    onnx_path: Path,
    engine_path: Path,
    trtexec: Path,
) -> list[str]:
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        raise ValueError(f"unknown model: {model_name}")
    return [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--avgTiming=1",
        "--builderOptimizationLevel=0",
        "--skipInference",
    ]


def convert_model(
    model_name: str,
    onnx_path: Path,
    engine_path: Path,
    trtexec: Path,
) -> None:
    validate_onnx_artifact(model_name, onnx_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_trtexec_command(model_name, onnx_path, engine_path, trtexec),
        check=True,
    )
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError(f"TensorRT engine was not created: {engine_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yunet", "sface", "all"], default="all")
    parser.add_argument("--model-dir", type=Path, default=Path("models/commercial_face"))
    parser.add_argument("--trtexec", type=Path)
    args = parser.parse_args()

    trtexec = args.trtexec or Path(
        shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    )
    if not trtexec.is_file():
        raise FileNotFoundError(f"trtexec not found: {trtexec}")
    selected = MODEL_SPECS if args.model == "all" else {args.model: MODEL_SPECS[args.model]}
    for model_name, spec in selected.items():
        convert_model(
            model_name,
            args.model_dir / spec.default_onnx.name,
            args.model_dir / spec.default_engine.name,
            trtexec,
        )
        print(f"TensorRT engine created: {args.model_dir / spec.default_engine.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
