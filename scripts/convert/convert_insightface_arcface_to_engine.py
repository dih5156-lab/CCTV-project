"""InsightFace ArcFace ONNX 모델을 Jetson TensorRT engine으로 변환한다."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

DEFAULT_ONNX_PATH = Path(
    "data/insightface/models/buffalo_l/w600k_r50.onnx"
)
DEFAULT_ENGINE_PATH = Path("models/insightface/w600k_r50_fp16.engine")


def validate_arcface_artifact(onnx_path: Path) -> None:
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ArcFace ONNX model not found: {onnx_path}")


def build_trtexec_command(
    onnx_path: Path,
    engine_path: Path,
    trtexec: Path,
) -> list[str]:
    shape = "input.1:1x3x112x112"
    return [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes={shape}",
        f"--optShapes={shape}",
        f"--maxShapes={shape}",
        "--skipInference",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE_PATH)
    parser.add_argument("--trtexec", type=Path)
    args = parser.parse_args()

    validate_arcface_artifact(args.onnx)
    trtexec = args.trtexec or Path(
        shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    )
    if not trtexec.is_file():
        raise FileNotFoundError(f"trtexec not found: {trtexec}")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    command = build_trtexec_command(args.onnx, args.engine, trtexec)
    print(f"실행: {' '.join(command)}")
    subprocess.run(command, check=True)
    if not args.engine.is_file() or args.engine.stat().st_size == 0:
        raise RuntimeError(f"TensorRT engine was not created: {args.engine}")
    print(f"TensorRT 엔진 생성 완료: {args.engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
