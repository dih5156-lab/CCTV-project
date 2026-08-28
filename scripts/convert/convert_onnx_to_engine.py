import os
import shutil
import subprocess
from dataclasses import dataclass

MODEL_DIR = "models"
INPUT_NAME = os.environ.get("TRT_INPUT_NAME", "input")
# Project helmet deployment is validated on a static 320x320 input for low latency
# and consistent TensorRT engine/profile matching.
MIN_IMGSZ = int(os.environ.get("TRT_MIN_IMGSZ", "320"))
OPT_IMGSZ = int(os.environ.get("TRT_OPT_IMGSZ", "320"))
MAX_IMGSZ = int(os.environ.get("TRT_MAX_IMGSZ", "320"))


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    input_name: str
    min_shape: str
    opt_shape: str
    max_shape: str


model_specs = [
    ModelSpec(
        "helmet_model.onnx",
        INPUT_NAME,
        f"1x3x{MIN_IMGSZ}x{MIN_IMGSZ}",
        f"1x3x{OPT_IMGSZ}x{OPT_IMGSZ}",
        f"1x3x{MAX_IMGSZ}x{MAX_IMGSZ}",
    ),
    ModelSpec(
        "helmet_model_ver0.5.onnx",
        INPUT_NAME,
        f"1x3x{MIN_IMGSZ}x{MIN_IMGSZ}",
        f"1x3x{OPT_IMGSZ}x{OPT_IMGSZ}",
        f"1x3x{MAX_IMGSZ}x{MAX_IMGSZ}",
    ),
    ModelSpec(
        "yolov8m-pose.onnx",
        INPUT_NAME,
        f"1x3x{MIN_IMGSZ}x{MIN_IMGSZ}",
        f"1x3x{OPT_IMGSZ}x{OPT_IMGSZ}",
        f"1x3x{MAX_IMGSZ}x{MAX_IMGSZ}",
    ),
    ModelSpec(
        "yolov8n-pose.onnx",
        INPUT_NAME,
        f"1x3x{MIN_IMGSZ}x{MIN_IMGSZ}",
        f"1x3x{OPT_IMGSZ}x{OPT_IMGSZ}",
        f"1x3x{MAX_IMGSZ}x{MAX_IMGSZ}",
    ),
    ModelSpec(
        "yolov8n.onnx",
        INPUT_NAME,
        f"1x3x{MIN_IMGSZ}x{MIN_IMGSZ}",
        f"1x3x{OPT_IMGSZ}x{OPT_IMGSZ}",
        f"1x3x{MAX_IMGSZ}x{MAX_IMGSZ}",
    ),
    ModelSpec(
        "pphuman_attribute.onnx",
        "x",
        "1x3x256x192",
        "4x3x256x192",
        "8x3x256x192",
    ),
]

TRTEXEC_CANDIDATES = [
    os.environ.get("TRTEXEC"),
    shutil.which("trtexec"),
    "/usr/src/tensorrt/bin/trtexec",
]


def find_trtexec():
    for path in TRTEXEC_CANDIDATES:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "trtexec를 찾을 수 없습니다. TensorRT가 설치되어 있는지 확인하거나 "
        "TRTEXEC=/path/to/trtexec 환경변수로 경로를 지정하세요."
    )


TRTEXEC = find_trtexec()


def convert_to_engine(spec):
    onnx_path = os.path.join(MODEL_DIR, spec.filename)
    engine_path = onnx_path.replace(".onnx", ".engine")
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes={spec.input_name}:{spec.min_shape}",
        f"--optShapes={spec.input_name}:{spec.opt_shape}",
        f"--maxShapes={spec.input_name}:{spec.max_shape}",
        "--skipInference",
    ]
    print(f"실행: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"TensorRT 엔진 생성 완료: {engine_path}")


def main():
    for spec in model_specs:
        onnx_path = os.path.join(MODEL_DIR, spec.filename)
        if not os.path.exists(onnx_path):
            print(f"ONNX 파일 없음: {onnx_path}")
            continue

        convert_to_engine(spec)


if __name__ == "__main__":
    main()
