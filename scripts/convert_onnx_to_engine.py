import os
import shutil
import subprocess

MODEL_DIR = "models"
INPUT_NAME = os.environ.get("TRT_INPUT_NAME", "input")
MIN_IMGSZ = int(os.environ.get("TRT_MIN_IMGSZ", "320"))
OPT_IMGSZ = int(os.environ.get("TRT_OPT_IMGSZ", "416"))
MAX_IMGSZ = int(os.environ.get("TRT_MAX_IMGSZ", "640"))

model_files = [
    "helmet_model.onnx",
    "helmet_model_ver0.5.onnx",
    "yolov8m-pose.onnx",
    "yolov8n-pose.onnx",
    "yolov8n.onnx",
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


def convert_to_engine(onnx_path, engine_path):
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes={INPUT_NAME}:1x3x{MIN_IMGSZ}x{MIN_IMGSZ}",
        f"--optShapes={INPUT_NAME}:1x3x{OPT_IMGSZ}x{OPT_IMGSZ}",
        f"--maxShapes={INPUT_NAME}:1x3x{MAX_IMGSZ}x{MAX_IMGSZ}",
        "--skipInference",
    ]
    print(f"실행: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"TensorRT 엔진 생성 완료: {engine_path}")


def main():
    for fname in model_files:
        onnx_path = os.path.join(MODEL_DIR, fname)
        engine_path = onnx_path.replace(".onnx", ".engine")
        if not os.path.exists(onnx_path):
            print(f"ONNX 파일 없음: {onnx_path}")
            continue

        convert_to_engine(onnx_path, engine_path)


if __name__ == "__main__":
    main()
