import inspect
import os

import torch

MODEL_DIR = "models"
OUTPUT_DIR = "models"

# 변환할 모델 리스트
model_files = [
    "helmet_model.pt",
    "helmet_model_ver0.5.pt",
    "yolov8m-pose.pt",
    "yolov8n-pose.pt",
    "yolov8n.pt",
]

# 더미 입력 (예시: yolov8n 기준)
dummy_input = torch.randn(1, 3, 640, 640)


def _torch_onnx_export_supports(arg_name):
    return arg_name in inspect.signature(torch.onnx.export).parameters


def _extract_model(data, model_path):
    if not isinstance(data, dict):
        return data

    for key in ("ema", "model", "module"):
        model = data.get(key)
        if model is not None and hasattr(model, "modules"):
            return model

    available = ", ".join(sorted(data.keys()))
    raise ValueError(
        f"PyTorch model not found in {model_path}. "
        f"Expected one of: ema, model, module. Available keys: {available}"
    )


def convert_to_onnx(model_path, onnx_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(model_path, map_location=device, weights_only=False)
    model = _extract_model(data, model_path)
    if hasattr(model, 'to'):
        model.to(device)
    if hasattr(model, 'float'):
        # Export as FP32 ONNX; TensorRT can still build an FP16 engine later.
        model.float()
    if hasattr(model, 'eval'):
        model.eval()

    export_kwargs = {
        "export_params": True,
        "opset_version": 12,
        "do_constant_folding": True,
        "input_names": ['input'],
        "output_names": ['output'],
        "dynamic_axes": {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'},
        },
    }
    if _torch_onnx_export_supports("dynamo"):
        # dynamic_shapes is for the newer torch.export path and must use the
        # model's real arg names. dynamic_axes keeps this script compatible.
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input.to(device=device, dtype=torch.float32),),
            onnx_path,
            **export_kwargs,
        )
    print(f"ONNX로 변환 완료: {onnx_path}")

if __name__ == "__main__":
    for fname in model_files:
        pt_path = os.path.join(MODEL_DIR, fname)
        onnx_path = os.path.splitext(pt_path)[0] + ".onnx"
        if os.path.exists(pt_path):
            convert_to_onnx(pt_path, onnx_path)
        else:
            print(f"파일 없음: {pt_path}")
