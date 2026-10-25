import inspect
import os

import torch
from ultralytics.nn.modules.head import Pose

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


class _PredictionOnly(torch.nn.Module):
    """YOLO 보조 특징이 배포 엔진의 외부 출력으로 노출되지 않게 한다."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        # 출력 이름만 지정하면 보조 튜플도 export되므로 최종 예측을 명시한다.
        predictions = outputs[0] if isinstance(outputs, tuple) else outputs
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("최종 YOLO 예측은 Tensor여야 합니다.")
        return predictions


def convert_to_onnx(model_path, onnx_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(model_path, map_location=device, weights_only=False)
    model = _extract_model(data, model_path)
    if hasattr(model, 'to'):
        model.to(device)
    if hasattr(model, 'float'):
        # FP32 ONNX로 변환하며, TensorRT에서 이후 FP16 엔진을 생성할 수 있다.
        model.float()
    if hasattr(model, 'eval'):
        model.eval()
    for module in model.modules():
        if isinstance(module, Pose):
            # 일반 경로의 좌표 덮어쓰기 대신 배포용 계산으로 Scatter 연산을 피한다.
            module.export = True
            module.dynamic = True
            module.format = "onnx"

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
        # dynamic_shapes는 최신 torch.export 경로용이며 모델의 실제 인자명을
        # 사용해야 한다. dynamic_axes를 유지해 기존 실행 환경과 호환한다.
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            _PredictionOnly(model).eval(),
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
