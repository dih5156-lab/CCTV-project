import torch
import tensorrt as trt
import cv2
import numpy as np

print("=== Jetson Docker 컨테이너 검증 ===")
print(f"PyTorch    : {torch.__version__}")
print(f"TensorRT   : {trt.__version__}")
print(f"OpenCV     : {cv2.__version__}")
print(f"CUDA 사용  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 이름   : {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전  : {torch.version.cuda}")
    x = torch.randn(512, 512).cuda()
    y = torch.matmul(x, x)
    print(f"GPU 행렬곱 : OK {y.shape}")
else:
    print("WARNING: CUDA 미사용")

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print("TRT Builder: OK")

img = np.zeros((640, 640, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"OpenCV 처리: OK {gray.shape}")

from ultralytics import YOLO
print("Ultralytics: OK")
print("=== 모든 테스트 통과 ===")
