#!/usr/bin/env bash
set -euo pipefail

# Pose 학습은 항상 NVIDIA GPU가 연결된 AI Engine 컨테이너 내부에서 실행한다.
# 호스트의 Jetson 가상환경은 PyTorch를 불러올 수 있어도 GPU 장치 노드에
# 안정적으로 접근하지 못할 수 있다.
container_name="${CCTV_TRAIN_CONTAINER:-cctv-ai-engine}"

if ! docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null | grep -qx 'true'; then
  echo "ERROR: $container_name is not running" >&2
  echo "Start the Jetson stack before GPU training." >&2
  exit 1
fi

cuda_state="$(docker exec "$container_name" python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>&1)"
if [[ "$cuda_state" != "True 1" && "$cuda_state" != "True 2" ]]; then
  echo "ERROR: container CUDA is not ready: $cuda_state" >&2
  exit 1
fi

echo "GPU training container: $container_name ($cuda_state)"
exec docker exec -i "$container_name" \
  python /app/scripts/datasets/train_yolo_pose_fall_rf.py "$@"
