#!/usr/bin/env bash
set -euo pipefail

# Always run pose training inside the NVIDIA-enabled AI engine container.
# Host-side Jetson virtualenvs may import PyTorch but cannot reliably access
# the Jetson GPU device nodes.
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
