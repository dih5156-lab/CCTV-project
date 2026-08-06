#!/usr/bin/env bash
set -euo pipefail

container_name="${CCTV_TRAIN_CONTAINER:-cctv-ai-engine}"
if ! docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null | grep -qx 'true'; then
  echo "ERROR: $container_name is not running" >&2
  exit 1
fi
exec docker exec -i "$container_name" \
  python /app/scripts/datasets/train_fall_direction_rf.py "$@"
