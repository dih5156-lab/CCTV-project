#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec docker compose -f docker-compose.appearance-train.jetson.yml up \
  --build --abort-on-container-exit --exit-code-from appearance-multitask-train \
  appearance-multitask-train
