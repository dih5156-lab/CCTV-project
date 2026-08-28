#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export AIHUB_KREID_TRAIN_ROOT=${AIHUB_KREID_TRAIN_ROOT:-"${PROJECT_ROOT}/data/training/aihub_kreid_appearance_native11_30pp"}
export AIHUB_KREID_LOG_DIR=${AIHUB_KREID_LOG_DIR:-"${AIHUB_KREID_TRAIN_ROOT}/logs"}
export AIHUB_KREID_CFG=${AIHUB_KREID_CFG:-"configs/pedes_baseline/aihub_kreid_native11_smoke.yaml"}
export AIHUB_KREID_LOG_NAME=${AIHUB_KREID_LOG_NAME:-"smoke_epoch1.log"}

exec "${PROJECT_ROOT}/scripts/train/run_aihub_kreid_smoke.sh"
