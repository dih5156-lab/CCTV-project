#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-"${PROJECT_ROOT}/.venv/bin/python"}
TRAIN_ROOT="${PROJECT_ROOT}/Rethinking_of_PAR"
TRAIN_ROOT=${TRAIN_ROOT:-"${PROJECT_ROOT}/Rethinking_of_PAR"}
DATASET_DIR=${AIHUB_KREID_DIR:-"${PROJECT_ROOT}/data/datasets/aihub_kreid"}
DATA_ROOT=${AIHUB_KREID_TRAIN_ROOT:-"${PROJECT_ROOT}/data/training/aihub_kreid_appearance_30pp"}
NUMPY_ROOT=${NUMPY_ROOT:-"${PROJECT_ROOT}/.training_env/numpy1"}
LOG_DIR=${AIHUB_KREID_LOG_DIR:-"${DATA_ROOT}/logs"}
CFG_PATH=${AIHUB_KREID_CFG:-"configs/pedes_baseline/aihub_kreid_smoke.yaml"}
LOG_NAME=${AIHUB_KREID_LOG_NAME:-"smoke_epoch1_retry.log"}
MAX_IMAGES_PER_PERSON=${AIHUB_MAX_IMAGES_PER_PERSON:-30}
FORCE_REBUILD=${AIHUB_FORCE_REBUILD:-0}

TRAIN_LABEL_ZIP=${AIHUB_TRAIN_LABEL_ZIP:-}
TRAIN_SOURCE_ZIP=${AIHUB_TRAIN_SOURCE_ZIP:-}
VALIDATION_LABEL_ZIP=${AIHUB_VALIDATION_LABEL_ZIP:-}
VALIDATION_SOURCE_ZIP=${AIHUB_VALIDATION_SOURCE_ZIP:-}

MANIFEST_PATH="${DATA_ROOT}/manifest.csv"
PKL_PATH="${DATA_ROOT}/RAP2/dataset_all.pkl"
IMAGE_ROOT="${DATA_ROOT}/images"
SCRIPT_PYTHONPATH=${SCRIPT_PYTHONPATH:-"${PROJECT_ROOT}:${PYTHONPATH:-}"}

log() {
    printf '[aihub-kreid] %s\n' "$*"
}

find_label_zip() {
    local split=$1
    find "$DATASET_DIR" -type f -iname "*${split}*.zip" -path '*라벨*' -print -quit
}

find_source_zip() {
    local split=$1
    find "$DATASET_DIR" -type f -iname "*${split}*.zip" ! -path '*라벨*' -print -quit
}

require_file() {
    local path=$1
    local description=$2
    if [[ ! -f "$path" ]]; then
        printf 'ERROR: missing %s: %s\n' "$description" "$path" >&2
        exit 2
    fi
}

ensure_zip_inputs() {
    TRAIN_LABEL_ZIP=${TRAIN_LABEL_ZIP:-$(find_label_zip Training || true)}
    TRAIN_SOURCE_ZIP=${TRAIN_SOURCE_ZIP:-$(find_source_zip Training || true)}
    VALIDATION_LABEL_ZIP=${VALIDATION_LABEL_ZIP:-$(find_label_zip Validation || true)}
    VALIDATION_SOURCE_ZIP=${VALIDATION_SOURCE_ZIP:-$(find_source_zip Validation || true)}

    require_file "$TRAIN_LABEL_ZIP" 'Training label ZIP'
    require_file "$TRAIN_SOURCE_ZIP" 'Training source ZIP'
    require_file "$VALIDATION_LABEL_ZIP" 'Validation label ZIP'
    require_file "$VALIDATION_SOURCE_ZIP" 'Validation source ZIP'
}

prepare_manifest() {
    if [[ "$FORCE_REBUILD" != "1" && -f "$MANIFEST_PATH" ]]; then
        log "reusing existing manifest: $MANIFEST_PATH"
        return
    fi

    ensure_zip_inputs
    log "building appearance manifest under $DATA_ROOT"
    PYTHONPATH="$SCRIPT_PYTHONPATH" "$PYTHON" "$PROJECT_ROOT/scripts/datasets/prepare_aihub_kreid_appearance.py" \
        --train-label-zip "$TRAIN_LABEL_ZIP" \
        --train-source-zip "$TRAIN_SOURCE_ZIP" \
        --validation-label-zip "$VALIDATION_LABEL_ZIP" \
        --validation-source-zip "$VALIDATION_SOURCE_ZIP" \
        --output-dir "$DATA_ROOT" \
        --max-images-per-person "$MAX_IMAGES_PER_PERSON"
}

build_training_pkl() {
    if [[ "$FORCE_REBUILD" != "1" && -f "$PKL_PATH" ]]; then
        log "reusing existing dataset pkl: $PKL_PATH"
        return
    fi

    require_file "$MANIFEST_PATH" 'prepared manifest'
    log "building Rethinking_of_PAR dataset: $PKL_PATH"
    PYTHONPATH="$SCRIPT_PYTHONPATH" "$PYTHON" "$PROJECT_ROOT/scripts/datasets/build_rethinking_par_dataset.py" \
        --manifest "$MANIFEST_PATH" \
        --image-root "$DATA_ROOT" \
        --output-pkl "$PKL_PATH"
}

run_training() {
    require_file "$TRAIN_ROOT/train.py" 'Rethinking_of_PAR train.py'
    require_file "$TRAIN_ROOT/$CFG_PATH" 'Rethinking_of_PAR config'
    mkdir -p "$LOG_DIR"
    export PYTHONPATH="${NUMPY_ROOT}:${PYTHONPATH:-}"
    export PAR_DATA_ROOT="$DATA_ROOT"
    export MPLCONFIGDIR="${PROJECT_ROOT}/.training_env/matplotlib"
    mkdir -p "$MPLCONFIGDIR"

    log "starting smoke training with PAR_DATA_ROOT=$PAR_DATA_ROOT cfg=$CFG_PATH"
    (
        cd "$TRAIN_ROOT"
        exec "$PYTHON" train.py \
            --cfg "$CFG_PATH" \
            --debug false \
            >> "${LOG_DIR}/${LOG_NAME}" 2>&1
    )
}

mkdir -p "$LOG_DIR"
prepare_manifest
build_training_pkl
run_training
