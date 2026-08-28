#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATASET_DIR=${AIHUB_KREID_DIR:-"${PROJECT_ROOT}/data/datasets/aihub_kreid"}
REPORT_DIR=${AIHUB_POSTPROCESS_REPORT_DIR:-"${PROJECT_ROOT}/data/training/aihub_kreid"}
COMPLETED_FILE="${DATASET_DIR}/.aihubshell_completed_filekeys"
INSPECT_SCRIPT="${PROJECT_ROOT}/scripts/datasets/inspect_aihub_kreid_dataset.py"
PYTHON=${PYTHON:-"${PROJECT_ROOT}/.venv/bin/python"}
POLL_SECONDS=${AIHUB_POSTPROCESS_POLL_SECONDS:-60}
EXPECTED_KEYS=(38308 38309 38310 38311 38312 50394 50395 50396 50397 50398)

download_is_complete() {
    local key
    [[ -f "$COMPLETED_FILE" ]] || return 1
    for key in "${EXPECTED_KEYS[@]}"; do
        grep -qx "$key" "$COMPLETED_FILE" || return 1
    done
}

find_label_zip() {
    local split=$1
    find "$DATASET_DIR" -type f -iname "*${split}*.zip" -path '*라벨*' -print -quit
}

run_postprocess() {
    local zip_file training_labels validation_labels zip_count=0
    mkdir -p "$REPORT_DIR"
    : > "${REPORT_DIR}/zip_integrity.tsv"

    while IFS= read -r -d '' zip_file; do
        zip_count=$((zip_count + 1))
        printf 'checking ZIP: %s\n' "$zip_file"
        if unzip -tqq "$zip_file"; then
            printf 'ok\t%s\n' "$zip_file" >> "${REPORT_DIR}/zip_integrity.tsv"
        else
            printf 'failed\t%s\n' "$zip_file" >> "${REPORT_DIR}/zip_integrity.tsv"
            printf 'ERROR: corrupt ZIP: %s\n' "$zip_file" >&2
            return 20
        fi
    done < <(find "$DATASET_DIR" -type f -name '*.zip' -print0)

    if (( zip_count == 0 )); then
        printf 'ERROR: no ZIP files found in %s\n' "$DATASET_DIR" >&2
        return 21
    fi

    training_labels=$(find_label_zip Training)
    validation_labels=$(find_label_zip Validation)
    if [[ -z "$training_labels" || -z "$validation_labels" ]]; then
        printf 'ERROR: Training/Validation label ZIP was not found.\n' >&2
        return 22
    fi

    "$PYTHON" "$INSPECT_SCRIPT" "$training_labels" "$validation_labels" \
        --output "${REPORT_DIR}/label_report.json" \
        > "${REPORT_DIR}/label_report.stdout.json"
    date --iso-8601=seconds > "${REPORT_DIR}/postprocess_complete"
    printf 'post-processing completed: %s\n' "$REPORT_DIR"
}

mode=${1:---once}
case "$mode" in
    --once)
        if ! download_is_complete; then
            printf 'waiting: AI-Hub download is not complete.\n'
            exit 10
        fi
        run_postprocess
        ;;
    --watch)
        while ! download_is_complete; do
            printf '%s waiting for all AI-Hub file keys\n' "$(date --iso-8601=seconds)"
            sleep "$POLL_SECONDS"
        done
        run_postprocess
        ;;
    *)
        printf 'Usage: %s {--once|--watch}\n' "$0" >&2
        exit 2
        ;;
esac
