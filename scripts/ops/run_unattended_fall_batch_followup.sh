#!/usr/bin/env bash
set -uo pipefail

project_root="${CCTV_PROJECT_ROOT:-/app}"
manifest_path="${CCTV_FALL_MANIFEST:-${project_root}/data/fall_eval/open_data_test_20260914/balanced320_c1_c8.jsonl}"
comparison_path="${CCTV_FALL_COMPARISON:-${project_root}/data/fall_eval/open_data_test_20260914/balanced320_c1_c8_comparison.json}"
report_path="${CCTV_FALL_REPORT:-${project_root}/data/fall_eval/open_data_test_20260914/FALL_OPEN_DATA_C1_C8_FINAL_20260914.md}"
status_path="${CCTV_FALL_STATUS:-${project_root}/data/fall_eval/open_data_test_20260914/unattended_status.json}"
log_path="${CCTV_FALL_LOG:-${project_root}/data/fall_eval/open_data_test_20260914/unattended_followup.log}"
lock_dir="${CCTV_FALL_LOCK_DIR:-${project_root}/data/fall_eval/open_data_test_20260914/unattended_followup.lock}"
process_pattern="${CCTV_FALL_PROCESS_PATTERN:-compare_fall_temporal_runtime.py.*balanced320_c1_c8.jsonl}"
poll_seconds="${CCTV_FALL_POLL_SECONDS:-60}"
max_wait_seconds="${CCTV_FALL_MAX_WAIT_SECONDS:-86400}"
restart_limit="${CCTV_FALL_RESTART_LIMIT:-1}"
python_executable="${CCTV_FALL_PYTHON_EXECUTABLE:-python}"
finalizer="${CCTV_FALL_FINALIZER:-${project_root}/scripts/ops/finalize_fall_open_data_batch.py}"

mkdir -p "$(dirname "${log_path}")" "$(dirname "${status_path}")" "$(dirname "${report_path}")"

log_message() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"${log_path}"
}

if ! mkdir "${lock_dir}" 2>/dev/null; then
    log_message "another unattended follow-up is already running: ${lock_dir}"
    exit 4
fi
cleanup_lock() {
    rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup_lock EXIT
trap 'exit 130' INT TERM

run_finalizer() {
    "${python_executable}" "${finalizer}" \
        --manifest "${manifest_path}" \
        --comparison "${comparison_path}" \
        --report "${report_path}" \
        --status "${status_path}"
}

comparison_is_running() {
    pgrep -f "${process_pattern}" >/dev/null 2>&1
}

restart_comparison() {
    log_message "comparison process stopped before completion; restarting once with shared pose cache"
    "${python_executable}" "${project_root}/scripts/ops/compare_fall_temporal_runtime.py" \
        --manifest "${manifest_path}" \
        --baseline-model "${project_root}/models/fall/fall_temporal_baseline_same_split_20260910.pt" \
        --candidate-model "${project_root}/models/fall/fall_temporal_same_split_resume_20260908.pt" \
        --pose-model "${project_root}/models/fall/yolov8n-pose.pt" \
        --output-json "${comparison_path}" \
        --output-csv "${comparison_path%.json}.csv" \
        --python-executable "${python_executable}" \
        --project-root "${project_root}" \
        --container-project-root "${project_root}" \
        --device cuda \
        --imgsz 640 \
        --confidence-threshold 0.35 \
        --feature-cache-dir "${project_root}/data/fall_eval/open_data_test_20260914/shared_pose_cache" \
        >>"${log_path}" 2>&1 &
}

started_epoch="$(date +%s)"
restart_count=0
log_message "unattended fall batch follow-up started"

while true; do
    run_finalizer
    finalizer_status=$?
    if [[ ${finalizer_status} -eq 0 ]]; then
        log_message "batch finalization complete: report=${report_path} status=${status_path}"
        exit 0
    fi

    current_epoch="$(date +%s)"
    if (( current_epoch - started_epoch >= max_wait_seconds )); then
        log_message "maximum wait time exceeded: ${max_wait_seconds}s"
        exit 6
    fi

    if comparison_is_running; then
        if [[ ${finalizer_status} -eq 3 ]]; then
            log_message "comparison JSON is temporarily unreadable while writer is active; retrying"
        fi
        sleep "${poll_seconds}"
        continue
    fi

    if [[ ${finalizer_status} -eq 3 ]]; then
        log_message "batch output is invalid and comparison process is not running"
        exit 3
    fi
    if (( restart_count >= restart_limit )); then
        log_message "batch incomplete and automatic restart limit reached"
        exit 5
    fi

    restart_count=$((restart_count + 1))
    restart_comparison
    sleep "${poll_seconds}"
done
