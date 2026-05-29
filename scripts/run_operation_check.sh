#!/usr/bin/env bash
# run_operation_check.sh - 운영/데모 전 표준 점검 wrapper
# 사용:
#   ./scripts/run_operation_check.sh
#   ./scripts/run_operation_check.sh --with-deepstream 30 30

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_operation_check.sh [--with-deepstream [duration_min interval_sec]]

Defaults:
  Runs deployment smoke and data-flow smoke checks only.
  With --with-deepstream, also runs scripts/run_deepstream_stability_watch.sh.

Examples:
  ./scripts/run_operation_check.sh
  ./scripts/run_operation_check.sh --with-deepstream 30 30
  ./scripts/run_operation_check.sh --with-deepstream 720 60
EOF
}

WITH_DEEPSTREAM=0
DEEPSTREAM_DURATION_MIN=30
DEEPSTREAM_INTERVAL_SEC=30

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--with-deepstream" ]]; then
    WITH_DEEPSTREAM=1
    shift
    DEEPSTREAM_DURATION_MIN=${1:-30}
    DEEPSTREAM_INTERVAL_SEC=${2:-30}
fi

if [[ "$DEEPSTREAM_DURATION_MIN" =~ [^0-9] || "$DEEPSTREAM_DURATION_MIN" -le 0 ]]; then
    echo "ERROR: DeepStream 실행시간(분)은 양수여야 합니다: ${DEEPSTREAM_DURATION_MIN}" >&2
    exit 2
fi

if [[ "$DEEPSTREAM_INTERVAL_SEC" =~ [^0-9] || "$DEEPSTREAM_INTERVAL_SEC" -le 0 ]]; then
    echo "ERROR: DeepStream 간격(초)은 양수여야 합니다: ${DEEPSTREAM_INTERVAL_SEC}" >&2
    exit 2
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
REPORT_DIR=${OPERATION_CHECK_REPORT_DIR:-reports/operation-checks}
REPORT_FILE=${OPERATION_CHECK_REPORT_FILE:-"${REPORT_DIR}/operation_check_${RUN_ID}.log"}
mkdir -p "$REPORT_DIR"

log() {
    printf '%s\n' "$*" | tee -a "$REPORT_FILE"
}

run_step() {
    local name=$1
    shift

    log ""
    log "===== ${name} ====="
    if "$@" >>"$REPORT_FILE" 2>&1; then
        log "[PASS] ${name}"
        return 0
    fi

    log "[FAIL] ${name}"
    return 1
}

FAILED=0

log "=== CCTV 운영 점검 시작 ==="
log "  report: ${REPORT_FILE}"
log "  started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
log "  with_deepstream: ${WITH_DEEPSTREAM}"

run_step "runtime secret consistency" .venv/bin/python scripts/check_runtime_secret_consistency.py || FAILED=1
run_step "deployment smoke" .venv/bin/python scripts/smoke_test_deployment.py || FAILED=1
run_step "data flow smoke" .venv/bin/python scripts/smoke_test_data_flow.py || FAILED=1

if [[ "$WITH_DEEPSTREAM" -eq 1 ]]; then
    run_step "deepstream stability watch" \
        ./scripts/run_deepstream_stability_watch.sh \
        "$DEEPSTREAM_DURATION_MIN" \
        "$DEEPSTREAM_INTERVAL_SEC" || FAILED=1
fi

log ""
log "=== CCTV 운영 점검 완료 ==="
log "  result: $([[ "$FAILED" -eq 0 ]] && printf 'PASS' || printf 'FAIL')"
log "  finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
log "  report: ${REPORT_FILE}"

exit "$FAILED"
