#!/usr/bin/env bash
# run_operation_check.sh - 운영/데모 전 표준 점검 wrapper
# 사용:
#   ./scripts/ops/run_operation_check.sh
#   ./scripts/ops/run_operation_check.sh --with-deepstream 30 30
#   ./scripts/ops/run_operation_check.sh --with-fall-shadow 30 30
#   AIOT_PILOT_CHECK=1 ./scripts/ops/run_operation_check.sh

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/ops/run_operation_check.sh [--with-deepstream|--with-fall-shadow [duration_min interval_sec]]

Defaults:
  Runs deployment smoke and data-flow smoke checks only.
  With --with-deepstream, also runs scripts/ops/run_deepstream_stability_watch.sh.
  With --with-fall-shadow, runs the same watch and checks fall Shadow each sample.
  With AIOT_PILOT_CHECK=1, also checks the Jetson EdgeX stack and AIoT metrics.
    Runtime secret check uses RUNTIME_ENV_FILE when set. Otherwise it uses
    .env.jetson when the running compose project is edgex-jetson, then .env,
    and falls back to .env.jetson when .env is absent.

Examples:
  ./scripts/ops/run_operation_check.sh
  ./scripts/ops/run_operation_check.sh --with-deepstream 30 30
  ./scripts/ops/run_operation_check.sh --with-fall-shadow 30 30
  ./scripts/ops/run_operation_check.sh --with-deepstream 720 60
EOF
}

WITH_DEEPSTREAM=0
WITH_FALL_SHADOW=0
DEEPSTREAM_DURATION_MIN=30
DEEPSTREAM_INTERVAL_SEC=30
RUNTIME_ENV_FILE=${RUNTIME_ENV_FILE:-}

running_jetson_stack() {
    docker inspect \
        --format '{{ index .Config.Labels "com.docker.compose.project" }}' \
        cctv-ai-engine 2>/dev/null | grep -qx 'edgex-jetson'
}

if [[ -z "$RUNTIME_ENV_FILE" ]]; then
    if [[ -f .env.jetson ]] && running_jetson_stack; then
        RUNTIME_ENV_FILE=.env.jetson
    elif [[ -f .env ]]; then
        RUNTIME_ENV_FILE=.env
    elif [[ -f .env.jetson ]]; then
        RUNTIME_ENV_FILE=.env.jetson
    else
        RUNTIME_ENV_FILE=.env
    fi
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--with-deepstream" || "${1:-}" == "--with-fall-shadow" ]]; then
    WITH_DEEPSTREAM=1
    if [[ "${1:-}" == "--with-fall-shadow" ]]; then
        WITH_FALL_SHADOW=1
    fi
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

export_runtime_env_value() {
    local key=$1
    local value

    if [[ ! -f "$RUNTIME_ENV_FILE" ]]; then
        return 0
    fi

    value=$(grep -E "^${key}=" "$RUNTIME_ENV_FILE" | tail -n 1 | cut -d= -f2- || true)
    if [[ -n "$value" ]]; then
        export "${key}=${value}"
    fi
}

load_runtime_env_exports() {
    export_runtime_env_value INTERNAL_SERVICE_TOKEN
    export_runtime_env_value PUBLIC_API_KEY
    export_runtime_env_value STREAM_API_TOKEN
    export_runtime_env_value AIOT_COMMANDS_ENABLED
    export_runtime_env_value AIOT_METRICS_PORT
}

FAILED=0

log "=== CCTV 운영 점검 시작 ==="
log "  report: ${REPORT_FILE}"
log "  started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
log "  with_deepstream: ${WITH_DEEPSTREAM}"
log "  with_fall_shadow: ${WITH_FALL_SHADOW}"
log "  runtime_env_file: ${RUNTIME_ENV_FILE}"

load_runtime_env_exports

run_step "runtime secret consistency" .venv/bin/python scripts/health/check_runtime_secret_consistency.py --env-file "$RUNTIME_ENV_FILE" || FAILED=1
run_step "deployment smoke" .venv/bin/python scripts/smoke/smoke_test_deployment.py || FAILED=1
run_step "data flow smoke" .venv/bin/python scripts/smoke/smoke_test_data_flow.py || FAILED=1
run_step "public api fd stability" .venv/bin/python scripts/health/check_public_api_fd_stability.py || FAILED=1

if [[ "${AIOT_PILOT_CHECK:-0}" == "1" ]]; then
    run_step "jetson edgex stack" .venv/bin/python scripts/health/check_jetson_edgex_stack.py || FAILED=1
    run_step "aiot metrics" curl -fsS --max-time 5 "http://127.0.0.1:${AIOT_METRICS_PORT:-9105}/metrics" || FAILED=1
fi

if [[ "$WITH_DEEPSTREAM" -eq 1 ]]; then
    export FALL_SHADOW_CHECK="$WITH_FALL_SHADOW"
    if [[ "$WITH_FALL_SHADOW" -eq 1 ]]; then
        export DEEPSTREAM_RUN_DATA_FLOW_SMOKE=0
    fi
    run_step "deepstream stability watch" \
        ./scripts/ops/run_deepstream_stability_watch.sh \
        "$DEEPSTREAM_DURATION_MIN" \
        "$DEEPSTREAM_INTERVAL_SEC" || FAILED=1
fi

log ""
log "=== CCTV 운영 점검 완료 ==="
log "  result: $([[ "$FAILED" -eq 0 ]] && printf 'PASS' || printf 'FAIL')"
log "  finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
log "  report: ${REPORT_FILE}"

exit "$FAILED"
