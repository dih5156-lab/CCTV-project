#!/usr/bin/env bash
# run_deepstream_stability_watch.sh — DeepStream 컨테이너 장시간 안정성 관찰
# 사용:
#   ./scripts/ops/run_deepstream_stability_watch.sh [실행시간(분)] [간격(초)] [로그파일]
# 예시:
#   ./scripts/ops/run_deepstream_stability_watch.sh 720 60

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/ops/run_deepstream_stability_watch.sh [duration_min] [interval_sec] [log_file]

Defaults:
  duration_min: 720
  interval_sec: 60
  log_file: reports/deepstream-stability/deepstream_stability_<timestamp>.log

Environment:
  DEEPSTREAM_CONTAINER_NAME          default: cctv-ai-engine
  DEEPSTREAM_STABILITY_REPORT_DIR    default: reports/deepstream-stability
  DEEPSTREAM_STABILITY_SUMMARY_FILE  default: <log_file>.summary
  RUNTIME_ENV_FILE                   default: .env.jetson for Jetson compose, then .env
  DOCKER_USE_SUDO                    default: auto (docker, then sudo -n docker)
  PUBLIC_API_URL                     default: http://localhost:9000/api/v1/health
  ZONE_API_URL                       default: http://localhost:8765/health
  MODEL_API_URL                      default: http://localhost:8766/health
  FACE_API_URL                       default: http://localhost:8767/health
  DEEPSTREAM_LOG_SINCE               default: stability watch start time
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

DURATION_MIN=${1:-720}
INTERVAL_SEC=${2:-60}
RUN_ID=$(date +%Y%m%d_%H%M%S)
REPORT_DIR=${DEEPSTREAM_STABILITY_REPORT_DIR:-reports/deepstream-stability}
LOG_FILE=${3:-"${REPORT_DIR}/deepstream_stability_${RUN_ID}.log"}
SUMMARY_FILE=${DEEPSTREAM_STABILITY_SUMMARY_FILE:-"${LOG_FILE%.log}.summary"}
CONTAINER_NAME=${DEEPSTREAM_CONTAINER_NAME:-cctv-ai-engine}
PUBLIC_API_URL=${PUBLIC_API_URL:-http://localhost:9000/api/v1/health}
ZONE_API_URL=${ZONE_API_URL:-http://localhost:8765/health}
MODEL_API_URL=${MODEL_API_URL:-http://localhost:8766/health}
FACE_API_URL=${FACE_API_URL:-http://localhost:8767/health}
HTTP_CHECK_TIMEOUT_SEC=${HTTP_CHECK_TIMEOUT_SEC:-8}
HTTP_CHECK_RETRIES=${HTTP_CHECK_RETRIES:-3}
HTTP_CHECK_RETRY_DELAY_SEC=${HTTP_CHECK_RETRY_DELAY_SEC:-2}
RUNTIME_ENV_FILE=${RUNTIME_ENV_FILE:-}

running_jetson_stack() {
    command -v docker >/dev/null 2>&1 || return 1
    docker inspect \
        --format '{{ index .Config.Labels "com.docker.compose.project" }}' \
        "$CONTAINER_NAME" 2>/dev/null | grep -qx 'edgex-jetson'
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

if ! [[ "$DURATION_MIN" =~ ^[0-9]+$ && "$DURATION_MIN" -gt 0 ]]; then
    echo "ERROR: 실행시간(분)은 양수여야 합니다: ${DURATION_MIN}" >&2
    exit 2
fi

if ! [[ "$INTERVAL_SEC" =~ ^[0-9]+$ && "$INTERVAL_SEC" -gt 0 ]]; then
    echo "ERROR: 간격(초)은 양수여야 합니다: ${INTERVAL_SEC}" >&2
    exit 2
fi

DURATION_SEC=$((DURATION_MIN * 60))
START=$(date +%s)
LOG_SINCE=${DEEPSTREAM_LOG_SINCE:-$(date -d "@$START" --iso-8601=seconds)}
END=$((START + DURATION_SEC))
SAMPLE=0
PASS=0
FAIL=0

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$SUMMARY_FILE")"

log() {
    printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

require_command() {
    local command_name=$1

    if command -v "$command_name" >/dev/null 2>&1; then
        return 0
    fi

    log "[FAIL] required command missing: ${command_name}"
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
}

docker_access_check() {
    if [[ "${DOCKER_USE_SUDO:-auto}" == "0" ]]; then
        DOCKER_CMD=(docker)
    elif [[ "${DOCKER_USE_SUDO:-auto}" == "1" ]]; then
        DOCKER_CMD=(sudo -n docker)
    elif docker ps >/dev/null 2>&1; then
        DOCKER_CMD=(docker)
    else
        DOCKER_CMD=(sudo -n docker)
    fi

    if "${DOCKER_CMD[@]}" ps >/dev/null 2>&1; then
        printf 'docker_command=%s\n' "${DOCKER_CMD[*]}"
        return 0
    fi

    printf 'Docker 접근 실패: docker 그룹 권한이 없거나 sudo 권한이 비대화형으로 열려 있지 않습니다.\n'
    printf '해결: sudo -v 실행 후 재시도하거나, 사용자를 docker 그룹에 추가한 뒤 다시 로그인하세요.\n'
    return 1
}

run_check() {
    local name=$1
    shift

    if "$@" >>"$LOG_FILE" 2>&1; then
        log "[PASS] $name"
        return 0
    fi

    log "[FAIL] $name"
    return 1
}

http_check() {
    local url=$1
    local attempt=1

    while [[ "$attempt" -le "$HTTP_CHECK_RETRIES" ]]; do
        if curl -fsS --max-time "$HTTP_CHECK_TIMEOUT_SEC" "$url"; then
            return 0
        fi

        if [[ "$attempt" -lt "$HTTP_CHECK_RETRIES" ]]; then
            printf "http_check retry: url=%s attempt=%s/%s\n" "$url" "$attempt" "$HTTP_CHECK_RETRIES"
            sleep "$HTTP_CHECK_RETRY_DELAY_SEC"
        fi
        attempt=$((attempt + 1))
    done

    return 1
}

docker_inspect_check() {
    "${DOCKER_CMD[@]}" inspect "$CONTAINER_NAME" \
        --format 'Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} RestartCount={{.RestartCount}} StartedAt={{.State.StartedAt}} Runtime={{.HostConfig.Runtime}}'
}

docker_stats_check() {
    "${DOCKER_CMD[@]}" stats "$CONTAINER_NAME" --no-stream
}

deepstream_log_check() {
    "${DOCKER_CMD[@]}" logs --since "$LOG_SINCE" --tail 160 "$CONTAINER_NAME" 2>&1 | grep -E 'DeepStream stats|ERROR|WARNING|NvDsInfer|RestartCount|dropped=' || true
}

smoke_check() {
    .venv/bin/python scripts/smoke/smoke_test_data_flow.py --timeout 8 --retries 3 --retry-delay 2
}

write_summary() {
    local result=$1
    local failure_rate="0.0%"

    if [[ "$SAMPLE" -gt 0 ]]; then
        local failure_per_mille=$((FAIL * 1000 / SAMPLE))
        failure_rate="$((failure_per_mille / 10)).$((failure_per_mille % 10))%"
    fi

    {
        printf 'result=%s\n' "$result"
        printf 'started_at=%s\n' "$(date -d "@$START" '+%Y-%m-%d %H:%M:%S %Z')"
        printf 'finished_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
        printf 'duration_min=%s\n' "$DURATION_MIN"
        printf 'interval_sec=%s\n' "$INTERVAL_SEC"
        printf 'container=%s\n' "$CONTAINER_NAME"
        printf 'samples=%s\n' "$SAMPLE"
        printf 'pass=%s\n' "$PASS"
        printf 'fail=%s\n' "$FAIL"
        printf 'failure_rate=%s\n' "$failure_rate"
        printf 'log_file=%s\n' "$LOG_FILE"
    } >"$SUMMARY_FILE"
}

log "=== DeepStream 장시간 안정성 관찰 시작 ==="
log "  실행 시간: ${DURATION_MIN}분"
log "  간격: ${INTERVAL_SEC}초"
log "  컨테이너: ${CONTAINER_NAME}"
log "  로그 파일: ${LOG_FILE}"
log "  요약 파일: ${SUMMARY_FILE}"
log "  런타임 env 파일: ${RUNTIME_ENV_FILE}"
log "  컨테이너 로그 기준: ${LOG_SINCE} 이후"
log "  시작: $(date '+%Y-%m-%d %H:%M:%S %Z')"
log "=========================================="

PRECHECK_FAILED=0
run_check "required command: curl" require_command curl || PRECHECK_FAILED=1
run_check "required command: docker" require_command docker || PRECHECK_FAILED=1
run_check "docker access" docker_access_check || PRECHECK_FAILED=1
run_check "smoke test script exists" test -f scripts/smoke/smoke_test_data_flow.py || PRECHECK_FAILED=1
run_check "python virtualenv exists" test -x .venv/bin/python || PRECHECK_FAILED=1

if [[ "$PRECHECK_FAILED" -ne 0 ]]; then
    log ""
    log "=== DeepStream 장시간 안정성 관찰 중단: 사전 점검 실패 ==="
    write_summary "precheck_failed"
    exit 2
fi

load_runtime_env_exports

while [[ $(date +%s) -lt "$END" ]]; do
    SAMPLE=$((SAMPLE + 1))
    SAMPLE_FAILED=0

    log ""
    log "===== sample ${SAMPLE} $(date --iso-8601=seconds) ====="

    run_check "docker inspect" docker_inspect_check || SAMPLE_FAILED=1
    run_check "docker stats" docker_stats_check || SAMPLE_FAILED=1
    run_check "public api health" http_check "$PUBLIC_API_URL" || SAMPLE_FAILED=1
    run_check "zone api health" http_check "$ZONE_API_URL" || SAMPLE_FAILED=1
    run_check "camera model api health" http_check "$MODEL_API_URL" || SAMPLE_FAILED=1
    run_check "face api health" http_check "$FACE_API_URL" || SAMPLE_FAILED=1
    run_check "data flow smoke" smoke_check || SAMPLE_FAILED=1
    run_check "deepstream recent logs" deepstream_log_check || SAMPLE_FAILED=1

    if command -v tegrastats >/dev/null 2>&1; then
        timeout 3s tegrastats --interval 1000 >>"$LOG_FILE" 2>&1 || true
    fi

    if [[ "$SAMPLE_FAILED" -eq 0 ]]; then
        PASS=$((PASS + 1))
        log "[SAMPLE PASS] sample=${SAMPLE} pass=${PASS} fail=${FAIL}"
    else
        FAIL=$((FAIL + 1))
        log "[SAMPLE FAIL] sample=${SAMPLE} pass=${PASS} fail=${FAIL}"
    fi

    REMAINING=$((END - $(date +%s)))
    if [[ "$REMAINING" -le 0 ]]; then
        break
    fi

    SLEEP=$((INTERVAL_SEC < REMAINING ? INTERVAL_SEC : REMAINING))
    sleep "$SLEEP"
done

log ""
log "=== DeepStream 장시간 안정성 관찰 완료 ==="
log "  총 샘플: ${SAMPLE}"
log "  PASS: ${PASS}"
log "  FAIL: ${FAIL}"
if [[ "$SAMPLE" -gt 0 ]]; then
    FAILURE_PER_MILLE=$((FAIL * 1000 / SAMPLE))
    log "  실패율: $((FAILURE_PER_MILLE / 10)).$((FAILURE_PER_MILLE % 10))%"
fi
log "  종료: $(date '+%Y-%m-%d %H:%M:%S %Z')"
log "  요약 파일: ${SUMMARY_FILE}"

if [[ "$FAIL" -eq 0 ]]; then
    write_summary "pass"
    exit 0
fi

write_summary "fail"
exit 1
