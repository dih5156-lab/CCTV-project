#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AIHUB_SHELL=${AIHUB_SHELL:-"${PROJECT_ROOT}/tools/aihub/aihubshell"}
OUTPUT_DIR=${AIHUB_KREID_DIR:-"${PROJECT_ROOT}/data/datasets/aihub_kreid"}
DATASET_KEY=84
COMPLETED_FILE="${OUTPUT_DIR}/.aihubshell_completed_filekeys"
MINIMUM_RESERVE_GB=${AIHUB_MINIMUM_RESERVE_GB:-30}

# file key : description : approximate compressed size (GB)
FULL_FILE_KEYS=(
    "38309:Training label:1"
    "38311:Validation label:1"
    "38310:Training source:5"
    "38312:Validation source:4"
    "38308:Full OUTDOOR3:23"
    "50394:Full INDOOR:27"
    "50395:Full INDOOR2:24"
    "50396:Full INDOOR3:25"
    "50397:Full OUTDOOR:29"
    "50398:Full OUTDOOR2:26"
)

usage() {
    printf '%s\n' \
        "Usage:" \
        "  ./scripts/datasets/download_aihub_kreid_sample.sh plan" \
        "  ./scripts/datasets/download_aihub_kreid_sample.sh list" \
        "  AIHUB_API_KEY='issued-key' ./scripts/datasets/download_aihub_kreid_sample.sh labels" \
        "  AIHUB_API_KEY='issued-key' ./scripts/datasets/download_aihub_kreid_sample.sh validation" \
        "  AIHUB_API_KEY='issued-key' ./scripts/datasets/download_aihub_kreid_sample.sh full" \
        "" \
        "Modes:" \
        "  plan        Show the full download plan without downloading." \
        "  list        Show the official dataset file tree." \
        "  labels      Download Training/Validation labels only." \
        "  validation  Download Validation labels and source images." \
        "  full        Download all ten files sequentially with resume tracking."
}

available_gb() {
    df -Pk "$OUTPUT_DIR" | awk 'NR == 2 {print int($4 / 1024 / 1024)}'
}

require_api_key() {
    if [[ -z "${AIHUB_API_KEY:-}" && -t 0 ]]; then
        read -r -s -p "AI-Hub API key (input is hidden): " AIHUB_API_KEY
        printf '\n'
        export AIHUB_API_KEY
    fi
    if [[ -z "${AIHUB_API_KEY:-}" ]]; then
        printf 'ERROR: AIHUB_API_KEY is not set.\n' >&2
        printf 'Run this command in a terminal so the key is not exposed in chat:\n' >&2
        printf '  read -rsp "AI-Hub API key: " AIHUB_API_KEY; echo; export AIHUB_API_KEY\n' >&2
        exit 3
    fi
}

download_file_key() {
    local file_key=$1
    local command_status log_file payload_count_before payload_count_after
    log_file=$(mktemp "${OUTPUT_DIR}/.aihubshell_${file_key}.XXXXXX.log")
    payload_count_before=$(find "$OUTPUT_DIR" -type f \
        ! -name '.aihubshell_completed_filekeys' \
        ! -name '.aihubshell_*.log' | wc -l)

    set +e
    (
        cd "$OUTPUT_DIR"
        "$AIHUB_SHELL" -mode d -datasetkey "$DATASET_KEY" \
            -filekey "$file_key" -aihubapikey "$AIHUB_API_KEY" 2>&1 | tee "$log_file"
    )
    command_status=$?
    set -e
    payload_count_after=$(find "$OUTPUT_DIR" -type f \
        ! -name '.aihubshell_completed_filekeys' \
        ! -name '.aihubshell_*.log' | wc -l)

    if (( command_status != 0 )) || \
        ! grep -q '^Request successful with HTTP status 200\.$' "$log_file" || \
        ! grep -q '^Download successful\.$' "$log_file" || \
        grep -q '^Download failed' "$log_file" || \
        (( payload_count_after <= payload_count_before )); then
        printf 'ERROR: AI-Hub download was not confirmed for file key %s.\n' "$file_key" >&2
        printf 'Diagnostic log: %s\n' "$log_file" >&2
        return 5
    fi

    rm "$log_file"
}

is_completed() {
    local file_key=$1
    [[ -f "$COMPLETED_FILE" ]] && grep -qx "$file_key" "$COMPLETED_FILE"
}

print_plan() {
    local entry file_key description compressed_gb
    printf 'output=%s\n' "$OUTPUT_DIR"
    printf 'compressed_total=about 163GB (plan rounds to 165GB) recommended_free=at least 420GB\n'
    for entry in "${FULL_FILE_KEYS[@]}"; do
        IFS=: read -r file_key description compressed_gb <<< "$entry"
        printf '%-6s %3sGB  %s\n' "$file_key" "$compressed_gb" "$description"
    done
}

download_full() {
    local entry file_key description compressed_gb required_gb free_gb
    touch "$COMPLETED_FILE"
    printf 'Full download started; completed keys are recorded in %s\n' "$COMPLETED_FILE"

    for entry in "${FULL_FILE_KEYS[@]}"; do
        IFS=: read -r file_key description compressed_gb <<< "$entry"
        if is_completed "$file_key"; then
            printf '[SKIP] %s %s (already completed)\n' "$file_key" "$description"
            continue
        fi

        free_gb=$(available_gb)
        required_gb=$((compressed_gb * 3 + MINIMUM_RESERVE_GB))
        if (( free_gb < required_gb )); then
            printf 'ERROR: insufficient space before file key %s.\n' "$file_key" >&2
            printf 'free=%sGB safety_requirement=%sGB\n' "$free_gb" "$required_gb" >&2
            printf 'Free space and rerun the same command; completed keys will be skipped.\n' >&2
            exit 4
        fi

        printf '[START] %s %s compressed~%sGB free=%sGB\n' \
            "$file_key" "$description" "$compressed_gb" "$free_gb"
        download_file_key "$file_key"
        printf '%s\n' "$file_key" >> "$COMPLETED_FILE"
        printf '[DONE] %s %s\n' "$file_key" "$description"
    done

    printf 'Full download completed: %s\n' "$OUTPUT_DIR"
}

mode=${1:-}
if [[ "$mode" == "-h" || "$mode" == "--help" || -z "$mode" ]]; then
    usage
    [[ -n "$mode" ]] && exit 0
    exit 2
fi

if [[ "$mode" == "plan" ]]; then
    print_plan
    exit 0
fi

if [[ ! -x "$AIHUB_SHELL" ]]; then
    printf 'ERROR: aihubshell not found or not executable: %s\n' "$AIHUB_SHELL" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

case "$mode" in
    list)
        exec "$AIHUB_SHELL" -mode l -datasetkey "$DATASET_KEY"
        ;;
    labels)
        require_api_key
        download_file_key 38309
        download_file_key 38311
        ;;
    validation)
        require_api_key
        download_file_key 38311
        download_file_key 38312
        ;;
    full)
        require_api_key
        download_full
        ;;
    *)
        printf 'ERROR: unsupported mode: %s\n' "$mode" >&2
        usage >&2
        exit 2
        ;;
esac
