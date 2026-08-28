#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="${PROJECT_ROOT}/scripts/datasets/download_aihub_kreid_sample.sh"
TEST_ROOT=$(mktemp -d "${PROJECT_ROOT}/.tmp_aihub_test.XXXXXX")
trap 'rm -rf "$TEST_ROOT"' EXIT

make_fake_shell() {
    local path=$1
    local behavior=$2
    apply_patch <<PATCH
*** Begin Patch
*** Add File: $path
+#!/usr/bin/env bash
+set -euo pipefail
+file_key=""
+while [[ \$# -gt 0 ]]; do
+    if [[ \$1 == "-filekey" ]]; then
+        file_key=\$2
+        shift
+    fi
+    shift
+done
+$behavior
*** End Patch
PATCH
    chmod +x "$path"
}

failed_shell="${TEST_ROOT}/failed_aihubshell"
make_fake_shell "$failed_shell" 'printf "Download failed with HTTP status 401.\n"; exit 0'
failed_output="${TEST_ROOT}/failed_output"

set +e
AIHUB_API_KEY=test AIHUB_SHELL="$failed_shell" AIHUB_KREID_DIR="$failed_output" \
    "$SCRIPT" full >/dev/null 2>&1
failed_status=$?
set -e

if [[ $failed_status -eq 0 ]]; then
    printf 'FAIL: an HTTP failure reported with exit 0 was recorded as success.\n' >&2
    exit 1
fi
if [[ -s "${failed_output}/.aihubshell_completed_filekeys" ]]; then
    printf 'FAIL: failed file key was written to the completion record.\n' >&2
    exit 1
fi

empty_shell="${TEST_ROOT}/empty_aihubshell"
make_fake_shell "$empty_shell" 'printf "Request successful with HTTP status 200.\nDownload successful.\n"'
empty_output="${TEST_ROOT}/empty_output"

set +e
AIHUB_API_KEY=test AIHUB_SHELL="$empty_shell" AIHUB_KREID_DIR="$empty_output" \
    "$SCRIPT" full >/dev/null 2>&1
empty_status=$?
set -e

if [[ $empty_status -eq 0 ]]; then
    printf 'FAIL: success text without an extracted payload was recorded as success.\n' >&2
    exit 1
fi
if [[ -s "${empty_output}/.aihubshell_completed_filekeys" ]]; then
    printf 'FAIL: empty download was written to the completion record.\n' >&2
    exit 1
fi

successful_shell="${TEST_ROOT}/successful_aihubshell"
make_fake_shell "$successful_shell" 'printf "Request successful with HTTP status 200.\nDownload successful.\n"; printf "downloaded\n" > "payload_${file_key}.jpg"'
successful_output="${TEST_ROOT}/successful_output"
AIHUB_API_KEY=test AIHUB_SHELL="$successful_shell" AIHUB_KREID_DIR="$successful_output" \
    "$SCRIPT" full >/dev/null

completed_count=$(sort -u "${successful_output}/.aihubshell_completed_filekeys" | wc -l)
payload_count=$(find "$successful_output" -name 'payload_*.jpg' -type f | wc -l)
if [[ $completed_count -ne 10 || $payload_count -ne 10 ]]; then
    printf 'FAIL: successful downloads were not recorded correctly.\n' >&2
    exit 1
fi

printf 'PASS: AI-Hub completion tracking requires downloaded payloads.\n'
