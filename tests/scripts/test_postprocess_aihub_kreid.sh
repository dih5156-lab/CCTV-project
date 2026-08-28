#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="${PROJECT_ROOT}/scripts/datasets/postprocess_aihub_kreid.sh"
TEST_ROOT=$(mktemp -d "${PROJECT_ROOT}/.tmp_aihub_postprocess_test.XXXXXX")
trap 'rm -rf "$TEST_ROOT"' EXIT

set +e
AIHUB_KREID_DIR="$TEST_ROOT" "$SCRIPT" --once >/dev/null 2>&1
waiting_status=$?
set -e

if [[ $waiting_status -ne 10 ]]; then
    printf 'FAIL: incomplete download must return waiting status 10.\n' >&2
    exit 1
fi

printf '%s\n' 38308 38309 38310 38311 38312 50394 50395 50396 50397 50398 \
    > "${TEST_ROOT}/.aihubshell_completed_filekeys"

TEST_ROOT="$TEST_ROOT" .venv/bin/python - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["TEST_ROOT"])
label_xml = """<xml><FILE><name>H00001_frame.png</name></FILE><OBJECT ID="H00001" TYPE="Human"><upperclothes>long_sleeve</upperclothes><upperclothes_color>white</upperclothes_color><defined_upperclothes_color>true</defined_upperclothes_color><lowerclothes>long_pants</lowerclothes><lowerclothes_color>black</lowerclothes_color><defined_lowerclothes_color>true</defined_lowerclothes_color></OBJECT></xml>"""
for index in range(10):
    path = root / f"archive_{index}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"payload_{index}.txt", "ok")
for split in ("Training", "Validation"):
    path = root / split / f"[라벨]{split}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("H00001_frame.xml", label_xml)
PY

AIHUB_KREID_DIR="$TEST_ROOT" AIHUB_POSTPROCESS_REPORT_DIR="${TEST_ROOT}/reports" \
    "$SCRIPT" --once >/dev/null

if [[ ! -s "${TEST_ROOT}/reports/label_report.json" ]]; then
    printf 'FAIL: label report was not generated.\n' >&2
    exit 1
fi
if [[ ! -f "${TEST_ROOT}/reports/postprocess_complete" ]]; then
    printf 'FAIL: completion marker was not generated.\n' >&2
    exit 1
fi

printf 'PASS: AI-Hub post-processing waits for all keys and generates reports.\n'
