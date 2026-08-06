#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRAPH_PYTHON="${GRAPH_PYTHON:-${PROJECT_ROOT}/.venv-code-review-graph/bin/python}"
GRAPH_MODULE="${GRAPH_MODULE:-code_review_graph}"
BASE_REF="${BASE_REF:-HEAD}"
REPORT_DIR="${REPORT_DIR:-${PROJECT_ROOT}/reports/code-review-graph}"

if [[ ! -x "${GRAPH_PYTHON}" ]]; then
    echo "code-review-graph Python not found: ${GRAPH_PYTHON}" >&2
    echo "Create .venv-code-review-graph or set GRAPH_PYTHON." >&2
    exit 2
fi

mkdir -p "${REPORT_DIR}"

cd "${PROJECT_ROOT}"
"${GRAPH_PYTHON}" -m "${GRAPH_MODULE}" update --repo "${PROJECT_ROOT}"
"${GRAPH_PYTHON}" -m "${GRAPH_MODULE}" detect-changes \
    --repo "${PROJECT_ROOT}" --base "${BASE_REF}" --brief \
    > "${REPORT_DIR}/detect-changes.txt"
"${GRAPH_PYTHON}" -m "${GRAPH_MODULE}" architecture \
    --repo "${PROJECT_ROOT}" --detail-level standard \
    > "${REPORT_DIR}/architecture.json"
"${GRAPH_PYTHON}" -m "${GRAPH_MODULE}" impact \
    --repo "${PROJECT_ROOT}" --base "${BASE_REF}" --depth 3 \
    > "${REPORT_DIR}/impact.json"
"${GRAPH_PYTHON}" -m "${GRAPH_MODULE}" large-functions \
    --repo "${PROJECT_ROOT}" --min-lines 300 --kind Function --limit 50 \
    > "${REPORT_DIR}/large-functions.json"
"${GRAPH_PYTHON}" -m "${GRAPH_MODULE}" dead-code \
    --repo "${PROJECT_ROOT}" --kind Function --limit 100 --json \
    > "${REPORT_DIR}/dead-code.json"
"${GRAPH_PYTHON}" -m "${GRAPH_MODULE}" wiki \
    --repo "${PROJECT_ROOT}" --force \
    > "${REPORT_DIR}/wiki.log"

echo "Code review graph report: ${REPORT_DIR}"
