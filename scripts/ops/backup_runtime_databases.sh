#!/usr/bin/env bash
set -euo pipefail

# SQLite runtime DB를 온라인 backup API로 복사한다. 서비스 중지 없이 실행 가능하다.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${CCTV_RUNTIME_DATA_DIR:-${PROJECT_ROOT}/data}"
BACKUP_ROOT="${CCTV_RUNTIME_BACKUP_DIR:-${DATA_ROOT}/backups/runtime}"
RETENTION_DAYS="${CCTV_RUNTIME_BACKUP_RETENTION_DAYS:-7}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST_DIR="${BACKUP_ROOT}/${STAMP}"

mkdir -p "${DEST_DIR}"

backup_sqlite() {
    local source_path="$1"
    local name
    name="$(basename "${source_path}")"
    if [[ ! -f "${source_path}" ]]; then
        echo "skip: ${source_path} (not found)"
        return 0
    fi
    python3 - "${source_path}" "${DEST_DIR}/${name}" <<'PY'
import sqlite3
import sys

source, destination = sys.argv[1:]
with sqlite3.connect(source) as source_conn, sqlite3.connect(destination) as destination_conn:
    source_conn.backup(destination_conn)
PY
    echo "backup: ${source_path} -> ${DEST_DIR}/${name}"
}

backup_sqlite "${DATA_ROOT}/runtime/appearances.db"
backup_sqlite "${DATA_ROOT}/runtime/action_events.db"
backup_sqlite "${DATA_ROOT}/runtime/event_reviews.db"
backup_sqlite "${DATA_ROOT}/runtime/commercial_faces.db"

if [[ -f "${DATA_ROOT}/fall_dataset/annotations/review.jsonl" ]]; then
    cp --reflink=auto "${DATA_ROOT}/fall_dataset/annotations/review.jsonl" "${DEST_DIR}/review.jsonl"
    echo "backup: review.jsonl"
fi

find "${BACKUP_ROOT}" -mindepth 1 -maxdepth 1 -type d -mtime "+${RETENTION_DAYS}" -print -exec rm -rf -- {} +
echo "backup complete: ${DEST_DIR}"
