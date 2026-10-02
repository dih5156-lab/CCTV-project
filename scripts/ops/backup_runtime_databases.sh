#!/usr/bin/env bash
set -euo pipefail

# SQLite runtime DB를 온라인 backup API로 복사한다. 서비스 중지 없이 실행 가능하다.
# 이벤트 본문이 포함된 새 백업은 실행 사용자만 읽을 수 있게 생성한다.
umask 077
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${CCTV_RUNTIME_DATA_DIR:-${PROJECT_ROOT}/data}"
BACKUP_ROOT="${CCTV_RUNTIME_BACKUP_DIR:-${DATA_ROOT}/backups/runtime}"
RETENTION_DAYS="${CCTV_RUNTIME_BACKUP_RETENTION_DAYS:-7}"
PRUNE_BACKUPS="${CCTV_RUNTIME_BACKUP_PRUNE:-1}"
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
from contextlib import closing
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
# 원본을 생성/변경하지 않으며 WAL의 커밋까지 SQLite API로 일관되게 복사한다.
if destination.exists():
    raise FileExistsError(f"backup already exists: {destination}")
temporary = destination.with_suffix(destination.suffix + ".partial")
with temporary.open("xb"):
    pass
try:
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as source_conn:
        with closing(sqlite3.connect(temporary)) as destination_conn:
            source_conn.backup(destination_conn)
            if destination_conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise RuntimeError(f"invalid SQLite backup: {source.name}")
    # 검사가 끝난 파일만 복원 가능한 .db 이름으로 노출한다.
    temporary.replace(destination)
finally:
    temporary.unlink(missing_ok=True)
PY
    echo "backup: ${source_path} -> ${DEST_DIR}/${name}"
}

backup_sqlite "${DATA_ROOT}/runtime/appearances.db"
backup_sqlite "${DATA_ROOT}/runtime/action_events.db"
backup_sqlite "${DATA_ROOT}/runtime/event_reviews.db"
backup_sqlite "${DATA_ROOT}/runtime/commercial_faces.db"
# 최종 전달 전 대기 중인 이벤트와 장치 응답도 복구 대상에 포함한다.
backup_sqlite "${DATA_ROOT}/runtime/mqtt_event_outbox.db"
backup_sqlite "${DATA_ROOT}/runtime/action_http_outbox.db"
backup_sqlite "${DATA_ROOT}/runtime/event_outbox.db"
backup_sqlite "${DATA_ROOT}/runtime/edgex_command_results.db"

if [[ -f "${DATA_ROOT}/fall_dataset/annotations/review.jsonl" ]]; then
    cp --reflink=auto "${DATA_ROOT}/fall_dataset/annotations/review.jsonl" "${DEST_DIR}/review.jsonl"
    echo "backup: review.jsonl"
fi

# 복구 리허설에서는 이전 근거를 보존한다. 정기 작업의 기존 보관 정책은 유지한다.
if [[ "${PRUNE_BACKUPS}" == "1" ]]; then
    find "${BACKUP_ROOT}" -mindepth 1 -maxdepth 1 -type d -mtime "+${RETENTION_DAYS}" -print -exec rm -rf -- {} +
fi
echo "backup complete: ${DEST_DIR}"
