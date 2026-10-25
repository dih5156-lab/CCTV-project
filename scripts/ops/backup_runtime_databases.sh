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
ADAPTER_DATA_DIR="${CCTV_EDGEX_ADAPTER_DATA_DIR:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST_DIR="${BACKUP_ROOT}/${STAMP}"

# 전용 볼륨을 지정한 경우 누락을 성공으로 숨기지 않고, 큰 백업 시작 전에 확인한다.
if [[ -n "${ADAPTER_DATA_DIR}" ]]; then
    for name in event_outbox.db aiot_commands.db; do
        if [[ ! -f "${ADAPTER_DATA_DIR}/${name}" || ! -r "${ADAPTER_DATA_DIR}/${name}" ]]; then
            echo "필수 어댑터 DB를 읽을 수 없습니다: ${ADAPTER_DATA_DIR}/${name}" >&2
            exit 1
        fi
    done
fi

mkdir -p "${DEST_DIR}"

backup_sqlite() {
    local source_path="$1"
    local destination_dir="${2:-${DEST_DIR}}"
    local required="${3:-0}"
    local name
    name="$(basename "${source_path}")"
    if [[ ! -f "${source_path}" ]]; then
        if [[ "${required}" == "1" ]]; then
            echo "필수 DB가 백업 전에 사라졌습니다: ${source_path}" >&2
            return 1
        fi
        echo "skip: ${source_path} (not found)"
        return 0
    fi
    mkdir -p "${destination_dir}"
    python3 - "${source_path}" "${destination_dir}/${name}" <<'PY'
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
    echo "backup: ${source_path} -> ${destination_dir}/${name}"
}

backup_sqlite "${DATA_ROOT}/runtime/appearances.db"
backup_sqlite "${DATA_ROOT}/runtime/action_events.db"
# 저장 전용 수신과 실행 중 종료의 보류 원본도 복구 시 함께 보존한다.
backup_sqlite "${DATA_ROOT}/runtime/action_mqtt_intake.db"
backup_sqlite "${DATA_ROOT}/runtime/event_reviews.db"
backup_sqlite "${DATA_ROOT}/runtime/commercial_faces.db"
# 최종 전달 전 대기 중인 이벤트와 장치 응답도 복구 대상에 포함한다.
backup_sqlite "${DATA_ROOT}/runtime/mqtt_event_outbox.db"
backup_sqlite "${DATA_ROOT}/runtime/action_http_outbox.db"
backup_sqlite "${DATA_ROOT}/runtime/event_outbox.db"
backup_sqlite "${DATA_ROOT}/runtime/edgex_command_results.db"
# 장치 실행 이력과 미확정 선점을 함께 보존해 복구 후 중복 출력을 막는다.
backup_sqlite "${DATA_ROOT}/device_requests/signboard.sqlite3"
backup_sqlite "${DATA_ROOT}/device_requests/speaker.sqlite3"
backup_sqlite "${DATA_ROOT}/device_requests/siren.sqlite3"

# 동명 event_outbox.db와 충돌하지 않도록 전용 볼륨의 두 DB를 함께 구분해 보관한다.
if [[ -n "${ADAPTER_DATA_DIR}" ]]; then
    backup_sqlite "${ADAPTER_DATA_DIR}/event_outbox.db" "${DEST_DIR}/edgex-adapter" 1
    backup_sqlite "${ADAPTER_DATA_DIR}/aiot_commands.db" "${DEST_DIR}/edgex-adapter" 1
fi

if [[ -f "${DATA_ROOT}/fall_dataset/annotations/review.jsonl" ]]; then
    cp --reflink=auto "${DATA_ROOT}/fall_dataset/annotations/review.jsonl" "${DEST_DIR}/review.jsonl"
    echo "backup: review.jsonl"
fi

# 복구 리허설에서는 이전 근거를 보존한다. 정기 작업의 기존 보관 정책은 유지한다.
if [[ "${PRUNE_BACKUPS}" == "1" ]]; then
    find "${BACKUP_ROOT}" -mindepth 1 -maxdepth 1 -type d -mtime "+${RETENTION_DAYS}" -print -exec rm -rf -- {} +
fi
echo "backup complete: ${DEST_DIR}"
