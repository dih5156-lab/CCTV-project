#!/usr/bin/env bash
# 런타임 데이터 정리 도우미
#
# 기본 실행은 대상만 출력하는 미리보기 모드입니다.
# 실제 삭제와 로그 회전은 --apply를 명시한 경우에만 수행합니다.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_DIR=${RUNTIME_DATA_DIR:-"${PROJECT_ROOT}/data"}
RUNTIME_DIR=${RUNTIME_DIR:-"${DATA_DIR}/runtime"}
LOG_DIR=${RUNTIME_LOG_DIR:-"${DATA_DIR}/logs"}
CROP_DIR=${APPEARANCE_CROP_DIR:-"${RUNTIME_DIR}/appearance_crops"}
ALERT_LOG=${ALERT_LOG_PATH:-"${LOG_DIR}/alert_api_events.jsonl"}
SENSOR_LOG=${SENSOR_LOG_PATH:-"${LOG_DIR}/sensor_readings.jsonl"}
FALL_SHADOW_LOG=${FALL_SHADOW_REVIEW_LOG_PATH:-"${LOG_DIR}/fall_shadow_review.jsonl"}
FALL_REVIEW_CLIP_DIR=${FALL_SHADOW_CLIP_DIR:-"${DATA_DIR}/fall_review_clips"}
APPEARANCES_DB=${APPEARANCES_DB:-"${RUNTIME_DIR}/appearances.db"}
CROP_RETENTION_DAYS=${CROP_RETENTION_DAYS:-7}
FALL_REVIEW_RETENTION_DAYS=${FALL_REVIEW_RETENTION_DAYS:-3}
LOG_MAX_MB=${LOG_MAX_MB:-200}
FALL_SHADOW_LOG_MAX_MB=${FALL_SHADOW_LOG_MAX_MB:-50}
PYTHON_BIN=${PYTHON_BIN:-}
APPLY=0

usage() {
  cat <<'EOF'
사용법:
  ./scripts/cleanup/cleanup_runtime_data.sh [--apply]

환경변수:
  RUNTIME_DATA_DIR        기본값: <project>/data
  RUNTIME_DIR             기본값: <data>/runtime
  RUNTIME_LOG_DIR         기본값: <data>/logs
  APPEARANCE_CROP_DIR     기본값: <data>/runtime/appearance_crops
  ALERT_LOG_PATH          기본값: <data>/logs/alert_api_events.jsonl
  SENSOR_LOG_PATH         기본값: <data>/logs/sensor_readings.jsonl
  FALL_SHADOW_REVIEW_LOG_PATH
                          기본값: <data>/logs/fall_shadow_review.jsonl
  FALL_SHADOW_CLIP_DIR    기본값: <data>/fall_review_clips
  APPEARANCES_DB          기본값: <data>/runtime/appearances.db
  CROP_RETENTION_DAYS     기본값: 7
  FALL_REVIEW_RETENTION_DAYS
                          기본값: 3
  LOG_MAX_MB              기본값: 200
  FALL_SHADOW_LOG_MAX_MB  기본값: 50
  PYTHON_BIN              기본값: <project>/.venv/bin/python, 없으면 python3

동작:
  기본 실행은 미리보기입니다.
  --apply를 지정하면 보존 기간이 지난 crop 이미지와 낙상 검토 클립을 삭제하고,
  삭제된 crop의 DB 참조를 비운 뒤 크기 제한을 넘은 JSONL 로그를 회전합니다.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
  shift
fi

if [[ "$#" -ne 0 ]]; then
  echo "알 수 없는 인자: $*" >&2
  usage >&2
  exit 2
fi

if ! [[ "$CROP_RETENTION_DAYS" =~ ^[0-9]+$ ]]; then
  echo "CROP_RETENTION_DAYS는 0 이상의 정수여야 합니다: ${CROP_RETENTION_DAYS}" >&2
  exit 2
fi

if ! [[ "$FALL_REVIEW_RETENTION_DAYS" =~ ^[0-9]+$ ]]; then
  echo "FALL_REVIEW_RETENTION_DAYS는 0 이상의 정수여야 합니다: ${FALL_REVIEW_RETENTION_DAYS}" >&2
  exit 2
fi

if ! [[ "$LOG_MAX_MB" =~ ^[1-9][0-9]*$ ]]; then
  echo "LOG_MAX_MB는 양의 정수여야 합니다: ${LOG_MAX_MB}" >&2
  exit 2
fi

if ! [[ "$FALL_SHADOW_LOG_MAX_MB" =~ ^[1-9][0-9]*$ ]]; then
  echo "FALL_SHADOW_LOG_MAX_MB는 양의 정수여야 합니다: ${FALL_SHADOW_LOG_MAX_MB}" >&2
  exit 2
fi

print_file_size() {
  local path=$1
  if [[ -f "$path" ]]; then
    du -h "$path" | cut -f1
  else
    printf '0'
  fi
}

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

CROP_REF_ARGS=()
if [[ "$APPLY" -eq 1 ]]; then
  CROP_REF_ARGS=(--apply)
fi

EXPIRED_COUNT=0
if [[ -d "$CROP_DIR" ]]; then
  EXPIRED_COUNT=$(find "$CROP_DIR" -type f -mtime "+${CROP_RETENTION_DAYS}" | wc -l)
fi

EXPIRED_FALL_CLIP_COUNT=0
if [[ -d "$FALL_REVIEW_CLIP_DIR" ]]; then
  EXPIRED_FALL_CLIP_COUNT=$(find "$FALL_REVIEW_CLIP_DIR" -type f -mtime "+${FALL_REVIEW_RETENTION_DAYS}" | wc -l)
fi

echo "=== CCTV 런타임 데이터 정리 ==="
echo "모드: $([[ "$APPLY" -eq 1 ]] && printf '적용' || printf '미리보기')"
echo "runtime 경로: ${DATA_DIR}"
echo "runtime 산출물 경로: ${RUNTIME_DIR}"
echo "log 경로: ${LOG_DIR}"
echo "crop 경로: ${CROP_DIR}"
echo "crop 보존 기간: ${CROP_RETENTION_DAYS}일"
echo "삭제 대상 crop 파일 수: ${EXPIRED_COUNT}"
echo "낙상 검토 클립 경로: ${FALL_REVIEW_CLIP_DIR}"
echo "낙상 검토 클립 보존 기간: ${FALL_REVIEW_RETENTION_DAYS}일"
echo "삭제 대상 낙상 검토 클립 수: ${EXPIRED_FALL_CLIP_COUNT}"
echo "이벤트 로그: ${ALERT_LOG} ($(print_file_size "$ALERT_LOG"))"
echo "센서 로그: ${SENSOR_LOG} ($(print_file_size "$SENSOR_LOG"))"
echo "낙상 shadow 로그: ${FALL_SHADOW_LOG} ($(print_file_size "$FALL_SHADOW_LOG"))"
echo "로그 회전 기준: ${LOG_MAX_MB}MB"
echo "낙상 shadow 로그 회전 기준: ${FALL_SHADOW_LOG_MAX_MB}MB"

if [[ "$APPLY" -eq 0 ]]; then
  "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/cleanup/cleanup_appearance_crop_refs.py" \
    "${CROP_REF_ARGS[@]}" \
    --db-path "$APPEARANCES_DB" \
    --crop-dir "$CROP_DIR"
  echo "실제 반영하려면 --apply를 추가하세요."
  exit 0
fi

mkdir -p "$RUNTIME_DIR" "$LOG_DIR"

if [[ "$EXPIRED_COUNT" -gt 0 && -d "$CROP_DIR" && ! -w "$CROP_DIR" ]]; then
  echo "crop 경로에 삭제 권한이 없습니다: ${CROP_DIR}" >&2
  echo "컨테이너가 생성한 파일이면 sudo로 다시 실행하세요." >&2
  exit 1
fi

if [[ "$EXPIRED_FALL_CLIP_COUNT" -gt 0 && -d "$FALL_REVIEW_CLIP_DIR" && ! -w "$FALL_REVIEW_CLIP_DIR" ]]; then
  echo "낙상 검토 클립 경로에 삭제 권한이 없습니다: ${FALL_REVIEW_CLIP_DIR}" >&2
  echo "컨테이너가 생성한 파일이면 sudo로 다시 실행하세요." >&2
  exit 1
fi

if [[ -d "$CROP_DIR" ]]; then
  find "$CROP_DIR" -type f -mtime "+${CROP_RETENTION_DAYS}" -delete
fi

if [[ -d "$FALL_REVIEW_CLIP_DIR" ]]; then
  find "$FALL_REVIEW_CLIP_DIR" -type f -mtime "+${FALL_REVIEW_RETENTION_DAYS}" -delete
fi

"$PYTHON_BIN" "${PROJECT_ROOT}/scripts/cleanup/cleanup_appearance_crop_refs.py" \
  "${CROP_REF_ARGS[@]}" \
  --db-path "$APPEARANCES_DB" \
  --crop-dir "$CROP_DIR"
"${PROJECT_ROOT}/scripts/ops/rotate_alert_log.sh" "$ALERT_LOG" "$LOG_MAX_MB"
"${PROJECT_ROOT}/scripts/ops/rotate_alert_log.sh" "$SENSOR_LOG" "$LOG_MAX_MB"
"${PROJECT_ROOT}/scripts/ops/rotate_alert_log.sh" "$FALL_SHADOW_LOG" "$FALL_SHADOW_LOG_MAX_MB"

echo "런타임 데이터 정리 완료"
