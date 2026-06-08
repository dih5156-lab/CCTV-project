#!/usr/bin/env bash
# rotate_alert_log.sh — alert_api_events.jsonl 크기 기반 로테이션
#
# 사용법:
#   ./scripts/ops/rotate_alert_log.sh [경로] [최대크기MB]
#
# 예시 (cron에서 매 시간 실행):
#   0 * * * * /path/to/project/scripts/ops/rotate_alert_log.sh /path/to/data/logs/alert_api_events.jsonl 200
#
# 동작:
#   - 파일 크기가 MAX_MB 이상이면 .1 로 rotate
#   - 최대 5개 보관 (alert_api_events.jsonl.1 ~ .5)

set -euo pipefail

LOG_PATH="${1:-/app/data/logs/alert_api_events.jsonl}"
MAX_MB="${2:-200}"
MAX_KEEP=5

if ! [[ "$MAX_MB" =~ ^[1-9][0-9]*$ ]]; then
  echo "최대 크기는 양의 정수 MB여야 합니다: ${MAX_MB}" >&2
  exit 2
fi

if [[ ! -f "$LOG_PATH" ]]; then
  echo "파일 없음: $LOG_PATH"
  exit 0
fi

SIZE_BYTES=$(stat -c %s "$LOG_PATH")
MAX_BYTES=$((MAX_MB * 1024 * 1024))

if (( SIZE_BYTES < MAX_BYTES )); then
  echo "$(date -Iseconds) 크기 ${SIZE_BYTES}바이트 < ${MAX_BYTES}바이트 — 로테이션 불필요"
  exit 0
fi

echo "$(date -Iseconds) 로테이션 시작: ${SIZE_BYTES}바이트 >= ${MAX_BYTES}바이트"

# 기존 백업 순차 이동
for i in $(seq $((MAX_KEEP - 1)) -1 1); do
  src="${LOG_PATH}.${i}"
  dst="${LOG_PATH}.$((i + 1))"
  [[ -f "$src" ]] && mv "$src" "$dst"
done

# 현재 파일을 .1로 복사 후 비우기 (fd 유지)
cp "$LOG_PATH" "${LOG_PATH}.1"
truncate -s 0 "$LOG_PATH"

echo "$(date -Iseconds) 로테이션 완료: ${LOG_PATH}.1 로 이동, 원본 초기화"
