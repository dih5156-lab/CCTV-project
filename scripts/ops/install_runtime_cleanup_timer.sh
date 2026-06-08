#!/usr/bin/env bash
# CCTV 런타임 데이터 정리 systemd 타이머 설치 도우미

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SYSTEMD_DIR=${SYSTEMD_DIR:-/etc/systemd/system}
UNIT_NAME=cctv-runtime-cleanup
SERVICE_TEMPLATE="${PROJECT_ROOT}/deploy/systemd/cctv-runtime-cleanup.service.in"
TIMER_TEMPLATE="${PROJECT_ROOT}/deploy/systemd/cctv-runtime-cleanup.timer"
SERVICE_TARGET="${SYSTEMD_DIR}/${UNIT_NAME}.service"
TIMER_TARGET="${SYSTEMD_DIR}/${UNIT_NAME}.timer"
DRY_RUN=0

usage() {
  cat <<'EOF'
사용법:
  sudo ./scripts/ops/install_runtime_cleanup_timer.sh
  ./scripts/ops/install_runtime_cleanup_timer.sh --dry-run

동작:
  매일 한국시간 09:00에 런타임 정리를 실행하는
  systemd service와 timer를 설치하고 활성화합니다.

환경변수:
  SYSTEMD_DIR  기본값: /etc/systemd/system
EOF
}

render_service() {
  sed "s#@PROJECT_ROOT@#${PROJECT_ROOT//\#/\\#}#g" "$SERVICE_TEMPLATE"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "알 수 없는 인자: $*" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

for template in "$SERVICE_TEMPLATE" "$TIMER_TEMPLATE"; do
  if [[ ! -f "$template" ]]; then
    echo "템플릿 파일 없음: ${template}" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== ${SERVICE_TARGET} ==="
  render_service
  echo
  echo "=== ${TIMER_TARGET} ==="
  cat "$TIMER_TEMPLATE"
  exit 0
fi

if [[ "$EUID" -ne 0 ]]; then
  echo "systemd 설치는 root 권한이 필요합니다. sudo로 다시 실행하세요." >&2
  exit 1
fi

install -d -m 0755 "$SYSTEMD_DIR"
render_service > "$SERVICE_TARGET"
install -m 0644 "$TIMER_TEMPLATE" "$TIMER_TARGET"
systemctl daemon-reload
systemctl enable --now "${UNIT_NAME}.timer"
systemctl list-timers --all "${UNIT_NAME}.timer"

echo "CCTV 런타임 데이터 정리 타이머 설치 완료"
