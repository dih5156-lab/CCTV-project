#!/usr/bin/env bash
# with_docker_group.sh - 현재 셸에 docker 그룹이 반영되지 않았을 때 docker 명령 실행 도우미

set -euo pipefail

usage() {
  cat <<'EOF'
사용법:
  ./scripts/ops/with_docker_group.sh <command> [args...]

예시:
  ./scripts/ops/with_docker_group.sh docker ps
  ./scripts/ops/with_docker_group.sh docker compose -f docker-compose.jetson.yml ps
  ./scripts/ops/with_docker_group.sh ./scripts/ops/run_operation_check.sh

설명:
  사용자가 이미 docker 그룹에 속하지만 현재 셸 세션이 변경을 반영하지 못한 경우,
  sg docker 를 사용해 명령을 실행합니다.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -eq 0 ]]; then
  usage
  exit 0
fi

if [[ ! -S /var/run/docker.sock ]]; then
  echo "Docker socket을 찾을 수 없습니다: /var/run/docker.sock" >&2
  exit 1
fi

if id -nG | tr ' ' '\n' | grep -Fxq docker; then
  exec "$@"
fi

if ! getent group docker >/dev/null; then
  echo "docker 그룹이 없습니다. Docker 설치 또는 그룹 구성을 먼저 확인하세요." >&2
  exit 1
fi

if ! getent group docker | cut -d: -f4 | tr ',' '\n' | grep -Fxq "${USER}"; then
  echo "현재 사용자 ${USER}는 docker 그룹에 속해 있지 않습니다." >&2
  echo "다음 명령 후 재로그인하세요: sudo usermod -aG docker ${USER}" >&2
  exit 1
fi

quoted_cmd=""
for arg in "$@"; do
  printf -v escaped_arg '%q' "$arg"
  quoted_cmd+="${quoted_cmd:+ }${escaped_arg}"
done

exec sg docker -c "$quoted_cmd"