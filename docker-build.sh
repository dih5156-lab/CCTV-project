#!/usr/bin/env bash
# docker-build.sh
#
# Windows PowerShell에서 'wsl -- bash -c "./docker-build.sh"' 또는
# WSL2 터미널에서 직접 'bash docker-build.sh' 로 실행
#
# 이유: Docker Desktop for Windows + BuildKit 조합에서 Windows 경로(C:\...)로
#       Dockerfile을 읽는 file request가 실패하는 버그가 있어,
#       WSL2(Linux) 경로에서 빌드해야 합니다.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== CCTV 이미지 빌드 (WSL2 모드) ==="
echo "프로젝트 경로: $PROJECT_DIR"

# 사용 예:
#   ./docker-build.sh cctv-public-api
#   COMPOSE_FILE=docker-compose.jetson.yml ./docker-build.sh cctv-alert-api
#   START_AFTER_BUILD=0 ./docker-build.sh cctv-action-layer

is_jetson_host() {
  [[ "$(uname -s)" == "Linux" ]] || return 1
  local machine
  machine="$(uname -m)"
  [[ "$machine" == "aarch64" || "$machine" == "arm64" ]] || return 1

  if [[ -f /etc/nv_tegra_release ]]; then
    return 0
  fi
  if [[ -r /proc/device-tree/model ]] && grep -qi "jetson" /proc/device-tree/model; then
    return 0
  fi
  return 1
}

if [[ -n "${COMPOSE_FILE:-}" ]]; then
  COMPOSE_FILE_PATH="$COMPOSE_FILE"
else
  if is_jetson_host; then
    COMPOSE_FILE_PATH="docker-compose.jetson.yml"
  else
    COMPOSE_FILE_PATH="docker-compose.yml"
  fi
fi

if [[ -n "${COMPOSE_ENV_FILE:-}" ]]; then
  ENV_FILE_PATH="$COMPOSE_ENV_FILE"
elif [[ "$COMPOSE_FILE_PATH" == "docker-compose.jetson.yml" ]]; then
  ENV_FILE_PATH=".env.jetson"
else
  ENV_FILE_PATH=".env"
fi

START_AFTER_BUILD="${START_AFTER_BUILD:-1}"

if [[ ! -f "$COMPOSE_FILE_PATH" ]]; then
  echo "Compose 파일을 찾을 수 없습니다: $COMPOSE_FILE_PATH" >&2
  exit 1
fi

if [[ -f "$ENV_FILE_PATH" ]]; then
  COMPOSE_ENV_ARGS=(--env-file "$ENV_FILE_PATH")
else
  COMPOSE_ENV_ARGS=()
fi

if [[ "$#" -gt 0 ]]; then
  SERVICES=("$@")
else
  SERVICES=("cctv-public-api")
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon에 접근할 수 없습니다." >&2
  echo "확인 방법:" >&2
  echo "  1) Docker Desktop 또는 Docker daemon이 실행 중인지 확인" >&2
  echo "  2) Linux/WSL이면 현재 사용자가 docker 그룹에 포함되어 있는지 확인: groups" >&2
  echo "  3) 필요 시 새 터미널에서 다시 실행하거나 sudo 권한으로 실행" >&2
  exit 1
fi

echo "Compose 파일: $COMPOSE_FILE_PATH"
if [[ -f "$ENV_FILE_PATH" ]]; then
  echo "환경 파일: $ENV_FILE_PATH"
else
  echo "환경 파일: 없음 (기본 docker compose 환경 사용)"
fi
echo "빌드 대상: ${SERVICES[*]}"
docker compose "${COMPOSE_ENV_ARGS[@]}" -f "$COMPOSE_FILE_PATH" build "${SERVICES[@]}"

if [[ "$START_AFTER_BUILD" == "1" || "$START_AFTER_BUILD" == "true" ]]; then
  echo ""
  echo "빌드 완료. 서비스 시작:"
  docker compose "${COMPOSE_ENV_ARGS[@]}" -f "$COMPOSE_FILE_PATH" up "${SERVICES[@]}" -d
else
  echo ""
  echo "빌드 완료. START_AFTER_BUILD=$START_AFTER_BUILD 이므로 서비스 시작은 건너뜁니다."
fi

echo ""
docker compose "${COMPOSE_ENV_ARGS[@]}" -f "$COMPOSE_FILE_PATH" ps "${SERVICES[@]}"
