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

# 어떤 서비스를 빌드할지 인자로 받거나, 기본값 사용
SERVICES="${*:-cctv-public-api}"

echo "빌드 대상: $SERVICES"
docker compose build $SERVICES

echo ""
echo "빌드 완료. 서비스 시작:"
docker compose up $SERVICES -d

echo ""
docker compose ps $SERVICES
