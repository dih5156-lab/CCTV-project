#!/usr/bin/env bash
# 호스트에 연결한 DeepStream 7.1 SDK의 런타임 의존성을 재현 가능하게 준비한다.
# SDK/JetPack 교체는 하지 않으며 기본 실행은 설치 예정 항목만 보여 준다.
set -euo pipefail

mode="${1:---dry-run}"
if [[ "$#" -gt 1 || ( "$mode" != "--dry-run" && "$mode" != "--apply" ) ]]; then
    echo "사용법: bash $0 [--dry-run|--apply]" >&2
    exit 2
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Jetson ARM64 호스트에서 실행해야 합니다." >&2
    exit 1
fi
sdk_status="$(dpkg-query -W -f='${db:Status-Status} ${Version}' deepstream-7.1 2>/dev/null || true)"
if [[ "$sdk_status" != installed\ 7.1.* ]]; then
    echo "DeepStream 7.1 SDK를 먼저 설치해야 합니다. 현재 SDK/JetPack은 변경하지 않습니다." >&2
    exit 1
fi

# 컨테이너용 Python/model 의존성과 구분하며, 호스트 실행에 필요한 Ubuntu 패키지만 설치한다.
packages=(
    libssl3 libgstreamer1.0-0 gstreamer1.0-tools
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
    gstreamer1.0-plugins-ugly gstreamer1.0-libav
    libgstrtspserver-1.0-0 libyaml-cpp0.7 libjsoncpp25 libjansson4
)
apt_args=(install --no-install-recommends --no-upgrade --no-remove)
if [[ "$mode" == "--dry-run" ]]; then
    apt-get --simulate "${apt_args[@]}" "${packages[@]}"
    exit 0
fi
if [[ "$EUID" -ne 0 ]]; then
    echo "실제 설치는 sudo bash $0 --apply 로 실행하세요." >&2
    exit 1
fi
apt-get -y "${apt_args[@]}" "${packages[@]}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$project_root/scripts/health/check_deepstream_env.py" --native-only
