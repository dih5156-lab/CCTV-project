#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
    echo "X11 DISPLAY가 설정되지 않았습니다." >&2
    exit 1
fi
# 소켓과 인증 파일의 존재만으로는 로그인 세션 준비를 보장하지 못한다.
# 실제 인증을 재시도하며 현재 컨테이너의 로컬 root 사용자만 허용한다.
for attempt in {1..30}; do
    if timeout 3 xhost +SI:localuser:root; then
        exit 0
    fi
    sleep 2
done
echo "X11 인증 준비 실패: DISPLAY=$DISPLAY" >&2
exit 1
