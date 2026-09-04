#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/fall_demo}"
VIDEO="$(find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name overlay.mp4 -size +1k -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "${VIDEO//[$' \t\r\n']/}" ]]; then
  VIDEO="$(find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name demo.mp4 -size +1k -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
fi
if [[ -z "${VIDEO:-}" ]]; then
  echo "재생 가능한 demo.mp4/overlay.mp4를 찾지 못했습니다: $ROOT" >&2
  exit 1
fi
echo "재생: $VIDEO"
exec ffplay -hide_banner -autoexit -loglevel warning "$VIDEO"
