#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

if ! git diff --cached --quiet; then
  echo "[weekly-git] 기존 staged 상태가 있어 중단합니다." >&2
  exit 1
fi

week_label="$(date +%Y-%m-%d)"
if ! grep -qE "^### ${week_label//-/\\-}$" README.md; then
  temporary_readme="$(mktemp)"
  awk -v date_label="$week_label" '
    /^## 업데이트 이력$/ {
      print
      print ""
      print "### " date_label
      print "- 주간 자동 업데이트: 코드·설정·문서·테스트 변경사항 반영"
      print ""
      next
    }
    { print }
  ' README.md >"$temporary_readme"
  mv "$temporary_readme" README.md
fi

# 소스/설정/문서/테스트만 자동 반영한다. 데이터·모델·검수 산출물은 수동 검토 대상이다.
git add -- .github config deploy docs scripts src tests README.md pyproject.toml \
  docker-compose*.yml Dockerfile* requirements 2>/dev/null || true

if git diff --cached --quiet; then
  echo "[weekly-git] 이번 주 반영할 코드 변경이 없습니다."
  exit 0
fi

if ! git diff --cached --check; then
  echo "[weekly-git] 공백 오류가 있어 commit을 중단합니다." >&2
  exit 1
fi

changed_count="$(git diff --cached --name-only | wc -l | tr -d ' ')"
git commit -m "chore: weekly project update ${week_label} (${changed_count} files)"
git push origin HEAD
echo "[weekly-git] ${week_label}: ${changed_count}개 파일 push 완료"
