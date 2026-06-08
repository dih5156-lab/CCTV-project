#!/usr/bin/env sh
set -eu

ENV_FILE="${1:-.env}"

if [ ! -f "$ENV_FILE" ]; then
  cp .env.example "$ENV_FILE"
fi

if grep -Eq '^PUBLIC_API_KEY=.+$' "$ENV_FILE"; then
  echo "PUBLIC_API_KEY already set in $ENV_FILE"
  exit 0
fi

if command -v openssl >/dev/null 2>&1; then
  KEY="$(openssl rand -base64 32 | tr '+/' '-_' | tr -d '=')"
else
  KEY="$(date +%s%N | sha256sum | awk '{print $1}')"
fi

if grep -Eq '^PUBLIC_API_KEY=' "$ENV_FILE"; then
  sed -i "s|^PUBLIC_API_KEY=.*|PUBLIC_API_KEY=${KEY}|" "$ENV_FILE"
else
  printf '\nPUBLIC_API_KEY=%s\n' "$KEY" >> "$ENV_FILE"
fi

echo "PUBLIC_API_KEY generated in $ENV_FILE"
