#!/usr/bin/env sh
set -eu

ENV_FILE="${1:-.env}"

if [ ! -f "$ENV_FILE" ]; then
  cp .env.example "$ENV_FILE"
fi

generate_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 32 | tr '+/' '-_' | tr -d '='
  else
    date +%s%N | sha256sum | awk '{print $1}'
  fi
}

is_placeholder_or_empty() {
  KEY_NAME="$1"
  ! grep -Eq "^${KEY_NAME}=.+$" "$ENV_FILE" \
    || grep -Eq "^${KEY_NAME}=\$\{${KEY_NAME}:-\}$" "$ENV_FILE"
}

ensure_secret() {
  KEY_NAME="$1"

  if ! is_placeholder_or_empty "$KEY_NAME"; then
    echo "$KEY_NAME already set in $ENV_FILE"
    return
  fi

  VALUE="$(generate_secret)"
  if grep -Eq "^${KEY_NAME}=" "$ENV_FILE"; then
    sed -i "s|^${KEY_NAME}=.*|${KEY_NAME}=${VALUE}|" "$ENV_FILE"
  else
    printf '\n%s=%s\n' "$KEY_NAME" "$VALUE" >> "$ENV_FILE"
  fi
  echo "$KEY_NAME generated in $ENV_FILE"
}

ensure_secret PUBLIC_API_KEY
ensure_secret INTERNAL_SERVICE_TOKEN
