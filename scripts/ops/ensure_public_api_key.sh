#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"
ENV_FILE="${1:-.env}"

if [ ! -f "$ENV_FILE" ]; then
  cp "$ENV_EXAMPLE" "$ENV_FILE"
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
    || grep -Fxq "${KEY_NAME}=\${${KEY_NAME}:-}" "$ENV_FILE"
}

set_env_value() {
  KEY_NAME="$1"
  VALUE="$2"
  TEMP_FILE="${ENV_FILE}.tmp"

  awk -v key="$KEY_NAME" -v value="$VALUE" '
    BEGIN { found = 0 }
    index($0, key "=") == 1 {
      print key "=" value
      found = 1
      next
    }
    { print }
    END {
      if (!found) {
        print key "=" value
      }
    }
  ' "$ENV_FILE" > "$TEMP_FILE"
  mv "$TEMP_FILE" "$ENV_FILE"
}

ensure_secret() {
  KEY_NAME="$1"

  if ! is_placeholder_or_empty "$KEY_NAME"; then
    echo "$KEY_NAME already set in $ENV_FILE"
    return
  fi

  VALUE="$(generate_secret)"
  set_env_value "$KEY_NAME" "$VALUE"
  echo "$KEY_NAME generated in $ENV_FILE"
}

ensure_secret PUBLIC_API_KEY
ensure_secret INTERNAL_SERVICE_TOKEN
