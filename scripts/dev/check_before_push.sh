#!/usr/bin/env bash
# Run the local checks that most often fail after pushing to GitHub Actions.

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$ROOT_DIR"

export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
RUFF_BIN=${RUFF_BIN:-.venv/bin/ruff}
PYTEST_TARGETS=${PYTEST_TARGETS:-tests/}
read -r -a PYTEST_ARGS <<< "$PYTEST_TARGETS"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    echo "Run this from a prepared dev environment, or set PYTHON_BIN." >&2
    exit 127
fi

if [[ ! -x "$RUFF_BIN" ]]; then
    echo "ERROR: ruff executable not found: $RUFF_BIN" >&2
    echo "Install dev dependencies first: $PYTHON_BIN -m pip install -r requirements-dev.txt" >&2
    exit 127
fi

echo "[1/5] sensitive defaults"
"$PYTHON_BIN" scripts/health/check_sensitive_defaults.py

echo "[2/5] root generated files"
"$PYTHON_BIN" scripts/health/check_root_generated_files.py

echo "[3/5] ruff auto-fix"
"$RUFF_BIN" check . --fix

echo "[4/5] ruff check"
"$RUFF_BIN" check .

echo "[5/5] pytest"
"$PYTHON_BIN" -m pytest "${PYTEST_ARGS[@]}" -v --tb=short

echo "Local pre-push checks passed."
