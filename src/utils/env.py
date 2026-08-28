"""Small helpers for reading environment variables safely."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union


def load_dotenv_file(path: Union[str, os.PathLike[str], None] = None, *, override: bool = False) -> bool:
    """Load key=value entries from a .env file into os.environ.

    Missing files are ignored. Existing environment variables are preserved unless
    override=True is explicitly requested.
    """
    if path is None:
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / ".env"

    env_path = Path(path).expanduser()
    if not env_path.exists() or not env_path.is_file():
        return False

    loaded = False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key_part, value_part = line.split("=", 1)
        key = key_part.strip()
        if not key or key.startswith("export "):
            key = key.replace("export ", "", 1).strip()

        value = value_part.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
        loaded = True

    return loaded


def get_env_bool(name: str, default: bool = False) -> bool:
    """Return a boolean environment value using common truthy strings."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_env_int(
    name: str,
    default: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Return an integer environment value, clamped when bounds are provided."""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        value = default
    else:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            if logger:
                logger.warning("Invalid %s=%r; using default %s", name, raw_value, default)
            value = default

    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def get_env_float(
    name: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Return a float environment value, clamped when bounds are provided."""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        value = default
    else:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            if logger:
                logger.warning("Invalid %s=%r; using default %s", name, raw_value, default)
            value = default

    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value
