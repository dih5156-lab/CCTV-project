"""Small helpers for reading environment variables safely."""

from __future__ import annotations

import logging
import os
from typing import Optional


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
