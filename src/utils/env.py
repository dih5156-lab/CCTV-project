"""환경 변수를 안전하게 읽기 위한 보조 함수."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union


def load_dotenv_file(path: Union[str, os.PathLike[str], None] = None, *, override: bool = False) -> bool:
    """.env 파일의 key=value 항목을 os.environ에 로드한다.

    파일이 없으면 무시한다. override=True를 명시한 경우를 제외하고
    기존 환경 변수 값은 유지한다.
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
    """일반적인 참 문자열을 기준으로 환경 변수의 불리언 값을 반환한다."""
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
    """환경 변수의 정수 값을 반환하고, 범위가 있으면 범위 안으로 제한한다."""
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
    """환경 변수의 실수 값을 반환하고, 범위가 있으면 범위 안으로 제한한다."""
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
