"""DeepStream 환경변수/설정값 파싱 헬퍼."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int = 0) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        logger.warning("잘못된 %s 값입니다: %r, 기본값 %d 사용", name, raw_value, default)
        return default


def read_float_setting(env_name: str, config_value: Any, default: float) -> float:
    raw_value = os.environ.get(env_name)
    if raw_value is not None:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "잘못된 %s 값입니다: %r, 기본값 %.2f 사용",
                env_name,
                raw_value,
                default,
            )
            return default
    if isinstance(config_value, bool):
        return default
    if isinstance(config_value, (int, float)):
        return float(config_value)
    return default


def read_int_setting(env_name: str, config_value: Any, default: int) -> int:
    raw_value = os.environ.get(env_name)
    if raw_value is not None:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "잘못된 %s 값입니다: %r, 기본값 %d 사용",
                env_name,
                raw_value,
                default,
            )
            return default
    if isinstance(config_value, bool):
        return default
    if isinstance(config_value, int):
        return config_value
    return default


def read_preview_max_fps() -> float:
    """DeepStream preview 샘플링 FPS를 읽는다."""
    raw_value = os.environ.get("DS_PREVIEW_MAX_FPS") or os.environ.get("STREAM_FPS") or "30.0"
    try:
        preview_fps = float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "잘못된 DS_PREVIEW_MAX_FPS/STREAM_FPS 값입니다: %r, 기본값 30.0 사용",
            raw_value,
        )
        return 30.0
    return max(0.0, min(preview_fps, 60.0))


def parse_class_ids(name: str, default: Optional[set[int]] = None) -> set[int]:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return set(default or set())
    return {int(value.strip()) for value in raw_value.split(",") if value.strip()}
