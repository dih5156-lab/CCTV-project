"""runner 공통 유틸리티.

각 runner에서 반복되던 프로젝트 루트 등록과 로깅 초기화를 공통화한다.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_runner_logging(level: int = logging.INFO) -> None:
    """runner용 기본 로깅을 한 번만 구성한다."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    logging.basicConfig(level=level, format=_LOG_FORMAT)
