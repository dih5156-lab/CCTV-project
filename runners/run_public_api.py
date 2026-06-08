"""run_public_api.py — CCTV 공개 API 서버 엔트리포인트.

서버팀과 공유하는 FastAPI 기반 공개 API 서버를 시작한다.
포트 기본값: 9000 (cctv-alert-api: 8000, cctv-action-layer: 8080과 구분)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_RUNNER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _RUNNER_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runners._shared import setup_runner_logging

logger = logging.getLogger("run-public-api")


def main() -> None:
    parser = argparse.ArgumentParser(description="CCTV Public API 서버")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 호스트")
    parser.add_argument("--port", type=int, default=9000, help="바인드 포트")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="개발 모드 핫 리로드 (프로덕션에서는 사용 금지)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="uvicorn worker 수 (reload 활성화 시 무시됨)",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        logger.error(
            "uvicorn이 설치되지 않았습니다. pip install 'uvicorn[standard]' 를 실행하세요."
        )
        raise SystemExit(1)

    setup_runner_logging()

    logger.info("CCTV Public API 서버 시작: http://%s:%d", args.host, args.port)
    logger.info("API 문서: http://%s:%d/docs", args.host, args.port)

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
