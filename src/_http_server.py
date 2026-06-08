"""_base_http_server.py - 내부 HTTP API 서버 공통 베이스.

camera_model_api, zone_api, face_api 세 모듈에서 반복되던
_ThreadingHTTPServer, _read_json(), _respond() 보일러플레이트를
한 곳에서 관리한다.

Usage::

    from src.services._base_http_server import BaseApiHandler, ThreadingApiServer

    class MyHandler(BaseApiHandler):
        _LOG_PREFIX = "[MyAPI]"

        def do_GET(self):
            self._respond(200, {"ok": True})

    server = ThreadingApiServer(("0.0.0.0", 8080), MyHandler)
    server.my_attr = ...               # 서버 인스턴스 속성으로 공유 상태 주입
    threading.Thread(target=server.serve_forever, daemon=True).start()
"""

import json
import logging
import os
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from .time_utils import now_kst_iso

logger = logging.getLogger(__name__)


class ThreadingApiServer(ThreadingMixIn, HTTPServer):
    """다중 요청 처리를 위한 멀티스레드 HTTP 서버."""

    daemon_threads = True


class BaseApiHandler(BaseHTTPRequestHandler):
    """JSON REST API 공통 헬퍼를 제공하는 베이스 핸들러.

    서브클래스에서 ``_LOG_PREFIX`` 클래스 변수를 재정의하면
    log_message 출력 접두어를 변경할 수 있다.

    사용 가능한 메서드::

        self._read_json()         → dict | None  (요청 본문 파싱)
        self._respond(code, body) → None          (JSON 응답 전송)
        self._consume_body()      → None          (본문 소비 후 버림)
    """

    _LOG_PREFIX: str = "[API]"

    # ------------------------------------------------------------------
    # 로깅
    # ------------------------------------------------------------------

    def log_message(self, fmt, *args):  # noqa: A002
        logger.debug(self._LOG_PREFIX + " " + fmt, *args)

    # ------------------------------------------------------------------
    # 공통 헬퍼
    # ------------------------------------------------------------------

    def _read_json(self):
        """요청 본문을 JSON으로 파싱한다. 실패 시 None을 반환한다."""
        try:
            length = max(0, int(self.headers.get("Content-Length", 0)))
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            logger.warning("%s JSON 파싱 실패: %s", self._LOG_PREFIX, exc)
            return None

    def _respond(self, code: int, body) -> None:
        """JSON 응답을 전송한다. CORS 헤더를 포함한다."""
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _respond_options(self, methods: str, headers: str = "Content-Type") -> None:
        """CORS preflight 응답을 전송한다."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", methods)
        if headers:
            self.send_header("Access-Control-Allow-Headers", headers)
        self.end_headers()

    def _consume_body(self) -> None:
        """요청 본문을 읽어 버린다.

        Windows 에서 본문을 소비하지 않으면 연결 리셋(WinError 10053)이
        발생할 수 있으므로 404 응답 전에 호출한다.
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                self.rfile.read(length)
        except Exception:
            pass

    def _build_health_payload(
        self,
        *,
        service: str,
        status: str = "up",
        **extra,
    ) -> dict:
        """내부 HTTP 서버용 표준 health payload를 생성한다."""
        payload = {
            "service": service,
            "status": status,
            "checked_at": now_kst_iso(),
        }
        payload.update(extra)
        return payload

    def _check_internal_token(
        self,
        *,
        exempt_paths: tuple[str, ...] = ("/health",),
    ) -> bool:
        """내부 관리 HTTP 서버용 X-Internal-Token 헤더를 검증한다."""
        configured = os.environ.get("INTERNAL_SERVICE_TOKEN") or None
        if configured is None:
            return True

        path = self.path.split("?")[0].rstrip("/") or "/"
        if path in exempt_paths:
            return True

        provided = self.headers.get("X-Internal-Token", "")
        if not secrets.compare_digest(provided, configured):
            self._respond(401, {"error": "Unauthorized"})
            return False
        return True
