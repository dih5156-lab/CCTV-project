"""REST 이벤트 수신 서버

ActionBridge 용 내부 HTTP 수신 엔드포인트.
_RestHandler 는 server.action_layer 에 연결된 ActionBridge 인스턴스로
요청을 위임한다.

Routes:
    GET    /sites               → 전체 사이트 목록
    POST   /sites               → 사이트 추가
    DELETE /sites/{site_id}     → 사이트 삭제
    GET    /pending             → 수동 승인 대기 이벤트 목록
    POST   /approve/{event_id}  → 이벤트 승인
    POST   /reject/{event_id}   → 이벤트 거부
    GET    /mode                → 현재 모드 조회
    POST   /mode                → 전역/사이트 모드 설정
    POST   /events              → 이벤트 수신
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class _RestHandler(BaseHTTPRequestHandler):
    """경량 HTTP 핸들러 - 복수 경로 지원.

    서버 객체(self.server)에 action_layer 참조가 있어야 한다.
    action_layer 는 ActionBridge 인스턴스여야 하며 아래 메서드를 제공해야 한다:
        list_sites(), get_pending_events(), default_mode,
        add_site_from_dict(), remove_site(), set_mode_str(),
        approve_event(), reject_event(), _handle_event()
    """

    def log_message(self, fmt, *args):  # noqa: A002
        logger.debug("[REST] " + fmt, *args)

    def _layer(self):
        return self.server.action_layer  # type: ignore[attr-defined]

    def _read_json(self) -> Optional[Dict]:
        try:
            length = max(0, int(self.headers.get("Content-Length", 0)))
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            logger.warning("[REST] JSON 파싱 실패: %s", exc)
            return None

    def _respond(self, code: int, body: Any) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        layer = self._layer()
        if self.path in ("/health", "/ping"):
            mqtt_ok = False
            try:
                mc = getattr(layer, "_mqtt_client", None)
                mqtt_ok = bool(mc and mc.is_connected())
            except Exception:
                pass
            status = "up" if (getattr(layer, "_running", False) and mqtt_ok) else "degraded"
            self._respond(
                200 if status == "up" else 503,
                {
                    "status": status,
                    "mqtt": "connected" if mqtt_ok else "disconnected",
                    "mode": layer.default_mode.value,
                    "sites": len(layer.list_sites()),
                    "pending": len(layer.get_pending_events()),
                },
            )
        elif self.path == "/sites":
            self._respond(200, layer.list_sites())
        elif self.path == "/pending":
            self._respond(200, layer.get_pending_events())
        elif self.path == "/mode":
            self._respond(200, {"mode": layer.default_mode.value})
        else:
            self._respond(404, {"error": "Not Found"})

    def do_POST(self):  # noqa: N802
        layer = self._layer()
        path  = self.path

        if path == "/events":
            payload = self._read_json()
            if payload is None:
                self._respond(400, {"error": "Invalid JSON"})
                return
            layer._handle_event(payload)
            self._respond(200, {"status": "ok"})

        elif path == "/sites":
            data = self._read_json()
            if data is None:
                self._respond(400, {"error": "Invalid JSON"})
                return
            try:
                site_id = layer.add_site_from_dict(data)
                self._respond(200, {"status": "ok", "site_id": site_id})
            except Exception as exc:
                self._respond(400, {"error": str(exc)})

        elif path.startswith("/approve/"):
            event_id = path[len("/approve/"):]
            ok, msg = layer.approve_event(event_id)
            self._respond(
                200 if ok else 404,
                {"status": "ok" if ok else "error", "message": msg},
            )

        elif path.startswith("/reject/"):
            event_id = path[len("/reject/"):]
            ok, msg = layer.reject_event(event_id)
            self._respond(
                200 if ok else 404,
                {"status": "ok" if ok else "error", "message": msg},
            )

        elif path == "/mode":
            data = self._read_json()
            if data is None or "mode" not in data:
                self._respond(400, {"error": "mode 필드 필요"})
                return
            try:
                mode_str = data["mode"]
                site_id  = data.get("site_id")
                layer.set_mode_str(mode_str, site_id=site_id)
                self._respond(200, {"status": "ok", "mode": mode_str, "site_id": site_id})
            except ValueError:
                self._respond(400, {"error": f"Invalid mode: {data['mode']!r}"})

        else:
            self._respond(404, {"error": "Not Found"})

    def do_DELETE(self):  # noqa: N802
        layer = self._layer()
        if self.path.startswith("/sites/"):
            site_id = self.path[len("/sites/"):]
            removed = layer.remove_site(site_id)
            self._respond(
                200 if removed else 404,
                {
                    "status":  "ok" if removed else "error",
                    "message": f"사이트 {'삭제 완료' if removed else '없음'}: {site_id}",
                },
            )
        else:
            self._respond(404, {"error": "Not Found"})


class RestEventReceiver:
    """GET/POST/DELETE 다중 경로 HTTP 서버.

    매개변수:
        host:         바인딩할 주소 (기본 0.0.0.0).
        port:         수신 포트 (기본 8080).
        action_layer: 요청을 위임할 ActionBridge 인스턴스.
    """

    def __init__(
        self,
        host:         str = "0.0.0.0",
        port:         int = 8080,
        action_layer=None,
    ) -> None:
        self.host          = host
        self.port          = port
        self._action_layer = action_layer
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[Thread]     = None

    def start(self) -> None:
        self._server = HTTPServer((self.host, self.port), _RestHandler)
        self._server.action_layer = self._action_layer  # type: ignore[attr-defined]
        self._thread = Thread(
            target=self._server.serve_forever, daemon=True, name="RestReceiver"
        )
        self._thread.start()
        logger.info("REST API 서버 시작: http://%s:%d/", self.host, self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()


__all__ = ["RestEventReceiver"]
