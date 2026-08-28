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
    GET    /commands            → 출력 제어 명령 상태 조회
    POST   /mode                → 전역/사이트 모드 설정
    POST   /events              → 이벤트 수신
"""

import logging
import os
import secrets
from threading import Thread
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .._http_server import BaseApiHandler, ThreadingApiServer
from ..services._action_bridge_support import AlarmDevice
from ..time_utils import now_kst_iso

logger = logging.getLogger(__name__)

# 내부 서비스 간 공유 시크릿 — 미설정 시 검증 건너뜀 (개발/단일 컨테이너 환경)
_INTERNAL_TOKEN: str | None = os.environ.get("INTERNAL_SERVICE_TOKEN") or None


class _RestHandler(BaseApiHandler):
    """경량 HTTP 핸들러 - 복수 경로 지원.

    서버 객체(self.server)에 action_layer 참조가 있어야 한다.
    action_layer 는 ActionBridge 인스턴스여야 하며 아래 메서드를 제공해야 한다:
        list_sites(), get_pending_events(), default_mode,
        add_site_from_dict(), remove_site(), set_mode_str(),
        approve_event(), reject_event(), _handle_event()
    """

    _LOG_PREFIX = "[REST]"

    def _layer(self):
        return self.server.action_layer  # type: ignore[attr-defined]

    def _check_internal_token(self) -> bool:
        """X-Internal-Token 헤더를 검증한다.

        INTERNAL_SERVICE_TOKEN 환경변수가 설정되지 않은 경우(개발 환경)에는
        검증을 건너뛰고 True를 반환한다.
        /health, /ping, /metrics 는 토큰 없이 접근 허용한다.
        """
        if _INTERNAL_TOKEN is None:
            return True
        if self.path in ("/", "/health", "/ping", "/metrics"):
            return True
        provided = self.headers.get("X-Internal-Token", "")
        if not secrets.compare_digest(provided, _INTERNAL_TOKEN):
            self._respond(401, {"error": "Unauthorized"})
            return False
        return True

    def _root_payload(self) -> dict:
        """브라우저로 루트 경로를 열었을 때 사용할 서비스 안내."""
        return {
            "service": "cctv-action-layer",
            "description": "Internal action and control API",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "sites": "GET/POST /sites, DELETE /sites/{site_id}",
            "mode": "GET/POST /mode",
            "commands": "GET /commands",
            "pending": "GET /pending",
            "events": "POST /events",
            "approve": "POST /approve/{event_id}",
            "reject": "POST /reject/{event_id}",
        }

    def do_GET(self):  # noqa: N802
        if not self._check_internal_token():
            return
        layer = self._layer()
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/":
            self._respond(200, self._root_payload())
        elif path in ("/health", "/ping"):
            self._respond_health(layer)
        elif path == "/sites":
            self._respond(200, layer.list_sites())
        elif path == "/pending":
            self._respond(200, layer.get_pending_events())
        elif path == "/mode":
            self._respond(200, layer.get_default_mode_settings())
        elif path == "/devices":
            self._respond(200, layer.list_output_devices())
        elif path == "/events":
            try:
                limit = int((query.get("limit") or ["20"])[0])
            except ValueError:
                limit = 20
            self._respond(200, layer.list_recent_events(limit=limit))
        elif path == "/commands":
            try:
                limit = int((query.get("limit") or ["50"])[0])
            except ValueError:
                limit = 50
            self._respond(200, layer.list_commands(limit=limit))
        elif path == "/metrics":
            self._respond_metrics()
        else:
            self._respond(404, {"error": "Not Found"})

    def do_POST(self):  # noqa: N802
        if not self._check_internal_token():
            return
        layer = self._layer()
        path  = self.path

        if path == "/events":
            payload = self._read_json()
            if payload is None:
                self._respond(400, {"error": "Invalid JSON"})
                return
            topic = str(payload.pop("topic", "rest/inbound") or "rest/inbound")
            if hasattr(layer, "enqueue_rest_event"):
                accepted = layer.enqueue_rest_event(payload, topic=topic)
            else:
                layer._handle_event(payload, topic=topic)
                accepted = True
            if not accepted:
                self._respond(503, {"status": "error", "error": "action queue full"})
                return
            self._respond(200, {"status": "ok", "queued": True})

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
            self._respond_action_result(*layer.approve_event(event_id))

        elif path.startswith("/reject/"):
            event_id = path[len("/reject/"):]
            self._respond_action_result(*layer.reject_event(event_id))

        elif path == "/mode":
            data = self._read_json()
            if data is None or "mode" not in data:
                self._respond(400, {"error": "mode 필드 필요"})
                return
            try:
                mode_str = data["mode"]
                site_id  = data.get("site_id")
                layer.set_mode_str(mode_str, site_id=site_id)
                if not site_id:
                    alarm_devices = data.get("alarm_devices")
                    layer.set_default_action_settings(
                        alarm_devices=[
                            AlarmDevice(device)
                            for device in alarm_devices
                        ] if alarm_devices is not None else None,
                        confidence_threshold=data.get("confidence_threshold"),
                        display_message=data.get("display_message"),
                        tts_message=data.get("tts_message"),
                    )
                response = {"status": "ok", "mode": mode_str, "site_id": site_id}
                if not site_id:
                    response.update(layer.get_default_mode_settings())
                self._respond(200, response)
            except ValueError:
                self._respond(400, {"error": f"Invalid mode: {data['mode']!r}"})

        else:
            self._respond(404, {"error": "Not Found"})

    def _respond_health(self, layer) -> None:
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
                "service": "cctv-action-layer",
                "status": status,
                "checked_at": now_kst_iso(),
                "mqtt": "connected" if mqtt_ok else "disconnected",
                "mode": layer.default_mode.value,
                "sites": len(layer.list_sites()),
                "pending": len(layer.get_pending_events()),
            },
        )

    def _respond_action_result(self, ok: bool, message: str) -> None:
        self._respond(
            200 if ok else 404,
            {"status": "ok" if ok else "error", "message": message},
        )

    def _respond_metrics(self) -> None:
        """Prometheus テキスト형식 메트릭을 반환한다."""
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            from ..services.cctv_metrics import REGISTRY
            body: bytes = generate_latest(REGISTRY)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            logger.warning("메트릭 생성 실패: %s", exc)
            self._respond(500, {"error": "metrics unavailable"})

    def do_DELETE(self):  # noqa: N802
        if not self._check_internal_token():
            return
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
        self._server: Optional[ThreadingApiServer] = None
        self._thread: Optional[Thread]              = None

    def start(self) -> None:
        self._server = ThreadingApiServer((self.host, self.port), _RestHandler)
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
