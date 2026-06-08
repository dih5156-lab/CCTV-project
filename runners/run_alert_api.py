"""run_alert_api.py - 내부 Alert API 수신 서버"""

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_RUNNER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _RUNNER_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runners._shared import setup_runner_logging
from src.time_utils import now_kst_iso

logger = logging.getLogger("alert-api")


class AlertHandler(BaseHTTPRequestHandler):
    """``server.log_path`` / ``server.sensor_log_path`` 속성을 읽어 동작한다.

    클래스 속성 대신 서버 인스턴스 속성을 사용하므로
    멀티스레드 환경에서도 안전하게 여러 서버 인스턴스를 띄울 수 있다.
    """

    def _log_path(self) -> Path:
        return self.server.log_path  # type: ignore[attr-defined]

    def _sensor_log_path(self) -> Path:
        return self.server.sensor_log_path  # type: ignore[attr-defined]

    def _health_payload(self) -> dict:
        """헬스체크 응답 본문을 생성한다."""
        return {
            "service": "cctv-alert-api",
            "status": "up",
            "checked_at": now_kst_iso(),
        }

    def _root_payload(self) -> dict:
        """브라우저로 루트 경로를 열었을 때 사용할 서비스 안내."""
        return {
            "service": "cctv-alert-api",
            "description": "Internal alert ingestion API",
            "health": "GET /health",
            "alerts": "POST /api/alerts",
            "sensor_readings": "POST /api/sensor-readings",
        }

    def _method_not_allowed_payload(self, path: str, allowed: str) -> dict:
        """정의된 경로를 잘못된 HTTP method로 호출했을 때의 안내."""
        return {
            "error": "method not allowed",
            "path": path,
            "allowed": allowed,
            "hint": f"{path}는 {allowed} 요청으로 호출해야 합니다.",
        }

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            logger.debug("클라이언트가 응답 전에 연결을 종료함 (BrokenPipe)")

    def do_GET(self):
        if self.path == "/":
            self._send_json(200, self._root_payload())
            return
        if self.path in ["/health", "/ping"]:
            self._send_json(200, self._health_payload())
            return
        if self.path in ["/api/alerts", "/api/sensor-readings"]:
            self._send_json(405, self._method_not_allowed_payload(self.path, "POST"))
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/sensor-readings":
            self._handle_post(
                self._sensor_log_path(),
                "Sensor reading 수신 완료: /api/sensor-readings",
            )
            return
        if self.path != "/api/alerts":
            self._send_json(404, {"error": "not found"})
            return
        self._handle_post(self._log_path(), "Alert 수신 완료: /api/alerts")

    def _handle_post(self, log_path: Path, log_msg: str) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid json"})
            return

        entry = {
            "receivedAt": now_kst_iso(),
            "payload": payload,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(log_msg)
        self._send_json(202, {"accepted": True})

    def log_message(self, format, *args):
        return


def main() -> None:
    setup_runner_logging()

    parser = argparse.ArgumentParser(description="CCTV 내부 Alert API 서버")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-path", default="/app/data/logs/alert_api_events.jsonl")
    parser.add_argument("--sensor-log-path", default="/app/data/logs/sensor_readings.jsonl")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AlertHandler)
    server.log_path = Path(args.log_path)              # type: ignore[attr-defined]
    server.sensor_log_path = Path(args.sensor_log_path)  # type: ignore[attr-defined]
    logger.info("Alert API 서버 시작: http://%s:%s", args.host, args.port)
    logger.info("이벤트 로그 경로: %s", server.log_path)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.info("Alert API 서버 종료")


if __name__ == "__main__":
    main()
