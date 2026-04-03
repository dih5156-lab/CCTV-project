"""run_alert_api.py - 내부 Alert API 수신 서버"""

import argparse
import json
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
)
logger = logging.getLogger("alert-api")


class AlertHandler(BaseHTTPRequestHandler):
    log_path: Path = Path("alert_api_events.jsonl")
    sensor_log_path: Path = Path("sensor_readings.jsonl")

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ["/health", "/ping"]:
            self._send_json(200, {"status": "up"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/sensor-readings":
            self._handle_post(
                self.sensor_log_path,
                "Sensor reading 수신 완료: /api/sensor-readings",
            )
            return
        if self.path != "/api/alerts":
            self._send_json(404, {"error": "not found"})
            return
        self._handle_post(self.log_path, "Alert 수신 완료: /api/alerts")

    def _handle_post(self, log_path: Path, log_msg: str) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid json"})
            return

        entry = {
            "receivedAt": datetime.utcnow().isoformat(),
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
    parser = argparse.ArgumentParser(description="CCTV 내부 Alert API 서버")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-path", default="/app/alert_api_events.jsonl")
    parser.add_argument("--sensor-log-path", default="/app/sensor_readings.jsonl")
    args = parser.parse_args()

    AlertHandler.log_path = Path(args.log_path)
    AlertHandler.sensor_log_path = Path(args.sensor_log_path)

    server = ThreadingHTTPServer((args.host, args.port), AlertHandler)
    logger.info(f"Alert API 서버 시작: http://{args.host}:{args.port}")
    logger.info(f"이벤트 로그 경로: {AlertHandler.log_path}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.info("Alert API 서버 종료")


if __name__ == "__main__":
    main()
