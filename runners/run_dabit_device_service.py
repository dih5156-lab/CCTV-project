"""Dabit 전광판 Device Service 경계 프로세스."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping
from urllib.parse import urlparse

from src.devices.signboard import SignboardConfig
from src.edgex.dabit_device_service import DabitDeviceService

try:
    import redis
except ImportError:  # pragma: no cover - image에는 포함됨
    redis = None


def resolve_device_id(environment: Mapping[str, str]) -> str:
    return environment.get("SIGNBOARD_DEVICE_ID", "cctv-signboard-01")


class _Handler(BaseHTTPRequestHandler):
    service: DabitDeviceService

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/health"):
            self._json(200, {"service": "cctv-device-dabit", "status": "up"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/command":
            self._json(404, {"error": "not found"})
            return
        self._handle_command("/command")

    def do_PUT(self):  # noqa: N802
        prefix = "/api/v3/device/name/"
        if not urlparse(self.path).path.startswith(prefix):
            self._json(404, {"error": "not found"})
            return
        self._handle_command(urlparse(self.path).path)

    def _handle_command(self, path: str):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size) or b"{}")
            if path.startswith("/api/v3/device/name/"):
                parts = path.removeprefix("/api/v3/device/name/").split("/")
                if len(parts) < 2:
                    self._json(400, {"error": "device and command are required"})
                    return
                command_id = self.headers.get("X-Command-Id", "edgex-command")
                command = parts[1]
                parameters = payload if isinstance(payload, dict) else {}
            else:
                command_id = str(payload.get("command_id") or "missing")
                command = str(payload.get("command") or "")
                parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
            result = self.service.execute(
                command_id, command, parameters,
            )
            self._json(200 if result.status == "acknowledged" else 502, result.__dict__)
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "invalid json"})

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def main() -> None:
    config = SignboardConfig(
        host=os.getenv("SIGNBOARD_HOST", ""),
        port=int(os.getenv("SIGNBOARD_PORT", "5000")),
        brightness=int(os.getenv("SIGNBOARD_BRIGHTNESS", "10")),
        text_color=int(os.getenv("SIGNBOARD_TEXT_COLOR", "7")),
        back_color=int(os.getenv("SIGNBOARD_BACK_COLOR", "0")),
    )
    service = DabitDeviceService(
        device_id=resolve_device_id(os.environ), config=config
    )
    _Handler.service = service
    validation_stop = threading.Event()
    validation_thread = _start_validation_responder(validation_stop)
    server = ThreadingHTTPServer(
        (os.getenv("DABIT_SERVICE_HOST", "0.0.0.0"), int(os.getenv("DABIT_SERVICE_PORT", "59990"))),
        _Handler,
    )
    try:
        server.serve_forever()
    finally:
        validation_stop.set()
        if validation_thread:
            validation_thread.join(timeout=2)
        service.close()
        server.server_close()


def _start_validation_responder(stop: threading.Event):
    """EdgeX Core Metadata의 device validation 요청에 응답한다."""
    if redis is None:
        return None

    def run() -> None:
        try:
            client = redis.Redis(
                host=os.getenv("REDIS_HOST", "edgex-redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            client.ping()
            pubsub = client.pubsub(ignore_subscribe_messages=True)
            channels = ("edgex.cctv-device-dabit.validate.device", "edgex/cctv-device-dabit/validate/device")
            pubsub.subscribe(*channels)
            while not stop.is_set():
                message = pubsub.get_message(timeout=1.0)
                if not message or message.get("type") != "message":
                    continue
                try:
                    envelope = json.loads(message.get("data") or "{}")
                except json.JSONDecodeError:
                    continue
                request_id = envelope.get("requestID") or envelope.get("requestId")
                if not request_id:
                    continue
                response = {
                    "apiVersion": "", "receivedTopic": message.get("channel", ""),
                    "correlationID": envelope.get("correlationID", ""),
                    "requestID": request_id, "errorCode": 0, "payload": "",
                    "contentType": "application/json",
                }
                body = json.dumps(response, ensure_ascii=False)
                client.publish(f"edgex.response.cctv-device-dabit.{request_id}", body)
                client.publish(f"edgex/response/cctv-device-dabit/{request_id}", body)
        except Exception:
            return

    thread = threading.Thread(target=run, daemon=True, name="DabitValidationResponder")
    thread.start()
    return thread


if __name__ == "__main__":
    main()
