"""EdgeX 경광등 Command를 InterM 장치로 전달하는 MQTT 러너."""

from __future__ import annotations

import json
import logging
import os
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

from src.devices.siren import SensorConfig, SirenDevice
from src.edgex.command_contract import build_command_topic
from src.edgex.command_http import handle_command_request
from src.edgex.device_registry import DeviceRegistry
from src.edgex.siren_device_service import SirenDeviceService
from src.edgex.validation_responder import start_validation_responder
from src.protocols._mqtt_factory import create_mqtt_client

logger = logging.getLogger("run-siren-device-service")


class _CommandHandler(BaseHTTPRequestHandler):
    """EdgeX Core Command HTTP 요청을 경광등 서비스로 전달한다."""

    service: SirenDeviceService
    device_id: str | tuple[str, ...]

    def do_GET(self):
        """경광등 Device Service의 상태 확인 요청에 응답한다."""
        if self.path == "/health":
            self._write_json(200, {"service": "cctv-device-siren", "status": "up"})
            return
        self._write_json(404, {"error_code": "not_found"})

    def do_PUT(self):  # noqa: N802
        """EdgeX v3 경광등 Command PUT 요청을 실행한다."""
        self._handle_command()

    def _handle_command(self) -> None:
        """요청 본문과 헤더를 공통 Command 실행 함수에 전달한다."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("요청 본문은 JSON object여야 합니다")
            status, result = handle_command_request(
                self.service,
                self.device_id,
                self.path,
                payload,
                self.headers.get("X-Command-Id", "edgex-command"),
                device_type="siren",
            )
            self._write_json(status, result)
        except (ValueError, json.JSONDecodeError):
            self._write_json(400, {"error_code": "invalid_json"})

    def _write_json(self, status: int, payload: dict) -> None:
        """HTTP JSON 응답을 UTF-8로 반환한다."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        """기본 HTTP 접근 로그를 중복 기록하지 않는다."""
        return


def _env(key: str, default: str = "") -> str:
    """환경변수 값을 읽고 미설정이면 기본값을 반환한다."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """환경변수 정수값을 읽는다."""
    return int(_env(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    """환경변수의 참·거짓 값을 안전하게 변환한다."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def create_service() -> SirenDeviceService:
    """환경변수 또는 장치 레지스트리로 경광등 서비스를 생성한다."""
    default_id = _env("SIREN_DEVICE_ID", "cctv-siren-01")
    base_config = _siren_config()
    registry_path = _env("EDGEX_DEVICE_REGISTRY_PATH")
    targets = DeviceRegistry.from_file(registry_path).targets("siren") if registry_path else []
    if not targets:
        return SirenDeviceService(device_id=default_id, config=base_config, dry_run=_env_bool("SIREN_DRY_RUN"))
    devices = {
        target.device_id: SirenDevice(
            SensorConfig(**{**base_config.__dict__, **_connection_values(target.connection)})
        )
        for target in targets
    }
    return SirenDeviceService(
        device_id=targets[0].device_id,
        config=base_config,
        devices=devices,
        dry_run=_env_bool("SIREN_DRY_RUN"),
    )


def _siren_config() -> SensorConfig:
    """환경변수 기반 경광등 공통 설정을 만든다."""
    return SensorConfig(
        host=_env("SIREN_HOST"), port=_env_int("SIREN_PORT", 80),
        username=_env("SIREN_USER"), password=_env("SIREN_PASSWORD"),
        auto_stop_seconds=float(_env("SIREN_AUTO_STOP", "10")),
    )


def _connection_values(connection: dict) -> dict:
    """레지스트리의 비밀정보 없는 연결값을 장치 설정에 반영한다."""
    return {key: connection[key] for key in ("host", "port") if key in connection}


def main() -> None:
    """경광등 명령 토픽을 구독하고 처리 결과를 발행한다."""
    logging.basicConfig(level=logging.INFO)
    service = create_service()
    broker = _env("MQTT_BROKER", "localhost")
    port = _env_int("MQTT_PORT", 1883)
    jetson_id = _env("EDGEX_JETSON_ID", "jetson-01")
    command_topic = build_command_topic(
        _env("EDGEX_COMMAND_TOPIC_PREFIX", "edgex/commands/cctv"), jetson_id, "siren"
    )
    result_topic = build_command_topic(
        _env("EDGEX_RESULT_TOPIC_PREFIX", "edgex/results/cctv"), jetson_id, "siren"
    )
    stop_event = Event()
    client = create_mqtt_client("cctv-edgex-siren-device")
    validation_thread = start_validation_responder("cctv-device-siren", stop_event)
    _CommandHandler.service = service
    _CommandHandler.device_id = service.device_ids
    http_server = ThreadingHTTPServer(
        (_env("SIREN_SERVICE_HOST", "0.0.0.0"), _env_int("SIREN_SERVICE_PORT", 59992)),
        _CommandHandler,
    )

    def on_connect(mqtt_client, userdata, flags, rc, *args):
        """MQTT 연결 후 경광등 명령 토픽을 구독한다."""
        if rc == 0:
            mqtt_client.subscribe(command_topic + "/#", qos=1)
            logger.info("경광등 EdgeX Command 구독 시작: %s", command_topic)

    def on_message(mqtt_client, userdata, message):
        """수신한 Command를 실행하고 표준 결과를 발행한다."""
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Command payload는 JSON object여야 합니다")
            result = service.execute_request(payload)
            mqtt_client.publish(
                result_topic,
                json.dumps(result.to_dict(), ensure_ascii=False),
                qos=1,
            )
        except Exception as exc:
            logger.warning("경광등 Command 처리 실패: %s", exc)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=60)
    client.loop_start()
    http_server_thread = Thread(
        target=http_server.serve_forever,
        daemon=True,
        name="siren-command-http",
    )
    http_server_thread.start()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    logger.info("경광등 EdgeX Device Service 시작: %s:%s", broker, port)
    try:
        stop_event.wait()
    finally:
        service.close()
        http_server.shutdown()
        http_server.server_close()
        client.loop_stop()
        client.disconnect()
        validation_thread.join(timeout=2)


if __name__ == "__main__":
    main()
