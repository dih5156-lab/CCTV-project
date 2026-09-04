"""EdgeX 장치 결과 MQTT를 구독해 SQLite에 저장하는 러너."""

from __future__ import annotations

import json
import logging
import os
import signal
from threading import Event

from src.edgex.command_result_collector import CommandResultStore
from src.protocols._mqtt_factory import create_mqtt_client

logger = logging.getLogger("run-command-result-collector")


def _env(key: str, default: str = "") -> str:
    """환경변수 값을 읽고 미설정이면 기본값을 반환한다."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """환경변수 정수값을 읽는다."""
    return int(_env(key, str(default)))


def main() -> None:
    """세 장치의 EdgeX 결과 토픽을 구독하고 저장한다."""
    logging.basicConfig(level=logging.INFO)
    broker = _env("MQTT_BROKER", "localhost")
    port = _env_int("MQTT_PORT", 1883)
    jetson_id = _env("EDGEX_JETSON_ID", "jetson-01")
    topic_prefix = _env("EDGEX_RESULT_TOPIC_PREFIX", "edgex/results/cctv")
    result_topic = f"{topic_prefix.rstrip('/')}/{jetson_id}/#"
    store = CommandResultStore(
        _env("EDGEX_COMMAND_RESULT_DB", "/app/data/runtime/edgex_command_results.db")
    )
    stop_event = Event()
    client = create_mqtt_client("cctv-edgex-command-result")

    def on_connect(mqtt_client, userdata, flags, rc, *args):
        """MQTT 연결 후 모든 장치 결과 토픽을 구독한다."""
        if rc == 0:
            mqtt_client.subscribe(result_topic, qos=1)
            logger.info("EdgeX 결과 토픽 구독 시작: %s", result_topic)

    def on_message(mqtt_client, userdata, message):
        """수신한 결과 JSON을 공통 SQLite 저장소에 기록한다."""
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("결과 payload는 JSON object여야 합니다")
            if not store.record(message.topic, payload):
                logger.warning("필수 필드가 없는 EdgeX 결과 무시: topic=%s", message.topic)
        except Exception as exc:
            logger.warning("EdgeX 결과 처리 실패: %s", exc)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=60)
    client.loop_start()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    logger.info("EdgeX Command 결과 수집기 시작: %s:%s", broker, port)
    try:
        stop_event.wait()
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
