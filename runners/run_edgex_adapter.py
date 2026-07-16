"""run_edgex_adapter.py - 경량 EdgeX 디바이스 어댑터 실행 진입점"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Mapping

from prometheus_client import start_http_server

_RUNNER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _RUNNER_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runners._shared import setup_runner_logging
from src.aiot.command_store import CommandStore
from src.aiot.media_uploader import MediaUploader
from src.aiot.metrics import AiotMetrics
from src.aiot.query_service import AiQueryService, RecentAppearanceLiveProvider
from src.edgex.adapter_service import EdgeXDeviceAdapterService
from src.services.aiot_command_service import AiotCommandService
from src.services.appearance_log import AppearanceLog

logger = logging.getLogger("run-edgex-adapter")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_aiot_commands(adapter: EdgeXDeviceAdapterService) -> None:
    if not _env_bool("AIOT_COMMANDS_ENABLED"):
        return
    appearance_log = AppearanceLog(os.environ.get("APPEARANCES_DB"))
    live_provider = RecentAppearanceLiveProvider(
        appearance_log,
        window_seconds=float(os.environ.get("AIOT_LIVE_WINDOW_SECONDS", "30")),
    )
    query_service = AiQueryService(appearance_log, live_provider)
    crop_dir = Path(
        os.environ.get("APPEARANCE_CROP_DIR", "data/runtime/appearance_crops")
    )
    allowed_hosts = {
        value.strip()
        for value in os.environ.get("AIOT_ALLOWED_UPLOAD_HOSTS", "").split(",")
        if value.strip()
    }
    media_uploader = MediaUploader(
        allowed_hosts=allowed_hosts,
        media_roots=[crop_dir],
    )

    def publish_result(payload: Mapping) -> bool:
        return adapter.edgex_service.publish_device_event(
            adapter.aiot_jetson_id,
            "cctv",
            "aiot_command_result",
            dict(payload),
        )

    adapter.aiot_jetson_id = os.environ.get("AIOT_JETSON_ID", "jetson-01")
    topic_prefix = os.environ.get(
        "AIOT_COMMAND_TOPIC_PREFIX", "edgex/commands/cctv"
    ).rstrip("/")
    adapter.aiot_command_topic = f"{topic_prefix}/{adapter.aiot_jetson_id}/#"
    adapter.aiot_command_service = AiotCommandService(
        command_store=CommandStore(
            os.environ.get("AIOT_COMMAND_DB", "data/runtime/aiot_commands.db")
        ),
        query_service=query_service,
        media_uploader=media_uploader,
        resolve_match=query_service.resolve_media,
        publish_result=publish_result,
        result_outbox=adapter.edgex_service,
        max_results=int(os.environ.get("AIOT_QUERY_MAX_RESULTS", "20")),
        metrics=AiotMetrics(),
    )
    adapter.aiot_commands_enabled = True


def main() -> None:
    setup_runner_logging()

    parser = argparse.ArgumentParser(
        description="경량 EdgeX 디바이스 어댑터 (MQTT 구독 -> EdgeX 발행)"
    )

    parser.add_argument("--ai-mqtt-broker", default="localhost", help="AI 엔진 MQTT 브로커 호스트")
    parser.add_argument("--ai-mqtt-port", type=int, default=1883, help="AI 엔진 MQTT 브로커 포트")
    parser.add_argument("--ai-topic-prefix", default="cctv/ai/events", help="AI 엔진 토픽 접두사")

    parser.add_argument("--edgex-metadata-url", default="http://localhost:59881", help="EdgeX Core Metadata URL")
    parser.add_argument("--edgex-data-url", default="http://localhost:59880", help="EdgeX Core Data URL")
    parser.add_argument("--edgex-mqtt-broker", default="localhost", help="EdgeX 메시지버스 MQTT 브로커 호스트")
    parser.add_argument("--edgex-mqtt-port", type=int, default=1883, help="EdgeX 메시지버스 MQTT 브로커 포트")
    parser.add_argument("--edgex-topic-prefix", default="edgex/events/device", help="EdgeX 토픽 접두사")
    parser.add_argument("--service-name", default="cctv-device-service", help="EdgeX 디바이스 서비스 이름")

    args = parser.parse_args()

    if args.ai_mqtt_port <= 0 or args.edgex_mqtt_port <= 0:
        parser.error("MQTT 포트는 양수여야 합니다")

    service = EdgeXDeviceAdapterService(
        ai_mqtt_broker=args.ai_mqtt_broker,
        ai_mqtt_port=args.ai_mqtt_port,
        ai_topic_prefix=args.ai_topic_prefix,
        metadata_url=args.edgex_metadata_url,
        data_url=args.edgex_data_url,
        edgex_mqtt_broker=args.edgex_mqtt_broker,
        edgex_mqtt_port=args.edgex_mqtt_port,
        edgex_topic_prefix=args.edgex_topic_prefix,
        service_name=args.service_name,
    )
    configure_aiot_commands(service)
    if service.aiot_commands_enabled:
        metrics_port = int(os.environ.get("AIOT_METRICS_PORT", "9105"))
        start_http_server(metrics_port)
        logger.info("AIoT Prometheus 메트릭 서버 시작: port=%s", metrics_port)
    logger.info(
        "EdgeX Adapter 시작: ai-mqtt=%s:%s edgex-data=%s",
        args.ai_mqtt_broker,
        args.ai_mqtt_port,
        args.edgex_data_url,
    )
    service.start()


if __name__ == "__main__":
    main()
