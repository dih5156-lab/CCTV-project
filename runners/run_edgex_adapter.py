"""run_edgex_adapter.py - 경량 EdgeX 디바이스 어댑터 실행 진입점"""

import argparse
import logging
import sys
from pathlib import Path

# runners/ 오프라인 실행 시 프로젝트 루트를 sys.path에 등록
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edgex.adapter_service import EdgeXDeviceAdapterService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)


def main() -> None:
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
    service.start()


if __name__ == "__main__":
    main()
