"""외부 MQTT 입력 수신 MVP 실행 스크립트."""

from __future__ import annotations

import argparse
import logging

from src.config import AppConfig
from src.services import ExternalIngestService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="외부 MQTT 입력을 수신해 내부 이벤트로 정규화합니다.")
    parser.add_argument("--mqtt-broker", help="외부 MQTT 브로커 호스트")
    parser.add_argument("--mqtt-port", type=int, help="외부 MQTT 브로커 포트")
    parser.add_argument("--topic", action="append", dest="topics", help="구독할 토픽. 여러 번 지정 가능")
    parser.add_argument("--mqtt-client-id", help="외부 MQTT client id. 브로커가 고정 ID를 요구할 때 사용")
    parser.add_argument("--mqtt-username", help="외부 MQTT 사용자명")
    parser.add_argument("--mqtt-password", help="외부 MQTT 비밀번호")
    parser.add_argument("--db-path", help="원시 수신 이벤트 저장 SQLite 경로")
    parser.add_argument("--republish", action="store_true", help="정규화 이벤트를 내부 MQTT 토픽으로 재발행")
    parser.add_argument("--republish-broker", help="내부 MQTT 브로커 호스트")
    parser.add_argument("--republish-port", type=int, help="내부 MQTT 브로커 포트")
    parser.add_argument("--republish-topic-prefix", help="내부 MQTT 토픽 prefix")
    parser.add_argument("--log-level", default="INFO", help="로그 레벨")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = AppConfig.from_env()
    ingest = config.external_ingest

    if args.mqtt_broker:
        ingest.mqtt_broker = args.mqtt_broker
    if args.mqtt_port:
        ingest.mqtt_port = args.mqtt_port
    if args.topics:
        ingest.topics = tuple(args.topics)
    if args.mqtt_client_id:
        ingest.mqtt_client_id = args.mqtt_client_id
    if args.mqtt_username:
        ingest.mqtt_username = args.mqtt_username
    if args.mqtt_password:
        ingest.mqtt_password = args.mqtt_password
    if args.db_path:
        ingest.db_path = args.db_path
    if args.republish:
        ingest.republish_enabled = True
    if args.republish_broker:
        config.mqtt.broker = args.republish_broker
    if args.republish_port:
        config.mqtt.port = args.republish_port
    if args.republish_topic_prefix:
        config.mqtt.topic_prefix = args.republish_topic_prefix

    service = ExternalIngestService.from_app_config(config)
    service.run_forever()


if __name__ == "__main__":
    main()
