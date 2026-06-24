import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Set

_RUNNER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _RUNNER_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runners._shared import setup_runner_logging
from src.devices.signboard import SignboardConfig
from src.devices.siren import SensorConfig
from src.devices.speaker import SpeakerConfig
from src.services.action_bridge import (
    ActionBridge,
    default_alarm_topics,
    default_subscribe_topics,
)

logger = logging.getLogger("run-action-bridge")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _csv_set(value: str) -> Set[str]:
    return {topic.strip() for topic in value.split(",") if topic.strip()}


def _topic_csv(topics: Set[str]) -> str:
    return ",".join(sorted(topics))


@dataclass(frozen=True)
class ActionBridgeRuntimeConfig:
    mqtt_broker: str
    mqtt_port: int
    subscribe_topics: Set[str]
    db_path: str
    external_api_url: Optional[str]
    speaker_config: SpeakerConfig
    signboard_config: SignboardConfig
    siren_config: SensorConfig
    alarm_topics: Set[str]
    alarm_cooldown_seconds: int
    rest_enabled: bool
    rest_host: str
    rest_port: int


def _add_mqtt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mqtt-broker", default=_env("MQTT_BROKER", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=_env_int("MQTT_PORT", 1883))
    parser.add_argument(
        "--subscribe-topics",
        default=_env(
            "SUBSCRIBE_TOPICS",
            _topic_csv(default_subscribe_topics()),
        ),
    )
    parser.add_argument(
        "--alarm-topics",
        default=_env("ALARM_TOPICS", _topic_csv(default_alarm_topics())),
    )


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db-path",
        default=_env("DB_PATH", "/app/data/runtime/action_events.db"),
    )
    parser.add_argument("--external-api-url", default=_env("EXTERNAL_API_URL", ""))
    parser.add_argument(
        "--alarm-cooldown",
        type=int,
        default=_env_int("ALARM_COOLDOWN", 10),
    )


def _add_speaker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--speaker-host", default=_env("SPEAKER_HOST", ""))
    parser.add_argument(
        "--speaker-port",
        type=int,
        default=_env_int("SPEAKER_PORT", 80),
    )
    parser.add_argument("--speaker-user", default=_env("SPEAKER_USER", ""))
    parser.add_argument("--speaker-password", default=_env("SPEAKER_PASSWORD", ""))
    parser.add_argument(
        "--speaker-volume",
        type=int,
        default=_env_int("SPEAKER_VOLUME", 1),
    )
    parser.add_argument(
        "--speaker-tts-language",
        default=_env("SPEAKER_TTS_LANGUAGE", "kor"),
    )
    parser.add_argument(
        "--speaker-tts-gender",
        default=_env("SPEAKER_TTS_GENDER", "female"),
    )
    parser.add_argument(
        "--speaker-tts-pitch",
        type=int,
        default=_env_int("SPEAKER_TTS_PITCH", 100),
    )
    parser.add_argument(
        "--speaker-tts-speed",
        type=int,
        default=_env_int("SPEAKER_TTS_SPEED", 100),
    )
    parser.add_argument(
        "--speaker-tts-volume",
        type=int,
        default=_env_int("SPEAKER_TTS_VOLUME", 1),
    )


def _add_signboard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--signboard-host", default=_env("SIGNBOARD_HOST", ""))
    parser.add_argument(
        "--signboard-port",
        type=int,
        default=_env_int("SIGNBOARD_PORT", 5000),
    )
    parser.add_argument(
        "--signboard-brightness",
        type=int,
        default=_env_int("SIGNBOARD_BRIGHTNESS", 10),
    )
    parser.add_argument(
        "--signboard-text-color",
        type=int,
        default=_env_int("SIGNBOARD_TEXT_COLOR", 7),
    )
    parser.add_argument(
        "--signboard-back-color",
        type=int,
        default=_env_int("SIGNBOARD_BACK_COLOR", 1),
    )
    parser.add_argument(
        "--signboard-text-size",
        type=int,
        default=_env_int("SIGNBOARD_TEXT_SIZE", 2),
    )
    parser.add_argument(
        "--signboard-text-speed",
        type=int,
        default=_env_int("SIGNBOARD_TEXT_SPEED", 10),
    )


def _add_siren_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--siren-host", default=_env("SIREN_HOST", ""))
    parser.add_argument("--siren-port", type=int, default=_env_int("SIREN_PORT", 80))
    parser.add_argument("--siren-user", default=_env("SIREN_USER", ""))
    parser.add_argument("--siren-password", default=_env("SIREN_PASSWORD", ""))
    parser.add_argument(
        "--siren-auto-stop",
        type=float,
        default=_env_float("SIREN_AUTO_STOP", 10),
    )


def _add_rest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rest-enabled",
        action="store_true",
        default=_env_bool("REST_ENABLED"),
    )
    parser.add_argument("--rest-host", default=_env("REST_HOST", "0.0.0.0"))
    parser.add_argument("--rest-port", type=int, default=_env_int("REST_PORT", 8080))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Action-Bridge 액션 레이어 (알람 디바이스 + 외부 API + DB)"
    )

    _add_mqtt_arguments(parser)
    _add_storage_arguments(parser)
    _add_speaker_arguments(parser)
    _add_signboard_arguments(parser)
    _add_siren_arguments(parser)
    _add_rest_arguments(parser)

    return parser


def _build_speaker_config(args: argparse.Namespace) -> SpeakerConfig:
    return SpeakerConfig(
        host=args.speaker_host,
        port=args.speaker_port,
        username=args.speaker_user,
        password=args.speaker_password,
        volume=args.speaker_volume,
        tts_language=args.speaker_tts_language,
        tts_gender=args.speaker_tts_gender,
        tts_pitch=args.speaker_tts_pitch,
        tts_speed=args.speaker_tts_speed,
        tts_volume=args.speaker_tts_volume,
    )


def _build_signboard_config(args: argparse.Namespace) -> SignboardConfig:
    return SignboardConfig(
        host=args.signboard_host,
        port=args.signboard_port,
        brightness=args.signboard_brightness,
        text_color=args.signboard_text_color,
        back_color=args.signboard_back_color,
        text_size=args.signboard_text_size,
        text_speed=args.signboard_text_speed,
    )


def _build_siren_config(args: argparse.Namespace) -> SensorConfig:
    return SensorConfig(
        host=args.siren_host,
        port=args.siren_port,
        username=args.siren_user,
        password=args.siren_password,
        auto_stop_seconds=args.siren_auto_stop,
    )


def parse_runtime_config(
    argv: Optional[Sequence[str]] = None,
) -> ActionBridgeRuntimeConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mqtt_port <= 0:
        parser.error("--mqtt-port는 양수여야 합니다")

    return ActionBridgeRuntimeConfig(
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        subscribe_topics=_csv_set(args.subscribe_topics),
        db_path=args.db_path,
        external_api_url=args.external_api_url or None,
        speaker_config=_build_speaker_config(args),
        signboard_config=_build_signboard_config(args),
        siren_config=_build_siren_config(args),
        alarm_topics=_csv_set(args.alarm_topics),
        alarm_cooldown_seconds=args.alarm_cooldown,
        rest_enabled=args.rest_enabled,
        rest_host=args.rest_host,
        rest_port=args.rest_port,
    )


def create_action_bridge(config: ActionBridgeRuntimeConfig) -> ActionBridge:
    return ActionBridge(
        mqtt_broker=config.mqtt_broker,
        mqtt_port=config.mqtt_port,
        subscribe_topics=config.subscribe_topics,
        db_path=config.db_path,
        external_api_url=config.external_api_url,
        speaker_config=config.speaker_config,
        signboard_config=config.signboard_config,
        siren_config=config.siren_config,
        alarm_topics=config.alarm_topics,
        alarm_cooldown_seconds=config.alarm_cooldown_seconds,
        rest_enabled=config.rest_enabled,
        rest_host=config.rest_host,
        rest_port=config.rest_port,
    )


def main() -> None:
    setup_runner_logging()
    config = parse_runtime_config()

    service = create_action_bridge(config)
    logger.info(
        "Action Bridge 시작: mqtt=%s:%s rest=%s",
        config.mqtt_broker,
        config.mqtt_port,
        f"{config.rest_host}:{config.rest_port}" if config.rest_enabled else "disabled",
    )
    service.start()


if __name__ == "__main__":
    main()
