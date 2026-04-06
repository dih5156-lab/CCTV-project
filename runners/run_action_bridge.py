"""run_action_bridge.py - 액션 레이어(action-bridge) 실행 진입점

환경 변수 우선 / CLI 인수 병행 지원.
Docker 배포 시 docker-compose.yml 의 environment 섹션으로 모든 값을 주입한다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# runners/ 오프라인 실행 시 프로젝트 루트를 sys.path에 등록
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.action_bridge import ActionBridge
from src.devices.speaker   import SpeakerConfig
from src.devices.signboard import SignboardConfig
from src.devices.sensor    import SensorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
)


def _env(key: str, default: str = "") -> str:
    """환경 변수를 읽는다 (없으면 default)."""
    return os.environ.get(key, default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action-Bridge 액션 레이어 (알람 디바이스 + 외부 API + DB)"
    )

    # ── MQTT ─────────────────────────────────────────────────────────────
    parser.add_argument("--mqtt-broker",  default=_env("MQTT_BROKER", "localhost"))
    parser.add_argument("--mqtt-port",    type=int, default=int(_env("MQTT_PORT", "1883")))
    parser.add_argument(
        "--subscribe-topics",
        default=_env(
            "SUBSCRIBE_TOPICS",
            "cctv/rules/intrusion/filtered,"
            "cctv/rules/intrusion/persisted,"
            "cctv/rules/intrusion/critical,"
            "cctv/ai/events/+/head,"
            "cctv/ai/events/+/fall_detected,"
            "cctv/ai/events/+/zone_entered,"
            "cctv/ai/events/+/zone_dwelling,"
            "aiot/rules/sensor/tilt,"
            "aiot/rules/sensor/temperature,"
            "aiot/rules/sensor/vibration",
        ),
    )
    parser.add_argument(
        "--alarm-topics",
        default=_env(
            "ALARM_TOPICS",
            "cctv/rules/intrusion/persisted,"
            "cctv/rules/intrusion/critical,"
            "aiot/rules/sensor/tilt,"
            "aiot/rules/sensor/temperature,"
            "aiot/rules/sensor/vibration",
        ),
    )

    # ── DB / 외부 API ─────────────────────────────────────────────────────
    parser.add_argument("--db-path",          default=_env("DB_PATH", "/app/action_events.db"))
    parser.add_argument("--external-api-url", default=_env("EXTERNAL_API_URL", ""))
    parser.add_argument("--alarm-cooldown",   type=int, default=int(_env("ALARM_COOLDOWN", "10")))

    # ── 스피커 (InterM HTTP REST + Digest 인증) ───────────────────────────
    parser.add_argument("--speaker-host",         default=_env("SPEAKER_HOST", ""))
    parser.add_argument("--speaker-port",         type=int, default=int(_env("SPEAKER_PORT", "80")))
    parser.add_argument("--speaker-user",         default=_env("SPEAKER_USER", ""))
    parser.add_argument("--speaker-password",     default=_env("SPEAKER_PASSWORD", ""))
    parser.add_argument("--speaker-volume",       type=int, default=int(_env("SPEAKER_VOLUME", "1")))
    parser.add_argument("--speaker-tts-language", default=_env("SPEAKER_TTS_LANGUAGE", "kor"))
    parser.add_argument("--speaker-tts-gender",   default=_env("SPEAKER_TTS_GENDER", "female"))
    parser.add_argument("--speaker-tts-pitch",    type=int, default=int(_env("SPEAKER_TTS_PITCH", "100")))
    parser.add_argument("--speaker-tts-speed",    type=int, default=int(_env("SPEAKER_TTS_SPEED", "100")))
    parser.add_argument("--speaker-tts-volume",   type=int, default=int(_env("SPEAKER_TTS_VOLUME", "1")))

    # ── 전광판 (Dabit TCP 소켓) ───────────────────────────────────────────
    parser.add_argument("--signboard-host",       default=_env("SIGNBOARD_HOST", ""))
    parser.add_argument("--signboard-port",       type=int, default=int(_env("SIGNBOARD_PORT", "5000")))
    parser.add_argument("--signboard-brightness", type=int, default=int(_env("SIGNBOARD_BRIGHTNESS", "10")))
    parser.add_argument("--signboard-text-color", type=int, default=int(_env("SIGNBOARD_TEXT_COLOR", "7")))
    parser.add_argument("--signboard-back-color", type=int, default=int(_env("SIGNBOARD_BACK_COLOR", "1")))
    parser.add_argument("--signboard-text-size",  type=int, default=int(_env("SIGNBOARD_TEXT_SIZE", "2")))
    parser.add_argument("--signboard-text-speed", type=int, default=int(_env("SIGNBOARD_TEXT_SPEED", "10")))

    # ── 경광등 (InterM HTTP REST + Digest 인증) ───────────────────────────
    parser.add_argument("--siren-host",         default=_env("SIREN_HOST", ""))
    parser.add_argument("--siren-port",         type=int, default=int(_env("SIREN_PORT", "80")))
    parser.add_argument("--siren-user",         default=_env("SIREN_USER", ""))
    parser.add_argument("--siren-password",     default=_env("SIREN_PASSWORD", ""))
    parser.add_argument("--siren-auto-stop",    type=float, default=float(_env("SIREN_AUTO_STOP", "10")))

    # ── REST 수신 서버 ────────────────────────────────────────────────────
    parser.add_argument("--rest-enabled", action="store_true",
                        default=_env("REST_ENABLED", "").lower() in ("1", "true", "yes"))
    parser.add_argument("--rest-host",    default=_env("REST_HOST", "0.0.0.0"))
    parser.add_argument("--rest-port",    type=int, default=int(_env("REST_PORT", "8080")))

    args = parser.parse_args()

    # ── 유효성 검사 ──────────────────────────────────────────────────────
    if args.mqtt_port <= 0:
        parser.error("--mqtt-port는 양수여야 합니다")

    subscribe_topics = {t.strip() for t in args.subscribe_topics.split(",") if t.strip()}
    alarm_topics     = {t.strip() for t in args.alarm_topics.split(",")     if t.strip()}

    # ── 디바이스 Config 조립 ──────────────────────────────────────────────
    speaker_cfg = SpeakerConfig(
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

    signboard_cfg = SignboardConfig(
        host=args.signboard_host,
        port=args.signboard_port,
        brightness=args.signboard_brightness,
        text_color=args.signboard_text_color,
        back_color=args.signboard_back_color,
        text_size=args.signboard_text_size,
        text_speed=args.signboard_text_speed,
    )

    siren_cfg = SensorConfig(
        host=args.siren_host,
        port=args.siren_port,
        username=args.siren_user,
        password=args.siren_password,
        auto_stop_seconds=args.siren_auto_stop,
    )

    # ── 서비스 실행 ───────────────────────────────────────────────────────
    service = ActionBridge(
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        subscribe_topics=subscribe_topics,
        db_path=args.db_path,
        external_api_url=args.external_api_url or None,
        speaker_config=speaker_cfg,
        signboard_config=signboard_cfg,
        siren_config=siren_cfg,
        alarm_topics=alarm_topics,
        alarm_cooldown_seconds=args.alarm_cooldown,
        rest_enabled=args.rest_enabled,
        rest_host=args.rest_host,
        rest_port=args.rest_port,
    )
    service.start()


if __name__ == "__main__":
    main()

