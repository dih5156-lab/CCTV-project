import pytest

from runners.run_action_bridge import parse_runtime_config
from src.services.action_bridge import default_alarm_topics, default_subscribe_topics


def test_parse_runtime_config_builds_device_configs():
    config = parse_runtime_config(
        [
            "--mqtt-broker",
            "mqtt.local",
            "--mqtt-port",
            "1884",
            "--subscribe-topics",
            "topic/a, topic/b,,",
            "--alarm-topics",
            "alarm/a,alarm/b",
            "--db-path",
            "/tmp/action.db",
            "--external-api-url",
            "http://api.local/events",
            "--speaker-host",
            "speaker.local",
            "--speaker-port",
            "81",
            "--speaker-user",
            "speaker-user",
            "--speaker-password",
            "speaker-pass",
            "--signboard-host",
            "signboard.local",
            "--signboard-port",
            "5001",
            "--signboard-idle-refresh-interval",
            "12.5",
            "--siren-host",
            "siren.local",
            "--siren-port",
            "82",
            "--siren-user",
            "siren-user",
            "--siren-password",
            "siren-pass",
            "--rest-enabled",
            "--rest-host",
            "127.0.0.1",
            "--rest-port",
            "8090",
        ]
    )

    assert config.mqtt_broker == "mqtt.local"
    assert config.mqtt_port == 1884
    assert config.subscribe_topics == {"topic/a", "topic/b"}
    assert config.alarm_topics == {"alarm/a", "alarm/b"}
    assert config.db_path == "/tmp/action.db"
    assert config.external_api_url == "http://api.local/events"
    assert config.speaker_config.host == "speaker.local"
    assert config.speaker_config.port == 81
    assert config.speaker_config.username == "speaker-user"
    assert config.speaker_config.password == "speaker-pass"
    assert config.signboard_config.host == "signboard.local"
    assert config.signboard_config.port == 5001
    assert config.signboard_config.idle_refresh_interval == 12.5
    assert config.siren_config.host == "siren.local"
    assert config.siren_config.port == 82
    assert config.siren_config.username == "siren-user"
    assert config.siren_config.password == "siren-pass"
    assert config.rest_enabled is True
    assert config.rest_host == "127.0.0.1"
    assert config.rest_port == 8090


def test_parse_runtime_config_uses_rest_env_default(monkeypatch):
    monkeypatch.setenv("REST_ENABLED", "true")

    config = parse_runtime_config([])

    assert config.rest_enabled is True


def test_parse_runtime_config_uses_action_bridge_default_topics(monkeypatch):
    monkeypatch.delenv("SUBSCRIBE_TOPICS", raising=False)
    monkeypatch.delenv("ALARM_TOPICS", raising=False)

    config = parse_runtime_config([])

    assert config.subscribe_topics == default_subscribe_topics()
    assert config.alarm_topics == default_alarm_topics()
    assert "cctv/ai/events/+/face_unknown" in config.subscribe_topics
    assert "cctv/ai/events/+/face_recognized" in config.subscribe_topics


def test_parse_runtime_config_rejects_invalid_mqtt_port():
    with pytest.raises(SystemExit):
        parse_runtime_config(["--mqtt-port", "0"])


def test_parse_runtime_config_supports_edgex_command_mode():
    config = parse_runtime_config(["--edgex-command-mode", "edgex"])

    assert config.edgex_command_mode == "edgex"


def test_parse_runtime_config_keeps_shadow_compatibility(monkeypatch):
    monkeypatch.setenv("EDGEX_SHADOW_ENABLED", "true")
    monkeypatch.delenv("EDGEX_COMMAND_MODE", raising=False)

    config = parse_runtime_config([])

    assert config.edgex_command_mode == "shadow"
