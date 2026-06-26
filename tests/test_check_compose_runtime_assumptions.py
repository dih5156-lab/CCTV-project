import importlib.util
import sys
from pathlib import Path


def _load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime_checks = _load_script_module(
    "check_compose_runtime_assumptions",
    "scripts/health/check_compose_runtime_assumptions.py",
)


def test_default_compose_architecture_passes_on_amd64():
    result = runtime_checks.check_default_compose_architecture(
        machine="x86_64",
        compose_text="image: edgexfoundry/core-data:3.1.0",
    )
    assert result["passed"] is True


def test_default_compose_architecture_fails_on_arm64_risky_edgex_images():
    result = runtime_checks.check_default_compose_architecture(
        machine="aarch64",
        compose_text="image: edgexfoundry/core-data:3.1.0",
        arm64_override_text="",
    )
    assert result["passed"] is False
    assert "arm64 host detected" in result["detail"]
    assert "docker-compose.jetson.yml" in result["detail"]


def test_default_compose_architecture_passes_on_arm64_with_platform_override():
    result = runtime_checks.check_default_compose_architecture(
        machine="aarch64",
        compose_text="platform: linux/arm64\nimage: edgexfoundry/core-data:3.1.0",
    )
    assert result["passed"] is True


def test_default_compose_architecture_fails_on_arm64_even_if_old_override_text_is_provided():
    result = runtime_checks.check_default_compose_architecture(
        machine="aarch64",
        compose_text="image: edgexfoundry/core-data:3.1.0",
        arm64_override_text="""
services:
  core-common-config-bootstrapper:
    platform: linux/arm64
  core-data:
    platform: linux/arm64
  core-metadata:
    platform: linux/arm64
  device-rest:
    platform: linux/arm64
  ui:
    profiles:
      - amd64-ui
""",
    )
    assert result["passed"] is False
    assert "docker-compose.jetson.yml" in result["detail"]


def test_parser_db_defaults_fail_when_db_host_is_localhost():
    result = runtime_checks.check_parser_db_defaults(
        "DB_HOST=localhost\n",
        compose_text="services:\n  aiot-parser:\n    environment: {}\n",
    )
    assert result["passed"] is False
    assert "DB_HOST=localhost" in result["detail"]


def test_parser_db_defaults_pass_when_compose_overrides_localhost():
    result = runtime_checks.check_parser_db_defaults(
        "DB_HOST=localhost\n",
        compose_text="""
services:
  aiot-parser-db:
    image: postgres:16-alpine
  aiot-parser:
    environment:
      DB_HOST: aiot-parser-db
""",
    )
    assert result["passed"] is True
    assert "overrides DB_HOST" in result["detail"]


def test_parser_db_defaults_pass_when_db_host_is_service_name():
    result = runtime_checks.check_parser_db_defaults("DB_HOST=aiot-parser-db\n")
    assert result["passed"] is True


def test_mqtt_auth_config_passes_when_auth_artifacts_are_wired(tmp_path):
    passwd = tmp_path / "passwd"
    passwd.write_text("cctv:$7$hash\n", encoding="utf-8")
    compose = """
services:
  edgex-mqtt-broker:
    environment:
      MQTT_USER: ${MQTT_USER:-}
      MQTT_PASSWORD: ${MQTT_PASSWORD:-}
    volumes:
      - ./mosquitto/passwd:/mosquitto/config/passwd:ro
  cctv-action-layer:
    environment:
      MQTT_USER: ${MQTT_USER:-}
      MQTT_PASSWORD: ${MQTT_PASSWORD:-}
"""

    result = runtime_checks.check_mqtt_auth_config(
        mosquitto_text="allow_anonymous false\npassword_file /mosquitto/config/passwd\n",
        compose_text=compose,
        jetson_compose_text=compose,
        passwd_path=passwd,
    )

    assert result["passed"] is True


def test_mqtt_auth_config_fails_when_passwd_is_missing(tmp_path):
    compose = """
services:
  edgex-mqtt-broker:
    environment:
      MQTT_USER: ${MQTT_USER:-}
      MQTT_PASSWORD: ${MQTT_PASSWORD:-}
    volumes:
      - ./mosquitto/passwd:/mosquitto/config/passwd:ro
"""

    result = runtime_checks.check_mqtt_auth_config(
        mosquitto_text="allow_anonymous false\npassword_file /mosquitto/config/passwd\n",
        compose_text=compose,
        jetson_compose_text=compose,
        passwd_path=tmp_path / "passwd",
    )

    assert result["passed"] is False
    assert "non-empty mosquitto/passwd" in result["detail"]



def test_required_runtime_secrets_fails_when_env_missing_values():
    result = runtime_checks.check_required_runtime_secrets(
        "MQTT_USER=cctv\nMQTT_PASSWORD=\n"
    )
    assert result["passed"] is False
    assert "MQTT_PASSWORD" in result["detail"]
    assert "AIOT_DB_PASSWORD" in result["detail"]


def test_required_runtime_secrets_passes_when_env_has_required_values():
    result = runtime_checks.check_required_runtime_secrets(
        "MQTT_USER=cctv\nMQTT_PASSWORD=secret\nAIOT_DB_PASSWORD=dbsecret\n"
    )
    assert result["passed"] is True


def test_aiot_db_secret_wiring_requires_same_env_source():
    compose = """
services:
  aiot-parser-db:
    environment:
      POSTGRES_PASSWORD: ${AIOT_DB_PASSWORD:-}
  aiot-parser:
    environment:
      DB_PW: ${AIOT_DB_PASSWORD:-}
"""
    result = runtime_checks.check_aiot_db_secret_wiring(
        compose_text=compose,
        jetson_compose_text=compose,
    )
    assert result["passed"] is True


def test_aiot_db_secret_wiring_fails_on_split_secret_sources():
    result = runtime_checks.check_aiot_db_secret_wiring(
        compose_text="POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-}\nDB_PW: ${AIOT_DB_PASSWORD:-}\n",
        jetson_compose_text="POSTGRES_PASSWORD: ${AIOT_DB_PASSWORD:-}\nDB_PW: ${DB_PASSWORD:-}\n",
    )
    assert result["passed"] is False
    assert "AIOT_DB_PASSWORD" in result["detail"]


def test_runtime_path_convergence_requires_runtime_and_logs_paths():
    compose = """
services:
  cctv-alert-api:
    command:
      - /app/data/logs/alert_api_events.jsonl
      - /app/data/logs/sensor_readings.jsonl
  cctv-action-layer:
    environment:
      DB_PATH: /app/data/runtime/action_events.db
  cctv-public-api:
    environment:
      APPEARANCES_DB: /app/data/runtime/appearances.db
      APPEARANCE_CROP_DIR: /app/data/runtime/appearance_crops
      ALERT_FALLBACK_LOG: /app/data/logs/public_api_fallback.jsonl
"""

    result = runtime_checks.check_runtime_path_convergence(
        compose_text=compose,
        jetson_compose_text=compose,
        env_example_text=compose,
        jetson_env_example_text=compose,
    )

    assert result["passed"] is True


def test_runtime_path_convergence_rejects_legacy_paths():
    result = runtime_checks.check_runtime_path_convergence(
        compose_text="/app/data/appearances.db\n/app/data/logs/alert_api_events.jsonl",
        jetson_compose_text="/app/logs/alert_api_events.jsonl",
        env_example_text="/app/data/appearance_crops",
        jetson_env_example_text="/app/action_events.db",
    )

    assert result["passed"] is False
    assert "/app/data/appearances.db" in result["detail"]
    assert "/app/logs" in result["detail"]


def test_mqtt_auth_config_requires_app_rules_engine_rendered_config(tmp_path):
    passwd = tmp_path / "passwd"
    passwd.write_text("cctv:$7$hash\n", encoding="utf-8")
    compose = """
services:
  edgex-mqtt-broker:
    environment:
      MQTT_USER: ${MQTT_USER:-}
      MQTT_PASSWORD: ${MQTT_PASSWORD:-}
    volumes:
      - ./mosquitto/passwd:/mosquitto/config/passwd:ro
"""
    jetson = compose + """
  app-rules-engine:
    environment:
      MQTT_USER: ${MQTT_USER:-}
      MQTT_PASSWORD: ${MQTT_PASSWORD:-}
"""

    result = runtime_checks.check_mqtt_auth_config(
        mosquitto_text="allow_anonymous false\npassword_file /mosquitto/config/passwd\n",
        compose_text=compose,
        jetson_compose_text=jetson,
        passwd_path=passwd,
    )

    assert result["passed"] is False
    assert "app-rules-engine rendered config entrypoint" in result["detail"]
