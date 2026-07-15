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


def test_edgex_adapter_outbox_path_is_isolated_from_aiot_parser():
    compose = """
services:
  cctv-edgex-adapter:
    environment:
      EDGEX_OUTBOX_DB: /data/cctv-edgex-adapter/event_outbox.db
  aiot-parser:
    environment:
      EDGEX_OUTBOX_DB: /data/runtime/event_outbox.db
"""

    result = runtime_checks.check_edgex_outbox_path_isolation(
        compose_text=compose,
        jetson_compose_text=compose,
    )

    assert result["passed"] is True


def test_edgex_adapter_outbox_path_rejects_shared_parser_database():
    compose = """
services:
  cctv-edgex-adapter:
    environment:
      EDGEX_OUTBOX_DB: /data/runtime/event_outbox.db
  aiot-parser:
    environment:
      EDGEX_OUTBOX_DB: /data/runtime/event_outbox.db
"""

    result = runtime_checks.check_edgex_outbox_path_isolation(
        compose_text=compose,
        jetson_compose_text=compose,
    )

    assert result["passed"] is False
    assert "shared outbox path" in result["detail"]


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


def test_appearance_model_wiring_requires_label_maps_and_runtimes():
    compose = """
x-appearance-search-runtime:
  APPEARANCE_BACKEND: ${APPEARANCE_BACKEND:-pphuman}
  APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/pphuman_attribute.onnx}
  APPEARANCE_LABEL_MAP_PATH: ${APPEARANCE_LABEL_MAP_PATH:-config/appearance_pphuman_labels.example.json}
  APPEARANCE_RUNTIME: ${APPEARANCE_RUNTIME:-onnxruntime}
"""
    jetson = """
x-appearance-runtime:
  DS_PPHUMAN_SGIE_ENABLED: ${DS_PPHUMAN_SGIE_ENABLED:-1}
  DS_PPHUMAN_INFER_CONFIG: ${DS_PPHUMAN_INFER_CONFIG:-config/deepstream/config_infer_pa100k.txt}
  APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/pa100k_resnet50_attr.engine}
  APPEARANCE_LABEL_MAP_PATH: ${APPEARANCE_LABEL_MAP_PATH:-config/appearance_pa100k_labels.json}
  APPEARANCE_RUNTIME: ${APPEARANCE_RUNTIME:-tensorrt}
"""

    result = runtime_checks.check_appearance_model_wiring(
        compose_text=compose,
        jetson_compose_text=jetson,
    )

    assert result["passed"] is True


def test_appearance_model_wiring_fails_when_pphuman_label_map_is_missing():
    compose = """
x-appearance-search-runtime:
  APPEARANCE_BACKEND: ${APPEARANCE_BACKEND:-pphuman}
  APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/pphuman_attribute.onnx}
  APPEARANCE_RUNTIME: ${APPEARANCE_RUNTIME:-onnxruntime}
"""
    jetson = """
x-appearance-runtime:
  DS_PPHUMAN_SGIE_ENABLED: ${DS_PPHUMAN_SGIE_ENABLED:-1}
  DS_PPHUMAN_INFER_CONFIG: ${DS_PPHUMAN_INFER_CONFIG:-config/deepstream/config_infer_pa100k.txt}
  APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/pa100k_resnet50_attr.engine}
  APPEARANCE_LABEL_MAP_PATH: ${APPEARANCE_LABEL_MAP_PATH:-config/appearance_pa100k_labels.json}
  APPEARANCE_RUNTIME: ${APPEARANCE_RUNTIME:-tensorrt}
"""

    result = runtime_checks.check_appearance_model_wiring(
        compose_text=compose,
        jetson_compose_text=jetson,
    )

    assert result["passed"] is False
    assert "appearance_pphuman_labels.example.json" in result["detail"]


def test_falldata_aux_wiring_requires_fail_open_and_jetson_paths():
    compose = """
services:
  cctv-ai-engine:
    environment:
      FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE: ${FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE:-true}
"""
    jetson = """
services:
  cctv-ai-engine:
    environment:
      FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE: ${FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE:-true}
      FALLDATA_AUX_MEDIAPIPE_PYTHON: ${FALLDATA_AUX_MEDIAPIPE_PYTHON:-/app/.venv-mediapipe/bin/python}
      FALLDATA_AUX_MODEL_PYTHON: ${FALLDATA_AUX_MODEL_PYTHON:-/app/.venv-falldata/bin/python}
    volumes:
      - type: bind
        source: ./falldata
      - type: bind
        source: ./.venv-mediapipe
      - type: bind
        source: ./.venv-falldata
"""
    result = runtime_checks.check_falldata_aux_wiring(
        compose_text=compose,
        jetson_compose_text=jetson,
        env_example_text="FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true\n",
        jetson_env_example_text=(
            "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true\n"
            "FALLDATA_AUX_CONFIRM_BORDERLINE=true\n"
            "FALLDATA_AUX_MEDIAPIPE_PYTHON=/app/.venv-mediapipe/bin/python\n"
            "FALLDATA_AUX_MODEL_PYTHON=/app/.venv-falldata/bin/python\n"
        ),
    )

    assert result["passed"] is True


def test_falldata_aux_wiring_fails_without_fail_open() -> None:
    result = runtime_checks.check_falldata_aux_wiring(
        compose_text="services: {}\n",
        jetson_compose_text="services: {}\n",
        env_example_text="FALLDATA_AUX_ENABLED=false\n",
        jetson_env_example_text="FALLDATA_AUX_ENABLED=false\n",
    )

    assert result["passed"] is False
    assert "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE" in result["detail"]


def test_h264_webrtc_wiring_requires_jetson_poc_guard() -> None:
    compose = "DS_H264_POC_FIX_ENABLED: ${DS_H264_POC_FIX_ENABLED:-true}\n"
    jetson = (
        "DS_H264_ENCODER: ${DS_H264_ENCODER:-nvv4l2h264enc}\n"
        "DS_H264_POC_FIX_ENABLED: ${DS_H264_POC_FIX_ENABLED:-true}\n"
        "DS_H264_POC_TYPE: ${DS_H264_POC_TYPE:-2}\n"
    )

    result = runtime_checks.check_h264_webrtc_wiring(
        compose_text=compose,
        jetson_compose_text=jetson,
    )

    assert result["passed"] is True


def test_h264_webrtc_wiring_rejects_disabled_poc_fix_default() -> None:
    result = runtime_checks.check_h264_webrtc_wiring(
        compose_text="DS_H264_POC_FIX_ENABLED: ${DS_H264_POC_FIX_ENABLED:-false}\n",
        jetson_compose_text=(
            "DS_H264_ENCODER: ${DS_H264_ENCODER:-nvv4l2h264enc}\n"
            "DS_H264_POC_FIX_ENABLED: ${DS_H264_POC_FIX_ENABLED:-false}\n"
            "DS_H264_POC_TYPE: ${DS_H264_POC_TYPE:-2}\n"
        ),
    )

    assert result["passed"] is False
    assert "DS_H264_POC_FIX_ENABLED" in result["detail"]


def test_public_api_exposure_defaults_require_localhost_bind() -> None:
    compose = """
services:
  edgex-mqtt-broker:
    ports:
      - target: 1883
        host_ip: ${MQTT_BIND_HOST:-127.0.0.1}
  cctv-public-api:
    ports:
      - target: 9000
        host_ip: ${PUBLIC_API_BIND_HOST:-127.0.0.1}
  public-demo-ui:
    ports:
      - target: 7000
        host_ip: ${PUBLIC_DEMO_BIND_HOST:-127.0.0.1}
  cctv-media-server:
    ports:
      - target: 8554
        host_ip: ${MEDIA_BIND_HOST:-127.0.0.1}
      - target: 9997
        host_ip: ${MEDIA_API_BIND_HOST:-127.0.0.1}
"""
    env_example = (
        "MQTT_BIND_HOST=127.0.0.1\n"
        "PUBLIC_API_BIND_HOST=127.0.0.1\n"
        "PUBLIC_DEMO_BIND_HOST=127.0.0.1\n"
        "MEDIA_BIND_HOST=127.0.0.1\n"
        "MEDIA_API_BIND_HOST=127.0.0.1\n"
    )

    result = runtime_checks.check_public_api_exposure_defaults(
        compose_text=compose,
        jetson_compose_text=compose,
        env_example_text=env_example,
        jetson_env_example_text=env_example,
    )

    assert result["passed"] is True


def test_public_api_exposure_defaults_reject_hardcoded_external_bind() -> None:
    compose = """
services:
  cctv-public-api:
    ports:
      - target: 9000
        host_ip: 0.0.0.0
"""

    result = runtime_checks.check_public_api_exposure_defaults(
        compose_text=compose,
        jetson_compose_text=compose,
        env_example_text="",
        jetson_env_example_text="",
    )

    assert result["passed"] is False
    assert "0.0.0.0" in result["detail"]
    assert "PUBLIC_API_BIND_HOST" in result["detail"]


def test_public_api_shared_secret_alignment_passes_when_env_files_match() -> None:
    env_text = """
PUBLIC_API_KEY=shared-key
INTERNAL_SERVICE_TOKEN=shared-token
"""

    result = runtime_checks.check_public_api_shared_secret_alignment(
        env_text=env_text,
        jetson_env_text=env_text,
    )

    assert result["passed"] is True


def test_public_api_shared_secret_alignment_fails_when_env_files_drift() -> None:
    result = runtime_checks.check_public_api_shared_secret_alignment(
        env_text="""
PUBLIC_API_KEY=local-key
INTERNAL_SERVICE_TOKEN=shared-token
""",
        jetson_env_text="""
PUBLIC_API_KEY=jetson-key
INTERNAL_SERVICE_TOKEN=shared-token
""",
    )

    assert result["passed"] is False
    assert "PUBLIC_API_KEY" in result["detail"]


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
