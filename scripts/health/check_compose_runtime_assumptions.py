"""Check runtime assumptions that docker compose config alone cannot catch."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# These images are known to be a risk on arm64 hosts when pulled without an
# explicit arm64-compatible tag or platform override in the default compose file.
ARM64_RISK_IMAGES = (
    "edgexfoundry/core-common-config-bootstrapper:",
    "edgexfoundry/core-data:",
    "edgexfoundry/core-metadata:",
    "edgexfoundry/device-rest:",
    "edgexfoundry/edgex-ui:",
)

REQUIRED_RUNTIME_SECRETS = (
    "MQTT_USER",
    "MQTT_PASSWORD",
    "AIOT_DB_PASSWORD",
)

REQUIRED_RUNTIME_PATHS = (
    "/app/data/runtime/appearances.db",
    "/app/data/runtime/appearance_crops",
    "/app/data/runtime/action_events.db",
    "/app/data/logs/alert_api_events.jsonl",
    "/app/data/logs/sensor_readings.jsonl",
    "/app/data/logs/public_api_fallback.jsonl",
)

LEGACY_RUNTIME_PATHS = (
    "/app/logs",
    "/app/data/appearances.db",
    "/app/data/appearance_crops",
    "/app/data/face_snapshots",
    "/app/data/action_http_outbox.db",
    "/app/action_events.db",
    "/data/event_outbox.db",
    "data/event_outbox.db",
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _normalize_machine(machine: str | None = None) -> str:
    value = (machine or platform.machine()).strip().lower()
    if value in {"aarch64", "arm64"}:
        return "arm64"
    if value in {"x86_64", "amd64"}:
        return "amd64"
    return value or "unknown"


def _parse_env_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _compose_service_env_value(text: str, service_name: str, key: str) -> str:
    """Return one environment value from a top-level Compose service block."""
    service_block = _compose_service_block(text, service_name)
    for line in service_block.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return ""


def _compose_service_block(text: str, service_name: str) -> str:
    """Return a top-level Compose service block."""
    service_marker = f"  {service_name}:"
    in_service = False
    lines: list[str] = []
    for line in text.splitlines():
        if line == service_marker:
            in_service = True
            lines.append(line)
            continue
        if in_service and line.startswith("  ") and not line.startswith("    "):
            break
        if in_service:
            lines.append(line)
    return "\n".join(lines)


def check_edgex_outbox_path_isolation(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
) -> dict[str, Any]:
    """Ensure the CCTV adapter and AIoT parser do not share one SQLite file."""
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )
    failures: list[str] = []
    for label, text in (("docker-compose.yml", compose), ("docker-compose.jetson.yml", jetson)):
        adapter_path = _compose_service_env_value(text, "cctv-edgex-adapter", "EDGEX_OUTBOX_DB")
        parser_path = _compose_service_env_value(text, "aiot-parser", "EDGEX_OUTBOX_DB")
        if not adapter_path or not parser_path:
            failures.append(f"{label} missing EDGEX_OUTBOX_DB")
        elif adapter_path == parser_path:
            failures.append(f"{label} shared outbox path: {adapter_path}")

    return {
        "name": "EdgeX outbox path isolation",
        "passed": not failures,
        "detail": ", ".join(failures),
    }


def check_default_compose_architecture(
    *,
    machine: str | None = None,
    compose_text: str | None = None,
    arm64_override_text: str | None = None,
) -> dict[str, Any]:
    """Detect default compose services likely to fail with exec format errors."""
    arch = _normalize_machine(machine)
    text = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    risky_images = [image for image in ARM64_RISK_IMAGES if image in text]

    deployment_target = os.environ.get("CCTV_DEPLOYMENT_TARGET", "").strip().lower()
    if deployment_target == "jetson":
        return {
            "name": "default compose architecture",
            "passed": True,
            "detail": "Jetson target selected; deployment uses docker-compose.jetson.yml",
        }

    if arch != "arm64" or not risky_images:
        return {
            "name": "default compose architecture",
            "passed": True,
            "detail": "",
        }

    has_platform_override = "platform:" in text
    if has_platform_override:
        return {
            "name": "default compose architecture",
            "passed": True,
            "detail": "arm64 host detected; compose contains platform override",
        }

    return {
        "name": "default compose architecture",
        "passed": False,
        "detail": (
            "arm64 host detected but docker-compose.yml includes EdgeX images that may be amd64-only: "
            f"{', '.join(risky_images)}. Use docker-compose.jetson.yml for Jetson deployment, or run only the "
            "API/action services from docker-compose.yml."
        ),
    }


def check_parser_db_defaults(
    env_text: str | None = None,
    compose_text: str | None = None,
) -> dict[str, Any]:
    """Warn when parser DB config is likely to resolve to localhost inside a container."""
    env_path = PROJECT_ROOT / "parser-python" / ".env"
    text = env_text if env_text is not None else _read_text(env_path)
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    if not text:
        return {
            "name": "aiot parser database config",
            "passed": True,
            "detail": "",
        }

    values = _parse_env_values(text)

    db_host = values.get("DB_HOST", "localhost")
    if db_host in {"localhost", "127.0.0.1", "::1"}:
        if "aiot-parser-db:" in compose and "DB_HOST: aiot-parser-db" in compose:
            return {
                "name": "aiot parser database config",
                "passed": True,
                "detail": "parser .env uses localhost, but docker-compose.yml overrides DB_HOST to aiot-parser-db",
            }
        return {
            "name": "aiot parser database config",
            "passed": False,
            "detail": (
                "parser-python/.env uses DB_HOST=localhost. Inside the aiot-parser container, "
                "localhost is the container itself, so PostgreSQL must run in the same container "
                "or DB_HOST must point to a compose service/external host."
            ),
        }

    return {
        "name": "aiot parser database config",
        "passed": True,
        "detail": "",
    }


def check_required_runtime_secrets(env_text: str | None = None) -> dict[str, Any]:
    """Fail fast when compose-required runtime secrets are missing from .env."""
    text = env_text if env_text is not None else _read_text(PROJECT_ROOT / ".env")
    values = _parse_env_values(text)
    missing = [key for key in REQUIRED_RUNTIME_SECRETS if not values.get(key)]

    if missing:
        return {
            "name": "runtime secret source",
            "passed": False,
            "detail": "missing non-empty .env values: " + ", ".join(missing),
        }

    return {
        "name": "runtime secret source",
        "passed": True,
        "detail": ".env provides MQTT and AIoT DB secrets",
    }


def check_aiot_db_secret_wiring(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
) -> dict[str, Any]:
    """Ensure parser and Postgres are wired to the same DB password source."""
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )

    missing: list[str] = []
    for label, text in (("docker-compose.yml", compose), ("docker-compose.jetson.yml", jetson)):
        if "POSTGRES_PASSWORD: ${AIOT_DB_PASSWORD:-}" not in text:
            missing.append(f"{label} aiot-parser-db POSTGRES_PASSWORD from AIOT_DB_PASSWORD")
        if "DB_PW: ${AIOT_DB_PASSWORD:-}" not in text:
            missing.append(f"{label} aiot-parser DB_PW from AIOT_DB_PASSWORD")

    if missing:
        return {
            "name": "aiot database secret wiring",
            "passed": False,
            "detail": "missing: " + ", ".join(missing),
        }

    return {
        "name": "aiot database secret wiring",
        "passed": True,
        "detail": "",
    }


def check_runtime_path_convergence(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
    env_example_text: str | None = None,
    jetson_env_example_text: str | None = None,
) -> dict[str, Any]:
    """Ensure runtime artifacts converge under data/runtime and data/logs."""
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )
    env_example = env_example_text if env_example_text is not None else _read_text(PROJECT_ROOT / ".env.example")
    jetson_env = (
        jetson_env_example_text
        if jetson_env_example_text is not None
        else _read_text(PROJECT_ROOT / ".env.jetson.example")
    )

    checked_files = {
        "docker-compose.yml": compose,
        "docker-compose.jetson.yml": jetson,
        ".env.example": env_example,
        ".env.jetson.example": jetson_env,
    }

    missing: list[str] = []
    for label, text in checked_files.items():
        for path in REQUIRED_RUNTIME_PATHS:
            if path not in text:
                missing.append(f"{label} missing {path}")

    legacy_hits = [
        f"{label} contains {path}"
        for label, text in checked_files.items()
        for path in LEGACY_RUNTIME_PATHS
        if path in text
    ]

    if missing or legacy_hits:
        return {
            "name": "runtime path convergence",
            "passed": False,
            "detail": "missing/legacy paths: " + ", ".join(missing + legacy_hits),
        }

    return {
        "name": "runtime path convergence",
        "passed": True,
        "detail": "runtime artifacts use /app/data/runtime and logs use /app/data/logs",
    }


def check_appearance_model_wiring(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
) -> dict[str, Any]:
    """Ensure appearance model config includes the decoder metadata it needs."""
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )

    required_entries = (
        ("docker-compose.yml", compose, "APPEARANCE_BACKEND: ${APPEARANCE_BACKEND:-pphuman}"),
        ("docker-compose.yml", compose, "APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/pphuman_attribute.onnx}"),
        (
            "docker-compose.yml",
            compose,
            "APPEARANCE_LABEL_MAP_PATH: ${APPEARANCE_LABEL_MAP_PATH:-config/appearance_pphuman_labels.example.json}",
        ),
        ("docker-compose.yml", compose, "APPEARANCE_RUNTIME: ${APPEARANCE_RUNTIME:-onnxruntime}"),
        ("docker-compose.jetson.yml", jetson, "DS_PPHUMAN_SGIE_ENABLED: ${DS_PPHUMAN_SGIE_ENABLED:-1}"),
        (
            "docker-compose.jetson.yml",
            jetson,
            "DS_PPHUMAN_INFER_CONFIG: ${DS_PPHUMAN_INFER_CONFIG:-config/deepstream/config_infer_pa100k.txt}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/pa100k_resnet50_attr.engine}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "APPEARANCE_LABEL_MAP_PATH: ${APPEARANCE_LABEL_MAP_PATH:-config/appearance_pa100k_labels.json}",
        ),
        ("docker-compose.jetson.yml", jetson, "APPEARANCE_RUNTIME: ${APPEARANCE_RUNTIME:-tensorrt}"),
    )
    missing = [f"{label} missing {entry}" for label, text, entry in required_entries if entry not in text]

    if missing:
        return {
            "name": "appearance model wiring",
            "passed": False,
            "detail": "missing: " + ", ".join(missing),
        }

    return {
        "name": "appearance model wiring",
        "passed": True,
        "detail": "PP-Human/PA100K model paths, label maps, and runtimes are wired",
    }


def check_falldata_aux_wiring(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
    env_example_text: str | None = None,
    jetson_env_example_text: str | None = None,
) -> dict[str, Any]:
    """Ensure falldata aux deployment keeps fail-open safety and Jetson paths wired."""
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )
    env_example = env_example_text if env_example_text is not None else _read_text(PROJECT_ROOT / ".env.example")
    jetson_env = (
        jetson_env_example_text
        if jetson_env_example_text is not None
        else _read_text(PROJECT_ROOT / ".env.jetson.example")
    )

    required_entries = (
        (
            "docker-compose.yml",
            compose,
            "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE: ${FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE:-true}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE: ${FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE:-true}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "FALLDATA_AUX_MEDIAPIPE_PYTHON: ${FALLDATA_AUX_MEDIAPIPE_PYTHON:-/app/.venv-mediapipe/bin/python}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "FALLDATA_AUX_MODEL_PYTHON: ${FALLDATA_AUX_MODEL_PYTHON:-/app/.venv-falldata/bin/python}",
        ),
        ("docker-compose.jetson.yml", jetson, "source: ./falldata"),
        ("docker-compose.jetson.yml", jetson, "source: ./.venv-mediapipe"),
        ("docker-compose.jetson.yml", jetson, "source: ./.venv-falldata"),
        (".env.example", env_example, "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true"),
        (".env.jetson.example", jetson_env, "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true"),
        (".env.jetson.example", jetson_env, "FALLDATA_AUX_CONFIRM_BORDERLINE=true"),
        (
            ".env.jetson.example",
            jetson_env,
            "FALLDATA_AUX_MEDIAPIPE_PYTHON=/app/.venv-mediapipe/bin/python",
        ),
        (
            ".env.jetson.example",
            jetson_env,
            "FALLDATA_AUX_MODEL_PYTHON=/app/.venv-falldata/bin/python",
        ),
    )
    missing = [f"{label} missing {entry}" for label, text, entry in required_entries if entry not in text]

    if missing:
        return {
            "name": "falldata aux safety wiring",
            "passed": False,
            "detail": "missing: " + ", ".join(missing),
        }

    return {
        "name": "falldata aux safety wiring",
        "passed": True,
        "detail": "fail-open policy and Jetson aux paths are wired",
    }


def check_h264_webrtc_wiring(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
) -> dict[str, Any]:
    """Ensure Jetson NVENC output keeps the WebRTC POC compatibility guard."""
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )
    required_entries = (
        (
            "docker-compose.yml",
            compose,
            "DS_H264_POC_FIX_ENABLED: ${DS_H264_POC_FIX_ENABLED:-true}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "DS_H264_ENCODER: ${DS_H264_ENCODER:-nvv4l2h264enc}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "DS_H264_POC_FIX_ENABLED: ${DS_H264_POC_FIX_ENABLED:-true}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "DS_H264_POC_TYPE: ${DS_H264_POC_TYPE:-2}",
        ),
    )
    missing = [f"{label} missing {entry}" for label, text, entry in required_entries if entry not in text]
    return {
        "name": "H.264 WebRTC compatibility wiring",
        "passed": not missing,
        "detail": "missing: " + ", ".join(missing) if missing else "Jetson NVENC POC guard is wired",
    }


def check_public_api_exposure_defaults(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
    env_example_text: str | None = None,
    jetson_env_example_text: str | None = None,
) -> dict[str, Any]:
    """Ensure Public API is not externally bound by default."""
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )
    env_example = env_example_text if env_example_text is not None else _read_text(PROJECT_ROOT / ".env.example")
    jetson_env = (
        jetson_env_example_text
        if jetson_env_example_text is not None
        else _read_text(PROJECT_ROOT / ".env.jetson.example")
    )

    required_entries = (
        ("docker-compose.yml", compose, "host_ip: ${PUBLIC_API_BIND_HOST:-127.0.0.1}"),
        ("docker-compose.yml", compose, "host_ip: ${PUBLIC_DEMO_BIND_HOST:-127.0.0.1}"),
        ("docker-compose.jetson.yml", jetson, "host_ip: ${MQTT_BIND_HOST:-127.0.0.1}"),
        ("docker-compose.jetson.yml", jetson, "host_ip: ${PUBLIC_API_BIND_HOST:-127.0.0.1}"),
        ("docker-compose.jetson.yml", jetson, "host_ip: ${PUBLIC_DEMO_BIND_HOST:-127.0.0.1}"),
        ("docker-compose.jetson.yml", jetson, "host_ip: ${MEDIA_BIND_HOST:-127.0.0.1}"),
        ("docker-compose.jetson.yml", jetson, "host_ip: ${MEDIA_API_BIND_HOST:-127.0.0.1}"),
        (".env.example", env_example, "PUBLIC_API_BIND_HOST=127.0.0.1"),
        (".env.example", env_example, "PUBLIC_DEMO_BIND_HOST=127.0.0.1"),
        (".env.jetson.example", jetson_env, "MQTT_BIND_HOST=127.0.0.1"),
        (".env.jetson.example", jetson_env, "PUBLIC_API_BIND_HOST=127.0.0.1"),
        (".env.jetson.example", jetson_env, "PUBLIC_DEMO_BIND_HOST=127.0.0.1"),
        (".env.jetson.example", jetson_env, "MEDIA_BIND_HOST=127.0.0.1"),
        (".env.jetson.example", jetson_env, "MEDIA_API_BIND_HOST=127.0.0.1"),
    )
    missing = [f"{label} missing {entry}" for label, text, entry in required_entries if entry not in text]
    unsafe_blocks = (
        ("docker-compose.yml", compose, "cctv-public-api"),
        ("docker-compose.yml", compose, "public-demo-ui"),
        ("docker-compose.jetson.yml", jetson, "edgex-mqtt-broker"),
        ("docker-compose.jetson.yml", jetson, "public-demo-ui"),
        ("docker-compose.jetson.yml", jetson, "cctv-media-server"),
        ("docker-compose.jetson.yml", jetson, "cctv-public-api"),
    )
    unsafe = [
        f"{label} hardcodes {service_name} host_ip 0.0.0.0"
        for label, text, service_name in unsafe_blocks
        if "host_ip: 0.0.0.0" in _compose_service_block(text, service_name)
    ]

    failures = missing + unsafe
    return {
        "name": "network exposure defaults",
        "passed": not failures,
        "detail": "missing/unsafe entries: " + ", ".join(failures) if failures else "externally reachable services bind to localhost by default",
    }


def check_public_api_shared_secret_alignment(
    *,
    env_text: str | None = None,
    jetson_env_text: str | None = None,
) -> dict[str, Any]:
    """Ensure the demo UI and public API share the same runtime secrets across env files."""
    env = env_text if env_text is not None else _read_text(PROJECT_ROOT / ".env")
    jetson = (
        jetson_env_text
        if jetson_env_text is not None
        else _read_text(PROJECT_ROOT / ".env.jetson")
    )

    env_values = _parse_env_values(env)
    jetson_values = _parse_env_values(jetson)
    tracked_keys = ("PUBLIC_API_KEY", "INTERNAL_SERVICE_TOKEN")

    failures: list[str] = []
    for key in tracked_keys:
        env_value = env_values.get(key, "")
        jetson_value = jetson_values.get(key, "")
        if not env_value or not jetson_value:
            failures.append(f"{key} missing in .env or .env.jetson")
            continue
        if env_value != jetson_value:
            failures.append(f"{key} differs between .env and .env.jetson")

    return {
        "name": "public API shared secret alignment",
        "passed": not failures,
        "detail": ", ".join(failures),
    }


def check_mqtt_auth_config(
    *,
    mosquitto_text: str | None = None,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
    passwd_path: Path | None = None,
) -> dict[str, Any]:
    """Catch common MQTT auth rollout mistakes before Docker startup."""
    broker_config = (
        mosquitto_text
        if mosquitto_text is not None
        else _read_text(PROJECT_ROOT / "mosquitto" / "mosquitto.conf")
    )
    if "allow_anonymous false" not in broker_config:
        return {
            "name": "mqtt authentication config",
            "passed": True,
            "detail": "anonymous MQTT access is not disabled",
        }

    missing: list[str] = []
    if "password_file /mosquitto/config/passwd" not in broker_config:
        missing.append("mosquitto.conf password_file /mosquitto/config/passwd")

    actual_passwd_path = passwd_path or PROJECT_ROOT / "mosquitto" / "passwd"
    if not actual_passwd_path.exists() or actual_passwd_path.stat().st_size == 0:
        missing.append("non-empty mosquitto/passwd")

    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )

    for label, text in (("docker-compose.yml", compose), ("docker-compose.jetson.yml", jetson)):
        if "./mosquitto/passwd" not in text:
            missing.append(f"{label} bind mount for ./mosquitto/passwd")
        if "MQTT_USER" not in text:
            missing.append(f"{label} MQTT_USER propagation")
        if "MQTT_PASSWORD" not in text:
            missing.append(f"{label} MQTT_PASSWORD propagation")

    if "app-rules-engine:" in jetson:
        if "MQTT_USER: ${MQTT_USER:-}" not in jetson:
            missing.append("docker-compose.jetson.yml app-rules-engine MQTT_USER render input")
        if "MQTT_PASSWORD: ${MQTT_PASSWORD:-}" not in jetson:
            missing.append("docker-compose.jetson.yml app-rules-engine MQTT_PASSWORD render input")
        if 'entrypoint: ["/res/cctv-external-http/render-and-run.sh"]' not in jetson:
            missing.append("docker-compose.jetson.yml app-rules-engine rendered config entrypoint")
        if 'command: ["-cp=consul.http://edgex-core-consul:8500", "--registry", "-o"]' not in jetson:
            missing.append("docker-compose.jetson.yml app-rules-engine consul command")

    app_config = _read_text(PROJECT_ROOT / "edgex" / "asc" / "cctv-external-http" / "configuration.yaml")
    render_script = _read_text(PROJECT_ROOT / "edgex" / "asc" / "cctv-external-http" / "render-and-run.sh")
    if 'Type: "external-mqtt"' in app_config:
        if 'SecretName: "mqtt"' not in app_config and 'SecretPath: "mqtt"' not in app_config:
            missing.append("cctv-external-http ExternalMqtt mqtt secret reference")
        if 'AuthMode: "usernamepassword"' not in app_config:
            missing.append("cctv-external-http ExternalMqtt AuthMode usernamepassword")
        if 'MQTT_USER' not in render_script or 'MQTT_PASSWORD' not in render_script:
            missing.append("cctv-external-http render-and-run MQTT credential rendering")

    if missing:
        return {
            "name": "mqtt authentication config",
            "passed": False,
            "detail": "missing: " + ", ".join(missing),
        }

    return {
        "name": "mqtt authentication config",
        "passed": True,
        "detail": "",
    }


def run_checks() -> list[dict[str, Any]]:
    return [
        check_default_compose_architecture(),
        check_parser_db_defaults(),
        check_required_runtime_secrets(),
        check_aiot_db_secret_wiring(),
        check_runtime_path_convergence(),
        check_appearance_model_wiring(),
        check_falldata_aux_wiring(),
        check_h264_webrtc_wiring(),
        check_public_api_exposure_defaults(),
        check_public_api_shared_secret_alignment(),
        check_mqtt_auth_config(),
        check_edgex_outbox_path_isolation(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check runtime assumptions for CCTV compose deployment.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    checks = run_checks()
    passed = all(check["passed"] for check in checks)
    payload = {"passed": passed, "checks": checks}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for check in checks:
            status = "PASS" if check["passed"] else "FAIL"
            detail = f" - {check['detail']}" if check["detail"] else ""
            print(f"[{status}] {check['name']}{detail}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
