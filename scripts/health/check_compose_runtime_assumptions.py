"""Docker Compose 설정만으로 확인할 수 없는 실행 환경 가정을 점검한다."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 기본 Compose 파일에서 ARM64 호환 태그나 플랫폼 지정 없이 내려받으면
# ARM64 호스트에서 실행 형식 오류가 발생할 가능성이 있는 이미지 목록이다.
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


def _pip_tensorrt_cuda_variant_majors(distribution_names: list[str] | None = None) -> set[int]:
    names = distribution_names
    if names is None:
        names = []
        for dist in importlib.metadata.distributions():
            name = str(dist.metadata.get("Name") or "").strip()
            if name:
                names.append(name)

    majors: set[int] = set()
    for raw_name in names:
        match = re.match(r"^tensorrt_cu(\d+)$", str(raw_name).strip().lower())
        if match:
            majors.add(int(match.group(1)))
    return majors


def _compose_service_env_value(text: str, service_name: str, key: str) -> str:
    """최상위 Compose 서비스 블록에서 환경 변수 하나의 값을 반환한다."""
    service_block = _compose_service_block(text, service_name)
    for line in service_block.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return ""


def _compose_service_block(text: str, service_name: str) -> str:
    """최상위 Compose 서비스 블록을 반환한다."""
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
    """CCTV 어댑터와 AIoT Parser가 하나의 SQLite 파일을 공유하지 않는지 확인한다."""
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


def check_edgex_mode_rollback_contract(
    *,
    jetson_compose_text: str | None = None,
    uat_compose_text: str | None = None,
    env_text: str | None = None,
    jetson_env_text: str | None = None,
) -> dict[str, Any]:
    """기본 운영은 Direct로 복귀하고 UAT만 EdgeX를 쓰는지 확인한다."""
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )
    uat = (
        uat_compose_text
        if uat_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.edgex-uat.yml")
    )
    env = env_text if env_text is not None else _read_text(PROJECT_ROOT / ".env")
    jetson_env = (
        jetson_env_text
        if jetson_env_text is not None
        else _read_text(PROJECT_ROOT / ".env.jetson")
    )
    failures: list[str] = []

    expected_base_values = {
        "EDGEX_COMMAND_MODE": "${EDGEX_COMMAND_MODE:-}",
        "EDGEX_ALLOWED_DEVICES": "${EDGEX_ALLOWED_DEVICES:-}",
    }
    for key, expected in expected_base_values.items():
        actual = _compose_service_env_value(jetson, "cctv-action-layer", key)
        if actual != expected:
            failures.append(f"docker-compose.jetson.yml {key} must default to empty")

    if _compose_service_env_value(uat, "cctv-action-layer", "EDGEX_COMMAND_MODE") != "edgex":
        failures.append("docker-compose.edgex-uat.yml EDGEX_COMMAND_MODE must be edgex")
    if _compose_service_env_value(uat, "cctv-action-layer", "EDGEX_ALLOWED_DEVICES") != "signboard":
        failures.append("docker-compose.edgex-uat.yml EDGEX_ALLOWED_DEVICES must be signboard")
    if "cctv-action-layer:" not in jetson:
        failures.append("docker-compose.jetson.yml missing cctv-action-layer")
    if "cctv-action-layer:" not in uat:
        failures.append("docker-compose.edgex-uat.yml missing cctv-action-layer")
    for label, text in ((".env", env), (".env.jetson", jetson_env)):
        values = _parse_env_values(text)
        for key in ("EDGEX_COMMAND_MODE", "EDGEX_ALLOWED_DEVICES"):
            if values.get(key, "").strip():
                failures.append(f"{label} must not override {key} during Direct rollback")

    return {
        "name": "EdgeX mode rollback contract",
        "passed": not failures,
        "detail": ", ".join(failures),
    }


def check_default_compose_architecture(
    *,
    machine: str | None = None,
    compose_text: str | None = None,
    arm64_override_text: str | None = None,
) -> dict[str, Any]:
    """기본 Compose 서비스 중 실행 형식 오류가 예상되는 항목을 찾는다."""
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
    """컨테이너 내부에서 Parser DB 설정이 localhost로 해석될 가능성을 경고한다."""
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
    """Compose 실행에 필요한 보안 값이 .env에 없으면 즉시 실패시킨다."""
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
    """Parser와 Postgres가 동일한 DB 비밀번호 공급원을 사용하는지 확인한다."""
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
    """실행 산출물이 data/runtime과 data/logs 아래로 모이는지 확인한다."""
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
    """외관 모델 설정에 디코더 실행에 필요한 메타데이터가 포함됐는지 확인한다."""
    compose = compose_text if compose_text is not None else _read_text(PROJECT_ROOT / "docker-compose.yml")
    jetson = (
        jetson_compose_text
        if jetson_compose_text is not None
        else _read_text(PROJECT_ROOT / "docker-compose.jetson.yml")
    )

    required_entries = (
        ("docker-compose.yml", compose, "APPEARANCE_BACKEND: ${APPEARANCE_BACKEND:-pphuman}"),
        ("docker-compose.yml", compose, "APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/appearance/pphuman_attribute.onnx}"),
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
            "APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/appearance/pa100k_resnet50_attr.engine}",
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


def check_tensorrt_binding_cuda_alignment(
    *,
    machine: str | None = None,
    distribution_names: list[str] | None = None,
    expected_cuda_major: str | None = None,
) -> dict[str, Any]:
    """pip TensorRT CUDA 변형이 ARM64 기준 환경과 호환되지 않으면 조기에 실패시킨다."""
    arch = _normalize_machine(machine)
    if arch != "arm64":
        return {
            "name": "TensorRT CUDA binding alignment",
            "passed": True,
            "detail": "",
        }

    raw_expected = expected_cuda_major or os.environ.get("TENSORRT_EXPECTED_CUDA_MAJOR", "12")
    try:
        expected_major = int(str(raw_expected).strip())
    except (TypeError, ValueError):
        return {
            "name": "TensorRT CUDA binding alignment",
            "passed": True,
            "detail": "",
        }

    variant_majors = _pip_tensorrt_cuda_variant_majors(distribution_names)
    if not variant_majors:
        return {
            "name": "TensorRT CUDA binding alignment",
            "passed": True,
            "detail": "",
        }

    if expected_major in variant_majors:
        return {
            "name": "TensorRT CUDA binding alignment",
            "passed": True,
            "detail": "",
        }

    found = ", ".join(f"cu{major}" for major in sorted(variant_majors))
    return {
        "name": "TensorRT CUDA binding alignment",
        "passed": False,
        "detail": (
            "arm64 host with incompatible pip TensorRT CUDA variant "
            f"[{found}] while expected is cu{expected_major}. "
            "Remove pip tensorrt_cu* packages and use Jetson system/container TensorRT binding, "
            "or set TENSORRT_EXPECTED_CUDA_MAJOR for the intended baseline."
        ),
    }


def check_falldata_aux_wiring(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
    env_example_text: str | None = None,
    jetson_env_example_text: str | None = None,
) -> dict[str, Any]:
    """falldata 보조 배포가 현재 사용하는 인라인 Pose RF를 안전하게 참조하는지 확인한다."""
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
            "FALLDATA_AUX_INLINE_POSE_RF: ${FALLDATA_AUX_INLINE_POSE_RF:-true}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "FALLDATA_AUX_COMPARE_MODEL_PATH: ${FALLDATA_AUX_COMPARE_MODEL_PATH:-models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "FALLDATA_AUX_COMPARE_MODEL_KIND: ${FALLDATA_AUX_COMPARE_MODEL_KIND:-yolo_pose_rf}",
        ),
        (
            "docker-compose.jetson.yml",
            jetson,
            "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH: ${FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH:-}",
        ),
        (".env.example", env_example, "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true"),
        (".env.jetson.example", jetson_env, "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true"),
        (".env.jetson.example", jetson_env, "FALLDATA_AUX_CONFIRM_BORDERLINE=true"),
        (".env.jetson.example", jetson_env, "FALLDATA_AUX_INLINE_POSE_RF=true"),
        (
            ".env.jetson.example",
            jetson_env,
            "FALLDATA_AUX_COMPARE_MODEL_PATH=models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl",
        ),
        (".env.jetson.example", jetson_env, "FALLDATA_AUX_COMPARE_MODEL_KIND=yolo_pose_rf"),
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
        "detail": "fail-open policy and Jetson inline pose RF are wired",
    }


def check_h264_webrtc_wiring(
    *,
    compose_text: str | None = None,
    jetson_compose_text: str | None = None,
) -> dict[str, Any]:
    """Jetson NVENC 출력이 WebRTC POC 호환성 보호 조건을 유지하는지 확인한다."""
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
    """Public API가 기본값으로 외부에 직접 바인딩되지 않는지 확인한다."""
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
    """데모 UI와 Public API가 환경 파일 간에 동일한 실행 보안 값을 사용하는지 확인한다."""
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
    """Docker 시작 전에 흔한 MQTT 인증 배포 실수를 찾아낸다."""
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
        check_tensorrt_binding_cuda_alignment(),
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
        check_edgex_mode_rollback_contract(),
    ]


def check_candidate_profile(config: dict[str, Any], *, check_paths: bool = True) -> dict[str, Any]:
    """병합된 후보 Compose의 영속 저장·고정 코드·비밀값 연결만 검사한다."""
    # 비밀값을 포함하는 config는 메모리에서만 읽고 실패에도 값이나 전체 명령을 출력하지 않는다.
    failures: list[str] = []
    service = config.get("services", {}).get("cctv-ai-engine", {})
    environment = service.get("environment", {})
    mounts = service.get("volumes", [])
    by_target = {mount["target"]: mount for mount in mounts}
    # tmpfs 별칭과 다른 컨테이너의 볼륨 상속도 영속 경로를 가릴 수 있어 후보에서 허용하지 않는다.
    if service.get("tmpfs") or service.get("volumes_from"):
        failures.append("tmpfs and volumes_from are not allowed in candidate")
    # 후보 Compose와 동일한 실검증 이미지여야 health/계측 보완이 함께 배포된다.
    expected_image = "sha256:0c11b835eb42cd245f4dda0b8bc8088733c0323bf75a2f4fefe9e7705d8dbf34"
    if service.get("image") != expected_image or service.get("build"):
        failures.append("tested image required; build must be absent")
    if service.get("pull_policy") != "never":
        failures.append("pull_policy must be never")
    # 정규화된 Compose JSON은 60초를 1m0s로 표시한다. 회수 중 강제 종료되는 후보를 막는다.
    if service.get("stop_grace_period") not in ("1m0s", "60s"):
        failures.append("candidate stop_grace_period must be 60 seconds")
    for key in ("MQTT_USER", "MQTT_PASSWORD", "INTERNAL_SERVICE_TOKEN"):
        if not str(environment.get(key) or "").strip():
            failures.append(f"missing secret: {key}")
    if environment.get("STREAM_API_ENABLED") != "0":
        failures.append("candidate STREAM_API_ENABLED must be 0")
    # 검증 후보는 CPU OSD로 첫 버퍼의 GPU 초기 지연을 우회한다. 누락 시 SDK 기본값으로 되돌아간다.
    if environment.get("DS_OSD_PROCESS_MODE") != "0":
        failures.append("candidate DS_OSD_PROCESS_MODE must be 0")
    for key, name in (
        ("APPEARANCES_DB", "appearances.db"),
        ("MQTT_EVENT_OUTBOX_DB", "mqtt_event_outbox.db"),
        ("COMMERCIAL_FACE_DATABASE", "commercial_faces.db"),
    ):
        if environment.get(key) != f"/app/data/runtime/{name}":
            failures.append(f"durable runtime path required: {key}")
    # 상위 경로도 비교해 /app 전체나 파일 단위 마운트가 검증 코드를 덮지 못하게 한다.
    frozen_paths = ("/app/src", "/app/scripts", "/app/config", "/app/main.py",
                    "/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstrtspclientsink.so")
    for mount in mounts:
        target = Path(mount["target"])
        if any(target == Path(p) or target in Path(p).parents or Path(p) in target.parents
               for p in frozen_paths):
            failures.append("mount shadows tested code/config/plugin")
        if Path("/app/data") in target.parents:
            failures.append("nested data mount may shadow durable databases")
        if mount.get("type") == "bind":
            if mount.get("bind", {}).get("create_host_path", True):
                failures.append("bind must not auto-create missing source")
            if check_paths and not Path(mount["source"]).exists():
                failures.append(f"missing bind source for {mount['target']}")
    required_mounts = {
        "/app/data": False, "/app/models": True, "/app/cameras.json": False,
        "/app/zones_config.json": True, "/app/known_faces.json": False,
        "/app/known_faces": False, "/root/.insightface": False,
        "/app/.venv-jetson-train": True, "/app/.venv-mediapipe": True,
        "/app/.venv-falldata": True, "/opt/nvidia/deepstream": True,
        "/usr/lib/aarch64-linux-gnu/nvidia": True, "/tmp/.X11-unix": False,
    }
    for target, read_only in required_mounts.items():
        mount = by_target.get(target, {})
        if mount.get("type") != "bind" or bool(mount.get("read_only")) != read_only:
            failures.append(f"required bind/read_only mismatch: {target}")
    cache = by_target.get("/app/models/trt_cache", {})
    if (cache.get("type") != "volume" or cache.get("source") != "trt-cache"
            or cache.get("read_only")):
        failures.append("persistent writable trt-cache volume required")
    # Docker는 읽기 전용 상위 bind 안에 중첩 볼륨 연결 디렉터리를 새로 만들 수 없다.
    model_source = by_target.get("/app/models", {}).get("source", "")
    if check_paths and not (Path(model_source) / "trt_cache").is_dir():
        failures.append("readonly models require existing trt_cache mountpoint")
    # 실제 운영 데이터 루트를 유지하는지 확인한다. 격리 시험은 검사를 마친 뒤 별도 경로로 치환한다.
    data_source = by_target.get("/app/data", {}).get("source", "")
    if Path(data_source).resolve() != (PROJECT_ROOT / "data").resolve():
        failures.append("operating data root must be preserved")
    for target in ("/app/cameras.json", "/app/zones_config.json", "/app/known_faces.json"):
        source = by_target.get(target, {}).get("source", "")
        if check_paths and not Path(source).is_file():
            failures.append(f"file bind required: {target}")
    if "edgex-network" not in service.get("networks", {}) or service.get("network_mode"):
        failures.append("operating edgex-network required")
    return {"name": "candidate profile contract", "passed": not failures,
            "detail": "; ".join(dict.fromkeys(failures))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check runtime assumptions for CCTV compose deployment.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--candidate-compose-json", metavar="PATH",
                        help="Validate merged candidate Compose JSON; '-' reads stdin without saving secrets.")
    args = parser.parse_args()

    # 후보 검사만 선택하면 기존 전체 스택 검사를 실행하거나 운영 서비스를 변경하지 않는다.
    if args.candidate_compose_json:
        try:
            raw = (sys.stdin.read() if args.candidate_compose_json == "-"
                   else Path(args.candidate_compose_json).read_text(encoding="utf-8"))
            checks = [check_candidate_profile(json.loads(raw))]
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            checks = [{"name": "candidate profile contract", "passed": False,
                       "detail": "invalid or unreadable Compose JSON"}]
    else:
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
