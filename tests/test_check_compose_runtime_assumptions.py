import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


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


def test_jetson_compose_keeps_signboard_recreation_path():
    # 실행 중인 장치가 Compose에서 누락되면 재생성 시 복구할 수 없으므로 원본 정의를 검사한다.
    compose = (runtime_checks.PROJECT_ROOT / "docker-compose.jetson.yml").read_text(encoding="utf-8")
    service = runtime_checks._compose_service_block(compose, "cctv-device-signboard")
    assert "runners/run_signboard_device_service.py" in service
    assert "target: action" in service
    assert "path: ./.env.jetson" in service
    assert "edgex-mqtt-broker:" in service
    assert "edgex-network:" in service
    # 장치 실행 이력이 이미지 교체 시 초기화되면 재전송으로 실물이 반복 작동한다.
    assert "./data/device_requests:/app/data/device_requests" in service
    assert runtime_checks._compose_service_env_value(
        compose, "cctv-device-signboard", "DEVICE_REQUEST_DB_PATH"
    ) == "/app/data/device_requests/signboard.sqlite3"
    # 빈 Compose 보간값으로 env_file의 장치 주소·인증값을 덮어쓰지 않는다.
    for key in ("SIGNBOARD_HOST", "MQTT_USER", "MQTT_PASSWORD"):
        assert runtime_checks._compose_service_env_value(compose, "cctv-device-signboard", key) == ""


@pytest.mark.parametrize(
    ("mode", "devices", "override", "passed"),
    [
        ("edgex", "speaker,signboard,siren", "", True),
        ("", "speaker,signboard,siren", "", False),
        ("shadow", "speaker,signboard,siren", "", False),
        ("edgex", "", "", False),
        ("edgex", "speaker,signboard,siren,unknown", "", False),
        ("edgex", "speaker,signboard,siren", "      EDGEX_COMMAND_MODE: shadow\n", False),
        ("edgex", "speaker,signboard,siren", "      EDGEX_COMMAND_MODE:\n", False),
    ],
)
def test_edgex_deployment_profile_checks_env_file_and_override(mode, devices, override, passed):
    # 현행 env_file 전달과 과거 Direct 복귀를 구분하되 environment 우선순위로 인한 우회를 막는다.
    base = (
        "services:\n  cctv-action-layer:\n    env_file:\n"
        "      - path: ./.env.jetson\n        required: false\n"
        "    environment:\n" + override
    )
    result = runtime_checks.check_edgex_mode_rollback_contract(
        profile="deployment",
        jetson_compose_text=base,
        uat_compose_text=("services:\n  cctv-action-layer:\n    environment:\n"
                          "      EDGEX_COMMAND_MODE: edgex\n      EDGEX_ALLOWED_DEVICES: signboard\n"),
        env_text="",
        jetson_env_text=f"EDGEX_COMMAND_MODE={mode}\nEDGEX_ALLOWED_DEVICES={devices}\n",
    )
    assert result["passed"] is passed


@pytest.mark.parametrize("missing", ["env_file", "uat_limit"])
def test_edgex_deployment_rejects_missing_env_source_or_broader_uat(missing):
    base = "services:\n  cctv-action-layer:\n"
    if missing != "env_file":
        base += "    env_file:\n      - path: ./.env.jetson\n"
    devices = "speaker,signboard" if missing == "uat_limit" else "signboard"
    result = runtime_checks.check_edgex_mode_rollback_contract(
        profile="deployment", jetson_compose_text=base,
        uat_compose_text=("services:\n  cctv-action-layer:\n    environment:\n"
                          f"      EDGEX_COMMAND_MODE: edgex\n      EDGEX_ALLOWED_DEVICES: {devices}\n"),
        env_text="", jetson_env_text="EDGEX_COMMAND_MODE=edgex\nEDGEX_ALLOWED_DEVICES=speaker,signboard,siren\n",
    )
    assert not result["passed"]


def test_jetson_model_defaults_match_deployment_contract():
    # 저장소 기본값 자체가 회귀하면 합성 문자열 시험이 통과해도 실제 배포 검사에서 잡는다.
    assert runtime_checks.check_appearance_model_wiring()["passed"]
    assert runtime_checks.check_falldata_aux_wiring()["passed"]


@pytest.mark.parametrize("profile", ["deployment", "direct-rollback"])
def test_cli_selects_edgex_profile(monkeypatch, capsys, profile):
    seen = []
    monkeypatch.setattr(sys, "argv", ["check", "--json", "--edgex-profile", profile])
    monkeypatch.setattr(runtime_checks, "run_checks", lambda **kwargs: seen.append(kwargs) or [])
    assert runtime_checks.main() == 0
    assert seen == [{"edgex_profile": profile}]
    assert '"passed": true' in capsys.readouterr().out


def test_edgex_mode_rollback_contract_accepts_direct_base_and_signboard_uat():
    base = """
services:
  cctv-action-layer:
    environment:
      EDGEX_COMMAND_MODE: ${EDGEX_COMMAND_MODE:-}
      EDGEX_ALLOWED_DEVICES: ${EDGEX_ALLOWED_DEVICES:-}
"""
    uat = """
services:
  cctv-action-layer:
    environment:
      EDGEX_COMMAND_MODE: edgex
      EDGEX_ALLOWED_DEVICES: signboard
"""

    result = runtime_checks.check_edgex_mode_rollback_contract(
        jetson_compose_text=base,
        uat_compose_text=uat,
        # 단위 검사는 개발자 PC의 실제 EdgeX 운영 모드에 좌우되지 않게 한다.
        env_text="",
        jetson_env_text="",
    )

    assert result["passed"] is True


def test_jetson_compose_loads_pose_crop_opt_in_from_env_file():
    compose = (Path(__file__).resolve().parents[1] / "docker-compose.jetson.yml").read_text(
        encoding="utf-8"
    )
    # 현재 옵션은 서비스 env_file로 전달된다. 삭제된 Compose 직접 선언을 요구하지 않는다.
    service = runtime_checks._compose_service_block(compose, "cctv-ai-engine")
    assert "env_file:" in service
    assert "path: ./.env.jetson" in service

    env_example = (Path(__file__).resolve().parents[1] / ".env.jetson.example").read_text(
        encoding="utf-8"
    )
    assert "POSE_CROP_MODE=0" in env_example
    assert "DS_CROP_POSE_ENABLED=0" in env_example


def test_edgex_mode_rollback_contract_rejects_edge_x_as_base_default():
    base = """
services:
  cctv-action-layer:
    environment:
      EDGEX_COMMAND_MODE: edgex
      EDGEX_ALLOWED_DEVICES: signboard
"""
    uat = """
services:
  cctv-action-layer:
    environment:
      EDGEX_COMMAND_MODE: edgex
      EDGEX_ALLOWED_DEVICES: signboard
"""

    result = runtime_checks.check_edgex_mode_rollback_contract(
        jetson_compose_text=base,
        uat_compose_text=uat,
    )

    assert result["passed"] is False
    assert "EDGEX_COMMAND_MODE" in result["detail"]


def test_edgex_mode_rollback_contract_rejects_env_override():
    base = """
services:
  cctv-action-layer:
    environment:
      EDGEX_COMMAND_MODE: ${EDGEX_COMMAND_MODE:-}
      EDGEX_ALLOWED_DEVICES: ${EDGEX_ALLOWED_DEVICES:-}
"""
    uat = """
services:
  cctv-action-layer:
    environment:
      EDGEX_COMMAND_MODE: edgex
      EDGEX_ALLOWED_DEVICES: signboard
"""

    result = runtime_checks.check_edgex_mode_rollback_contract(
        jetson_compose_text=base,
        uat_compose_text=uat,
        env_text="EDGEX_COMMAND_MODE=edgex\n",
        jetson_env_text="",
    )

    assert result["passed"] is False
    assert ".env must not override" in result["detail"]


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


def test_default_compose_architecture_passes_for_jetson_target(monkeypatch):
    monkeypatch.setenv("CCTV_DEPLOYMENT_TARGET", "jetson")
    result = runtime_checks.check_default_compose_architecture(
        machine="aarch64",
        compose_text="image: edgexfoundry/core-data:3.1.0",
    )
    assert result["passed"] is True
    assert "Jetson target" in result["detail"]


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


def test_tensorrt_binding_cuda_alignment_fails_on_arm64_mismatch():
    result = runtime_checks.check_tensorrt_binding_cuda_alignment(
        machine="aarch64",
        distribution_names=["tensorrt", "tensorrt_cu13"],
        expected_cuda_major="12",
    )
    assert result["passed"] is False
    assert "incompatible pip TensorRT CUDA variant" in result["detail"]


def test_tensorrt_binding_cuda_alignment_passes_on_arm64_match():
    result = runtime_checks.check_tensorrt_binding_cuda_alignment(
        machine="aarch64",
        distribution_names=["tensorrt", "tensorrt_cu12"],
        expected_cuda_major="12",
    )
    assert result["passed"] is True


def test_tensorrt_binding_cuda_alignment_ignored_on_amd64():
    result = runtime_checks.check_tensorrt_binding_cuda_alignment(
        machine="x86_64",
        distribution_names=["tensorrt_cu13"],
        expected_cuda_major="12",
    )
    assert result["passed"] is True


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
  APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/appearance/pphuman_attribute.onnx}
  APPEARANCE_LABEL_MAP_PATH: ${APPEARANCE_LABEL_MAP_PATH:-config/appearance_pphuman_labels.example.json}
  APPEARANCE_RUNTIME: ${APPEARANCE_RUNTIME:-onnxruntime}
"""
    jetson = """
x-appearance-runtime:
  DS_PPHUMAN_SGIE_ENABLED: ${DS_PPHUMAN_SGIE_ENABLED:-1}
  DS_PPHUMAN_INFER_CONFIG: ${DS_PPHUMAN_INFER_CONFIG:-config/deepstream/config_infer_pa100k.txt}
  APPEARANCE_MODEL_PATH: ${APPEARANCE_MODEL_PATH:-models/appearance/pa100k_resnet50_attr.engine}
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


def test_falldata_aux_wiring_requires_fail_open_and_inline_pose_rf():
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
      FALLDATA_AUX_INLINE_POSE_RF: ${FALLDATA_AUX_INLINE_POSE_RF:-true}
      FALLDATA_AUX_COMPARE_MODEL_PATH: ${FALLDATA_AUX_COMPARE_MODEL_PATH:-models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl}
      FALLDATA_AUX_COMPARE_MODEL_KIND: ${FALLDATA_AUX_COMPARE_MODEL_KIND:-yolo_pose_rf}
      FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH: ${FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH:-}
"""
    result = runtime_checks.check_falldata_aux_wiring(
        compose_text=compose,
        jetson_compose_text=jetson,
        env_example_text="FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true\n",
        jetson_env_example_text=(
            "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true\n"
            "FALLDATA_AUX_CONFIRM_BORDERLINE=true\n"
            "FALLDATA_AUX_INLINE_POSE_RF=true\n"
            "FALLDATA_AUX_COMPARE_MODEL_PATH="
            "models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl\n"
            "FALLDATA_AUX_COMPARE_MODEL_KIND=yolo_pose_rf\n"
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


def test_falldata_aux_wiring_fails_without_disabled_feature_capture_default() -> None:
    result = runtime_checks.check_falldata_aux_wiring(
        compose_text=(
            "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE: "
            "${FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE:-true}\n"
        ),
        jetson_compose_text=(
            "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE: "
            "${FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE:-true}\n"
            "FALLDATA_AUX_INLINE_POSE_RF: ${FALLDATA_AUX_INLINE_POSE_RF:-true}\n"
            "FALLDATA_AUX_COMPARE_MODEL_PATH: "
            "${FALLDATA_AUX_COMPARE_MODEL_PATH:-models/experiments/"
            "yolo_pose_fall_cam2_continuous_200_80_640.pkl}\n"
            "FALLDATA_AUX_COMPARE_MODEL_KIND: "
            "${FALLDATA_AUX_COMPARE_MODEL_KIND:-yolo_pose_rf}\n"
        ),
        env_example_text="FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true\n",
        jetson_env_example_text=(
            "FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE=true\n"
            "FALLDATA_AUX_CONFIRM_BORDERLINE=true\n"
            "FALLDATA_AUX_INLINE_POSE_RF=true\n"
            "FALLDATA_AUX_COMPARE_MODEL_PATH="
            "models/experiments/yolo_pose_fall_cam2_continuous_200_80_640.pkl\n"
            "FALLDATA_AUX_COMPARE_MODEL_KIND=yolo_pose_rf\n"
        ),
    )

    assert result["passed"] is False
    assert "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH" in result["detail"]


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


def _candidate_config(tmp_path):
    # 실제 비밀값·운영 DB를 사용하지 않고 검사 경계만 구성한다.
    readonly = {"/app/models", "/app/zones_config.json", "/app/.venv-jetson-train",
                "/app/.venv-mediapipe", "/app/.venv-falldata", "/opt/nvidia/deepstream",
                "/usr/lib/aarch64-linux-gnu/nvidia"}
    writable = {"/app/data", "/app/cameras.json", "/app/known_faces.json",
                "/app/known_faces", "/root/.insightface", "/tmp/.X11-unix"}
    mounts = []
    for target in sorted(readonly | writable):
        source = tmp_path / target.lstrip("/")
        source.parent.mkdir(parents=True, exist_ok=True)
        if target.endswith(".json"):
            source.write_text("{}")
        else:
            source.mkdir(exist_ok=True)
        if target == "/app/data":
            source = runtime_checks.PROJECT_ROOT / "data"
        mounts.append({"type": "bind", "source": str(source), "target": target,
                       "read_only": target in readonly, "bind": {"create_host_path": False}})
    mounts.append({"type": "volume", "source": "trt-cache", "target": "/app/models/trt_cache"})
    # 읽기 전용 모델 원본 안에 중첩 캐시 볼륨의 연결 지점이 미리 있어야 한다.
    (tmp_path / "app/models/trt_cache").mkdir(exist_ok=True)
    return {"services": {"cctv-ai-engine": {
        "image": "sha256:0c11b835eb42cd245f4dda0b8bc8088733c0323bf75a2f4fefe9e7705d8dbf34",
        "pull_policy": "never", "networks": {"edgex-network": {}}, "volumes": mounts,
        "stop_grace_period": "1m0s",
        "environment": {"MQTT_USER": "test", "MQTT_PASSWORD": "private-test-value",
                        "INTERNAL_SERVICE_TOKEN": "private-test-value", "STREAM_API_ENABLED": "0",
                        "DS_OSD_PROCESS_MODE": "0",
                        "APPEARANCES_DB": "/app/data/runtime/appearances.db",
                        "MQTT_EVENT_OUTBOX_DB": "/app/data/runtime/mqtt_event_outbox.db",
                        "COMMERCIAL_FACE_DATABASE": "/app/data/runtime/commercial_faces.db"},
    }}}


def test_candidate_accepts_existing_durable_mounts(tmp_path):
    assert runtime_checks.check_candidate_profile(_candidate_config(tmp_path))["passed"]


def test_candidate_accepts_explicit_image_id_without_changing_default(tmp_path):
    config = _candidate_config(tmp_path)
    image_id = "sha256:" + "a" * 64
    config["services"]["cctv-ai-engine"]["image"] = image_id
    assert not runtime_checks.check_candidate_profile(config)["passed"]
    assert runtime_checks.check_candidate_profile(config, expected_image_id=image_id)["passed"]


@pytest.mark.parametrize("image_id", ["", "cctv-ai:latest", "sha256:abc", "sha256:" + "z" * 64])
def test_candidate_rejects_mutable_or_invalid_expected_image(tmp_path, image_id):
    config = _candidate_config(tmp_path)
    config["services"]["cctv-ai-engine"]["image"] = image_id
    result = runtime_checks.check_candidate_profile(config, expected_image_id=image_id)
    assert not result["passed"]
    assert "private-test-value" not in str(result)


def test_candidate_new_image_still_enforces_data_and_code_protection(tmp_path):
    config = _candidate_config(tmp_path)
    image_id = "sha256:" + "b" * 64
    config["services"]["cctv-ai-engine"]["image"] = image_id
    config["services"]["cctv-ai-engine"]["volumes"].append(
        {"type": "bind", "source": str(tmp_path), "target": "/app/data/runtime"})
    result = runtime_checks.check_candidate_profile(config, expected_image_id=image_id)
    assert not result["passed"]
    assert "nested data mount" in result["detail"]


def test_candidate_cli_checks_selected_image_without_printing_config(tmp_path, monkeypatch, capsys):
    config = _candidate_config(tmp_path)
    image_id = "sha256:" + "c" * 64
    config["services"]["cctv-ai-engine"]["image"] = image_id
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(config)))
    monkeypatch.setattr(sys, "argv", ["check", "--candidate-compose-json", "-",
                                     "--expected-image-id", image_id, "--json"])
    assert runtime_checks.main() == 0
    output = capsys.readouterr().out
    assert json.loads(output)["passed"]
    assert "private-test-value" not in output


def test_candidate_rejects_gpu_or_unspecified_osd(tmp_path):
    # GPU OSD 첫 버퍼 지연으로 매칭이 만료된 구성을 검증 후보로 통과시키지 않는다.
    for mode in (None, "", "1"):
        config = _candidate_config(tmp_path)
        environment = config["services"]["cctv-ai-engine"]["environment"]
        if mode is None:
            environment.pop("DS_OSD_PROCESS_MODE")
        else:
            environment["DS_OSD_PROCESS_MODE"] = mode
        result = runtime_checks.check_candidate_profile(config)
        assert not result["passed"], mode
        assert "DS_OSD_PROCESS_MODE" in result["detail"]
        assert "private-test-value" not in result["detail"]


def test_candidate_rejects_disabled_outbox_without_leaking_secrets(tmp_path):
    config = _candidate_config(tmp_path)
    env = config["services"]["cctv-ai-engine"]["environment"]
    del env["MQTT_EVENT_OUTBOX_DB"]
    result = runtime_checks.check_candidate_profile(config)
    assert not result["passed"]
    assert "MQTT_EVENT_OUTBOX_DB" in result["detail"]
    assert "private-test-value" not in str(result)


def test_candidate_rejects_secret_missing(tmp_path):
    config = _candidate_config(tmp_path)
    config["services"]["cctv-ai-engine"]["environment"]["MQTT_PASSWORD"] = " "
    assert not runtime_checks.check_candidate_profile(config)["passed"]


def test_candidate_rejects_short_or_missing_stop_grace(tmp_path):
    # 정상 종료가 기본 유예보다 길었던 실측을 반영해 명시적인 60초 계약을 유지한다.
    for value in (None, "10s"):
        config = _candidate_config(tmp_path)
        config["services"]["cctv-ai-engine"]["stop_grace_period"] = value
        assert not runtime_checks.check_candidate_profile(config)["passed"]


def test_candidate_rejects_missing_cache_mountpoint_in_readonly_models(tmp_path):
    config = _candidate_config(tmp_path)
    (tmp_path / "app/models/trt_cache").rmdir()
    result = runtime_checks.check_candidate_profile(config)
    assert not result["passed"]
    assert "trt_cache" in result["detail"]


def test_candidate_rejects_readonly_database_mount(tmp_path):
    config = _candidate_config(tmp_path)
    for mount in config["services"]["cctv-ai-engine"]["volumes"]:
        if mount["target"] == "/app/data":
            mount["read_only"] = True
    assert not runtime_checks.check_candidate_profile(config)["passed"]


def test_candidate_rejects_shadowed_image_and_database(tmp_path):
    for target in ("/app", "/app/src/core", "/app/config", "/app/data/runtime"):
        config = _candidate_config(tmp_path)
        config["services"]["cctv-ai-engine"]["volumes"].append(
            {"type": "bind", "source": str(tmp_path), "target": target})
        assert not runtime_checks.check_candidate_profile(config)["passed"], target


def test_candidate_rejects_missing_mount_or_automatic_directory(tmp_path):
    for mutation in ("missing", "auto_create"):
        config = _candidate_config(tmp_path)
        mount = config["services"]["cctv-ai-engine"]["volumes"][0]
        if mutation == "missing":
            mount["source"] = str(tmp_path / "missing")
        else:
            mount["bind"]["create_host_path"] = True
        assert not runtime_checks.check_candidate_profile(config)["passed"]


def test_candidate_rejects_build_tag_or_isolated_network(tmp_path):
    for key, value in (("build", {"context": "."}), ("image", "cctv-ai:latest"),
                       ("network_mode", "none"), ("tmpfs", ["/app/data"]),
                       ("volumes_from", ["another-service"])):
        config = _candidate_config(tmp_path)
        config["services"]["cctv-ai-engine"][key] = value
        assert not runtime_checks.check_candidate_profile(config)["passed"], key
