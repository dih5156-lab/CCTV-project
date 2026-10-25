import json
import os
import shutil
import subprocess

import pytest

from scripts.ops import compose_stack


def test_jetson_compose_does_not_inherit_host_display():
    compose_text = (compose_stack.PROJECT_ROOT / "docker-compose.jetson.yml").read_text()

    assert "DISPLAY: ${DS_DISPLAY:-" in compose_text
    assert "DISPLAY: ${DISPLAY:-}" not in compose_text


@pytest.mark.parametrize("enabled", ["false", "true"])
def test_jetson_compose_wires_temporal_fall_shadow_runtime(tmp_path, enabled):
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose 실행 도구 필요")
    compose_text = (compose_stack.PROJECT_ROOT / "docker-compose.jetson.yml").read_text()
    compose_file = tmp_path / "docker-compose.jetson.yml"
    compose_file.write_text(compose_text)
    settings = {
        "FALLDATA_AUX_TEMPORAL_COMPARE_MODEL_PATH": "/app/models/test-temporal.pt",
        "FALLDATA_AUX_TEMPORAL_PYTHON": "/app/.venv-jetson-train/bin/python",
        "FALLDATA_AUX_CLIP_BUFFER_ENABLED": enabled,
        "FALLDATA_AUX_CLIP_BUFFER_WIDTH": "320",
        "FALLDATA_AUX_CLIP_BUFFER_HEIGHT": "180",
        "FALLDATA_AUX_MIN_POSE_FRAMES": "12",
    }
    env_file = tmp_path / ".env.jetson"
    env_file.write_text("\n".join(f"{key}={value}" for key, value in settings.items()) + "\n")
    asc_env = tmp_path / "edgex/asc/cctv_asc.env"
    asc_env.parent.mkdir(parents=True)
    asc_env.write_text("")
    # env_file 경로의 정상 전달도 허용한다. 실제 비밀 파일 대신 임시 배포 구성을 해석한다.
    result = subprocess.run(
        ["docker", "compose", "--project-directory", str(tmp_path),
         "--env-file", str(env_file), "-f", str(compose_file), "config", "--format", "json"],
        env={key: value for key, value in os.environ.items() if key not in settings},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    service = json.loads(result.stdout)["services"]["cctv-ai-engine"]
    for key, value in settings.items():
        assert service["environment"][key] == value
    mounts = {mount["target"]: mount for mount in service["volumes"]}
    assert mounts["/app/.venv-jetson-train"]["source"] == str(tmp_path / ".venv-jetson-train")


def test_jetson_systemd_unit_uses_compose_service_names():
    unit_text = (
        compose_stack.PROJECT_ROOT / "deploy/systemd/cctv-jetson-stack.service"
    ).read_text()

    assert "cctv-ai-engine cctv-public-api public-demo-ui" in unit_text
    assert "cctv-ai-engine cctv-public-api cctv-public-demo-ui" not in unit_text


@pytest.mark.parametrize("board", ["agx", "nx16", "nx16_custom"])
def test_jetson_board_paths_preserve_runtime_contract(tmp_path, board):
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose 실행 도구 필요")
    root = compose_stack.PROJECT_ROOT
    compose_file = tmp_path / "docker-compose.jetson.yml"
    compose_file.write_text((root / compose_file.name).read_text())
    settings = {}
    if board.startswith("nx16"):
        for line in (root / ".env.jetson.nx16.example").read_text().splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                settings[key] = value
    if board == "nx16_custom":
        settings["MODELS_HOST_PATH"] = str(tmp_path / "NX models")
        settings["JETSON_TORCH_VENV_HOST_PATH"] = str(tmp_path / "NX torch venv")
    env_file = tmp_path / ".env.jetson"
    env_file.write_text("\n".join(f"{key}={value}" for key, value in settings.items()) + "\n")
    asc_env = tmp_path / "edgex/asc/cctv_asc.env"
    asc_env.parent.mkdir(parents=True)
    asc_env.write_text("")
    path_keys = {"MODELS_HOST_PATH", "DEEPSTREAM_CONFIG_HOST_PATH", "JETSON_TORCH_VENV_HOST_PATH", "TRT_CACHE_VOLUME"}
    result = subprocess.run(
        ["docker", "compose", "--project-directory", str(tmp_path), "--env-file", str(env_file),
         "-f", str(compose_file), "config", "--format", "json"],
        env={key: value for key, value in os.environ.items() if key not in path_keys},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    service = config["services"]["cctv-ai-engine"]
    mounts = {mount["target"]: mount for mount in service["volumes"]}
    for target, key, default in [
        ("/app/models", "MODELS_HOST_PATH", "./models"),
        ("/app/config/deepstream", "DEEPSTREAM_CONFIG_HOST_PATH", "./config/deepstream"),
        ("/app/.venv-jetson-train", "JETSON_TORCH_VENV_HOST_PATH", "./.venv-jetson-train"),
    ]:
        assert mounts[target]["source"] == str(tmp_path / settings.get(key, default))
        assert mounts[target].get("bind", {}).get("create_host_path", False) is False
    assert config["volumes"]["trt-cache"]["name"] == settings.get("TRT_CACHE_VOLUME", "edgex-jetson_trt-cache")
    assert config["volumes"]["trt-cache"]["external"] is True
    assert mounts["/app/models/trt_cache"]["source"] == "trt-cache"
    # 보드 경로 선택이 추론/기능 설정을 바꾸지 않아야 한다.
    assert service["environment"]["PYTHONPATH"] == "/app/.venv-jetson-train/lib/python3.10/site-packages:/app"
    assert "DS_HELMET_ENABLED" not in settings
    assert "--confidence" in service["command"]


def test_detects_jetson_from_model_marker(tmp_path):
    marker = tmp_path / "model"
    marker.write_text("NVIDIA Jetson AGX Orin")
    assert compose_stack.is_jetson_host(
        system="Linux", machine="aarch64", marker_paths=[marker]
    )


def test_arm_server_is_not_jetson_without_marker(tmp_path):
    assert not compose_stack.is_jetson_host(
        system="Linux", machine="aarch64", marker_paths=[tmp_path / "missing"]
    )


def test_jetson_command_uses_only_jetson_files(monkeypatch, tmp_path):
    monkeypatch.setattr(compose_stack, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env.jetson").write_text("")
    command = compose_stack.build_compose_command("jetson", ["up", "-d"])
    assert "edgex-jetson" in command
    assert str(tmp_path / "docker-compose.jetson.yml") in command
    assert str(tmp_path / ".env.jetson") in command
    assert str(tmp_path / "docker-compose.yml") not in command


def test_windows_command_uses_server_compose(monkeypatch, tmp_path):
    monkeypatch.setattr(compose_stack, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("")
    command = compose_stack.build_compose_command("windows", ["ps"])
    assert "edgex" in command
    assert str(tmp_path / "docker-compose.yml") in command
    assert str(tmp_path / ".env") in command
