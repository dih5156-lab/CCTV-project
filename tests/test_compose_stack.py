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
