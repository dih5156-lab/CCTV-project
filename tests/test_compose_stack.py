from scripts.ops import compose_stack


def test_jetson_compose_does_not_inherit_host_display():
    compose_text = (compose_stack.PROJECT_ROOT / "docker-compose.jetson.yml").read_text()

    assert "DISPLAY: ${DS_DISPLAY:-}" in compose_text
    assert "DISPLAY: ${DISPLAY:-}" not in compose_text


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
