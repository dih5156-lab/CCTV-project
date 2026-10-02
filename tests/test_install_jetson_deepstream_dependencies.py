import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ops/install_jetson_deepstream_dependencies.sh"


def test_default_is_simulation_and_does_not_run_runtime_check(tmp_path):
    # 설치 도구의 기본 실행이 실제 패키지를 변경하는 회귀를 막는다.
    for name, body in {
        "uname": "echo aarch64",
        "dpkg-query": "echo 'installed 7.1.0-1'",
        "apt-get": 'printf "%s\\n" "$@" > "$INSTALL_TEST_LOG"',
        "python3": "exit 99",
    }.items():
        binary = tmp_path / name
        binary.write_text("#!/bin/sh\n" + body + "\n")
        binary.chmod(0o755)
    log = tmp_path / "apt-arguments"
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)], capture_output=True, text=True,
        env={**os.environ, "PATH": str(tmp_path) + ":/usr/bin:/bin", "INSTALL_TEST_LOG": str(log)},
    )
    assert result.returncode == 0, result.stderr
    arguments = log.read_text().splitlines()
    assert "--simulate" in arguments
    assert "--no-upgrade" in arguments and "--no-remove" in arguments
    assert "libgstrtspserver-1.0-0" in arguments


def test_missing_sdk_stops_before_package_install(tmp_path):
    for name, body in {
        "uname": "echo aarch64",
        "dpkg-query": "exit 1",
        "apt-get": 'touch "$INSTALL_TEST_LOG"',
    }.items():
        binary = tmp_path / name
        binary.write_text("#!/bin/sh\n" + body + "\n")
        binary.chmod(0o755)
    log = tmp_path / "apt-called"
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)], capture_output=True, text=True,
        env={**os.environ, "PATH": str(tmp_path) + ":/usr/bin:/bin", "INSTALL_TEST_LOG": str(log)},
    )
    assert result.returncode == 1
    assert not log.exists()
    assert "DeepStream 7.1" in result.stderr
