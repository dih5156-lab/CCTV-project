import importlib.util
import socket
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


check_alarm_devices = _load_script_module(
    "check_alarm_devices",
    "scripts/check_alarm_devices.py",
)


def test_reports_missing_required_environment_variables():
    result = check_alarm_devices.run_checks(env={}, skip_network=True)

    assert result["passed"] is False
    speaker = next(item for item in result["checks"] if item["name"] == "speaker")
    assert speaker["configured"] is False
    assert speaker["missing_env"] == [
        "SPEAKER_HOST",
        "SPEAKER_USER",
        "SPEAKER_PASSWORD",
    ]


def test_skip_network_passes_when_all_devices_are_configured():
    env = {
        "SPEAKER_HOST": "192.0.2.10",
        "SPEAKER_USER": "admin",
        "SPEAKER_PASSWORD": "secret",
        "SIREN_HOST": "192.0.2.11",
        "SIREN_USER": "admin",
        "SIREN_PASSWORD": "secret",
        "SIGNBOARD_HOST": "192.0.2.12",
    }

    result = check_alarm_devices.run_checks(env=env, skip_network=True)

    assert result["passed"] is True
    assert all(item["configured"] for item in result["checks"])
    assert all(item["reachable"] is None for item in result["checks"])


def test_allow_unconfigured_returns_success_for_optional_devices():
    result = check_alarm_devices.run_checks(
        env={},
        skip_network=True,
        allow_unconfigured=True,
    )

    assert result["passed"] is True


def test_invalid_port_marks_device_unconfigured():
    env = {
        "SIGNBOARD_HOST": "192.0.2.12",
        "SIGNBOARD_PORT": "bad",
    }

    result = check_alarm_devices.run_device_check(
        check_alarm_devices.DEVICE_CHECKS[-1],
        env=env,
        timeout=0.1,
    )

    assert result["configured"] is False
    assert "invalid port" in result["detail"]


def test_load_env_file_parses_simple_key_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "SPEAKER_HOST=192.0.2.10",
                "SPEAKER_USER='admin'",
                'SPEAKER_PASSWORD="secret"',
            ]
        ),
        encoding="utf-8",
    )

    values = check_alarm_devices.load_env_file(env_file)

    assert values["SPEAKER_HOST"] == "192.0.2.10"
    assert values["SPEAKER_USER"] == "admin"
    assert values["SPEAKER_PASSWORD"] == "secret"


def test_reachability_uses_tcp_connection(monkeypatch):
    calls = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    def fake_create_connection(address, timeout):
        calls.append((address, timeout))
        return FakeSocket()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    env = {"SIGNBOARD_HOST": "192.0.2.12", "SIGNBOARD_PORT": "5000"}

    result = check_alarm_devices.run_device_check(
        check_alarm_devices.DEVICE_CHECKS[-1],
        env=env,
        timeout=1.5,
    )

    assert result["reachable"] is True
    assert calls == [(("192.0.2.12", 5000), 1.5)]
