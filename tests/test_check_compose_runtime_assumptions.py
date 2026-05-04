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
    "scripts/check_compose_runtime_assumptions.py",
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
    assert "docker-compose.arm64.yml" in result["detail"]


def test_default_compose_architecture_passes_on_arm64_with_platform_override():
    result = runtime_checks.check_default_compose_architecture(
        machine="aarch64",
        compose_text="platform: linux/arm64\nimage: edgexfoundry/core-data:3.1.0",
    )
    assert result["passed"] is True


def test_default_compose_architecture_passes_on_arm64_with_override_file():
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
    assert result["passed"] is True
    assert "docker-compose.arm64.yml" in result["detail"]
    assert "UI is excluded" in result["detail"]


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
