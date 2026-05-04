import importlib.util
import json
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


smoke_test_deployment = _load_script_module(
    "smoke_test_deployment",
    "scripts/smoke_test_deployment.py",
)


def test_run_http_check_passes_on_expected_status_and_text(monkeypatch):
    monkeypatch.setattr(
        smoke_test_deployment,
        "_read_url",
        lambda url, timeout: (True, 200, '{"service":"cctv-alert-api"}'),
    )

    result = smoke_test_deployment.run_http_check(
        smoke_test_deployment.HttpCheck(
            "alert api health",
            "http://localhost:8000/health",
            required_text="cctv-alert-api",
        ),
        timeout=1.0,
    )

    assert result["passed"] is True


def test_run_http_check_fails_on_missing_required_text(monkeypatch):
    monkeypatch.setattr(
        smoke_test_deployment,
        "_read_url",
        lambda url, timeout: (True, 200, '{"service":"other"}'),
    )

    result = smoke_test_deployment.run_http_check(
        smoke_test_deployment.HttpCheck(
            "alert api health",
            "http://localhost:8000/health",
            required_text="cctv-alert-api",
        ),
        timeout=1.0,
    )

    assert result["passed"] is False


def test_build_checks_includes_public_api_readiness():
    checks = smoke_test_deployment.build_checks("localhost")
    urls = {check.name: check.url for check in checks}
    assert urls["public api readiness"] == "http://localhost:9000/api/v1/readiness"


def test_prometheus_targets_require_action_and_public_api_up(monkeypatch):
    payload = {
        "data": {
            "activeTargets": [
                {"labels": {"job": "cctv-action-layer"}, "health": "up"},
                {"labels": {"job": "cctv-public-api"}, "health": "up"},
            ]
        }
    }
    monkeypatch.setattr(
        smoke_test_deployment,
        "_read_url",
        lambda url, timeout: (True, 200, json.dumps(payload)),
    )

    result = smoke_test_deployment.check_prometheus_targets(
        "http://localhost:9090",
        timeout=1.0,
    )

    assert result["passed"] is True


def test_prometheus_targets_report_missing_jobs(monkeypatch):
    payload = {"data": {"activeTargets": []}}
    monkeypatch.setattr(
        smoke_test_deployment,
        "_read_url",
        lambda url, timeout: (True, 200, json.dumps(payload)),
    )

    result = smoke_test_deployment.check_prometheus_targets(
        "http://localhost:9090",
        timeout=1.0,
    )

    assert result["passed"] is False
    assert "missing jobs" in result["detail"]
