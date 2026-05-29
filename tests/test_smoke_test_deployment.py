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


def test_run_http_check_fails_on_forbidden_text(monkeypatch):
    monkeypatch.setattr(
        smoke_test_deployment,
        "_read_url",
        lambda url, timeout: (True, 200, '<script src="https://cdn.jsdelivr.net/swagger.js"></script>'),
    )

    result = smoke_test_deployment.run_http_check(
        smoke_test_deployment.HttpCheck(
            "public api docs",
            "http://localhost:9000/docs",
            forbidden_texts=("cdn.jsdelivr.net", "unpkg.com"),
        ),
        timeout=1.0,
    )

    assert result["passed"] is False
    assert "forbidden text found" in result["detail"]


def test_build_checks_includes_public_api_readiness_and_docs():
    checks = smoke_test_deployment.build_checks("localhost")
    urls = {check.name: check.url for check in checks}
    checks_by_name = {check.name: check for check in checks}

    assert urls["public api readiness"] == "http://localhost:9000/api/v1/readiness"
    assert urls["public api docs"] == "http://localhost:9000/docs"
    assert urls["public api openapi schema"] == "http://localhost:9000/openapi.json"
    assert checks_by_name["public api docs"].required_text == "CCTV Platform API"
    assert checks_by_name["public api docs"].forbidden_texts == ("cdn.jsdelivr.net", "unpkg.com")
    assert "prometheus readiness" not in urls
    assert "grafana health" not in urls


def test_build_checks_includes_monitoring_when_requested():
    checks = smoke_test_deployment.build_checks("localhost", include_monitoring=True)
    urls = {check.name: check.url for check in checks}
    assert urls["prometheus readiness"] == "http://localhost:9090/-/ready"
    assert urls["grafana health"] == "http://localhost:3001/api/health"


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
