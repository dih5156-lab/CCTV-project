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


smoke_test_data_flow = _load_script_module(
    "smoke_test_data_flow",
    "scripts/smoke/smoke_test_data_flow.py",
)


def test_run_request_check_passes_expected_post(monkeypatch):
    monkeypatch.setattr(
        smoke_test_data_flow,
        "_request",
        lambda method, url, timeout, payload=None, headers=None: (True, 202, '{"accepted":true}'),
    )

    result = smoke_test_data_flow.run_request_check(
        smoke_test_data_flow.RequestCheck(
            "alert api accepts alert",
            "POST",
            "http://localhost:8000/api/alerts",
            (202,),
            payload={"camera_id": "cam-1"},
            required_text="accepted",
        ),
        timeout=1.0,
    )

    assert result["passed"] is True


def test_run_request_check_fails_on_wrong_status(monkeypatch):
    monkeypatch.setattr(
        smoke_test_data_flow,
        "_request",
        lambda method, url, timeout, payload=None, headers=None: (False, 500, "boom"),
    )

    result = smoke_test_data_flow.run_request_check(
        smoke_test_data_flow.RequestCheck(
            "alert api accepts alert",
            "POST",
            "http://localhost:8000/api/alerts",
            (202,),
            payload={"camera_id": "cam-1"},
        ),
        timeout=1.0,
    )

    assert result["passed"] is False
    assert result["detail"] == "boom"


def test_build_checks_contains_runtime_data_paths():
    checks = smoke_test_data_flow.build_checks("localhost")
    urls = {check.url for check in checks}

    assert "http://localhost:8000/api/alerts" in urls
    assert "http://localhost:8000/api/sensor-readings" in urls
    assert "http://localhost:8080/events" in urls
    assert "http://localhost:8080/metrics" in urls
    assert "http://localhost:9000/api/v1/metrics" in urls
