"""falldata aux readiness checker tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "health"
    / "check_falldata_aux.py"
)

spec = importlib.util.spec_from_file_location("check_falldata_aux", SCRIPT_PATH)
check_falldata_aux = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(check_falldata_aux)


def test_version_in_range_uses_exclusive_upper_bound() -> None:
    rule = {"min": "1.3.2", "max_exclusive": "1.4.0"}

    assert check_falldata_aux._version_in_range("1.3.2", rule) is True
    assert check_falldata_aux._version_in_range("1.3.9", rule) is True
    assert check_falldata_aux._version_in_range("1.4.0", rule) is False


def test_version_check_reports_failed_package(monkeypatch, tmp_path) -> None:
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")

    def fake_run(command, timeout):
        return {
            "passed": True,
            "command": command,
            "returncode": 0,
            "stdout": json.dumps({"numpy": "2.2.6"}),
            "stderr": "",
        }

    monkeypatch.setattr(check_falldata_aux, "_run", fake_run)

    result = check_falldata_aux._version_check(
        label="model_python",
        python_path=python_path,
        rules={"numpy": {"min": "1.26.1", "max_exclusive": "2.0.0"}},
        timeout=1.0,
    )

    assert result["passed"] is False
    assert result["checks"][0]["package"] == "numpy"
    assert result["checks"][0]["version"] == "2.2.6"


def test_version_check_accepts_expected_versions(monkeypatch, tmp_path) -> None:
    python_path = tmp_path / "python"
    python_path.write_text("", encoding="utf-8")

    def fake_run(command, timeout):
        return {
            "passed": True,
            "command": command,
            "returncode": 0,
            "stdout": json.dumps({"numpy": "1.26.1", "scikit-learn": "1.3.2"}),
            "stderr": "",
        }

    monkeypatch.setattr(check_falldata_aux, "_run", fake_run)

    result = check_falldata_aux._version_check(
        label="model_python",
        python_path=python_path,
        rules={
            "numpy": {"min": "1.26.1", "max_exclusive": "2.0.0"},
            "scikit-learn": {"min": "1.3.2", "max_exclusive": "1.4.0"},
        },
        timeout=1.0,
    )

    assert result["passed"] is True


def test_policy_check_rejects_confirm_without_fail_open() -> None:
    result = check_falldata_aux._policy_check(
        mode="confirm",
        fail_open_on_unavailable=False,
        confirm_borderline=False,
        compare_veto_enabled=False,
    )

    assert result["passed"] is False
    assert "confirm mode requires fail-open" in result["errors"][0]


def test_policy_check_allows_shadow_with_compare_warning() -> None:
    result = check_falldata_aux._policy_check(
        mode="shadow",
        fail_open_on_unavailable=True,
        confirm_borderline=False,
        compare_veto_enabled=True,
    )

    assert result["passed"] is True
    assert result["warnings"]


def test_inline_pose_rf_smoke_runs_predict_proba(monkeypatch, tmp_path) -> None:
    python_path = tmp_path / "python"
    model_path = tmp_path / "inline.pkl"
    python_path.write_text("", encoding="utf-8")
    model_path.write_bytes(b"model")

    def fake_run(command, timeout):
        assert command[0] == str(python_path)
        assert str(model_path) in command[-1]
        return {
            "passed": True,
            "command": command,
            "returncode": 0,
            "stdout": json.dumps(
                {
                    "feature_count": 50,
                    "classes": [0, 1],
                    "probability": [[0.9, 0.1]],
                }
            ),
            "stderr": "",
        }

    monkeypatch.setattr(check_falldata_aux, "_run", fake_run)

    result = check_falldata_aux._inline_pose_rf_smoke(
        python_path=python_path,
        model_path=model_path,
        timeout=1.0,
    )

    assert result["passed"] is True
    assert result["inference"]["feature_count"] == 50
    assert result["inference"]["probability"] == [[0.9, 0.1]]
