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


check_sensitive_defaults = _load_script_module(
    "check_sensitive_defaults",
    "scripts/check_sensitive_defaults.py",
)


def test_allowed_secret_values_include_empty_and_env_without_default():
    assert check_sensitive_defaults._is_allowed_value("")
    assert check_sensitive_defaults._is_allowed_value('""')
    assert check_sensitive_defaults._is_allowed_value("${SPEAKER_PASSWORD}")
    assert check_sensitive_defaults._is_allowed_value("${SPEAKER_PASSWORD:-}")


def test_non_empty_secret_defaults_are_rejected():
    assert not check_sensitive_defaults._is_allowed_value("plain-secret")
    assert not check_sensitive_defaults._is_allowed_value("${SPEAKER_PASSWORD:-plain-secret}")


def test_current_shared_configs_have_no_sensitive_defaults():
    assert check_sensitive_defaults.find_sensitive_defaults() == []
