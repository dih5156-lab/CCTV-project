import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock


def _load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_deepstream_env = _load_script_module(
    "check_deepstream_env",
    "scripts/health/check_deepstream_env.py",
)


def test_parse_property_file_ignores_comments_and_sections(tmp_path):
    config = tmp_path / "config.txt"
    config.write_text(
        """
[property]
model-engine-file=../../models/sample.engine
labelfile-path=labels.txt # inline comment

[class-attrs-all]
threshold=0.5
""",
        encoding="utf-8",
    )

    values = check_deepstream_env._parse_property_file(config)

    assert values["model-engine-file"] == "../../models/sample.engine"
    assert values["labelfile-path"] == "labels.txt"
    assert values["threshold"] == "0.5"


def test_check_infer_config_resolves_paths_relative_to_config_file(tmp_path):
    root = tmp_path
    config_dir = root / "config" / "deepstream"
    model_dir = root / "models"
    config_dir.mkdir(parents=True)
    model_dir.mkdir()
    (model_dir / "sample.engine").write_bytes(b"engine")
    (config_dir / "labels.txt").write_text("person\n", encoding="utf-8")
    config = config_dir / "config_infer_primary.txt"
    config.write_text(
        """
[property]
model-engine-file=../../models/sample.engine
labelfile-path=labels.txt
""",
        encoding="utf-8",
    )

    results = check_deepstream_env._check_infer_config(config)

    assert all(result.ok for result in results)
    assert {result.name for result in results} == {
        "nvinfer config config_infer_primary.txt",
        "config_infer_primary.txt model-engine-file",
        "config_infer_primary.txt labelfile-path",
    }


def test_check_infer_config_reports_missing_model(tmp_path):
    config = tmp_path / "config_infer_primary.txt"
    config.write_text(
        """
[property]
model-engine-file=missing.engine
""",
        encoding="utf-8",
    )

    results = check_deepstream_env._check_infer_config(config)
    failures = [result for result in results if not result.ok]

    assert [result.name for result in failures] == [
        "config_infer_primary.txt model-engine-file",
    ]


def test_check_gst_plugin_reports_success(monkeypatch):
    completed = Mock(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(check_deepstream_env.subprocess, "run", Mock(return_value=completed))

    result = check_deepstream_env._check_gst_plugin("nvinfer", timeout=1.0)

    assert result.ok is True
    assert result.name == "GStreamer plugin nvinfer"


def test_build_checks_can_skip_gst_plugins(monkeypatch, tmp_path):
    monkeypatch.setattr(
        check_deepstream_env,
        "_check_gstreamer_python",
        Mock(return_value=check_deepstream_env.CheckResult("gst", True, "ok")),
    )
    monkeypatch.setattr(
        check_deepstream_env,
        "_check_python_module",
        Mock(return_value=check_deepstream_env.CheckResult("pyds", True, "ok")),
    )
    config = tmp_path / "config.txt"
    config.write_text(
        """
[property]
model-engine-file=model.engine
""",
        encoding="utf-8",
    )
    (tmp_path / "model.engine").write_bytes(b"engine")

    results = check_deepstream_env.build_checks(
        root=tmp_path,
        config_paths=[str(config)],
        gst_plugins=["nvinfer"],
        timeout=1.0,
        skip_gst_plugins=True,
    )

    assert all("GStreamer plugin" not in result.name for result in results)
