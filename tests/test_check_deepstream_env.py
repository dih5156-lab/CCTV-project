import importlib.util
import json
import subprocess
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


def test_native_check_rejects_missing_library(monkeypatch):
    # 버전 조회가 실패했는데 Python import만으로 배포를 통과시키는 회귀를 막는다.
    monkeypatch.setattr(
        check_deepstream_env.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 127, "",
            "deepstream-app: error while loading shared libraries: "
            "libgstrtspserver-1.0.so.0: cannot open shared object file",
        ),
    )
    result = check_deepstream_env._check_deepstream_app(1.0)
    assert not result.ok
    assert "libgstrtspserver-1.0.so.0" in result.detail


def test_native_check_requires_expected_sdk_version(monkeypatch):
    for output, expected_ok in [
        ("deepstream-app version 7.1.0\nDeepStreamSDK 7.1.0\n", True),
        ("deepstream-app version 8.0.0\n", False),
        ("deepstream-app version 7.10.0\n", False),
        ("", False),
    ]:
        monkeypatch.setattr(
            check_deepstream_env.subprocess, "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
        )
        assert check_deepstream_env._check_deepstream_app(1.0).ok is expected_ok


def test_native_check_reports_missing_command_and_timeout(monkeypatch):
    for error in [FileNotFoundError("deepstream-app"), subprocess.TimeoutExpired("deepstream-app", 1)]:
        def run(*args, **kwargs):
            raise error
        monkeypatch.setattr(check_deepstream_env.subprocess, "run", run)
        assert not check_deepstream_env._check_deepstream_app(1.0).ok


def test_native_check_uses_mounted_sdk_when_app_is_not_on_path(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    def run(args, **kwargs):
        if args[0] != "/opt/nvidia/deepstream/deepstream/bin/deepstream-app":
            raise FileNotFoundError(args[0])
        return subprocess.CompletedProcess(args, 0, "deepstream-app version 7.1.0\n", "")
    monkeypatch.setattr(check_deepstream_env.subprocess, "run", run)
    assert check_deepstream_env._check_deepstream_app(1.0).ok


def test_native_only_cli_checks_real_command_failure_without_model_files(tmp_path, monkeypatch, capsys):
    # 외부 명령 경계만 대체하고 CLI의 실패 집계·종료 코드는 실제로 실행한다.
    binary = tmp_path / "deepstream-app"
    binary.write_text("#!/bin/sh\necho 'libgstrtspserver-1.0.so.0 missing' >&2\nexit 127\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    exit_code = check_deepstream_env.main([
        "--native-only", "--skip-gst-plugins", "--root", str(tmp_path), "--json",
    ])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert not output["passed"]
    assert len(output["results"]) == 1
    assert "libgstrtspserver" in output["results"][0]["detail"]


def test_native_only_cli_checks_plugins_and_passes_without_python_bindings(tmp_path, monkeypatch, capsys):
    binary = tmp_path / "deepstream-app"
    binary.write_text("#!/bin/sh\necho 'deepstream-app version 7.1.0'\n")
    binary.chmod(0o755)
    plugin = tmp_path / "gst-inspect-1.0"
    plugin.write_text("#!/bin/sh\nexit 0\n")
    plugin.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert check_deepstream_env.main(["--native-only", "--root", str(tmp_path), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["passed"]
    assert any("nvinfer" in result["name"] for result in output["results"])
    plugin.write_text("#!/bin/sh\necho 'No such element' >&2\nexit 1\n")
    assert check_deepstream_env.main(["--native-only", "--root", str(tmp_path), "--json"]) == 1
