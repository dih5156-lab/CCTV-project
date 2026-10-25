import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest


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


def test_parse_property_file_only_reads_property_section(tmp_path):
    config = tmp_path / "config.txt"
    config.write_text(
        """
[property]
model-engine-file=../../models/sample.engine
labelfile-path=labels.txt # inline comment

[class-attrs-all]
threshold=0.5
model-engine-file=wrong-section.engine
""",
        encoding="utf-8",
    )

    values = check_deepstream_env._parse_property_file(config)

    assert values["model-engine-file"] == "../../models/sample.engine"
    assert values["labelfile-path"] == "labels.txt"
    assert "threshold" not in values


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


@pytest.mark.parametrize("artifact", ["missing", "empty", "directory"])
def test_engine_check_rejects_unprepared_artifacts_without_loading(tmp_path, monkeypatch, artifact):
    config = tmp_path / "config.txt"
    config.write_text("[property]\nmodel-engine-file=model.engine\n")
    engine = tmp_path / "model.engine"
    if artifact == "empty":
        engine.touch()
    elif artifact == "directory":
        engine.mkdir()
    loader = Mock()
    monkeypatch.setattr(check_deepstream_env, "_check_engine_load", loader)
    results = check_deepstream_env._check_infer_config(config, load_engines=True)
    assert not all(result.ok for result in results)
    assert results[-1].name == "config.txt model-engine-file"
    loader.assert_not_called()


@pytest.mark.parametrize("load_engines", [False, True])
def test_engine_loading_is_opt_in_and_failure_reaches_cli(tmp_path, monkeypatch, capsys, load_engines):
    config = tmp_path / "config.txt"
    config.write_text("[property]\nmodel-engine-file=model.engine\n")
    (tmp_path / "model.engine").write_bytes(b"engine")
    tracker = tmp_path / "config/deepstream/config_tracker_iou.yml"
    tracker.parent.mkdir(parents=True)
    tracker.write_text("tracker")
    passed = check_deepstream_env.CheckResult("dependency", True, "ok")
    monkeypatch.setattr(check_deepstream_env, "_check_deepstream_app", lambda _: passed)
    monkeypatch.setattr(check_deepstream_env, "_check_gstreamer_python", lambda: passed)
    monkeypatch.setattr(check_deepstream_env, "_check_python_module", lambda _: passed)
    loader = Mock(return_value=check_deepstream_env.CheckResult("engine", False, "incompatible"))
    monkeypatch.setattr(check_deepstream_env, "_check_engine_load", loader)
    args = ["--root", str(tmp_path), "--config", str(config), "--skip-gst-plugins", "--json"]
    if load_engines:
        args.append("--load-engines")
    assert check_deepstream_env.main(args) == int(load_engines)
    assert json.loads(capsys.readouterr().out)["passed"] is not load_engines
    assert loader.call_count == int(load_engines)


@pytest.mark.parametrize("outcome", ["loaded", "incompatible", "exception", "plugin_failure", "missing_module"])
def test_engine_deserialization_result(tmp_path, monkeypatch, outcome):
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"serialized-model")
    trt = MagicMock()
    trt.init_libnvinfer_plugins.return_value = outcome != "plugin_failure"
    runtime = trt.Runtime.return_value.__enter__.return_value
    runtime.deserialize_cuda_engine.return_value = None if outcome == "incompatible" else object()
    if outcome == "exception":
        runtime.deserialize_cuda_engine.side_effect = RuntimeError("CUDA unavailable")
    importer = Mock(return_value=trt)
    if outcome == "missing_module":
        importer.side_effect = ImportError("tensorrt missing")
    monkeypatch.setattr(check_deepstream_env.importlib, "import_module", importer)
    result = check_deepstream_env._check_engine_load(engine_path)
    assert result.ok is (outcome == "loaded")
    if outcome in {"loaded", "incompatible", "exception"}:
        runtime.deserialize_cuda_engine.assert_called_once_with(b"serialized-model")
        trt.Runtime.return_value.__exit__.assert_called_once()
    else:
        runtime.deserialize_cuda_engine.assert_not_called()


def test_native_only_cannot_silently_skip_requested_engine_load():
    with pytest.raises(SystemExit) as exc:
        check_deepstream_env.main(["--native-only", "--load-engines"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "profile_batch,requested,expected",
    [((1, 1, 1), 1, True), ((1, 1, 1), 2, False),
     ((1, 2, 4), 2, True), ((1, 2, 4), 4, True),
     ((1, 2, 4), 5, False), ((2, 2, 4), 1, False)],
)
def test_engine_load_checks_requested_batch(tmp_path, monkeypatch, profile_batch, requested, expected):
    # GPU 없이 실제 회귀 조건인 고정 batch1/요청2와 동적 profile 경계를 검사한다.
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"serialized-model")
    trt = MagicMock()
    engine = trt.Runtime.return_value.__enter__.return_value.deserialize_cuda_engine.return_value
    engine.num_io_tensors = 2
    engine.num_optimization_profiles = 2
    engine.get_tensor_name.side_effect = ["images", "output"]
    engine.get_tensor_mode.side_effect = [trt.TensorIOMode.INPUT, trt.TensorIOMode.OUTPUT]
    engine.get_tensor_shape.return_value = (-1, 3, 640, 640)
    engine.get_tensor_profile_shape.return_value = [(batch, 3, 640, 640) for batch in profile_batch]
    monkeypatch.setattr(check_deepstream_env.importlib, "import_module", Mock(return_value=trt))

    config = tmp_path / "infer.txt"
    config.write_text(f"[property]\nmodel-engine-file={engine_path}\nbatch-size={requested}\n")
    results = check_deepstream_env._check_infer_config(config, load_engines=True)

    assert all(result.ok for result in results) is expected
    engine.get_tensor_profile_shape.assert_called_once_with("images", 0)
    assert any(f"batch={requested}" in result.detail for result in results)


@pytest.mark.parametrize("configured", ["0", "-1", "bad", "1.5"])
def test_invalid_configured_batch_fails_before_gpu_load(tmp_path, monkeypatch, configured):
    config = tmp_path / "infer.txt"
    config.write_text(f"[property]\nmodel-engine-file=model.engine\nbatch-size={configured}\n")
    (tmp_path / "model.engine").write_bytes(b"engine")
    loader = Mock()
    monkeypatch.setattr(check_deepstream_env, "_check_engine_load", loader)
    results = check_deepstream_env._check_infer_config(config, load_engines=True)
    assert not all(result.ok for result in results)
    loader.assert_not_called()


@pytest.mark.parametrize("shape,profile", [
    ((3, 640, 640), ((1,), (1,), (1,))),
    ((-1, 3, 640, 640), ((0,), (1,), (2,))),
    ((-1, 3, 640, 640), ((1,), (4,), (2,))),
    ((-1, 3, 640, 640), ()),
])
def test_unsupported_engine_shape_or_profile_fails_closed(tmp_path, monkeypatch, shape, profile):
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    trt = MagicMock()
    runtime = trt.Runtime.return_value.__enter__.return_value
    engine = runtime.deserialize_cuda_engine.return_value
    engine.num_io_tensors = 1
    engine.get_tensor_name.return_value = "images"
    engine.get_tensor_mode.return_value = trt.TensorIOMode.INPUT
    engine.get_tensor_shape.return_value = shape
    engine.get_tensor_profile_shape.return_value = profile
    monkeypatch.setattr(check_deepstream_env.importlib, "import_module", Mock(return_value=trt))
    result = check_deepstream_env._check_engine_load(engine_path, batch_size=2)
    assert not result.ok
    # 추론 컨텍스트를 만드는 실제 엔진 객체에서 GPU 추론 미실행을 확인한다.
    engine.create_execution_context.assert_not_called()
    trt.Runtime.return_value.__exit__.assert_called_once()


def test_runtime_batch_override_reaches_engine_check(tmp_path, monkeypatch, capsys):
    config = tmp_path / "infer.txt"
    config.write_text("[property]\nmodel-engine-file=model.engine\nbatch-size=8\n")
    (tmp_path / "model.engine").write_bytes(b"engine")
    tracker = tmp_path / "config/deepstream/config_tracker_iou.yml"
    tracker.parent.mkdir(parents=True)
    tracker.write_text("tracker")
    passed = check_deepstream_env.CheckResult("dependency", True, "ok")
    for name in ("_check_deepstream_app", "_check_gstreamer_python", "_check_python_module"):
        monkeypatch.setattr(check_deepstream_env, name, Mock(return_value=passed))
    loader = Mock(return_value=check_deepstream_env.CheckResult("engine", False, "batch mismatch"))
    monkeypatch.setattr(check_deepstream_env, "_check_engine_load", loader)
    assert check_deepstream_env.main([
        "--root", str(tmp_path), "--config", str(config), "--load-engines",
        "--batch-size", "2", "--skip-gst-plugins", "--json",
    ]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
    loader.assert_called_once_with(tmp_path / "model.engine", batch_size=2)


@pytest.mark.parametrize("args", [
    ["--files-only", "--batch-size", "2"],
    ["--load-engines", "--batch-size", "2"],
    ["--load-engines", "--config", "one.txt", "--config", "two.txt", "--batch-size", "2"],
    ["--load-engines", "--config", "one.txt", "--batch-size", "0"],
    ["--load-engines", "--config", "one.txt", "--batch-size", "-1"],
])
def test_runtime_batch_requires_one_config_and_engine_load(args):
    with pytest.raises(SystemExit) as exc:
        check_deepstream_env.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("missing", [None, "engine", "tracker", "broken_symlink"])
def test_files_only_checks_runtime_assets_without_sdk_or_gpu(tmp_path, monkeypatch, capsys, missing):
    config_dir = tmp_path / "config/deepstream"
    config_dir.mkdir(parents=True)
    (config_dir / "infer.txt").write_text("[property]\nmodel-engine-file=../../models/test.engine\n")
    (config_dir / "config_tracker.txt").write_text("legacy tracker")
    if missing != "tracker":
        (config_dir / "config_tracker_iou.yml").write_text("BaseConfig: {}\n")
    engine = tmp_path / "models/test.engine"
    engine.parent.mkdir()
    if missing == "broken_symlink":
        engine.symlink_to(tmp_path / "missing-agx-model.engine")
    elif missing != "engine":
        engine.write_bytes(b"test engine")
    # 오프라인 준비 검사는 SDK 실행·Python 바인딩·GPU 로드 없이 동작해야 한다.
    forbidden = Mock(side_effect=AssertionError("runtime access forbidden"))
    monkeypatch.setattr(check_deepstream_env.subprocess, "run", forbidden)
    monkeypatch.setattr(check_deepstream_env.importlib, "import_module", forbidden)
    assert check_deepstream_env.main([
        "--files-only", "--root", str(tmp_path), "--config", "config/deepstream/infer.txt", "--json",
    ]) == int(missing is not None)
    payload = json.loads(capsys.readouterr().out)
    assert payload["files_only"] is True
    assert payload["passed"] is (missing is None)
    if missing == "tracker":
        assert payload["results"][-1]["ok"] is False
        assert "config_tracker_iou.yml" in payload["results"][-1]["detail"]
    forbidden.assert_not_called()


@pytest.mark.parametrize("other_mode", ["--native-only", "--load-engines"])
def test_files_only_rejects_runtime_check_options(other_mode):
    with pytest.raises(SystemExit) as exc:
        check_deepstream_env.main(["--files-only", other_mode])
    assert exc.value.code == 2


@pytest.mark.parametrize("content", [b"model-engine-file=model.engine\n", b"[property]\n\xff"])
def test_invalid_config_does_not_pass_file_readiness(tmp_path, content):
    config = tmp_path / "infer.txt"
    config.write_bytes(content)
    (tmp_path / "model.engine").write_bytes(b"engine")
    assert not all(result.ok for result in check_deepstream_env._check_infer_config(config))


def test_check_gst_plugin_reports_success(monkeypatch):
    completed = Mock(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr(check_deepstream_env.subprocess, "run", Mock(return_value=completed))

    result = check_deepstream_env._check_gst_plugin("nvinfer", timeout=1.0)

    assert result.ok is True
    assert result.name == "GStreamer plugin nvinfer"


def test_build_checks_can_skip_gst_plugins(monkeypatch, tmp_path):
    # 파일 점검 시험이 병행 실장비 시험의 SDK를 실행하지 않도록 경계를 격리한다.
    monkeypatch.setattr(
        check_deepstream_env, "_check_deepstream_app",
        Mock(return_value=check_deepstream_env.CheckResult("sdk", True, "ok")),
    )
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
