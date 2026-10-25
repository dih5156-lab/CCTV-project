"""GPU 없이 엔진 생성 입력 검증과 기존 산출물 보존을 확인한다."""

import hashlib
import json
import subprocess
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from scripts.convert import convert_onnx_to_engine as converter


@pytest.fixture
def conversion(tmp_path):
    source = tmp_path / "pose.onnx"
    tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 960, 960])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 3, 960, 960])
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "pose", [tensor], [output])
    onnx.save(helper.make_model(graph), source)
    target = tmp_path / "nx" / "pose.engine"
    arguments = [
        "--onnx", str(source), "--engine", str(target),
        "--sha256", hashlib.sha256(source.read_bytes()).hexdigest(),
        "--input-name", "input", "--min-shape", "1x3x960x960",
        "--opt-shape", "1x3x960x960", "--max-shape", "2x3x960x960",
        "--trtexec", "/missing/trtexec",
    ]
    return source, target, arguments


def test_dry_run_validates_without_sdk_or_output(conversion, monkeypatch, capsys):
    _, target, arguments = conversion
    monkeypatch.setattr(converter.subprocess, "run", lambda *a, **k: pytest.fail("GPU 실행 금지"))
    assert converter.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "DRY_RUN"
    assert "--minShapes=input:1x3x960x960" in result["command"]
    assert "--maxShapes=input:2x3x960x960" in result["command"]
    assert "--memPoolSize=workspace:1024" in result["command"]
    assert not target.parent.exists()


@pytest.mark.parametrize("flag,value", [
    ("--sha256", "0" * 64), ("--input-name", "images"),
    ("--min-shape", "0x3x960x960"), ("--min-shape", "1x3x960"),
    ("--min-shape", "3x3x960x960"), ("--max-shape", "2x3x640x640"),
])
def test_invalid_artifact_or_profile_fails(conversion, flag, value):
    _, target, arguments = conversion
    arguments[arguments.index(flag) + 1] = value
    assert converter.main(arguments) == 1
    assert not target.parent.exists()


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_link"])
def test_existing_destination_is_preserved(conversion, kind):
    _, target, arguments = conversion
    target.parent.mkdir()
    if kind == "file":
        target.write_bytes(b"existing-engine")
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(target.parent / "absent")
    assert converter.main(arguments) == 1
    if kind == "file":
        assert target.read_bytes() == b"existing-engine"
    else:
        assert target.is_symlink() if kind == "dangling_link" else target.is_dir()


@pytest.mark.parametrize("kind", ["missing", "empty", "invalid"])
def test_bad_source_fails(conversion, kind):
    source, target, arguments = conversion
    if kind == "missing":
        source.unlink()
    else:
        source.write_bytes(b"" if kind == "empty" else b"not-onnx")
        arguments[arguments.index("--sha256") + 1] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert converter.main(arguments) == 1
    assert not target.parent.exists()


@pytest.mark.parametrize("nested", [False, True])
def test_external_weights_are_not_covered_by_single_file_hash(conversion, nested):
    source, _, arguments = conversion
    model = onnx.load(source)
    weight = model.graph.initializer.add(name="weights", data_type=TensorProto.FLOAT, dims=[1])
    weight.data_location = TensorProto.EXTERNAL
    weight.external_data.add(key="location", value="weights.bin")
    if nested:
        graph = helper.make_graph([], "nested", [], [], [weight])
        model.graph.node.append(helper.make_node("If", ["condition"], ["branch"], then_branch=graph))
        del model.graph.initializer[:]
    source.write_bytes(model.SerializeToString())
    arguments[arguments.index("--sha256") + 1] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert converter.main(arguments) == 1


def test_multiple_runtime_inputs_are_rejected(conversion):
    source, _, arguments = conversion
    model = onnx.load(source)
    model.graph.input.append(helper.make_tensor_value_info("second", TensorProto.FLOAT, [1]))
    source.write_bytes(model.SerializeToString())
    arguments[arguments.index("--sha256") + 1] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert converter.main(arguments) == 1


def test_build_requires_available_trtexec(conversion):
    _, target, arguments = conversion
    assert converter.main(arguments + ["--build"]) == 1
    assert not target.parent.exists()


@pytest.mark.parametrize("outcome", ["success", "failure", "missing", "empty", "race", "source_changed"])
def test_build_publishes_only_complete_output(conversion, monkeypatch, capsys, outcome):
    source, target, arguments = conversion
    arguments[arguments.index("--trtexec") + 1] = "/bin/true"

    def fake_builder(command, **kwargs):
        assert kwargs["check"] is True
        if command[-1] == "--help":
            return subprocess.CompletedProcess(command, 0, stdout="--skipInference --builderOptimizationLevel")
        temporary_engine = Path(next(part.split("=", 1)[1] for part in command if part.startswith("--saveEngine=")))
        assert temporary_engine != target
        if outcome != "missing":
            temporary_engine.write_bytes(b"" if outcome == "empty" else b"new-engine")
        if outcome == "failure":
            raise subprocess.CalledProcessError(1, command)
        if outcome == "race":
            target.write_bytes(b"another-engine")
        if outcome == "source_changed":
            source.write_bytes(b"changed-source")

    monkeypatch.setattr(converter.subprocess, "run", fake_builder)
    assert converter.main(arguments + ["--build"]) == (0 if outcome == "success" else 1)
    if outcome == "success":
        assert target.read_bytes() == b"new-engine"
        result = json.loads(capsys.readouterr().out)
        assert result["state"] == "BUILT_NOT_VALIDATED"
        assert result["engine_sha256"] == hashlib.sha256(b"new-engine").hexdigest()
    elif outcome == "race":
        assert target.read_bytes() == b"another-engine"
    else:
        assert not target.exists()
    assert not list(target.parent.glob(".trt-build-*"))


@pytest.mark.parametrize("help_text,expected,optimization", [
    ("--buildOnly --avgTiming", "--buildOnly", False),
    ("--skipInference --builderOptimizationLevel", "--skipInference", True),
])
def test_build_selects_installed_sdk_flags(conversion, monkeypatch, help_text, expected, optimization):
    _, target, arguments = conversion
    arguments[arguments.index("--trtexec") + 1] = "/bin/true"

    def builder(command, **kwargs):
        if command[-1] == "--help":
            return subprocess.CompletedProcess(command, 0, stdout=help_text)
        assert expected in command
        assert ("--builderOptimizationLevel=0" in command) is optimization
        assert not ({"--buildOnly", "--skipInference"} <= set(command))
        Path(next(part.split("=", 1)[1] for part in command if part.startswith("--saveEngine="))).write_bytes(b"engine")

    monkeypatch.setattr(converter.subprocess, "run", builder)
    assert converter.main(arguments + ["--build"]) == 0
    assert target.read_bytes() == b"engine"


def test_unknown_sdk_options_fail_before_output_creation(conversion, monkeypatch):
    _, target, arguments = conversion
    arguments[arguments.index("--trtexec") + 1] = "/bin/true"
    monkeypatch.setattr(converter.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="unknown"))
    assert converter.main(arguments + ["--build"]) == 1
    assert not target.parent.exists()


def test_static_onnx_omits_dynamic_profile(conversion, capsys):
    source, _, arguments = conversion
    model = onnx.load(source)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    source.write_bytes(model.SerializeToString())
    arguments[arguments.index("--sha256") + 1] = hashlib.sha256(source.read_bytes()).hexdigest()
    arguments[arguments.index("--max-shape") + 1] = "1x3x960x960"
    assert converter.main(arguments) == 0
    command = json.loads(capsys.readouterr().out)["command"]
    assert not any(part.startswith(("--minShapes=", "--optShapes=", "--maxShapes=")) for part in command)
