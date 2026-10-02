"""진단 기록 자체의 덮어쓰기·유실을 방지한다. CUDA 오류 재현 테스트는 아니다."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts.ops.trace_deepstream_startup import (
    NativeLogTail,
    StartupTrace,
    configure_gpu_core_dump,
)


def test_bounded_ring_preserves_first_events_and_full_stage_counts(tmp_path):
    trace = StartupTrace(tmp_path, capacity=4)
    trace.record("caps", element="primary", pad="sink", caps="NV12", properties={})
    for pts in range(150):
        trace.record("buffer", element="primary", pad="sink", pts=pts)
    snapshot = trace.snapshot()
    assert [row["pts"] for row in snapshot["recent"]] == [146, 147, 148, 149]
    assert len(snapshot["first"]) == 128
    assert snapshot["first"][0]["kind"] == "caps"
    assert snapshot["stages"]["primary:sink"]["count"] == 150
    assert snapshot["stages"]["primary:sink"]["configuration"]["caps"] == "NV12"
    trace.record("buffer", element="primary", pad="sink", pts=150)
    assert snapshot["stages"]["primary:sink"]["count"] == 150


def test_native_first_error_is_not_overwritten_by_cascade_or_bus_error(tmp_path):
    trace = StartupTrace(tmp_path)
    trace.capture("bus", {"message": "RTSP 404"})
    trace.record("buffer", element="primary", pad="sink", pts=17)
    trace.native_line("Cuda failure: status=700 in createTexture at line 783")
    original = (tmp_path / "first_cuda.json").read_bytes()
    trace.native_line("NVDSINFER_CUDA_ERROR cascade")
    trace.capture("cuda", {"message": "later bus error"})
    assert (tmp_path / "first_cuda.json").read_bytes() == original
    document = json.loads(original)
    assert "createTexture" in document["reason"]["message"]
    assert document["recent"][-1]["pts"] == 17


def test_concurrent_capture_has_exactly_one_winner(tmp_path):
    trace = StartupTrace(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda i: trace.capture("cuda", {"number": i}), range(32)))
    assert sum(results) == 1
    assert "number" in json.loads((tmp_path / "first_cuda.json").read_text())["reason"]


@pytest.mark.parametrize("ending", ["", "\n"])
def test_tail_handles_split_signature_and_missing_newline(tmp_path, ending):
    trace = StartupTrace(tmp_path)
    path = tmp_path / "app.log"
    tail = NativeLogTail(trace, path)
    path.write_text("init\nCuda failure: sta")
    tail.poll()
    assert not (tmp_path / "first_cuda.json").exists()
    with path.open("a") as handle:
        handle.write("tus=700" + ending)
    tail.poll()
    assert json.loads((tmp_path / "first_cuda.json").read_text())["reason"]["origin"] == "native_log"


def test_log_and_checkpoint_remain_bounded(tmp_path):
    trace = StartupTrace(tmp_path, capacity=2)
    for _ in range(100):
        trace.native_line("x" * 10000)
    trace.record("checkpoint")
    trace.checkpoint()
    document = json.loads((tmp_path / "trace_latest.json").read_text())
    assert len(document["native_lines"]) == 64
    assert all(len(row["line"]) == 4096 for row in document["native_lines"])
    assert not (tmp_path / "trace_latest.tmp").exists()


def test_existing_first_error_file_is_never_overwritten(tmp_path):
    path = tmp_path / "first_cuda.json"
    path.write_text("previous run evidence")
    trace = StartupTrace(tmp_path)
    with pytest.raises(FileExistsError):
        trace.capture("cuda", {"message": "new run"})
    assert path.read_text() == "previous run evidence"


def test_source_timeout_preserves_upstream_boundary_without_claiming_cuda(tmp_path):
    # 입력 경고 스냅샷은 후속 CUDA 오류의 첫 기록과 독립적이어야 한다.
    trace = StartupTrace(tmp_path)
    trace.record("buffer", element="source/depay", pad="src", pts=100)
    trace.record("buffer", element="source/decoder", pad="sink", pts=100)
    trace.native_line("No data from source since last 15 sec. Trying reconnection")
    document = json.loads((tmp_path / "first_source_timeout.json").read_text())
    assert document["stages"]["source/decoder:sink"]["last"]["pts"] == 100
    assert not (tmp_path / "first_cuda.json").exists()
    trace.native_line("status=700 subsequent error")
    assert (tmp_path / "first_cuda.json").exists()
    assert json.loads((tmp_path / "first_source_timeout.json").read_text()) == document


def test_source_timeout_split_log_without_newline(tmp_path):
    trace = StartupTrace(tmp_path)
    path = tmp_path / "app.log"
    tail = NativeLogTail(trace, path)
    path.write_text("No data from source since last 15 sec. Trying recon")
    tail.poll()
    assert not (tmp_path / "first_source_timeout.json").exists()
    with path.open("a") as handle:
        handle.write("nection")
    tail.poll()
    assert (tmp_path / "first_source_timeout.json").exists()


def test_gpu_core_capture_requires_separate_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_CUDA_COREDUMP", raising=False)
    monkeypatch.setenv("CUDA_COREDUMP_FILE", "caller-owned-path")
    assert configure_gpu_core_dump(tmp_path) is None
    assert not list(tmp_path.iterdir())
    assert os.environ["CUDA_COREDUMP_FILE"] == "caller-owned-path"


def test_gpu_core_capture_preserves_prior_run_and_disables_manual_trigger(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_CUDA_COREDUMP", "1")
    monkeypatch.setenv("CUDA_ENABLE_USER_TRIGGERED_COREDUMP", "1")
    # 외부 설정의 변경은 monkeypatch가 회수하고 이전 덤프 파일 보존을 검증한다.
    for key in ("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION",
                "CUDA_ENABLE_LIGHTWEIGHT_COREDUMP", "CUDA_COREDUMP_FILE",
                "CUDA_COREDUMP_GENERATION_FLAGS"):
        monkeypatch.setenv(key, "previous")
    first = configure_gpu_core_dump(tmp_path)
    first_path = Path(first["directory"])
    existing = first_path / "previous.nvcudmp"
    existing.write_bytes(b"original evidence")
    second = configure_gpu_core_dump(tmp_path)
    assert first["directory"] != second["directory"]
    assert existing.read_bytes() == b"original evidence"
    assert first_path.stat().st_mode & 0o777 == 0o700
    assert os.environ["CUDA_ENABLE_USER_TRIGGERED_COREDUMP"] == "0"
    assert os.environ["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] == "1"
    assert os.environ["CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION"] == "0"
    assert os.environ["CUDA_ENABLE_LIGHTWEIGHT_COREDUMP"] == "1"
    assert "CUDA_COREDUMP_GENERATION_FLAGS" not in os.environ
    assert os.environ["CUDA_COREDUMP_FILE"].endswith("gpu.%p.%t.nvcudmp")
    assert json.loads((first_path / "capture_config.json").read_text()) == first


def test_launch_failure_is_preserved_before_later_memory_copy_error(tmp_path):
    # 수동 덤프 뒤의719를 원700과 혼동하지 않도록 최초 CUDA 오류 문구를 보존한다.
    trace = StartupTrace(tmp_path)
    trace.native_line("cudaMemset2DAsync failed with error cudaErrorLaunchFailure while converting buffer")
    trace.native_line("Failed in mem copy")
    message = json.loads((tmp_path / "first_cuda.json").read_text())["reason"]["message"]
    assert "cudaErrorLaunchFailure" in message
