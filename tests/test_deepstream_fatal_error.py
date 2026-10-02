"""치명적 CUDA/추론 오류는 같은 프로세스에서 다시 시작하지 않는다."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.bootstrap import runtime
from src.core import deepstream_processor
from src.core.deepstream_processor import DeepStreamProcessor


@pytest.mark.parametrize(
    "error,debug",
    [
        ("cudaErrorIllegalAddress", ""),
        ("Preprocessor transform input data failed", "NVDSINFER_CUDA_ERROR"),
        ("Failed to queue input batch for inferencing", "gstnvinfer.cpp(1420): gst_nvinfer_input_queue_loop (): primary-infer"),
    ],
)
def test_fatal_bus_error_stops_without_source_restart(monkeypatch, error, debug):
    proc = object.__new__(DeepStreamProcessor)
    proc._cameras = {"camera_1": {}}
    proc._source_backoff_until = {}
    proc._source_last_error = {}
    proc._source_failure_backoff_sec = 10
    proc.stop_event = threading.Event()
    proc._main_loop = MagicMock()
    proc._request_pipeline_restart = MagicMock()
    monkeypatch.setattr(deepstream_processor, "Gst", SimpleNamespace(
        MessageType=SimpleNamespace(ERROR="ERROR", EOS="EOS", WARNING="WARNING")))
    message = MagicMock(type="ERROR")
    # 하위 요소 이름에 카메라 식별자가 있어도 CUDA 오류는 소스 재연결로 처리하지 않는다.
    message.src.get_name.return_value = "src-camera_1"
    message.parse_error.return_value = (RuntimeError(error), debug)

    proc._on_bus_message(None, message)

    assert proc.process_restart_reason
    assert proc.stop_event.is_set()
    proc._request_pipeline_restart.assert_not_called()
    assert not proc._source_backoff_until
    with pytest.raises(RuntimeError, match="process restart required"):
        proc.start()


@pytest.mark.parametrize("fatal", [True, False])
def test_runtime_exits_for_fatal_but_retries_normal_stop(monkeypatch, fatal):
    processor = SimpleNamespace(running=True, stop_event=threading.Event(),
                                process_restart_reason="cuda_fatal" if fatal else "")
    starts = []

    def start():
        starts.append(True)
        if len(starts) > 1:
            raise KeyboardInterrupt
        processor.stop_event.set()

    processor.start = start
    processor.stop = MagicMock()
    monkeypatch.setattr(runtime, "create_processor", lambda cfg: processor)
    monkeypatch.setattr(runtime, "collect_active_cameras", lambda cameras, proc: [("camera_1", "file", {}, {}, {})])
    monkeypatch.setattr(runtime, "connect_cameras_parallel", lambda cameras, proc: {"camera_1": True})
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("STREAM_PORT", "0")
    monkeypatch.setenv("STREAM_API_ENABLED", "0")
    monkeypatch.setenv("METRICS_PORT", "0")
    monkeypatch.setenv("AI_ENGINE_RESTART_ON_PIPELINE_ERROR", "1")

    if fatal:
        with pytest.raises(SystemExit) as exc:
            runtime.start_processor_runtime([{}], SimpleNamespace(display=False))
        assert exc.value.code == 1
        assert len(starts) == 1
    else:
        runtime.start_processor_runtime([{}], SimpleNamespace(display=False))
        assert len(starts) == 2
    assert processor.stop.called
