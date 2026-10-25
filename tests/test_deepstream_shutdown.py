"""영상 저장이 끝나기 전에 인터프리터 종료/자원 해제가 진행되지 않아야 한다."""

import signal
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.bootstrap import runtime
from src.core import deepstream_processor
from src.core.deepstream_processor import DeepStreamProcessor


def test_stop_waits_for_inflight_aux_work_before_releasing_resources(monkeypatch):
    # 기존 2초 join을 넘는 실제 스레드로 영상 쓰기와 종료의 경합을 재현한다.
    proc = object.__new__(DeepStreamProcessor)
    proc.stop_event = threading.Event()
    proc._pipeline = object()
    proc._main_loop = proc._publish_thread = proc._main_loop_thread = None
    proc._face_worker_thread = None
    proc.event_publisher = MagicMock()
    released = threading.Event()
    entered = threading.Event()
    stopped = threading.Event()
    writes_finished = threading.Event()
    cleanup_order = []
    errors = []

    def write_video():
        entered.set()
        released.wait(timeout=8)
        writes_finished.set()
        cleanup_order.append("writer_released")

    def close():
        assert writes_finished.is_set()
        cleanup_order.append("model_child_closed")

    proc._falldata_aux = SimpleNamespace(close=close)
    proc._falldata_aux_thread = threading.Thread(target=write_video)
    worker = proc._falldata_aux_thread
    worker.start()
    assert entered.wait(timeout=1)
    monkeypatch.setattr(deepstream_processor, "stop_pipeline_runtime", lambda **kw: None)

    def stop():
        try:
            proc.stop()
        except BaseException as exc:
            errors.append(exc)
        finally:
            stopped.set()

    stopper = threading.Thread(target=stop)
    stopper.start()
    try:
        assert proc.stop_event.wait(timeout=1)
        assert not stopped.wait(timeout=2.2), "영상 쓰기가 남아 있는데 stop()이 반환됨"
        proc.event_publisher.disconnect.assert_not_called()
        assert proc._falldata_aux_thread is worker
    finally:
        released.set()
        worker.join(timeout=2)
        stopper.join(timeout=2)
    assert not errors
    assert stopped.is_set()
    assert cleanup_order == ["writer_released", "model_child_closed"]
    assert proc._falldata_aux_thread is None
    proc.event_publisher.disconnect.assert_called_once()
    # 런타임 종료와 atexit에서 중복 호출해도 이미 끝난 스레드를 다시 기다리지 않는다.
    proc.stop()


def test_shutdown_request_prevents_background_pipeline_start():
    proc = object.__new__(DeepStreamProcessor)
    proc._shutdown_requested = True
    proc.running = False
    proc.stop_event = threading.Event()
    proc.stop_event.set()
    proc._build_pipeline = MagicMock()

    # 소스 복구 스레드가 늦게 start()를 호출해도 종료 이벤트를 지우면 안 된다.
    proc.start()

    proc._build_pipeline.assert_not_called()
    assert not proc.running
    assert proc.stop_event.is_set()


@pytest.mark.parametrize("signal_stage", ["pipeline", "aux_join", "model_close"])
def test_shutdown_signal_during_cleanup_does_not_reenter_or_restart(monkeypatch, signal_stage):
    # native 중지·작업 대기·모델 회수 중 시그널을 넣어 원래 정리가 끝까지 실행되는지 확인한다.
    proc = object.__new__(DeepStreamProcessor)
    proc.running = True
    proc.stop_event = threading.Event()
    proc._pipeline = object()
    proc._main_loop = proc._publish_thread = proc._main_loop_thread = None
    proc._face_worker_thread = None
    proc.next_source_retry_delay = lambda: None
    callbacks = {}
    atexit_callbacks = []
    cleanup = []
    starts = []
    signal_sent = False

    monkeypatch.setattr(runtime.signal, "signal", lambda sig, cb: callbacks.setdefault(sig, cb))
    monkeypatch.setattr(runtime.atexit, "register", atexit_callbacks.append)
    runtime.register_shutdown_handlers([proc])

    def interrupt_cleanup(stage):
        nonlocal signal_sent
        if stage == signal_stage and not signal_sent:
            signal_sent = True
            # 반복 종료 요청도 진행 중인 정리를 중단하거나 중복 실행하면 안 된다.
            callbacks[signal.SIGTERM](signal.SIGTERM, None)
            callbacks[signal.SIGTERM](signal.SIGTERM, None)

    def stop_pipeline(**kwargs):
        cleanup.append("pipeline_enter")
        interrupt_cleanup("pipeline")
        cleanup.append("pipeline_exit")

    def join():
        cleanup.append("aux_join")
        interrupt_cleanup("aux_join")
        cleanup.append("aux_finished")

    def close():
        cleanup.append("model_close")
        interrupt_cleanup("model_close")

    def start():
        starts.append(True)
        if len(starts) > 1:
            raise KeyboardInterrupt
        # 출력 장애로 런타임의 finally 정리가 이미 시작된 상황을 재현한다.
        proc.running = True
        proc.stop_event.set()

    proc.start = start
    proc._falldata_aux_thread = SimpleNamespace(is_alive=lambda: True, join=join)
    proc._falldata_aux = SimpleNamespace(close=close)
    proc.event_publisher = SimpleNamespace(disconnect=lambda: cleanup.append("disconnect"))
    monkeypatch.setattr(deepstream_processor, "stop_pipeline_runtime", stop_pipeline)
    monkeypatch.setattr(runtime, "create_processor", lambda cfg: proc)
    monkeypatch.setattr(runtime, "collect_active_cameras", lambda *args: [("cam", "file", {}, {}, {})])
    monkeypatch.setattr(runtime, "connect_cameras_parallel", lambda *args: {"cam": True})
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("STREAM_PORT", "0")
    monkeypatch.setenv("STREAM_API_ENABLED", "0")
    monkeypatch.setenv("METRICS_PORT", "0")
    monkeypatch.setenv("AI_ENGINE_RESTART_ON_PIPELINE_ERROR", "1")

    interrupted = None
    try:
        runtime.start_processor_runtime([{}], SimpleNamespace(display=False))
    except SystemExit as exc:
        interrupted = exc

    assert cleanup == [
        "pipeline_enter", "pipeline_exit", "aux_join", "aux_finished", "model_close", "disconnect"
    ]
    assert interrupted is None, "시그널이 진행 중인 정리를 SystemExit로 중단함"
    assert len(starts) == 1, "종료 요청 뒤 파이프라인을 다시 시작함"
    assert proc._pipeline is None
    assert proc._falldata_aux_thread is None
