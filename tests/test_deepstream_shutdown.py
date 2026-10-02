"""영상 저장이 끝나기 전에 인터프리터 종료/자원 해제가 진행되지 않아야 한다."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

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

