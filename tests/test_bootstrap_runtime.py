import json
import signal
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.bootstrap import runtime
from src.bootstrap.runtime import load_camera_list


def test_load_camera_list_accepts_existing_file_uri(tmp_path):
    video_file = tmp_path / "sample video.mp4"
    video_file.write_bytes(b"")
    cameras_file = tmp_path / "cameras.json"
    cameras_file.write_text(
        json.dumps([{"id": "sample_eval", "source": video_file.as_uri()}]),
        encoding="utf-8",
    )

    cameras = load_camera_list(str(cameras_file))

    assert cameras == [{"id": "sample_eval", "source": video_file.as_uri()}]


def test_load_camera_list_skips_missing_file_uri(tmp_path):
    missing_video = tmp_path / "missing.mp4"
    cameras_file = tmp_path / "cameras.json"
    cameras_file.write_text(
        json.dumps([{"id": "sample_eval", "source": missing_video.as_uri()}]),
        encoding="utf-8",
    )

    cameras = load_camera_list(str(cameras_file))

    assert cameras == []


@pytest.mark.parametrize("phase", ["idle", "initial_retry", "restart_wait", "startup"])
def test_shutdown_request_exits_wait_or_start_without_restarting(monkeypatch, phase):
    # 종료 신호를 정리 밖에서도 처리하고 초기 연결/재시작 대기를 즉시 빠져나와야 한다.
    processor = SimpleNamespace(running=True, stop_event=threading.Event(), cameras={})
    handlers = {}
    releases = []
    starts = []
    waits = []
    monkeypatch.setattr(runtime.signal, "signal", lambda sig, cb: handlers.setdefault(sig, cb))
    monkeypatch.setattr(runtime.atexit, "register", releases.append)
    runtime.register_shutdown_handlers([processor])

    def request_shutdown():
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    def sleep(seconds):
        waits.append(seconds)
        request_shutdown()

    def start():
        starts.append(True)
        if phase == "startup":
            request_shutdown()
        processor.stop_event.set()

    processor.start = start
    processor.stop = MagicMock()
    processor.add_camera = MagicMock(return_value=False)
    processor.enqueue_camera_retry = MagicMock()
    processor.release_all_cameras = MagicMock()
    monkeypatch.setattr(runtime, "create_processor", lambda cfg: processor)
    monkeypatch.setattr(runtime, "collect_active_cameras", lambda *args: [("cam", "file", {}, {}, {})])
    monkeypatch.setattr(runtime, "connect_cameras_parallel", lambda *args: {"cam": phase != "initial_retry"})
    monkeypatch.setattr(runtime.time, "sleep", sleep)
    monkeypatch.setenv("STREAM_PORT", "0")
    monkeypatch.setenv("STREAM_API_ENABLED", "0")
    monkeypatch.setenv("METRICS_PORT", "0")
    monkeypatch.setenv("AI_ENGINE_RESTART_ON_PIPELINE_ERROR", "1")

    runtime.start_processor_runtime([] if phase == "idle" else [{}], SimpleNamespace(display=False))

    assert processor._shutdown_requested is True
    assert processor.running is False
    assert len(starts) == int(phase in {"startup", "restart_wait"})
    assert processor.stop.call_count == len(starts)
    assert all(0 < seconds <= 0.5 for seconds in waits)
    processor.add_camera.assert_not_called()
    # 시그널 안에서 카메라를 해제하지 않고 기존 atexit 경로를 보존한다.
    processor.release_all_cameras.assert_not_called()
    assert len(releases) == 1
