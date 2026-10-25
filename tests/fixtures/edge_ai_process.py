"""실제 자식 프로세스·소켓 수명을 시험하는 모형 지표 서버이며 AI 추론은 하지 않는다."""

import logging
import os
import signal
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Queue

started = time.monotonic()
mode_path = Path(os.environ["FIXTURE_MODE_PATH"])
if os.environ.get("FIXTURE_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def read_mode():
    # 시험 제어기가 모드 파일을 제거하는 순간에도 작업자가 종료되지 않게 한다.
    try:
        return mode_path.read_text()
    except FileNotFoundError:
        return "running"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        mode = read_mode()
        count = int((time.monotonic() - started) * 30) + 1
        if mode == "stalled":
            count = 0
        session = os.environ["EDGE_SERVICE_SESSION_ID"]
        backend = "deepstream"
        if mode == "wrong-session":
            session = "old-session"
        if mode == "cpu":
            backend = "video_processor"
        rows = [
            f'cctv_edge_service_info{{backend="{backend}",session_id="{session}"}} 1'
        ]
        for name in ("primary", "helmet"):
            labels = f'camera_id="camera_1",model="{name}"'
            tensor = 0 if name == "helmet" else count
            empty = count if name == "helmet" else 0
            rows += [
                f'cctv_model_enabled{{model="{name}"}} 1',
                f'cctv_model_unique_id_conflict{{model="{name}"}} 0',
                f"cctv_model_frames_tensor_output_total{{{labels}}} {tensor}",
                f"cctv_model_frames_no_objects_total{{{labels}}} {empty}",
                f"cctv_model_last_frame_age_seconds{{{labels}}} 0.01",
            ]
        if mode == "duplicate":
            rows.append(rows[-1])
        body = ("\n".join(rows) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


server = HTTPServer(("127.0.0.1", int(os.environ["METRICS_PORT"])), Handler)
if os.environ.get("FIXTURE_EMIT_EVENTS") == "1":
    from src.core._event_publish import run_publish_loop
    from src.core.events import DetectionEvent, EventType
    from src.core.video_event_ingest import VideoEventIngest

    intake = VideoEventIngest.from_env(backend="deepstream", topic_prefix="poc/unused")
    # 제품의 별도 지표 서버·발행/복구 작업자를 사용해 DB 지연을 모형 프로세스 종료로 오판하지 않는다.
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    event_queue = Queue(maxsize=50)

    def publishing_enabled():
        # 접수 저장과 전송 큐 사이의 크래시를 실제 발행 작업자에서 재현한다.
        paused = mode_path.parent / "publish-paused"
        while read_mode() == "hold_publish":
            paused.touch()
            time.sleep(0.02)
        paused.unlink(missing_ok=True)
        return True

    threading.Thread(
        target=run_publish_loop,
        kwargs={
            "is_running": publishing_enabled,
            "stop_event": threading.Event(),
            "event_queue": event_queue,
            "topic_prefix": "poc/unused",
            "mqtt_publish": None,
            "event_publisher": None,
            "increment_stat": lambda name: 0,
            "queue_timeout_sec": 0.05,
            "video_ingest": intake,
        },
        daemon=True,
    ).start()
    event = None
    while True:
        if event is None and read_mode() in ("drain", "drained"):
            # 전수 대조 전에 입력 종료를 확인하고 이미 접수한 원본의 복구 작업자는 계속 돌린다.
            if read_mode() == "drain":
                mode_path.write_text("drained")
            time.sleep(0.05)
            continue
        if event is None:
            # 추론을 모사한 입력만 만들며 실제 원본 접수·중앙 송신 코드를 그대로 통과시킨다.
            event = DetectionEvent(
                EventType.PERSON,
                1,
                2,
                30,
                40,
                0.8,
                time.time(),
                metadata={"camera_id": "camera_1"},
            )
        try:
            event_queue.put(intake.accept(event, "camera_1"))
            event = None
        except sqlite3.Error as error:
            # 접수 전 실패는 같은 입력을 보유해 재시도하며 저장 성공으로 세지 않는다.
            logging.warning("fixture intake pending: %s", type(error).__name__)
        time.sleep(0.2)
else:
    server.serve_forever(poll_interval=0.05)
