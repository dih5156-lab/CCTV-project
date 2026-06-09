"""stream_api.py - 실시간 MJPEG 비디오 스트리밍 서버.

각 카메라의 최신 프레임을 MJPEG 형식(multipart/x-mixed-replace)으로 스트리밍한다.
원격 모니터링 대시보드나 웹 브라우저에서 직접 접근할 수 있다.

Routes:
    GET /cameras                → 사용 가능한 카메라 ID 목록 (JSON)
    GET /stream/<camera_id>     → MJPEG 스트림 (브라우저 img src로 사용 가능)
    GET /snapshot/<camera_id>   → 최신 프레임 1장을 JPEG로 반환 (구역 편집용)

사용법::

    from src.services.stream_api import start_stream_api_server

    start_stream_api_server(processor=processor, port=8769)

    # 브라우저에서: http://<host>:8769/stream/<camera_id>
    # HTML img 태그: <img src="http://<host>:8769/stream/camera-1">
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING

from .._http_server import BaseApiHandler, ThreadingApiServer
from ..utils.env import get_env_float, get_env_int

if TYPE_CHECKING:
    from ..core.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

_RE_STREAM = re.compile(r"^/stream/([^/]+)$")
_RE_SNAPSHOT = re.compile(r"^/snapshot/([^/]+)$")


def _read_stream_fps() -> float:
    """환경 변수에서 스트리밍 FPS를 읽고 안전한 범위로 보정한다."""
    return get_env_float("STREAM_FPS", 30.0, minimum=1.0, maximum=60.0, logger=logger)


def _read_jpeg_quality() -> int:
    """환경 변수에서 JPEG 품질을 읽고 안전한 범위로 보정한다."""
    return get_env_int("STREAM_JPEG_QUALITY", 75, minimum=30, maximum=95, logger=logger)


def _read_stream_size() -> tuple[int, int]:
    """환경 변수에서 MJPEG 송출 해상도를 읽는다. 0이면 원본 크기를 유지한다."""
    width = get_env_int("STREAM_WIDTH", 0, minimum=0, maximum=3840, logger=logger)
    height = get_env_int("STREAM_HEIGHT", 0, minimum=0, maximum=2160, logger=logger)
    if width <= 0 or height <= 0:
        return 0, 0
    return width, height


def _get_camera_frame_for_stream(proc: "BaseProcessor", camera_id: str):
    """프로세서 공통 인터페이스를 통해 복사본 없는 프레임을 요청한다."""
    return proc.get_camera_frame(camera_id, annotated=True, copy_frame=False)


def _resize_for_stream(cv2, frame, width: int, height: int):
    if width <= 0 or height <= 0:
        return frame
    frame_height, frame_width = frame.shape[:2]
    if frame_width == width and frame_height == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class StreamApiHandler(BaseApiHandler):
    """MJPEG 스트리밍 HTTP 핸들러.

    ``serve_forever()`` 호출 전에 HTTPServer 인스턴스에
    ``server.processor`` 속성이 설정되어 있어야 한다.
    """

    _LOG_PREFIX = "[StreamAPI]"

    def _processor(self) -> "BaseProcessor":
        return self.server.processor  # type: ignore[attr-defined]

    def do_OPTIONS(self):  # noqa: N802
        self._respond_options("GET, OPTIONS", headers="")

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/"):
            self._list_cameras()
        elif path == "/health":
            self._health()
        elif path == "/cameras":
            self._list_cameras()
        elif m := _RE_STREAM.match(path):
            self._stream(m.group(1))
        elif m := _RE_SNAPSHOT.match(path):
            self._snapshot(m.group(1))
        else:
            self._consume_body()
            self._respond(404, {"error": "Not Found"})

    def _list_cameras(self) -> None:
        cameras = list(self._processor().cameras.keys())
        self._respond(200, {"cameras": cameras})

    def _health(self) -> None:
        proc = self._processor()
        cameras = list(proc.cameras.keys())
        stream_width, stream_height = _read_stream_size()
        self._respond(
            200,
            self._build_health_payload(
                service="cctv-stream-api",
                status="ok",
                camera_count=len(cameras),
                cameras=cameras,
                stream_fps=_read_stream_fps(),
                jpeg_quality=_read_jpeg_quality(),
                stream_size={
                    "width": stream_width,
                    "height": stream_height,
                },
            ),
        )

    def _stream(self, camera_id: str) -> None:
        try:
            import cv2
        except ImportError:
            self._respond(503, {"error": "cv2 not available"})
            return

        proc = self._processor()
        if camera_id not in proc.cameras:
            self._respond(404, {"error": f"Camera '{camera_id}' not found"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        interval = 1.0 / _read_stream_fps()
        jpeg_quality = _read_jpeg_quality()
        stream_width, stream_height = _read_stream_size()
        try:
            # 단순 time.sleep(interval) 대신 deadline 보정으로 인코딩 시간 누적 드리프트 방지
            deadline = time.monotonic()
            while True:
                frame = _get_camera_frame_for_stream(proc, camera_id)
                if frame is not None:
                    frame = _resize_for_stream(cv2, frame, stream_width, stream_height)
                    ret, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )
                    if ret:
                        data = jpeg.tobytes()
                        header = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        )
                        self.wfile.write(header + data + b"\r\n")
                        self.wfile.flush()
                deadline += interval
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    # 처리 지연이 한 프레임 이상 누적된 경우 deadline 리셋
                    deadline = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 클라이언트 연결 끊김 — 정상 종료
        except Exception as exc:
            logger.debug("[%s] 스트리밍 중단: %s", camera_id, exc)

    def _snapshot(self, camera_id: str) -> None:
        try:
            import cv2
        except ImportError:
            self._respond(503, {"error": "cv2 not available"})
            return

        proc = self._processor()
        if camera_id not in proc.cameras:
            self._respond(404, {"error": f"Camera '{camera_id}' not found"})
            return

        frame = proc.get_camera_frame(camera_id, annotated=True)
        if frame is None:
            self._respond(503, {"error": "Frame not ready"})
            return

        ret, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _read_jpeg_quality()]
        )
        if not ret:
            self._respond(500, {"error": "JPEG encode failed"})
            return

        data = jpeg.tobytes()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 클라이언트가 snapshot 갱신 중 연결을 닫은 정상 케이스


def start_stream_api_server(
    processor: "BaseProcessor",
    port: int = 8769,
) -> None:
    """MJPEG 스트리밍 서버를 백그라운드 데몬 스레드로 시작한다.

    매개변수:
        processor: 프레임을 제공하는 프로세서 인스턴스
        port:      서버 포트 (기본값 8769, 환경 변수 STREAM_PORT로 덮어쓸 수 있음)

    환경 변수:
        STREAM_PORT         서버 포트 (기본값: 8769)
        STREAM_FPS          스트리밍 프레임 레이트 (기본값: 15)
        STREAM_JPEG_QUALITY JPEG 품질 0~100 (기본값: 75)
    """
    port = get_env_int("STREAM_PORT", port, minimum=1, maximum=65535, logger=logger)
    try:
        server = ThreadingApiServer(("", port), StreamApiHandler)
        server.processor = processor  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="StreamAPI")
        thread.start()
        logger.info(
            "MJPEG 스트리밍 서버 시작: http://0.0.0.0:%d/stream/<camera_id> "
            "(카메라 목록: http://0.0.0.0:%d/cameras, 상태: http://0.0.0.0:%d/health)",
            port,
            port,
            port,
        )
    except OSError as exc:
        logger.error("스트리밍 서버 시작 실패 (포트 %d): %s", port, exc)
