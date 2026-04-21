"""stream_api.py - 실시간 MJPEG 비디오 스트리밍 서버.

각 카메라의 최신 프레임을 MJPEG 형식(multipart/x-mixed-replace)으로 스트리밍한다.
원격 모니터링 대시보드나 웹 브라우저에서 직접 접근할 수 있다.

Routes:
    GET /cameras                → 사용 가능한 카메라 ID 목록 (JSON)
    GET /stream/<camera_id>     → MJPEG 스트림 (브라우저 img src로 사용 가능)

사용법::

    from src.services.stream_api import start_stream_api_server

    start_stream_api_server(processor=processor, port=8769)

    # 브라우저에서: http://<host>:8769/stream/<camera_id>
    # HTML img 태그: <img src="http://<host>:8769/stream/camera-1">
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .._http_server import BaseApiHandler, ThreadingApiServer

if TYPE_CHECKING:
    from ..core.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

_RE_STREAM = re.compile(r"^/stream/([^/]+)$")


def _read_stream_fps() -> float:
    """환경 변수에서 스트리밍 FPS를 읽고 안전한 범위로 보정한다."""
    raw = os.environ.get("STREAM_FPS", "15")
    try:
        fps = float(raw)
    except (TypeError, ValueError):
        logger.warning("잘못된 STREAM_FPS=%r → 기본값 15 사용", raw)
        return 15.0
    return min(max(fps, 1.0), 30.0)


def _read_jpeg_quality() -> int:
    """환경 변수에서 JPEG 품질을 읽고 안전한 범위로 보정한다."""
    raw = os.environ.get("STREAM_JPEG_QUALITY", "75")
    try:
        quality = int(raw)
    except (TypeError, ValueError):
        logger.warning("잘못된 STREAM_JPEG_QUALITY=%r → 기본값 75 사용", raw)
        return 75
    return min(max(quality, 30), 95)


class StreamApiHandler(BaseApiHandler):
    """MJPEG 스트리밍 HTTP 핸들러.

    ``serve_forever()`` 호출 전에 HTTPServer 인스턴스에
    ``server.processor`` 속성이 설정되어 있어야 한다.
    """

    _LOG_PREFIX = "[StreamAPI]"

    def _processor(self) -> "BaseProcessor":
        return self.server.processor  # type: ignore[attr-defined]

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

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
        else:
            self._consume_body()
            self._respond(404, {"error": "Not Found"})

    def _list_cameras(self) -> None:
        cameras = list(self._processor().cameras.keys())
        self._respond(200, {"cameras": cameras})

    def _health(self) -> None:
        proc = self._processor()
        cameras = list(proc.cameras.keys())
        self._respond(
            200,
            {
                "service": "cctv-stream-api",
                "status": "ok",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "camera_count": len(cameras),
                "cameras": cameras,
                "stream_fps": _read_stream_fps(),
                "jpeg_quality": _read_jpeg_quality(),
            },
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
        try:
            while True:
                frame = proc.get_camera_frame(camera_id, annotated=True)
                if frame is not None:
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
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 클라이언트 연결 끊김 — 정상 종료
        except Exception as exc:
            logger.debug("[%s] 스트리밍 중단: %s", camera_id, exc)


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
    port = int(os.environ.get("STREAM_PORT", port))
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
