"""camera_model_api.py - 카메라 AI 모델 설정 REST API 서버.

카메라별 AI 모델 on/off 설정(use_pose, use_helmet, use_person 등)을
조회·변경하는 경량 HTTP 서버를 제공한다.
HTTPServer + BaseHTTPRequestHandler 기반으로 의존성 없이 동작하며,
백그라운드 데몬 스레드로 실행된다.

사용법::

    from src.services.camera_model_api import start_camera_model_api_server

    start_camera_model_api_server(
        processor=processor,
        cameras_json_path='cameras.json',
        port=8766,
    )

Routes:
    GET   /cameras/{id}/models  → 특정 카메라 모델 on/off 상태 조회
    POST  /cameras/{id}/models  → 모델 on/off 상태 변경 (cameras.json에 저장)
"""

import json
import logging
import re
import threading
from typing import TYPE_CHECKING

from .._http_server import BaseApiHandler, ThreadingApiServer

if TYPE_CHECKING:
    from ..core import VideoProcessor

logger = logging.getLogger(__name__)

_RE_CAMERA_MODELS = re.compile(r"^/cameras/([^/]+)/models$")

_ALLOWED_KEYS = frozenset({
    "use_pose", "use_helmet", "use_person", "use_face",
    "use_appearance",
    "pose", "helmet", "person", "face", "appearance",
})


class CameraModelApiHandler(BaseApiHandler):
    """카메라 AI 모델 설정 REST API 핸들러.

    ``serve_forever()`` 호출 전에 HTTPServer 인스턴스에
    아래 두 속성이 반드시 설정되어 있어야 한다:
        server.processor          – VideoProcessor 인스턴스
        server.cameras_json_path  – cameras.json 파일 경로
    """

    _LOG_PREFIX = "[CameraModelAPI]"

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _processor(self) -> "VideoProcessor":
        return self.server.processor  # type: ignore[attr-defined]

    def _cameras_path(self) -> str:
        return self.server.cameras_json_path  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # 디스패치
    # ------------------------------------------------------------------

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if not self._check_internal_token():
            return
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            self._health()
        elif m := _RE_CAMERA_MODELS.match(path):
            self._get_camera_models(m.group(1))
        else:
            self._respond(404, {"error": "Not Found"})

    def do_POST(self):  # noqa: N802
        if not self._check_internal_token():
            return
        path = self.path.rstrip("/")
        if m := _RE_CAMERA_MODELS.match(path):
            self._post_camera_models(m.group(1))
        else:
            # 요청 본문을 소비해야 Windows에서 연결 리셋(WinError 10053) 방지
            self._consume_body()
            self._respond(404, {"error": "Not Found"})

    # ------------------------------------------------------------------
    # GET 핸들러
    # ------------------------------------------------------------------

    def _health(self) -> None:
        processor = self._processor()
        camera_status = processor.get_camera_status() if hasattr(processor, "get_camera_status") else {}
        self._respond(
            200,
            self._build_health_payload(
                service="cctv-camera-model-api",
                status="ok",
                camera_count=len(camera_status),
            ),
        )

    def _get_camera_models(self, camera_id: str) -> None:
        settings = self._processor().get_camera_model_settings(camera_id)
        if settings is None:
            self._respond(404, {"error": f"camera '{camera_id}' not found"})
            return
        self._respond(200, {"camera_id": camera_id, "model_settings": settings})

    # ------------------------------------------------------------------
    # POST 핸들러
    # ------------------------------------------------------------------

    def _post_camera_models(self, camera_id: str) -> None:
        body = self._read_json()
        if body is None:
            self._respond(400, {"error": "Invalid JSON"})
            return

        if not any(key in body for key in _ALLOWED_KEYS):
            self._respond(400, {"error": "model_settings payload is required"})
            return

        try:
            settings = self._processor().update_camera_model_settings(
                camera_id,
                body,
                self._cameras_path(),
            )
        except KeyError as exc:
            self._respond(404, {"error": str(exc)})
            return
        except Exception as exc:
            logger.error("[CameraModelAPI] 모델 설정 업데이트 실패: %s", exc)
            self._respond(500, {"error": "model settings update failed"})
            return

        if settings is None:
            self._respond(404, {"error": f"camera '{camera_id}' not found"})
            return

        processor = self._processor()
        pipeline_restarting = bool(getattr(processor, "_pipeline_restart_pending", False))

        self._respond(200, {
            "status": "ok",
            "camera_id": camera_id,
            "model_settings": settings,
            "pipeline_restarting": pipeline_restarting,
        })


# ===========================================================================
# 공개 API
# ===========================================================================


def start_camera_model_api_server(
    processor: "VideoProcessor",
    cameras_json_path: str,
    port: int,
) -> None:
    """Camera Model API HTTP 서버를 백그라운드 데몬 스레드로 시작한다.

    매개변수:
        processor:          VideoProcessor 인스턴스
        cameras_json_path:  cameras.json 경로 (모델 설정 저장 대상)
        port:               수신 TCP 포트 번호
    """
    server = ThreadingApiServer(("0.0.0.0", port), CameraModelApiHandler)
    server.processor = processor  # type: ignore[attr-defined]
    server.cameras_json_path = cameras_json_path  # type: ignore[attr-defined]
    threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="CameraModelApiServer",
    ).start()
    logger.info("Camera Model API 서버 시작: http://0.0.0.0:%d", port)
    logger.info("  GET   /health")
    logger.info("  GET   /cameras/{id}/models")
    logger.info("  POST  /cameras/{id}/models")


__all__ = ["CameraModelApiHandler", "start_camera_model_api_server"]
