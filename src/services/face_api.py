"""face_api.py - 얼굴 등록/조회/삭제 REST API 서버.

등록 얼굴 관리(CRUD) 및 얼굴 이미지 서빙을 담당하는 경량 HTTP 서버.
HTTPServer + BaseHTTPRequestHandler 기반으로 의존성 없이 동작하며,
백그라운드 데몬 스레드로 실행된다.

사용법::

    from src.services.face_api import start_face_api_server

    start_face_api_server(
        processor=processor,
        port=8767,
    )

Routes:
    GET    /faces                 → 등록 얼굴 목록
    POST   /faces                 → 얼굴 등록 (name + phone + image_base64)
    DELETE /faces/{face_id}       → 등록 얼굴 삭제
    GET    /known_faces/{filename} → 등록 얼굴 이미지 파일 서빙
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from .._http_server import BaseApiHandler, ThreadingApiServer

if TYPE_CHECKING:
    from ..core import VideoProcessor

logger = logging.getLogger(__name__)

# 라우트 패턴
_RE_FACE_ID        = re.compile(r"^/faces/([^/]+)$")
_RE_KNOWN_FACE_IMG = re.compile(r"^/known_faces/([\w.\-]+)$")

_KNOWN_FACES_DIR = Path(__file__).resolve().parents[2] / "known_faces"


# ===========================================================================
# HTTP 핸들러
# ===========================================================================


class FaceApiHandler(BaseApiHandler):
    """얼굴 등록·조회·삭제 REST API 핸들러.

    ``serve_forever()`` 호출 전에 HTTPServer 인스턴스에
    아래 속성이 반드시 설정되어 있어야 한다:
        server.processor  – VideoProcessor 인스턴스
    """

    _LOG_PREFIX = "[FaceAPI]"

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _processor(self) -> "VideoProcessor":
        return self.server.processor  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # 디스패치
    # ------------------------------------------------------------------

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if not self._check_internal_token():
            return
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            self._health()
        elif path == "/faces":
            self._get_faces()
        elif m := _RE_KNOWN_FACE_IMG.match(path):
            self._serve_image(m.group(1))
        else:
            self._respond(404, {"error": "Not Found"})

    def do_POST(self):  # noqa: N802
        if not self._check_internal_token():
            return
        path = self.path.rstrip("/")
        if path == "/faces":
            self._post_face()
        else:
            self._consume_body()
            self._respond(404, {"error": "Not Found"})

    def do_DELETE(self):  # noqa: N802
        if not self._check_internal_token():
            return
        path = self.path.rstrip("/")
        if m := _RE_FACE_ID.match(path):
            self._delete_face(m.group(1))
        else:
            self._respond(404, {"error": "Not Found"})

    # ------------------------------------------------------------------
    # GET 핸들러
    # ------------------------------------------------------------------

    def _health(self) -> None:
        processor = self._processor()
        faces = processor.list_registered_faces() if hasattr(processor, "list_registered_faces") else []
        self._respond(
            200,
            self._build_health_payload(
                service="cctv-face-api",
                status="ok",
                face_count=len(faces),
            ),
        )

    def _get_faces(self) -> None:
        try:
            faces = self._processor().list_registered_faces()
        except Exception as exc:
            logger.error("[FaceAPI] 얼굴 목록 조회 실패: %s", exc)
            self._respond(500, {"error": "face list failed"})
            return
        self._respond(200, {"faces": faces})

    def _serve_image(self, filename: str) -> None:
        # 경로 조작 방지: 영숫자·마침표·하이픈·밑줄만 허용
        if not re.match(r"^[\w.\-]+$", filename):
            self._respond(400, {"error": "invalid filename"})
            return
        image_path = (_KNOWN_FACES_DIR / filename).resolve()
        if _KNOWN_FACES_DIR.resolve() not in image_path.parents:
            self._respond(403, {"error": "forbidden"})
            return
        try:
            data = image_path.read_bytes()
        except FileNotFoundError:
            self._respond(404, {"error": "image not found"})
            return
        _MIME = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp",
        }
        mime = _MIME.get(image_path.suffix.lower().lstrip("."), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------
    # POST 핸들러
    # ------------------------------------------------------------------

    def _post_face(self) -> None:
        body = self._read_json()
        if body is None:
            self._respond(400, {"error": "Invalid JSON"})
            return

        name = str(body.get("name", "")).strip()
        phone = str(body.get("phone", "")).strip()
        image_base64 = str(body.get("image_base64", "")).strip()
        filename = body.get("filename")

        if not name:
            self._respond(400, {"error": "'name' field is required"})
            return
        if not phone:
            self._respond(400, {"error": "'phone' field is required"})
            return
        if not image_base64:
            self._respond(400, {"error": "'image_base64' field is required"})
            return

        # 선택 필드
        department = body.get("department") or None
        position = body.get("position") or None
        employee_id = body.get("employee_id") or None
        hired_at = body.get("hired_at") or None
        note = body.get("note") or None

        try:
            face = self._processor().register_face(
                name=name,
                phone=phone,
                image_base64=image_base64,
                filename=filename,
                department=department,
                position=position,
                employee_id=employee_id,
                hired_at=hired_at,
                note=note,
            )
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
            return
        except Exception as exc:
            logger.error("[FaceAPI] 얼굴 등록 실패: %s", exc)
            self._respond(500, {"error": "face register failed"})
            return

        self._respond(201, {"status": "ok", "face": face})

    # ------------------------------------------------------------------
    # DELETE 핸들러
    # ------------------------------------------------------------------

    def _delete_face(self, face_id: str) -> None:
        try:
            deleted = self._processor().delete_face(face_id)
        except Exception as exc:
            logger.error("[FaceAPI] 얼굴 삭제 실패: %s", exc)
            self._respond(500, {"error": "face delete failed"})
            return
        if deleted:
            self._respond(200, {"status": "ok", "deleted_face_id": face_id})
        else:
            self._respond(404, {"error": f"face '{face_id}' not found"})


# ===========================================================================
# 공개 API
# ===========================================================================


def start_face_api_server(
    processor: "VideoProcessor",
    port: int,
) -> None:
    """Face API HTTP 서버를 백그라운드 데몬 스레드로 시작한다.

    매개변수:
        processor:  VideoProcessor 인스턴스
        port:       수신 TCP 포트 번호
    """
    server = ThreadingApiServer(("0.0.0.0", port), FaceApiHandler)
    server.processor = processor  # type: ignore[attr-defined]
    threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="FaceApiServer",
    ).start()
    logger.info("Face API 서버 시작: http://0.0.0.0:%d", port)
    logger.info("  GET    /health")
    logger.info("  GET    /faces")
    logger.info("  POST   /faces")
    logger.info("  DELETE /faces/{face_id}")
    logger.info("  GET    /known_faces/{filename}")


__all__ = ["FaceApiHandler", "start_face_api_server"]
