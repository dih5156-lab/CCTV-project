"""zone_api.py - 위험구역 설정 REST API 서버.

카메라별 위험구역(폴리곤/라인)을 조회·저장·삭제하는 경량 HTTP 서버를 제공한다.
HTTPServer + BaseHTTPRequestHandler 기반으로 의존성 없이 동작하며,
백그라운드 데몬 스레드로 실행된다.

사용법::

    from src.services.zone_api import start_zone_api_server

    start_zone_api_server(
        processor=processor,
        cameras_json_path='cameras.json',
        port=8765,
        presets_path='zone_presets.json',
    )

Routes:
    GET    /cameras                              → 전체 카메라 목록 + zones
    GET    /cameras/{id}/zones                   → 특정 카메라 구역 목록
    GET    /cameras/{id}/models                  → 특정 카메라 모델 on/off 상태
    POST   /cameras/{id}/zones                   → 구역 전체 교체 (cameras.json에 저장)
    POST   /cameras/{id}/models                  → 모델 on/off 상태 변경 (cameras.json에 저장)
    DELETE /cameras/{id}/zones/{zone_id}         → 특정 구역 삭제
    GET    /zone-presets                         → 저장된 프리셋 목록 (드롭박스용)
    GET    /faces                                → 등록 얼굴 목록
    POST   /zone-presets                         → 새 프리셋 저장
    POST   /faces                                → 얼굴 등록 (name + image_base64)
    DELETE /zone-presets/{preset_id}             → 프리셋 삭제
    DELETE /faces/{face_id}                      → 등록 얼굴 삭제
    POST   /cameras/{id}/zones/from-preset/{pid} → 프리셋을 카메라에 적용
"""

import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..utils.zone_presets import ZonePresetStore

if TYPE_CHECKING:
    from ..core import VideoProcessor

logger = logging.getLogger(__name__)

# 라우트 패턴 — 모듈 로드 시 한 번만 컴파일
_RE_CAMERA_ZONES      = re.compile(r"^/cameras/([^/]+)/zones$")
_RE_CAMERA_MODELS     = re.compile(r"^/cameras/([^/]+)/models$")
_RE_CAMERA_ZONE_ID    = re.compile(r"^/cameras/([^/]+)/zones/([^/]+)$")
_RE_CAMERA_FROM_PRESET = re.compile(r"^/cameras/([^/]+)/zones/from-preset/([^/]+)$")
_RE_PRESET_ID         = re.compile(r"^/zone-presets/([^/]+)$")
_RE_FACE_ID           = re.compile(r"^/faces/([^/]+)$")


# ===========================================================================
# HTTP 핸들러
# ===========================================================================


class ZoneApiHandler(BaseHTTPRequestHandler):
    """카메라 위험구역 설정 REST API 핸들러.

    ``serve_forever()`` 호출 전에 HTTPServer 인스턴스에
    아래 세 속성이 반드시 설정되어 있어야 한다:
        server.processor          – VideoProcessor 인스턴스
        server.cameras_json_path  – cameras.json 파일 경로
        server.preset_store       – ZonePresetStore 인스턴스
    """

    def log_message(self, fmt, *args):  # noqa: A002
        logger.debug("[ZoneAPI] " + fmt, *args)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _processor(self) -> "VideoProcessor":
        return self.server.processor  # type: ignore[attr-defined]

    def _cameras_path(self) -> str:
        return self.server.cameras_json_path  # type: ignore[attr-defined]

    def _presets(self) -> ZonePresetStore:
        return self.server.preset_store  # type: ignore[attr-defined]

    def _read_json(self):
        try:
            length = max(0, int(self.headers.get("Content-Length", 0)))
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            logger.warning("[ZoneAPI] JSON 파싱 실패: %s", exc)
            return None

    def _respond(self, code: int, body) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _load_cameras(self) -> list:
        return json.loads(Path(self._cameras_path()).read_text(encoding="utf-8"))

    def _zones_from_memory(self, camera_id: str) -> Optional[List[dict]]:
        """메모리 상의 ZoneManager에서 camera_id의 구역 목록을 직렬화해 반환한다. 없으면 None."""
        zm = self._processor().zone_manager
        if zm and camera_id in zm.zones:
            return [z.to_dict() for z in zm.zones[camera_id].values()]
        return None

    def _parse_zone_list(self, body) -> Tuple[Optional[List[dict]], Optional[dict]]:
        """zone 목록 유효성 검사 후 (zones_data, None) 또는 (None, error_body)를 반환한다.

        body에 ``zones`` 키가 있고, 각 항목이 ``id``와 타입별 좌표를
        포함하는 dict인지 확인한다.
        """
        if body is None:
            return None, {"error": "Invalid JSON"}
        zones = body.get("zones")
        if not isinstance(zones, list):
            return None, {"error": "'zones' array is required"}
        for z in zones:
            if not isinstance(z, dict) or "id" not in z:
                return None, {"error": "each zone must have 'id'"}
            zone_type = z.get("type", "polygon")
            if zone_type == "line":
                points = z.get("points")
                if not isinstance(points, list) or len(points) != 2:
                    return None, {"error": "line zone must have exactly 2 points"}
            else:
                if "polygon" not in z:
                    return None, {"error": "polygon zone must have 'polygon'"}
                if not isinstance(z["polygon"], list) or len(z["polygon"]) < 3:
                    return None, {"error": "polygon must have at least 3 points"}
        return zones, None

    # ------------------------------------------------------------------
    # 디스패치
    # ------------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        path = self.path.rstrip("/")
        if path == "/cameras":
            self._get_cameras()
        elif path == "/faces":
            self._get_faces()
        elif path == "/zone-presets":
            self._get_presets()
        elif m := _RE_CAMERA_MODELS.match(path):
            self._get_camera_models(m.group(1))
        elif m := _RE_CAMERA_ZONES.match(path):
            self._get_camera_zones(m.group(1))
        else:
            self._respond(404, {"error": "Not Found"})

    def do_POST(self):  # noqa: N802
        path = self.path.rstrip("/")
        if path == "/zone-presets":
            self._post_preset()
        elif path == "/faces":
            self._post_face()
        elif m := _RE_CAMERA_MODELS.match(path):
            self._post_camera_models(m.group(1))
        elif m := _RE_CAMERA_FROM_PRESET.match(path):
            self._post_apply_preset(m.group(1), m.group(2))
        elif m := _RE_CAMERA_ZONES.match(path):
            self._post_camera_zones(m.group(1))
        else:
            # 요청 본문을 먼저 소비해야 Windows에서 연결 리셋(WinError 10053) 방지
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 0:
                    self.rfile.read(length)
            except Exception:
                pass
            self._respond(404, {"error": "Not Found"})

    def do_DELETE(self):  # noqa: N802
        path = self.path.rstrip("/")
        if m := _RE_CAMERA_ZONE_ID.match(path):
            self._delete_camera_zone(m.group(1), m.group(2))
        elif m := _RE_FACE_ID.match(path):
            self._delete_face(m.group(1))
        elif m := _RE_PRESET_ID.match(path):
            self._delete_preset(m.group(1))
        else:
            self._respond(404, {"error": "Not Found"})

    # ------------------------------------------------------------------
    # GET 핸들러
    # ------------------------------------------------------------------

    def _get_cameras(self) -> None:
        result = []
        for cam in self._load_cameras():
            cam_id = cam.get("id")
            zones = self._zones_from_memory(cam_id) or cam.get("zones", [])
            result.append({
                "id": cam_id,
                "name": cam.get("name", cam_id),
                "enabled": cam.get("enabled", True),
                "detections": cam.get("detections", []),
                "model_settings": self._processor().get_camera_model_settings(cam_id)
                or cam.get("model_settings", {}),
                "zones": zones,
            })
        self._respond(200, result)

    def _get_camera_zones(self, camera_id: str) -> None:
        zones = self._zones_from_memory(camera_id)
        if zones is None:
            zones = next(
                (c.get("zones", []) for c in self._load_cameras()
                 if c.get("id") == camera_id),
                [],
            )
        self._respond(200, {"camera_id": camera_id, "zones": zones})

    def _get_presets(self) -> None:
        self._respond(200, self._presets().list_all())

    def _get_faces(self) -> None:
        try:
            faces = self._processor().list_registered_faces()
        except Exception as exc:
            logger.error("[ZoneAPI] 얼굴 목록 조회 실패: %s", exc)
            self._respond(500, {"error": "face list failed"})
            return
        self._respond(200, {"faces": faces})

    def _get_camera_models(self, camera_id: str) -> None:
        settings = self._processor().get_camera_model_settings(camera_id)
        if settings is None:
            self._respond(404, {"error": f"camera '{camera_id}' not found"})
            return
        self._respond(200, {"camera_id": camera_id, "model_settings": settings})

    # ------------------------------------------------------------------
    # POST 핸들러
    # ------------------------------------------------------------------

    def _post_camera_zones(self, camera_id: str) -> None:
        zones, err = self._parse_zone_list(self._read_json())
        if err:
            self._respond(400, err)
            return
        ok = self._processor().update_zones(camera_id, zones, self._cameras_path())
        if ok:
            self._respond(200, {"status": "ok", "camera_id": camera_id,
                                "zones_count": len(zones)})
        else:
            self._respond(500, {"error": "zone update failed (zone_manager may be disabled)"})

    def _post_preset(self) -> None:
        body = self._read_json()
        if body is None:
            self._respond(400, {"error": "Invalid JSON"})
            return
        name = body.get("name", "").strip()
        if not name:
            self._respond(400, {"error": "'name' field is required"})
            return
        zones, err = self._parse_zone_list(body)
        if err:
            self._respond(400, err)
            return
        self._respond(201, self._presets().save(name, zones))

    def _post_face(self) -> None:
        body = self._read_json()
        if body is None:
            self._respond(400, {"error": "Invalid JSON"})
            return

        name = str(body.get("name", "")).strip()
        image_base64 = str(body.get("image_base64", "")).strip()
        filename = body.get("filename")
        if not name:
            self._respond(400, {"error": "'name' field is required"})
            return
        if not image_base64:
            self._respond(400, {"error": "'image_base64' field is required"})
            return

        try:
            face = self._processor().register_face(name, image_base64, filename)
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
            return
        except Exception as exc:
            logger.error("[ZoneAPI] 얼굴 등록 실패: %s", exc)
            self._respond(500, {"error": "face register failed"})
            return

        self._respond(201, {"status": "ok", "face": face})

    def _post_apply_preset(self, camera_id: str, preset_id: str) -> None:
        preset = self._presets().get(preset_id)
        if preset is None:
            self._respond(404, {"error": f"preset '{preset_id}' not found"})
            return
        ok = self._processor().update_zones(camera_id, preset["zones"],
                                            self._cameras_path())
        if ok:
            self._respond(200, {
                "status": "ok",
                "camera_id": camera_id,
                "preset_id": preset_id,
                "preset_name": preset["name"],
                "zones_count": len(preset["zones"]),
            })
        else:
            self._respond(500, {"error": "preset apply failed"})

    def _post_camera_models(self, camera_id: str) -> None:
        body = self._read_json()
        if body is None:
            self._respond(400, {"error": "Invalid JSON"})
            return

        allowed = {"use_pose", "use_helmet", "use_person", "use_face", "pose", "helmet", "person", "face"}
        if not any(key in body for key in allowed):
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
            logger.error("[ZoneAPI] 모델 설정 업데이트 실패: %s", exc)
            self._respond(500, {"error": "model settings update failed"})
            return

        if settings is None:
            self._respond(404, {"error": f"camera '{camera_id}' not found"})
            return

        self._respond(200, {"status": "ok", "camera_id": camera_id, "model_settings": settings})

    # ------------------------------------------------------------------
    # DELETE 핸들러
    # ------------------------------------------------------------------

    def _delete_camera_zone(self, camera_id: str, zone_id: str) -> None:
        processor = self._processor()
        if not processor.zone_manager:
            self._respond(503, {"error": "zone_manager is disabled"})
            return
        current = processor.zone_manager.zones.get(camera_id, {})
        if zone_id not in current:
            self._respond(404, {"error": f"zone '{zone_id}' not found"})
            return
        remaining = [
            z.to_dict()
            for z in current.values()
            if z.zone_id != zone_id
        ]
        ok = processor.update_zones(camera_id, remaining, self._cameras_path())
        if ok:
            self._respond(200, {"status": "ok", "camera_id": camera_id,
                                "deleted_zone_id": zone_id})
        else:
            self._respond(500, {"error": "zone delete failed"})

    def _delete_preset(self, preset_id: str) -> None:
        if self._presets().delete(preset_id):
            self._respond(200, {"status": "ok", "deleted_preset_id": preset_id})
        else:
            self._respond(404, {"error": f"preset '{preset_id}' not found"})

    def _delete_face(self, face_id: str) -> None:
        try:
            deleted = self._processor().delete_face(face_id)
        except Exception as exc:
            logger.error("[ZoneAPI] 얼굴 삭제 실패: %s", exc)
            self._respond(500, {"error": "face delete failed"})
            return
        if deleted:
            self._respond(200, {"status": "ok", "deleted_face_id": face_id})
        else:
            self._respond(404, {"error": f"face '{face_id}' not found"})


# ===========================================================================
# 공개 API
# ===========================================================================


def start_zone_api_server(
    processor: "VideoProcessor",
    cameras_json_path: str,
    port: int,
    presets_path: str = "zone_presets.json",
) -> None:
    """Zone API HTTP 서버를 백그라운드 데몬 스레드로 시작한다.

    매개변수:
        processor:          VideoProcessor 인스턴스
        cameras_json_path:  cameras.json 경로 (구역 저장 대상)
        port:               수신 TCP 포트 번호
        presets_path:       zone_presets.json 경로 (기본값: zone_presets.json)
    """
    server = HTTPServer(("0.0.0.0", port), ZoneApiHandler)
    server.processor = processor  # type: ignore[attr-defined]
    server.cameras_json_path = cameras_json_path  # type: ignore[attr-defined]
    server.preset_store = ZonePresetStore(presets_path)  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="ZoneApiServer").start()
    logger.info("Zone API 서버 시작: http://0.0.0.0:%d", port)
    for line in [
        "  GET    /cameras",
        "  GET    /cameras/{id}/zones",
        "  GET    /cameras/{id}/models",
        "  GET    /faces",
        "  POST   /cameras/{id}/zones",
        "  POST   /cameras/{id}/models",
        "  POST   /faces",
        "  DELETE /cameras/{id}/zones/{zone_id}",
        "  GET    /zone-presets",
        "  POST   /zone-presets",
        "  DELETE /faces/{face_id}",
        "  DELETE /zone-presets/{preset_id}",
        "  POST   /cameras/{id}/zones/from-preset/{pid}",
    ]:
        logger.info(line)


__all__ = ["ZoneApiHandler", "start_zone_api_server"]

