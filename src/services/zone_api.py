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
    GET    /                                     → 웹 대시보드 HTML
    GET    /events?limit=N                       → 이벤트 로그 (JSONL)
    GET    /health                               → 시스템 상태
    GET    /known_faces/{filename}               → 등록 얼굴 이미지
"""

import json
import logging
import re
import threading
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
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
_RE_KNOWN_FACE_IMG    = re.compile(r"^/known_faces/([\w.\-]+)$")

_DASHBOARD_HTML  = Path(__file__).resolve().parents[2] / "web" / "index.html"
_KNOWN_FACES_DIR = Path(__file__).resolve().parents[2] / "known_faces"
_SNAPSHOTS_DIR   = Path(__file__).resolve().parents[2] / "snapshots"

_RE_CAMERA_STREAM    = re.compile(r"^/cameras/([^/]+)/stream$")
_RE_CAMERA_SNAPSHOT  = re.compile(r"^/cameras/([^/]+)/snapshot$")
_RE_SNAPSHOT_FILE    = re.compile(r"^/snapshots/([\w.\-/]+)$")


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """MJPEG 스트리밍 지원을 위한 멀티스레드 HTTP 서버."""
    daemon_threads = True


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
        self.send_header("Access-Control-Allow-Origin", "*")
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

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/"):
            self._serve_dashboard()
        elif path == "/cameras":
            self._get_cameras()
        elif path == "/faces":
            self._get_faces()
        elif path == "/zone-presets":
            self._get_presets()
        elif path == "/events":
            self._get_events()
        elif path == "/health":
            self._get_health()
        elif path == "/snapshots":
            self._list_snapshots()
        elif m := _RE_CAMERA_STREAM.match(path):
            self._stream_mjpeg(m.group(1))
        elif m := _RE_CAMERA_SNAPSHOT.match(path):
            self._get_snapshot(m.group(1))
        elif m := _RE_CAMERA_MODELS.match(path):
            self._get_camera_models(m.group(1))
        elif m := _RE_CAMERA_ZONES.match(path):
            self._get_camera_zones(m.group(1))
        elif m := _RE_KNOWN_FACE_IMG.match(path):
            self._serve_image(m.group(1))
        elif m := _RE_SNAPSHOT_FILE.match(path):
            self._serve_snapshot_file(m.group(1))
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

    def _get_health(self) -> None:
        try:
            cameras = self._load_cameras()
            total = len(cameras)
            active = sum(1 for c in cameras if c.get("enabled", True))
        except Exception:
            total = active = 0
        try:
            faces = self._processor().list_registered_faces()
            faces_count = len(faces)
        except Exception:
            faces_count = 0
        events_today = 0
        log_path = getattr(self.server, "event_log_path", None)
        if log_path:
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                with open(log_path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get("receivedAt", "").startswith(today):
                                events_today += 1
                        except json.JSONDecodeError:
                            pass
            except (OSError, FileNotFoundError):
                pass
        self._respond(200, {
            "status": "ok",
            "cameras": {"total": total, "active": active},
            "faces_registered": faces_count,
            "events_today": events_today,
        })

    def _get_events(self) -> None:
        qs = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(qs)
        try:
            limit = max(1, min(int(params.get("limit", [50])[0]), 500))
        except (ValueError, IndexError):
            limit = 50
        log_path = getattr(self.server, "event_log_path", None)
        if not log_path:
            self._respond(200, {"events": []})
            return
        try:
            with open(log_path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except (OSError, FileNotFoundError):
            self._respond(200, {"events": []})
            return
        events = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(events) >= limit:
                break
        self._respond(200, {"events": events})

    def _serve_dashboard(self) -> None:
        try:
            html = _DASHBOARD_HTML.read_bytes()
        except FileNotFoundError:
            self._respond(404, {"error": "dashboard not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

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
        _MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "gif": "image/gif", "webp": "image/webp"}
        mime = _MIME.get(image_path.suffix.lower().lstrip("."), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------
    # 스트리밍 / 스냅샷
    # ------------------------------------------------------------------

    def _stream_mjpeg(self, camera_id: str) -> None:
        """MJPEG 스트리밍 -- 브라우저에서 <img src=".../stream"> 로 연결."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            self._respond(503, {"error": "cv2 not available"})
            return

        import cv2 as _cv2

        proc = self._processor()
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                frame = None
                try:
                    frame = proc.get_camera_frame(camera_id)
                except Exception:
                    pass
                if frame is None:
                    # 프레임 없으면 회색 placeholder 영상 전송
                    import numpy as _np
                    frame = _np.zeros((360, 640, 3), dtype=_np.uint8)
                    _cv2.putText(frame, f"No signal: {camera_id}", (20, 180),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                ok, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    continue
                jpg = buf.tobytes()
                try:
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                        + jpg + b"\r\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                import time as _t
                _t.sleep(0.05)  # ~20 fps 상한
        except Exception:
            pass

    def _get_snapshot(self, camera_id: str) -> None:
        """UD604재 프레임을 JPEG 한 장 반환."""
        try:
            import cv2 as _cv2
            import numpy as _np
        except ImportError:
            self._respond(503, {"error": "cv2 not available"})
            return
        proc = self._processor()
        frame = None
        try:
            frame = proc.get_camera_frame(camera_id)
        except Exception:
            pass
        if frame is None:
            self._respond(404, {"error": "no frame available"})
            return
        ok, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            self._respond(500, {"error": "encode failed"})
            return
        jpg = buf.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(jpg)

    def _list_snapshots(self) -> None:
        """snapshots/ 디렉토리 안의 파일 목록을 반환."""
        snap_dir = getattr(self.server, "snapshot_dir", str(_SNAPSHOTS_DIR))
        base = Path(snap_dir)
        result = []
        if base.exists():
            for cam_dir in sorted(base.iterdir()):
                if not cam_dir.is_dir():
                    continue
                for f in sorted(cam_dir.glob("*.jpg"), reverse=True):
                    result.append({
                        "camera_id": cam_dir.name,
                        "filename": f"{cam_dir.name}/{f.name}",
                        "name": f.name,
                        "size": f.stat().st_size,
                        "mtime": f.stat().st_mtime,
                        "url": f"/snapshots/{cam_dir.name}/{f.name}",
                    })
        self._respond(200, {"snapshots": result})

    def _serve_snapshot_file(self, rel_path: str) -> None:
        """snapshots/{camera_id}/{filename}.jpg 서빙."""
        # 경로 조작 방지
        if ".." in rel_path or rel_path.startswith("/"):
            self._respond(403, {"error": "forbidden"})
            return
        parts = rel_path.split("/")
        if len(parts) != 2 or not re.match(r"^[\w.\-]+$", parts[0]) or not re.match(r"^[\w.\-]+$", parts[1]):
            self._respond(400, {"error": "invalid path"})
            return
        snap_dir = getattr(self.server, "snapshot_dir", str(_SNAPSHOTS_DIR))
        image_path = (Path(snap_dir) / parts[0] / parts[1]).resolve()
        base_resolved = Path(snap_dir).resolve()
        if base_resolved not in image_path.parents:
            self._respond(403, {"error": "forbidden"})
            return
        try:
            data = image_path.read_bytes()
        except FileNotFoundError:
            self._respond(404, {"error": "not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
    event_log_path: Optional[str] = None,
    snapshot_dir: str = "snapshots",
) -> None:
    """Zone API HTTP 서버를 백그라운드 데몬 스레드로 시작한다.

    매개변수:
        processor:          VideoProcessor 인스턴스
        cameras_json_path:  cameras.json 경로 (구역 저장 대상)
        port:               수신 TCP 포트 번호
        presets_path:       zone_presets.json 경로 (기본값: zone_presets.json)
        event_log_path:     alert_api_events.jsonl 경로 (대시보드 이벤트 로그용)
        snapshot_dir:       자동 스냅샷 저장 디렉토리 (기본값: snapshots)
    """
    server = _ThreadingHTTPServer(("0.0.0.0", port), ZoneApiHandler)
    server.processor = processor  # type: ignore[attr-defined]
    server.cameras_json_path = cameras_json_path  # type: ignore[attr-defined]
    server.preset_store = ZonePresetStore(presets_path)  # type: ignore[attr-defined]
    server.event_log_path = event_log_path  # type: ignore[attr-defined]
    server.snapshot_dir = snapshot_dir  # type: ignore[attr-defined]
    # processor에 snapshot_dir 동기화 (자동 스냅샷 저장경로 공유)
    if processor is not None:
        processor.snapshot_dir = snapshot_dir  # type: ignore[attr-defined]
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

