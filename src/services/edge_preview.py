"""감독자가 활성화한 실행의 신선한 프레임만 제공하는 선택형 POC 미리보기."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import math
import re
import threading
import time

from src.services.edge_node_agent import RegistryClient
from src.services.edge_orchestrator import RegistryError, _fields, _identifier
from src.services.edge_result_context import EdgeResultContext
from src.services.edge_service import _private_json


class EdgePreviewSource:
    """기존 프레임 저장소를 사용하며 새 촬영·GPU 추론·결과 DB 쓰기를 만들지 않는다."""

    def __init__(self, processor, context_path, session_id, *, clock=time.monotonic):
        self.processor = processor
        self.context_path = context_path
        self.session_id = session_id
        self.clock = clock
        self.lock = threading.Lock()
        self.boundary = None
        self.boundary_at = None

    def frame(self, camera_id):
        import cv2

        with self.lock:
            context = EdgeResultContext(self.context_path)
            try:
                identity = context.preview_identity(self.session_id)
                if (
                    identity["camera_id"] != camera_id
                    or self.processor.edge_runtime_backend != "deepstream"
                    or self.processor.running is not True
                    or self.processor.edge_session_id != self.session_id
                ):
                    raise RegistryError(409, "preview_execution_mismatch")
                if identity != self.boundary:
                    # 첫 활성화 조회 이전에 쌓인 준비 프레임은 정상 영상으로 쓰지 않는다.
                    self.boundary, self.boundary_at = identity, self.clock()
                    raise RegistryError(503, "preview_waiting_for_new_frame")
                packet = self.processor.get_camera_preview(camera_id)
                if packet is None:
                    raise RegistryError(503, "preview_frame_unavailable")
                frame, info = packet
                received = info.get("received_monotonic")
                if (
                    type(received) not in (float, int)
                    or not self.boundary_at < received <= self.clock()
                    or self.clock() - received > 2
                    or info.get("camera_id") != camera_id
                ):
                    raise RegistryError(503, "preview_frame_stale")
                height, width = frame.shape[:2]
                if width > 640:
                    frame = cv2.resize(
                        frame,
                        (640, max(1, height * 640 // width)),
                        interpolation=cv2.INTER_AREA,
                    )
                success, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not success:
                    raise RegistryError(503, "preview_encode_failed")
                image = encoded.tobytes()
                if len(image) > 524288:
                    raise RegistryError(503, "preview_image_too_large")
                latest = self.processor.get_camera_preview(camera_id)
                if (
                    context.preview_identity(self.session_id) != identity
                    or self.processor.running is not True
                    or latest is None
                    or latest[1].get("generation") != info.get("generation")
                    or self.clock() - received > 2
                ):
                    raise RegistryError(409, "preview_execution_changed")
                return {
                    "edge": identity,
                    "frame": {
                        key: info[key]
                        for key in (
                            "camera_id",
                            "generation",
                            "frame_token",
                            "frame_num",
                            "source_pts_ns",
                            "received_at",
                        )
                    },
                    "age_seconds": self.clock() - received,
                    "jpeg_base64": base64.b64encode(image).decode("ascii"),
                    "sha256": hashlib.sha256(image).hexdigest(),
                }
            except cv2.error as error:
                raise RegistryError(503, "preview_encode_failed") from error
            finally:
                context.close()


class EdgePreviewRelay:
    """명시한 영상 끝점에서만 가져오며 전송 전후의 현재 담당을 대조한다."""

    @classmethod
    def from_file(cls, path, store):
        settings = _private_json(path)
        _fields(settings, {"nodes"})
        if not isinstance(settings["nodes"], dict) or not settings["nodes"]:
            raise ValueError("preview nodes are required")
        def peer(item):
            _fields(item, {"url", "ca_file", "credentials_file"})
            secret = _private_json(item["credentials_file"])
            _fields(secret, {"stream_token"})
            token = secret["stream_token"]
            if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token):
                raise ValueError("invalid preview token")
            return RegistryClient(item["url"], token, ca_file=item["ca_file"])

        clients = {}
        for node, item in settings["nodes"].items():
            _identifier(node)
            if isinstance(item, dict) and "cameras" in item:
                _fields(item, {"cameras"})
                if not isinstance(item["cameras"], dict) or not 1 <= len(item["cameras"]) <= 32:
                    raise ValueError("1-32 camera preview endpoints are required")
                clients[node] = {"cameras": {
                    _identifier(camera): peer(value)
                    for camera, value in item["cameras"].items()
                }}
            else:
                clients[node] = peer(item)
        return cls(store, clients)

    def __init__(self, store, clients):
        self.store, self.clients = store, clients

    def frame(self, camera_id, expected_assignment):
        view = self.store.view(camera_id)
        owner = view["active"]
        if not view["inference_ready"] or owner["assignment_id"] != expected_assignment:
            raise RegistryError(409, "preview_owner_unverified")
        client = self.clients.get(owner["node_id"])
        if isinstance(client, dict):
            client = client["cameras"].get(camera_id)
        if client is None:
            raise RegistryError(503, "preview_endpoint_missing")
        started = time.monotonic()
        connection = (
            http.client.HTTPSConnection(
                client.host, client.port, timeout=2, context=client.context
            )
            if client.context
            else http.client.HTTPConnection(client.host, client.port, timeout=2)
        )
        try:
            connection.request(
                "GET",
                "/edge-preview/" + _identifier(camera_id),
                headers={"Authorization": "Bearer " + client.token},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RegistryError(503, "preview_source_unavailable")
            body = response.read(1048577)
            if len(body) > 1048576:
                raise RegistryError(503, "preview_response_too_large")
            payload = json.loads(body)
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise RegistryError(503, "preview_source_unavailable") from error
        finally:
            connection.close()
        _fields(payload, {"edge", "frame", "age_seconds", "jpeg_base64", "sha256"})
        expected = {
            key: owner[key]
            for key in (
                "node_id",
                "camera_id",
                "assignment_id",
                "epoch",
                "supervisor_boot_id",
                "session_id",
                "activation_id",
            )
        }
        if (
            not isinstance(payload["edge"], dict)
            or type(payload["edge"].get("epoch")) is not int
            or payload["edge"] != expected
        ):
            raise RegistryError(409, "preview_execution_mismatch")
        info = payload["frame"]
        _fields(
            info,
            {
                "camera_id",
                "generation",
                "frame_token",
                "frame_num",
                "source_pts_ns",
                "received_at",
            },
        )
        if (
            info["camera_id"] != camera_id
            or any(
                type(info[key]) is not int or info[key] < minimum
                for key, minimum in (
                    ("generation", 0),
                    ("frame_token", 1),
                    ("frame_num", 0),
                    ("source_pts_ns", 0),
                )
            )
            or type(info["received_at"]) not in (int, float)
            or not math.isfinite(info["received_at"])
        ):
            raise RegistryError(503, "preview_frame_invalid")
        age = payload["age_seconds"]
        if type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= 2:
            raise RegistryError(503, "preview_frame_stale")
        try:
            image = base64.b64decode(payload["jpeg_base64"], validate=True)
        except (TypeError, ValueError) as error:
            raise RegistryError(503, "preview_image_invalid") from error
        if (
            not 4 <= len(image) <= 524288
            or not image.startswith(b"\xff\xd8")
            or not image.endswith(b"\xff\xd9")
            or hashlib.sha256(image).hexdigest() != payload["sha256"]
        ):
            raise RegistryError(503, "preview_image_invalid")
        current = self.store.view(camera_id)
        if (
            not current["inference_ready"]
            or current["active"]["assignment_id"] != expected_assignment
            or any(current["active"][key] != value for key, value in expected.items())
        ):
            raise RegistryError(409, "preview_owner_changed")
        payload["age_seconds"] += time.monotonic() - started
        if payload["age_seconds"] > 2:
            raise RegistryError(503, "preview_frame_stale")
        return payload
