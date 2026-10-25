"""기존 노드 등록 API에 명시적으로 활성화하는 중앙 담당·결과 저장 API."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from src.services.edge_orchestrator import (
    EdgeRegistryHandler,
    RegistryError,
    _fields,
    create_edge_server,
)


class EdgeControllerHandler(EdgeRegistryHandler):
    """노드 결과 접수·제어·최종 모의 수신의 인증 권한을 구분한다."""

    def end_headers(self):
        if self.path.endswith("/preview"):
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def _dispatch(self, method):
        if method == "GET" and self.path in ("/viewer", "/viewer.js"):
            return self._viewer_asset()
        if self.path in ("/v1/nodes", "/v1/nodes/register") or re.fullmatch(
            r"/v1/nodes/[A-Za-z0-9_.-]+/reports", self.path
        ):
            return super()._dispatch(method)
        self.close_connection = True
        store = self.server.ownership
        try:
            camera = re.fullmatch(
                r"/v1/cameras/([A-Za-z0-9_.-]+)(/(?:prepare|progress|view|preview))?",
                self.path,
            )
            assignment = re.fullmatch(
                r"/v1/assignments/([A-Za-z0-9_.-]+)/(activate|abort)", self.path
            )
            result = re.fullmatch(r"/v1/nodes/([A-Za-z0-9_.-]+)/results", self.path)
            receipt = re.fullmatch(
                r"/v1/nodes/([A-Za-z0-9_.-]+)/receipts/([A-Za-z0-9_.-]+)", self.path
            )
            consume = re.fullmatch(r"/v1/results/([A-Za-z0-9_.-]+)/consume", self.path)
            if method == "GET" and camera and not camera[2]:
                self._authorize()
                response = store.camera(camera[1])
            elif method == "GET" and camera and camera[2] == "/progress":
                self._authorize()
                response = store.result_progress(camera[1])
            elif method == "GET" and camera and camera[2] == "/view":
                self._authorize()
                response = store.view(camera[1])
                response["preview_available"] = self.server.preview_relay is not None
            elif method == "GET" and camera and camera[2] == "/preview":
                self._authorize()
                if self.server.preview_relay is None:
                    raise RegistryError(503, "preview_disabled")
                response = self.server.preview_relay.frame(
                    camera[1], self.headers.get("X-Edge-Assignment-Id")
                )
            elif method == "POST" and camera and camera[2] == "/prepare":
                self._authorize_token(self.server.role_tokens["control"])
                response = store.prepare(camera[1], self._body())
            elif method == "POST" and assignment:
                self._authorize_token(self.server.role_tokens["control"])
                action = store.activate if assignment[2] == "activate" else store.abort
                response = action(assignment[1], self._body())
            elif method == "POST" and result:
                self._authorize(result[1])
                response = store.receive(result[1], self._body())
            elif method == "GET" and receipt:
                self._authorize(receipt[1])
                response = store.receipt(receipt[1], receipt[2])
            elif method == "GET" and self.path == "/v1/results/pending":
                self._authorize_token(self.server.role_tokens["consumer"])
                response = {"results": store.pending()}
            elif method == "POST" and consume:
                self._authorize_token(self.server.role_tokens["consumer"])
                body = self._body()
                _fields(body, {"sha256"})
                response = store.consume(consume[1], body["sha256"])
            else:
                raise RegistryError(404, "not_found")
            self._respond(200, response)
        except RegistryError as error:
            self._respond(error.status, {"error": error.code})
        except sqlite3.Error as exc:
            self._storage_unavailable(exc)

    def _viewer_asset(self):
        # 빈 조회 화면만 공개한다. 실제 상태는 기존 읽기 토큰 없이는 조회할 수 없다.
        self.close_connection = True
        filename, content_type = (
            ("edge-viewer.html", "text/html; charset=utf-8")
            if self.path == "/viewer"
            else ("edge-viewer.js", "text/javascript; charset=utf-8")
        )
        try:
            body = (Path(__file__).resolve().parents[2] / "web" / filename).read_bytes()
        except OSError:
            self._respond(503, {"error": "viewer_asset_unavailable"})
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; "
            "connect-src 'self'; img-src data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(body)


def create_controller_server(
    registry,
    ownership,
    node_tokens,
    reader_token,
    control_token,
    consumer_token,
    preview_relay=None,
    **options,
):
    server = create_edge_server(
        EdgeControllerHandler,
        node_tokens,
        reader_token,
        role_tokens={"control": control_token, "consumer": consumer_token},
        allow_remote=registry.allow_remote_nodes,
        **options,
    )
    server.preview_relay = preview_relay
    server.registry = registry
    server.ownership = ownership
    return server
