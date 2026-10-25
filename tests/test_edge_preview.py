"""실제 JPEG/HTTP 경로의 담당·프레임 경계 시험. 영상은 합성이며 GPU 시험이 아니다."""

import base64
import copy
import http.client
import json
import threading
import time
import uuid
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src._http_server import ThreadingApiServer
from src.core._deepstream_context_worker import DeepStreamContextWorker
from src.core._preview_frame_store import PreviewFrameStore, read_frame_identity
from src.core.deepstream_processor import DeepStreamProcessor
from src.services.edge_node_agent import RegistryClient
from src.services.edge_orchestrator import RegistryError
from src.services.edge_preview import EdgePreviewRelay, EdgePreviewSource
from src.services.edge_result_context import EdgeResultContext
from src.services.stream_api import StreamApiHandler
from tests.test_edge_ownership import lab as lab
from tests.test_edge_view import running


def source_for_owner(root, owner, *, clock=time.monotonic):
    root.mkdir(mode=0o700)
    context = EdgeResultContext(root / "context.sqlite3", create=True)
    identity = {
        key: owner[key]
        for key in (
            "node_id",
            "camera_id",
            "assignment_id",
            "epoch",
            "supervisor_boot_id",
            "session_id",
        )
    }
    context.begin(identity)
    context.activate(
        identity["session_id"],
        {
            "command_id": str(uuid.uuid4()),
            "activation_id": owner["activation_id"],
            "expected_activation_id": None,
        },
    )
    worker = DeepStreamContextWorker.__new__(DeepStreamContextWorker)
    worker._state_lock = threading.RLock()
    worker._generations, worker._frame_token = {}, 0
    store = PreviewFrameStore(0)
    processor = SimpleNamespace(
        _preview_store=store,
        _context_worker=worker,
        edge_runtime_backend="deepstream",
        running=True,
        edge_session_id=identity["session_id"],
        cameras={"camera_1": {}},
    )
    processor.get_camera_preview = lambda camera: (
        DeepStreamProcessor.get_camera_preview(processor, camera)
    )
    source = EdgePreviewSource(
        processor, context.path, identity["session_id"], clock=clock
    )

    def feed():
        native = SimpleNamespace(
            misc_frame_info=[0, 0, 0, 0],
            source_id=0,
            frame_num=worker._frame_token + 1,
            buf_pts=worker._frame_token + 1,
        )
        worker.stamp_frame(native, "camera_1")
        pixels = np.zeros((720, 1280, 3), dtype=np.uint8)
        pixels[:, :, 1 if identity["node_id"] == "jetson-a" else 2] = 180
        store.put_frame(
            "camera_1",
            pixels,
            now_monotonic=clock(),
            identity=read_frame_identity(native, "camera_1"),
        )

    return SimpleNamespace(
        source=source,
        context=context,
        feed=feed,
        worker=worker,
        processor=processor,
        store=store,
    )


@pytest.fixture
def preview_nodes(lab, monkeypatch):
    token = uuid.uuid4().hex + uuid.uuid4().hex
    monkeypatch.setenv("STREAM_API_TOKEN", token)
    servers, threads, sources, clients = {}, [], {}, {}
    stop = threading.Event()
    for node in ("jetson-a", "jetson-b"):
        server = ThreadingApiServer(("127.0.0.1", 0), StreamApiHandler)
        server.edge_preview = None
        servers[node] = server
        clients[node] = RegistryClient(f"http://127.0.0.1:{server.server_port}", token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        threads.append(thread)

    def bind(owner):
        item = source_for_owner(
            lab.root.parent / ("preview-" + owner["assignment_id"]), owner
        )
        sources[owner["assignment_id"]] = item
        servers[owner["node_id"]].edge_preview = item.source
        return item

    def feed():
        while not stop.wait(0.1):
            for item in list(sources.values()):
                item.feed()

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        yield SimpleNamespace(
            bind=bind,
            relay=EdgePreviewRelay(lab.store, clients),
            clients=clients,
            servers=servers,
        )
    finally:
        stop.set()
        feeder.join(timeout=3)
        for server in servers.values():
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)
        for item in sources.values():
            item.context.close()


def test_source_needs_post_activation_frame_and_rejects_stale_generation(lab):
    owner = running(lab)
    clock = SimpleNamespace(now=100.0)
    item = source_for_owner(lab.root.parent / "source", owner, clock=lambda: clock.now)
    try:
        item.feed()
        with pytest.raises(RegistryError, match="waiting_for_new_frame"):
            item.source.frame("camera_1")
        with pytest.raises(RegistryError, match="frame_stale"):
            item.source.frame("camera_1")
        clock.now += 0.1
        item.feed()
        before = item.context.db.execute(
            "SELECT next_sequence FROM sessions"
        ).fetchall()
        payload = item.source.frame("camera_1")
        pixels = cv2.imdecode(
            np.frombuffer(base64.b64decode(payload["jpeg_base64"]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert pixels.shape == (360, 640, 3)
        assert (
            item.context.db.execute("SELECT next_sequence FROM sessions").fetchall()
            == before
        )
        assert (
            item.context.db.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 0
        )
        assert payload["edge"]["assignment_id"] == owner["assignment_id"]
        item.processor.running = False
        with pytest.raises(RegistryError, match="execution_mismatch"):
            item.source.frame("camera_1")
        item.processor.running = True
        item.worker._generations["camera_1"] = 1
        with pytest.raises(RegistryError, match="frame_unavailable"):
            item.source.frame("camera_1")
        item.feed()
        clock.now += 2.1
        with pytest.raises(RegistryError, match="frame_stale"):
            item.source.frame("camera_1")
        item.context.end(owner["session_id"])
        with pytest.raises(RegistryError, match="retired_result_session"):
            item.source.frame("camera_1")
    finally:
        item.context.close()


def wait_frame(relay, owner):
    deadline = time.monotonic() + 5
    while True:
        try:
            return relay.frame("camera_1", owner["assignment_id"])
        except RegistryError as error:
            if (
                error.code != "preview_source_unavailable"
                or time.monotonic() > deadline
            ):
                raise
            time.sleep(0.15)


def test_two_nodes_real_jpeg_http_relay_and_auth(lab, preview_nodes):
    owner = running(lab)
    preview_nodes.bind(owner)
    first = wait_frame(preview_nodes.relay, owner)
    assert first["edge"]["node_id"] == "jetson-a"
    owner = running(lab, "jetson-b")
    preview_nodes.bind(owner)
    second = wait_frame(preview_nodes.relay, owner)
    assert second["edge"]["node_id"] == "jetson-b"
    assert first["sha256"] != second["sha256"]
    with pytest.raises(RegistryError, match="owner_unverified"):
        preview_nodes.relay.frame("camera_1", first["edge"]["assignment_id"])
    server = preview_nodes.servers["jetson-b"]
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    connection.request("GET", "/edge-preview/camera_1")
    response = connection.getresponse()
    assert response.status == 401
    assert response.getheader("Cache-Control") == "no-store"
    assert "jpeg_base64" not in response.read().decode()
    connection.close()
    lab.nodes["jetson-b"]["freshness"] = "stale"
    with pytest.raises(RegistryError, match="owner_unverified"):
        preview_nodes.relay.frame("camera_1", owner["assignment_id"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda p: p["edge"].update(session_id="old"), "execution_mismatch"),
        (lambda p: p["edge"].update(epoch=True), "execution_mismatch"),
        (lambda p: p["frame"].update(frame_token=0), "frame_invalid"),
        (lambda p: p.update(age_seconds=3), "frame_stale"),
        (lambda p: p.update(age_seconds=float("nan")), "frame_stale"),
        (lambda p: p.update(jpeg_base64="bad!"), "image_invalid"),
        (lambda p: p.update(sha256="0" * 64), "image_invalid"),
    ],
)
def test_relay_rejects_invalid_remote_proof(lab, preview_nodes, mutation, reason):
    owner = running(lab)
    item = preview_nodes.bind(owner)
    payload = wait_frame(preview_nodes.relay, owner)
    malformed = copy.deepcopy(payload)
    mutation(malformed)
    item.source.frame = lambda camera: malformed
    with pytest.raises(RegistryError, match=reason):
        preview_nodes.relay.frame("camera_1", owner["assignment_id"])


def test_owner_change_during_fetch_discards_previous_image(lab, preview_nodes):
    owner = running(lab)
    item = preview_nodes.bind(owner)
    payload = wait_frame(preview_nodes.relay, owner)

    def change_owner(camera):
        running(lab, "jetson-b")
        return payload

    item.source.frame = change_owner
    with pytest.raises(RegistryError, match="owner_changed"):
        preview_nodes.relay.frame("camera_1", owner["assignment_id"])


def test_preview_config_requires_private_secrets_and_loopback_http(lab):
    secret = lab.root.parent / "preview-secret.json"
    secret.write_text(json.dumps({"stream_token": uuid.uuid4().hex}))
    secret.chmod(0o600)
    config = lab.root.parent / "preview-config.json"
    node = {
        "url": "http://127.0.0.1:18972",
        "ca_file": None,
        "credentials_file": str(secret),
    }
    config.write_text(json.dumps({"nodes": {"jetson-b": node}}))
    config.chmod(0o600)
    assert (
        EdgePreviewRelay.from_file(config, lab.store).clients["jetson-b"].port == 18972
    )
    secret.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        EdgePreviewRelay.from_file(config, lab.store)
    secret.chmod(0o600)
    node["url"] = "http://192.168.2.203:18972"
    config.write_text(json.dumps({"nodes": {"jetson-b": node}}))
    with pytest.raises(ValueError):
        EdgePreviewRelay.from_file(config, lab.store)


def test_preview_copy_holds_identity_lock():
    store = PreviewFrameStore(0)
    copying, updated = threading.Event(), threading.Event()

    class BlockingFrame:
        def copy(self):
            copying.set()
            assert not updated.wait(0.1)
            return "original pixels"

    store.put_frame(
        "camera_1", BlockingFrame(), now_monotonic=1, identity={"frame_token": 1}
    )

    def update():
        assert copying.wait(2)
        store.put_frame(
            "camera_1", "new pixels", now_monotonic=2, identity={"frame_token": 2}
        )
        updated.set()

    writer = threading.Thread(target=update)
    writer.start()
    try:
        pixels, info = store.get_frame_with_info("camera_1")
        assert (pixels, info["frame_token"]) == ("original pixels", 1)
        assert store.get_frame_with_info("missing") is None
    finally:
        writer.join(timeout=3)
    assert updated.is_set()
