"""담당 조회가 오래된 실행을 정상 표시하지 않고 인증/읽기 경계를 지키는지 검사한다."""

import copy
import http.client
import threading

import pytest

from src.services.edge_controller_api import create_controller_server
from src.services.edge_node_agent import RegistryClient
from src.services.edge_ownership import EdgeOwnershipStore
from tests.test_edge_ownership import activate, observed, prepare, result
from tests.test_edge_ownership import lab as lab


def running(lab, node="jetson-a"):
    candidate, _ = prepare(lab, node)
    owner, _ = activate(lab, candidate)
    lab.nodes[node]["report"]["service"]["status"] = "running"
    lab.store.receive(node, result(lab, owner, 2, kind="observation"))
    return owner


def test_view_tracks_owner_candidate_and_bounded_metadata_without_writes(lab):
    assert lab.store.view("camera_1")["status"] == "unassigned"
    owner = running(lab)
    before = lab.store.db.total_changes
    view = lab.store.view("camera_1")
    assert view["status"] == "running" and view["inference_ready"]
    assert view["active"]["assignment_id"] == owner["assignment_id"]
    assert view["media_verified"] is False
    assert lab.store.db.total_changes == before
    assert "event" not in view["latest_result"]
    assert set(view["history"][0]) == {
        "assignment_id",
        "node_id",
        "epoch",
        "manifest_id",
        "state",
        "prepared_at",
        "activated_at",
    }
    pending, _ = prepare(lab, "jetson-b")
    view = lab.store.view("camera_1")
    assert view["status"] == "switching" and view["inference_ready"]
    assert view["active"]["node_id"] == "jetson-a"
    assert view["pending"]["assignment_id"] == pending["assignment_id"]
    owner_b, _ = activate(lab, pending)
    lab.nodes["jetson-b"]["report"]["service"]["status"] = "running"
    assert lab.store.view("camera_1")["status"] == "awaiting_results"
    lab.store.receive("jetson-b", result(lab, owner_b, 2))
    view = lab.store.view("camera_1")
    assert view["inference_ready"] and view["active"]["node_id"] == "jetson-b"
    assert [row["state"] for row in view["history"]] == ["active", "retired"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("ready", False),
        ("status", "stopped"),
        ("backend", "video_processor"),
        ("models_verified", False),
        ("session_id", "other"),
        ("supervisor_boot_id", "other"),
        ("epoch", 99),
        ("manifest_id", "other"),
        ("assignment_id", "other"),
        ("activation", None),
    ],
)
def test_view_rejects_unverified_execution(lab, field, value):
    running(lab)
    lab.nodes["jetson-a"]["report"]["service"][field] = value
    view = lab.store.view("camera_1")
    assert view["status"] == "unavailable" and not view["inference_ready"]
    assert view["reason"] == "owner_execution_unverified"


def test_view_stale_report_result_and_controller_restart(lab):
    running(lab)
    original = copy.deepcopy(lab.nodes["jetson-a"])
    lab.nodes["jetson-a"]["freshness"] = "stale"
    assert lab.store.view("camera_1")["reason"] == "node_not_observed"
    lab.nodes["jetson-a"] = original
    lab.clock.now += 11
    assert lab.store.view("camera_1")["reason"] == "node_observation_stale"
    observed(lab, lab.store.camera("camera_1")["active"])
    view = lab.store.view("camera_1")
    assert view["status"] == "awaiting_results" and view["latest_result"] is None
    lab.clock.now -= 11
    lab.store.close()
    lab.store = EdgeOwnershipStore(lab.root, lab.registry, clock=lambda: lab.clock.now)
    view = lab.store.view("camera_1")
    assert view["status"] == "reconciling" and not view["inference_ready"]
    assert view["history"][0]["state"] == "active"
    assert view["latest_result"] is None


def test_view_history_is_camera_scoped_and_limited_to_twenty(lab):
    for _ in range(25):
        candidate, _ = prepare(lab, "jetson-a")
        lab.store.abort(
            candidate["assignment_id"],
            {
                "request_id": candidate["assignment_id"],
                "controller_boot_id": lab.store.boot_id,
            },
        )
    view = lab.store.view("camera_1")
    assert [row["epoch"] for row in view["history"]] == list(range(25, 5, -1))
    assert lab.store.view("camera_2")["history"] == []


def test_http_view_read_role_static_assets_and_path_boundary(lab):
    running(lab)
    lab.registry.allow_remote_nodes = False
    tokens = {
        name: letter * 40
        for name, letter in (
            ("reader", "r"),
            ("control", "c"),
            ("consumer", "s"),
            ("node", "n"),
        )
    }
    with create_controller_server(
        lab.registry,
        lab.store,
        {"jetson-a": tokens["node"]},
        tokens["reader"],
        tokens["control"],
        tokens["consumer"],
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            reader = RegistryClient(url, tokens["reader"])
            assert reader.request("GET", "/v1/cameras/camera_1/view")[1][
                "inference_ready"
            ]
            for role in ("control", "consumer", "node"):
                assert (
                    RegistryClient(url, tokens[role]).request(
                        "GET", "/v1/cameras/camera_1/view"
                    )[0]
                    == 401
                )
            assert reader.request("POST", "/v1/cameras/camera_1/view", {})[0] == 404
            for path in ("/viewer", "/viewer.js"):
                connection = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=3
                )
                connection.request("GET", path)
                response = connection.getresponse()
                body = response.read().decode()
                assert response.status == 200
                assert response.getheader("Cache-Control") == "no-store"
                assert "frame-ancestors 'none'" in response.getheader(
                    "Content-Security-Policy"
                )
                assert all(token not in body for token in tokens.values())
                connection.close()
            assert (
                reader.request("GET", "/viewer/../src/services/edge_ownership.py")[0]
                == 404
            )
        finally:
            server.shutdown()
            thread.join(timeout=3)
