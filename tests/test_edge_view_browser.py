"""선택 실행하는 Firefox 합성 JPEG 시험이며 실장비 GPU·연속 영상은 포함하지 않는다."""

import base64
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from contextlib import closing

import pytest

from src.services.edge_controller_api import create_controller_server
from tests.test_edge_ownership import activate, prepare, result
from tests.test_edge_ownership import lab as lab
from tests.test_edge_preview import preview_nodes as preview_nodes
from tests.test_edge_view import running


@pytest.mark.skipif(
    os.environ.get("CCTV_EDGE_FIREFOX_TEST") != "1",
    reason="설치된 Firefox/geckodriver를 쓰는 명시적 격리 브라우저 시험",
)
def test_firefox_owner_transition_stale_auth_keyboard_and_layout(
    lab, tmp_path_factory, preview_nodes
):
    artifacts = tmp_path_factory.getbasetemp()
    websocket = pytest.importorskip("websocket")
    profile_root = lab.root.parent / "browser-profiles"
    profile_root.mkdir(mode=0o700)
    driver = shutil.which("geckodriver")
    assert driver, "geckodriver가 필요합니다. 운영 환경에 자동 설치하지 않습니다."
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    driver_url = f"http://127.0.0.1:{port}"
    session = None

    def command(method, path, data=None, timeout=45):
        request = urllib.request.Request(
            driver_url + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["value"]

    def session_command(method, path, data=None):
        return command(method, f"/session/{session}" + path, data)

    def script(source, *arguments):
        return session_command(
            "POST", "/execute/sync", {"script": source, "args": arguments}
        )

    def wait_for(source, expected, seconds=12):
        deadline = time.monotonic() + seconds
        value = None
        while time.monotonic() < deadline:
            value = script(source)
            if value == expected:
                return
            time.sleep(0.1)
        assert value == expected

    def element(selector):
        found = session_command(
            "POST", "/element", {"using": "css selector", "value": selector}
        )
        return found["element-6066-11e4-a52e-4f735466cecf"]

    def keys(selector, text):
        session_command("POST", f"/element/{element(selector)}/value", {"text": text})

    def click(selector):
        session_command("POST", f"/element/{element(selector)}/click", {})

    reader = secrets.token_urlsafe(32)
    lab.registry.allow_remote_nodes = False
    with (
        create_controller_server(
            lab.registry,
            lab.store,
            {"jetson-a": secrets.token_urlsafe(32)},
            reader,
            secrets.token_urlsafe(32),
            secrets.token_urlsafe(32),
            preview_relay=preview_nodes.relay,
        ) as server,
        (artifacts / "geckodriver.log").open("w") as log,
    ):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        process = subprocess.Popen(
            [
                driver,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--profile-root",
                str(profile_root),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    command("GET", "/status")
                    break
                except OSError:
                    assert process.poll() is None, (
                        "geckodriver 시작 실패: 전용 로그 확인"
                    )
                    assert time.monotonic() < deadline, "geckodriver 시작 시간 초과"
                    time.sleep(0.2)
            created = command(
                "POST",
                "/session",
                {
                    "capabilities": {
                        "alwaysMatch": {
                            "browserName": "firefox",
                            "webSocketUrl": True,
                            "moz:firefoxOptions": {"args": ["-headless"]},
                        }
                    }
                },
                timeout=120,
            )
            session = created["sessionId"]
            session_command(
                "POST", "/url", {"url": f"http://127.0.0.1:{server.server_port}/viewer"}
            )
            script(
                "window.viewerErrors=[]; addEventListener('error', e=>viewerErrors.push(e.message));"
            )
            click("#camera-id")
            keys("#camera-id", "\ue004")
            assert script("return document.activeElement.id") == "reader-token"
            assert (
                script("return getComputedStyle(document.activeElement).outlineStyle")
                == "solid"
            )
            keys("#reader-token", reader + "\ue004")
            assert script("return document.activeElement.textContent") == "연결"
            keys("button[type=submit]", "\ue007")
            wait_for("return document.getElementById('state').textContent", "담당 없음")
            assert script("return document.getElementById('reader-token').value") == ""
            assert script(
                "return [localStorage.length,sessionStorage.length,document.cookie]"
            ) == [0, 0, ""]

            preview_nodes.bind(running(lab))
            wait_for(
                "return document.getElementById('state').textContent",
                "추론·결과 저장 확인",
            )
            assert (
                script("return document.getElementById('owner').textContent")
                == "jetson-a"
            )
            wait_for("return document.getElementById('preview-image').hidden", False)
            first_image = script("return document.getElementById('preview-image').src")
            assert (
                script("return document.getElementById('preview-image').naturalWidth")
                == 640
            )
            candidate, _ = prepare(lab, "jetson-b")
            wait_for(
                "return document.getElementById('candidate').textContent", "jetson-b"
            )
            assert (
                script("return document.getElementById('owner').textContent")
                == "jetson-a"
            )
            owner, _ = activate(lab, candidate)
            preview_nodes.bind(owner)
            lab.nodes["jetson-b"]["report"]["service"]["status"] = "running"
            wait_for(
                "return document.getElementById('state').textContent",
                "최근 결과 저장 대기",
            )
            lab.store.receive("jetson-b", result(lab, owner, 2))
            wait_for(
                "return document.getElementById('state').textContent",
                "추론·결과 저장 확인",
            )
            assert (
                script("return document.getElementById('owner').textContent")
                == "jetson-b"
            )
            assert script("return document.querySelectorAll('#history tr').length") == 2
            assert (
                script("return document.querySelectorAll('video,iframe').length") == 0
            )

            wait_for("return document.getElementById('preview-image').hidden", False)
            assert (
                script("return document.getElementById('preview-image').src")
                != first_image
            )
            layouts = []
            # 일반 창 크기는 Firefox에서 500px로 제한되므로 실제 뷰포트를 지정한다.
            with closing(
                websocket.create_connection(
                    created["capabilities"]["webSocketUrl"],
                    timeout=5,
                    suppress_origin=True,
                )
            ) as bidi:
                context = session_command("GET", "/window")
                for index, width in enumerate((320, 768, 1024, 1440), start=1):
                    bidi.send(
                        json.dumps(
                            {
                                "id": index,
                                "method": "browsingContext.setViewport",
                                "params": {
                                    "context": context,
                                    "viewport": {"width": width, "height": 1000},
                                },
                            }
                        )
                    )
                    answer = json.loads(bidi.recv())
                    assert answer["id"] == index and answer["type"] == "success", answer
                    geometry = script(
                        "return {width:innerWidth,scroll:document.documentElement.scrollWidth}"
                    )
                    assert geometry["width"] == width, geometry
                    assert geometry["scroll"] <= geometry["width"], geometry
                    layouts.append({"requested_width": width, **geometry})
                    screenshot = session_command("GET", "/screenshot")
                    (artifacts / f"viewer-{width}.png").write_bytes(
                        base64.b64decode(screenshot)
                    )
            lab.nodes["jetson-b"]["freshness"] = "stale"
            wait_for(
                "return document.getElementById('state').textContent",
                "현재 실행 확인 불가",
            )
            assert (
                script("return document.getElementById('result').textContent")
                == "현재 실행의 최근 저장 미확인"
            )
            assert script("return document.getElementById('preview-image').hidden")
            click("#disconnect")
            keys("#reader-token", secrets.token_urlsafe(32))
            click("button[type=submit]")
            wait_for(
                "return document.getElementById('connection').textContent",
                "인증에 실패했습니다. 읽기 전용 토큰을 다시 입력해 주세요.",
            )
            assert script("return document.getElementById('owner').textContent") == "—"
            lab.nodes["jetson-b"]["freshness"] = "online"
            keys("#reader-token", reader)
            click("button[type=submit]")
            wait_for(
                "return document.getElementById('state').textContent",
                "추론·결과 저장 확인",
            )
            server.shutdown()
            server.server_close()
            wait_for("return document.getElementById('state').textContent", "미확인")
            assert script("return document.getElementById('owner').textContent") == "—"
            assert script("return window.viewerErrors") == []
            (artifacts / "browser-result.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "browser": created["capabilities"]["browserVersion"],
                        "layouts": layouts,
                        "real_gpu_test": False,
                        "media_verified": False,
                        "preview_verified": True,
                        "continuous_video_verified": False,
                        "cases": [
                            "owner_bound_preview",
                            "jpeg_decode_640",
                            "stale_image_cleared",
                            "keyboard_focus",
                            "no_token_persistence",
                            "owner_a_to_b",
                            "pending_owner_preserved",
                            "result_wait",
                            "stale_status",
                            "auth_failure",
                            "network_failure",
                            "no_page_overflow",
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            if session:
                try:
                    session_command("DELETE", "")
                except OSError:
                    pass
            server.shutdown()
            thread.join(timeout=3)
            if process.poll() is None:
                # 이 시험에서 만든 프로세스 그룹만 정리한다.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
