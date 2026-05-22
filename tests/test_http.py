"""
test_http.py — HttpEventForwarder / HttpEventTarget 단위 테스트

전략: requests.post를 mock 처리하여 전송·재시도·큐 관련 로직을 검증한다.
"""
import time
from unittest.mock import MagicMock, patch, call

import pytest

from src.protocols.http import (
    HttpEventForwarder,
    HttpEventTarget,
    _RETRY_MAX_ATTEMPTS,
    _RETRY_QUEUE_MAX,
)


# ---------------------------------------------------------------------------
# HttpEventTarget 테스트
# ---------------------------------------------------------------------------

class TestHttpEventTarget:
    def test_defaults(self):
        t = HttpEventTarget(name="TEST", url="http://example.com/api")
        assert "Content-Type" in t.headers
        assert t.timeout == 5

    def test_custom_headers(self):
        t = HttpEventTarget(
            name="X",
            url="http://x.com",
            headers={"Authorization": "Bearer abc"},
        )
        assert t.headers["Authorization"] == "Bearer abc"


# ---------------------------------------------------------------------------
# HttpEventForwarder 테스트
# ---------------------------------------------------------------------------

def _make_target(name="T1", url="http://example.com/api") -> HttpEventTarget:
    return HttpEventTarget(name=name, url=url)


def _mock_resp(status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = "OK"
    return resp


class TestHttpEventForwarder:
    def test_add_target(self):
        fw = HttpEventForwarder()
        fw.add_target(_make_target("A"))
        fw.add_target(_make_target("B"))
        assert fw.target_count == 2

    def test_init_with_targets(self):
        targets = [_make_target("X"), _make_target("Y")]
        fw = HttpEventForwarder(targets=targets)
        assert fw.target_count == 2

    @patch("src.protocols.http.requests.post")
    def test_forward_sends_to_all_targets(self, mock_post):
        mock_post.return_value = _mock_resp(200)
        fw = HttpEventForwarder(targets=[_make_target("A"), _make_target("B")])

        fw.forward("test/topic", {"key": "val"})
        assert mock_post.call_count == 2

    @patch("src.protocols.http.requests.post")
    def test_forward_sends_correct_body_structure(self, mock_post):
        mock_post.return_value = _mock_resp(200)
        fw = HttpEventForwarder(targets=[_make_target()])

        fw.forward("cctv/events", {"camera_id": "cam1", "type": "head"})

        body = mock_post.call_args[1]["json"]
        assert body["topic"] == "cctv/events"
        assert body["event"]["camera_id"] == "cam1"
        assert body["source"] == "edge-ai"
        assert "timestamp" in body

    @patch("src.protocols.http.requests.post")
    def test_forward_no_targets_no_call(self, mock_post):
        fw = HttpEventForwarder()
        fw.forward("topic", {})
        mock_post.assert_not_called()

    @patch("src.protocols.http.requests.post")
    def test_failed_request_goes_to_retry_queue(self, mock_post):
        mock_post.return_value = _mock_resp(500)  # 서버 오류
        fw = HttpEventForwarder(targets=[_make_target()])

        fw.forward("topic", {"x": 1})
        # 실패 → 재시도 큐에 1건 들어감
        assert fw.retry_queue_size == 1

    @patch("src.protocols.http.requests.post")
    def test_202_accepted_is_success(self, mock_post):
        mock_post.return_value = _mock_resp(202)
        fw = HttpEventForwarder(targets=[_make_target()])

        fw.forward("topic", {})
        # 202는 성공 → 재시도 큐 비어있음
        assert fw.retry_queue_size == 0

    @patch("src.protocols.http.requests.post")
    def test_retry_queue_not_exceeded_max(self, mock_post):
        mock_post.return_value = _mock_resp(500)
        fw = HttpEventForwarder(targets=[_make_target()])

        # _RETRY_QUEUE_MAX + 10 개 전송 → 큐 크기는 최대 _RETRY_QUEUE_MAX
        for _ in range(_RETRY_QUEUE_MAX + 10):
            fw._send(fw.target_at(0), "topic", {}, attempt=1)
        assert fw.retry_queue_size <= _RETRY_QUEUE_MAX

    @patch("src.protocols.http.requests.post")
    def test_max_attempts_no_more_retry(self, mock_post):
        mock_post.return_value = _mock_resp(500)
        fw = HttpEventForwarder(targets=[_make_target()])

        # attempt == _RETRY_MAX_ATTEMPTS → 더 이상 큐에 넣지 않음
        fw._send(fw.target_at(0), "topic", {}, attempt=_RETRY_MAX_ATTEMPTS)
        assert fw.retry_queue_size == 0

    def test_start_stop_worker(self):
        fw = HttpEventForwarder()
        fw.start()
        assert fw._running is True
        assert fw._worker is not None
        assert fw._worker.is_alive()
        fw.stop()
        time.sleep(0.1)
        assert fw._running is False
