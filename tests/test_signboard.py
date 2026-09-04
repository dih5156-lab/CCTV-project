"""
test_signboard.py — SignboardDevice / _DabitClient / 헬퍼 함수 단위 테스트

전략: TCP 소켓을 mock 처리하여 실제 전광판 없이 Dabit 프로토콜 로직을 검증한다.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.devices.signboard import (
    CLASS_COLOR_MAP,
    SignboardConfig,
    SignboardDevice,
    _buf_brightness,
    _buf_context,
    _buf_title,
    _center_pad,
    _DabitClient,
    _display_width,
    build_display_text,
)

# ---------------------------------------------------------------------------
# 헬퍼 함수 테스트
# ---------------------------------------------------------------------------

class TestDisplayWidth:
    def test_ascii_only(self):
        assert _display_width("hello") == 5

    def test_korean_only(self):
        # 한글 1자 = EAW=W → 너비 2
        assert _display_width("안녕") == 4

    def test_mixed(self):
        assert _display_width("A안") == 3   # ascii(1) + 한글(2)

    def test_empty(self):
        assert _display_width("") == 0


class TestCenterPad:
    def test_short_text_is_padded(self):
        result = _center_pad("AB", width=10)
        assert len(result) >= 10
        assert result.strip() == "AB"

    def test_equal_width_no_pad(self):
        text = "ABCDE"     # 너비 5, width=5
        result = _center_pad(text, width=5)
        assert result == text

    def test_korean_centering(self):
        # 한글 2자 = 너비 4
        result = _center_pad("안녕", width=8)
        assert "안녕" in result
        # 좌우 합산 패딩 = 8 - 4 = 4 (공백 4자)
        assert result.count(" ") == 4


class TestBuildDisplayText:
    def test_known_event_type(self):
        text = build_display_text("head")
        assert "안전모" in text

    def test_known_event_with_camera(self):
        text = build_display_text("fall_detected", camera_id="cam3")
        assert "cam3" in text
        assert "낙상" in text

    def test_fall_variant(self):
        text = build_display_text("fall_other")
        assert "낙상" in text

    def test_critical_severity(self):
        text = build_display_text("unknown_type", severity="critical")
        assert "위험" in text

    def test_unknown_type_default(self):
        text = build_display_text("xyz123")
        assert text  # 빈 문자열 아님

    def test_intrusion(self):
        text = build_display_text("intrusion")
        assert "침입" in text


# ---------------------------------------------------------------------------
# SignboardConfig 테스트
# ---------------------------------------------------------------------------

class TestSignboardConfig:
    def test_is_configured_with_host(self):
        cfg = SignboardConfig(host="192.168.0.1")
        assert cfg.is_configured is True

    def test_is_configured_without_host(self):
        cfg = SignboardConfig(host="")
        assert cfg.is_configured is False

    def test_defaults(self):
        cfg = SignboardConfig()
        assert cfg.port == 5000
        assert cfg.brightness == 10
        assert cfg.text_color == 7
        assert cfg.back_color == 0
        assert cfg.idle_refresh_interval == 10.0


# ---------------------------------------------------------------------------
# _DabitClient 테스트 (소켓 mock)
# ---------------------------------------------------------------------------

class TestDabitClient:
    def test_brightness_uses_two_digit_protocol_value(self):
        assert _buf_brightness(8).decode("euc-kr").endswith("08!]")

    def _make_client(self):
        return _DabitClient(host="127.0.0.1", port=5000, timeout=3.0)

    def _mock_sock(self, recv_data=b"OK!"):
        """성공 응답을 반환하는 소켓 mock."""
        sock = MagicMock()
        sock.recv.return_value = recv_data
        return sock

    @patch("src.devices.signboard.socket.socket")
    def test_send_context_success(self, mock_socket_cls):
        sock = self._mock_sock(b"OK!")
        mock_socket_cls.return_value.__enter__ = MagicMock(return_value=sock)
        mock_socket_cls.return_value = sock

        client = self._make_client()
        # 성공 응답 → 예외 없음
        client.send_context("테스트", size=2, speed=10, back=0, color=7)

    @patch("src.devices.signboard.socket.socket")
    def test_check_raises_on_error_response(self, mock_socket_cls):
        # Dabit 오류 응답: 마지막에서 3번째 바이트가 'F'(70)
        error_resp = b"XX" + bytes([ord("F")]) + b"YZ"
        sock = self._mock_sock(error_resp)
        mock_socket_cls.return_value = sock

        client = self._make_client()
        with pytest.raises(RuntimeError, match="Dabit 오류"):
            client._check(error_resp)

    @patch("src.devices.signboard.socket.socket")
    def test_send_raises_on_connection_error(self, mock_socket_cls):
        sock = MagicMock()
        sock.connect.side_effect = ConnectionRefusedError("연결 거부")
        mock_socket_cls.return_value = sock

        client = self._make_client()
        with pytest.raises(ConnectionRefusedError):
            client._send(b"test")

    def test_check_ok_response(self):
        client = self._make_client()
        # 3바이트 이상이고 끝에서 3번째가 'F'가 아닌 경우 → 정상
        client._check(b"OK!")

    def test_check_short_response_no_raise(self):
        client = self._make_client()
        # 응답 길이 2 이하 → 체크 생략 (예외 없음)
        client._check(b"OK")


# ---------------------------------------------------------------------------
# _buf 함수: EUC-KR 인코딩 됨을 확인
# ---------------------------------------------------------------------------

class TestBufferBuilders:
    def test_buf_title_encodable(self):
        buf = _buf_title("경고")
        assert isinstance(buf, bytes)
        assert b"![00" in buf

    def test_buf_context_contains_effect(self):
        buf = _buf_context("테스트", size=2, speed=10, back=0, color=7)
        decoded = buf.decode("euc-kr", errors="replace")
        assert "/E0100" in decoded  # 고정 효과 코드

    def test_buf_context_contains_options(self):
        buf = _buf_context("텍스트", size=2, speed=10, back=0, color=5)
        decoded = buf.decode("euc-kr", errors="replace")
        assert "/C5" in decoded   # 색상 5 (자주)
        assert "/G0" in decoded   # 배경 0 (검정)


# ---------------------------------------------------------------------------
# SignboardDevice 테스트 (소켓 mock)
# ---------------------------------------------------------------------------

class TestSignboardDevice:
    def _make_device(self, **kwargs) -> SignboardDevice:
        cfg = SignboardConfig(host="192.168.0.1", **kwargs)
        # idle 스레드가 실제 소켓 접근하지 않도록 패치
        with patch("src.devices.signboard._DabitClient"):
            device = SignboardDevice(cfg)
        return device

    def test_display_returns_false_when_unconfigured(self):
        cfg = SignboardConfig(host="")
        device = SignboardDevice(cfg)
        result = device.display("테스트")
        assert result is False

    @patch("src.devices.signboard._DabitClient")
    def test_display_calls_send_context(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        cfg = SignboardConfig(host="192.168.0.1", display_time=0)
        with patch.object(SignboardDevice, "_start_idle_thread"):
            device = SignboardDevice(cfg)
        device._client = mock_client

        device.display("안전모 미착용", class_name="no_helmet")

        mock_client.send_context.assert_called_once()

    @patch("src.devices.signboard._DabitClient")
    def test_display_cooldown_skip(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        cfg = SignboardConfig(host="192.168.0.1", display_time=30)
        with patch.object(SignboardDevice, "_start_idle_thread"):
            device = SignboardDevice(cfg)
        device._client = mock_client

        # 첫 번째 호출 → 전송
        device.display("경고", title="제목", class_name="head")
        first_call_count = mock_client.send_context.call_count

        # 두 번째 호출 (쿨다운 내) → 스킵
        device.display("경고", title="제목", class_name="head")
        assert mock_client.send_context.call_count == first_call_count

    def test_get_color_by_class(self):
        assert SignboardDevice.get_color_by_class("helmet") == CLASS_COLOR_MAP["helmet"]
        assert SignboardDevice.get_color_by_class("no_helmet") == CLASS_COLOR_MAP["no_helmet"]
        assert SignboardDevice.get_color_by_class("unknown_xyz") == CLASS_COLOR_MAP["default"]

    @patch("src.devices.signboard._DabitClient")
    def test_clear_calls_show_basic(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        cfg = SignboardConfig(host="192.168.0.1")
        with patch.object(SignboardDevice, "_start_idle_thread"):
            device = SignboardDevice(cfg)
        device._client = mock_client

        result = device.clear()
        assert result is True
        mock_client.show_basic.assert_called_once()

    @patch("src.devices.signboard._DabitClient")
    def test_display_returns_false_on_exception(self, mock_cls):
        mock_client = MagicMock()
        mock_client.set_brightness.side_effect = OSError("연결 실패")
        mock_cls.return_value = mock_client

        cfg = SignboardConfig(host="192.168.0.1", display_time=0)
        with patch.object(SignboardDevice, "_start_idle_thread"):
            device = SignboardDevice(cfg)
        device._client = mock_client

        result = device.display("경고")
        assert result is False
