"""전광판(Signboard/Dabit Metrix) 디바이스 컨트롤러

인수인계 자료 Control-Metrix 코드를 기반으로 작성한 모듈이다.

통신 방식: TCP 소켓 (stateless 연결, EUC-KR 버퍼 프로토콜)

동작 흐름 (display):
  ① socket.send  brightness 버퍼   - 밝기 설정
  ② socket.send  title 버퍼        - 제목 전송
  ③ socket.send  context 버퍼      - 본문 전송

동작 흐름 (clear):
  ④ socket.send  basic 버퍼        - 기본 이미지로 복귀 (방송 종료)

버퍼 형식:  ![00 <payload> !]  (EUC-KR 인코딩)
"""

import logging
import socket
import unicodedata
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# 설정 DTO
# ===========================================================================

@dataclass
class SignboardConfig:
    """전광판(Dabit) TCP 연결 및 표시 설정.

    속성:
        host:           전광판 IP 주소
        port:           TCP 포트 (기본 5000)
        brightness:     밝기 (0~100)
        display_time:   표시 시간 (초, MetrixOption.displayTime)
        text_color:     글자 색상 코드 (0~7 Dabit 팔레트)
        back_color:     배경 색상 코드 (0~7 Dabit 팔레트)
        text_size:      글자 크기 (1~4)
        text_speed:     스크롤 속도 (1~99)
        socket_timeout: 소켓 타임아웃 (초)
    """
    host: str = ""
    port: int = 5000
    brightness: int = 10          # 0~100
    display_time: int = 10
    text_color: int = 7           # 7 = 흰색 (Dabit 팔레트)
    back_color: int = 1           # 1 = 빨간색
    text_size: int = 2            # 1~4
    text_speed: int = 10          # 1~99
    socket_timeout: float = 3.0

    @property
    def is_configured(self) -> bool:
        return bool(self.host)


# ===========================================================================
# Dabit 버퍼 빌더 (Control-Metrix/parameter.py 통합)
# ===========================================================================

# --- 버퍼 상수 ---
_STX            = "![00"
_ETX            = "!]"
_CODE_BROADCAST = "0"
_CODE_BASIC     = "621"
_CODE_ON        = "211"
_CODE_OFF       = "210"
_CODE_BRIGHTNESS = "50"

_POS_TITLE      = "/P0000/Y0004"
_POS_STRING     = "/P0001/Y0408"

_OPT_SIZE       = "/F"
_OPT_EFFECT     = "/E"
_OPT_SPEED      = "/S"
_OPT_TEXT_CLR   = "/C"
_OPT_BACK_CLR   = "/G"


def _encode_buffer(payload: str) -> bytes:
    """버퍼 메시지를 EUC-KR 바이트열로 변환한다."""
    return (_STX + payload + _ETX).encode("euc-kr")


def _center_title(string: str, width: int = 24) -> str:
    """제목 문자열을 전광판 가로 너비에 맞게 중앙 정렬한다."""
    now = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in string)
    padding = max(0, width - now)
    left = " " * (padding // 2)
    right = " " * (padding - padding // 2)
    return f"{left}{string}{right}"


def _buf_brightness(amount: int) -> bytes:
    return _encode_buffer(_CODE_BRIGHTNESS + str(amount))

def _buf_title(title_text: str) -> bytes:
    payload = _CODE_BROADCAST + _POS_TITLE + _OPT_TEXT_CLR + "7" + _center_title(title_text)
    return _encode_buffer(payload)

def _buf_context(text: str, text_size: int, text_speed: int,
                 back_color: int, text_color: int) -> bytes:
    payload = (
        _CODE_BROADCAST + _POS_STRING
        + _OPT_SIZE  + f"00{text_size:02}"
        + _OPT_EFFECT + "0606"
        + _OPT_SPEED + f"{text_speed:02}00"
        + _OPT_BACK_CLR + str(back_color)
        + _OPT_TEXT_CLR + str(text_color)
        + text
    )
    return _encode_buffer(payload)

def _buf_basic() -> bytes:
    return _encode_buffer(_CODE_BASIC)

def _buf_off() -> bytes:
    return _encode_buffer(_CODE_OFF)

def _buf_on() -> bytes:
    return _encode_buffer(_CODE_ON)


# ===========================================================================
# 이벤트 → 표시 문구
# ===========================================================================

_SIGNBOARD_MESSAGES: Dict[str, str] = {
    "head":          "안전모 미착용 감지",
    "fall_detected": "낙상 감지 - 즉시 확인",
    "danger_zone":   "위험 구역 침입 감지",
    "intrusion":     "위험 구역 침입 감지",
    "critical":      "위험 이벤트 감지",
}


def build_display_text(event_type: str, severity: str = "", camera_id: str = "") -> str:
    """이벤트 타입에 따른 전광판 본문 문구를 반환한다."""
    base = _SIGNBOARD_MESSAGES.get(event_type.lower())
    if not base:
        if "fall" in event_type.lower():
            base = _SIGNBOARD_MESSAGES["fall_detected"]
        elif severity.lower() == "critical":
            base = _SIGNBOARD_MESSAGES["critical"]
        else:
            base = "안전 이벤트 감지"
    return f"[{camera_id}] {base}" if camera_id else base


# ===========================================================================
# Dabit TCP 클라이언트 (Control-Metrix/communicate.py SocketCommunicator 통합)
# ===========================================================================

class _DabitClient:
    """Dabit 전광판 TCP 소켓 클라이언트.

    각 명령은 stateless 방식 (connect → send → recv → close) 으로 전송한다.
    """

    def __init__(self, host: str, port: int, timeout: float):
        self._host    = host
        self._port    = port
        self._timeout = timeout

    def _send_stateless(self, buffer: bytes) -> bytes:
        """소켓을 열고 버퍼를 전송한 뒤 응답을 수신하고 닫는다."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        try:
            sock.connect((self._host, self._port))
            sock.sendall(bytearray(buffer))
            response = sock.recv(1024)
        finally:
            sock.close()
        return response

    @staticmethod
    def _validate(response: bytes) -> None:
        """응답이 실패(…F…)를 포함하면 예외를 발생시킨다."""
        if len(response) >= 3 and response[-3] == ord("F"):
            raise RuntimeError(f"Dabit 오류 응답: {response!r}")

    # ── 커맨드 메서드 ────────────────────────────────────────────

    def set_brightness(self, amount: int) -> None:
        resp = self._send_stateless(_buf_brightness(amount))
        self._validate(resp)

    def send_title(self, title_text: str) -> None:
        resp = self._send_stateless(_buf_title(title_text))
        self._validate(resp)

    def send_context(self, text: str, text_size: int, text_speed: int,
                     back_color: int, text_color: int) -> None:
        resp = self._send_stateless(
            _buf_context(text, text_size, text_speed, back_color, text_color)
        )
        self._validate(resp)

    def show_basic(self) -> None:
        """기본 이미지로 복귀 (방송 종료)."""
        resp = self._send_stateless(_buf_basic())
        self._validate(resp)

    def turn_off(self) -> None:
        resp = self._send_stateless(_buf_off())
        self._validate(resp)

    def turn_on(self) -> None:
        resp = self._send_stateless(_buf_on())
        self._validate(resp)


# ===========================================================================
# SignboardDevice (Control-Metrix/control.py Metrix 클래스 통합)
# ===========================================================================

class SignboardDevice:
    """전광판 디바이스 컨트롤러.

    display() 흐름:
      1) set_brightness → 밝기 설정
      2) send_title     → 제목 전송
      3) send_context   → 본문 전송

    clear() 흐름:
      1) show_basic     → 기본 이미지로 복귀
    """

    def __init__(self, config: SignboardConfig):
        self.config = config
        self._client: Optional[_DabitClient] = None

    def _get_client(self) -> Optional[_DabitClient]:
        if not self.config.is_configured:
            logger.warning("[Signboard] host 미설정 - 전광판 비활성화")
            return None
        if self._client is None:
            self._client = _DabitClient(
                self.config.host,
                self.config.port,
                self.config.socket_timeout,
            )
        return self._client

    def display(self, event_type: str, severity: str = "", camera_id: str = "") -> bool:
        """CCTV 이벤트 내용을 전광판에 표시한다."""
        client = self._get_client()
        if client is None:
            return False

        cfg   = self.config
        text  = build_display_text(event_type, severity, camera_id)
        title = f"CCTV {camera_id}" if camera_id else "CCTV 경보"
        logger.info("[Signboard] 표시 요청: title=%r text=%r", title, text)

        try:
            client.set_brightness(cfg.brightness)
            client.send_title(title)
            client.send_context(
                text,
                text_size=cfg.text_size,
                text_speed=cfg.text_speed,
                back_color=cfg.back_color,
                text_color=cfg.text_color,
            )
            logger.info("[Signboard] 방송 시작 완료")
            return True
        except Exception as exc:
            logger.error("[Signboard] display() 오류: %s", exc)
            return False

    def clear(self) -> bool:
        """전광판을 기본 이미지로 복귀시킨다 (방송 종료)."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.show_basic()
            logger.info("[Signboard] 방송 종료(기본 화면 복귀) 완료")
            return True
        except Exception as exc:
            logger.error("[Signboard] clear() 오류: %s", exc)
            return False

    def power_off(self) -> bool:
        """전광판 화면을 끈다."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.turn_off()
            logger.info("[Signboard] 화면 OFF")
            return True
        except Exception as exc:
            logger.error("[Signboard] power_off() 오류: %s", exc)
            return False

    def power_on(self) -> bool:
        """전광판 화면을 켠다."""
        client = self._get_client()
        if client is None:
            return False
        try:
            client.turn_on()
            logger.info("[Signboard] 화면 ON")
            return True
        except Exception as exc:
            logger.error("[Signboard] power_on() 오류: %s", exc)
            return False
