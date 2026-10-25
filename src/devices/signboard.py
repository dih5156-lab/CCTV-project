"""전광판(Signboard / Dabit Metrix) 디바이스 컨트롤러

통신 방식: TCP 소켓 (stateless 연결, EUC-KR 버퍼 프로토콜)
버퍼 형식:  ![00 <payload> !]  (EUC-KR 인코딩)

Dabit 팔레트 색상 코드:
  0=검정  1=빨강  2=녹색  3=노랑  4=파랑  5=자주  6=하늘  7=흰색
"""

import logging
import socket
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional, Tuple

from ..config.event_type_map import event_type_map as _etm

logger = logging.getLogger(__name__)


# ===========================================================================
# 색상 매핑 (객체 탐지 클래스 → Dabit 색상 코드)
# ===========================================================================

# 내장 탐지 클래스 전용 색상 맵 (event_type는 config/event_type_map.json 사용)
CLASS_COLOR_MAP: Dict[str, int] = {
    "helmet":    2,   # 안전모 착용 → 녹색
    "no_helmet": 1,   # 안전모 미착용 → 빨강
    "person":    3,   # 사람 → 노랑
    "car":       4,   # 차량 → 파랑
    "fire":      5,   # 화재 → 자주
    "smoke":     6,   # 연기 → 하늘
    "default":   7,   # 기본 → 흰색
}

# 한국어 요일 (월=0 ~ 일=6)
_WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]
# 한국 표준시
_KST = timezone(timedelta(hours=9))
# Dabit 특수문자: '/' → /U04 행 7번
_CHAR_SLASH = "/U047"


# ===========================================================================
# 설정 DTO
# ===========================================================================

@dataclass
class SignboardConfig:
    """전광판(Dabit) TCP 연결 및 표시 설정."""
    host: str = ""
    port: int = 5000
    brightness: int = 10
    display_time: int = 10      # 동일 이벤트 재전송 차단 시간(초)
    text_color: int = 7         # 기본: 흰색
    back_color: int = 0         # 기본: 검정
    text_size: int = 2          # 1~4
    text_speed: int = 10        # 1~99
    socket_timeout: float = 3.0
    idle_refresh_interval: float = 10.0
    idle_enabled: bool = True  # 같은 전광판의 유휴 화면은 한 서비스만 갱신한다.

    @property
    def is_configured(self) -> bool:
        return bool(self.host)


# ===========================================================================
# Dabit 버퍼 빌더
# ===========================================================================

_STX             = "![00"
_ETX             = "!]"
_CODE_BROADCAST  = "0"
_CODE_BASIC      = "621"
_CODE_ON         = "211"
_CODE_OFF        = "210"
_CODE_BRIGHTNESS = "50"

_POS_TITLE       = "/P0000/Y0004"
_POS_STRING      = "/P0001/Y0408"

_OPT_SIZE        = "/F"
_OPT_EFFECT      = "/E"
_OPT_SPEED       = "/S"
_OPT_TEXT_CLR    = "/C"
_OPT_BACK_CLR    = "/G"


def _encode_buffer(payload: str) -> bytes:
    return (_STX + payload + _ETX).encode("euc-kr")


def _display_width(text: str) -> int:
    """EAW 기준 표시 너비 계산 (한글·전각=2, 그 외=1)."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _center_pad(text: str, width: int = 24) -> str:
    """전광판 너비에 맞게 텍스트를 가운데 정렬한다."""
    padding = max(0, width - _display_width(text))
    left = " " * (padding // 2)
    right = " " * (padding - padding // 2)
    return f"{left}{text}{right}"


def _buf_brightness(amount: int) -> bytes:
    return _encode_buffer(_CODE_BRIGHTNESS + f"{amount:02}")


def _buf_title(text: str) -> bytes:
    payload = _CODE_BROADCAST + _POS_TITLE + _OPT_TEXT_CLR + "7" + _center_pad(text)
    return _encode_buffer(payload)


def _buf_context(text: str, size: int, speed: int, back: int, color: int) -> bytes:
    payload = (
        _CODE_BROADCAST + _POS_STRING
        + _OPT_SIZE     + f"00{size:02}"
        + _OPT_EFFECT   + "0100"
        + _OPT_SPEED    + f"{speed:02}00"
        + _OPT_BACK_CLR + str(back)
        + _OPT_TEXT_CLR + str(color)
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
# 이벤트 → 표시 문구  (설정 파일 위임)
# ===========================================================================


def build_display_text(event_type: str, severity: str = "", camera_id: str = "") -> str:
    """이벤트 타입에 따른 전광판 본문 문구를 반환한다 (config/event_type_map.json 참조)."""
    return _etm.display_text(event_type, severity, camera_id)


# ===========================================================================
# Dabit TCP 클라이언트
# ===========================================================================

class _DabitClient:
    """Dabit 전광판 TCP 소켓 클라이언트 (stateless 방식)."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout

    def _send(self, buf: bytes) -> bytes:
        last_exc: Optional[OSError] = None
        for attempt in range(1, 4):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._timeout)
            try:
                sock.connect((self._host, self._port))
                sock.sendall(bytearray(buf))
                return sock.recv(1024)
            except OSError as exc:
                last_exc = exc
                if attempt >= 3:
                    break
                # Dabit 컨트롤러가 직전 명령 직후 포트를 잠깐 거부하는 경우가 있다.
                time.sleep(0.2 * attempt)
            finally:
                sock.close()
        raise last_exc or ConnectionError("Dabit 전광판 전송 실패")

    @staticmethod
    def _check(response: bytes) -> None:
        if len(response) >= 3 and response[-3] == ord("F"):
            raise RuntimeError(f"Dabit 오류 응답: {response!r}")

    def set_brightness(self, amount: int) -> bool:
        response = self._send(_buf_brightness(amount))
        self._check(response)
        return response == b"![00500!]"

    def send_title(self, text: str) -> bool:
        response = self._send(_buf_title(text))
        self._check(response)
        return response == b"![0000!]"

    def send_context(self, text: str, size: int, speed: int, back: int, color: int) -> bool:
        response = self._send(_buf_context(text, size, speed, back, color))
        self._check(response)
        # 10-08 실장비 캡처의 정상 프레임만 완료 근거로 쓴다. 빈/미등록 응답은 제외한다.
        return response == b"![0000!]"

    def show_basic(self) -> None:
        self._check(self._send(_buf_basic()))

    def turn_off(self) -> None:
        self._check(self._send(_buf_off()))

    def turn_on(self) -> None:
        self._check(self._send(_buf_on()))


# ===========================================================================
# SignboardDevice
# ===========================================================================

class SignboardDevice:
    """전광판 디바이스 컨트롤러.

    탐지 이벤트 발생 시 display()로 메시지를 전송하며,
    유휴 표시를 담당하는 인스턴스만 설정 주기에 따라 현재 시각을 갱신한다.
    """

    def __init__(self, config: SignboardConfig) -> None:
        self.config = config
        self._client: Optional[_DabitClient] = None
        # (title, class_name) → 마지막 전송 시각
        self._last_display_ts: Dict[Tuple[str, str], float] = {}
        # 마지막 경보 전송 완료 시각부터 유휴 화면 복귀를 지연한다.
        self._last_event_ts: float = 0.0
        self._on_idle: Optional[Callable[[], bool]] = None
        self._idle_stop = threading.Event()
        self._send_lock = threading.Lock()
        self._idle_thread: Optional[threading.Thread] = None
        if config.is_configured and config.idle_enabled:
            self._start_idle_thread()

    # ------------------------------------------------------------------
    # 퍼블릭 API
    # ------------------------------------------------------------------

    def display(
        self,
        text: str,
        title: Optional[str] = None,
        class_name: Optional[str] = None,
        text_color: Optional[int] = None,
        back_color: Optional[int] = None,
        text_size: Optional[int] = None,
        text_speed: Optional[int] = None,
        brightness: Optional[int] = None,
        on_idle: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """전광판에 이벤트 메시지를 표시한다.

        Args:
            text:       본문 (\\n 구분으로 여러 줄 지원)
            title:      상단 제목 (기본: 'CCTV 경보')
            class_name: 탐지 클래스명 → 색상 자동 매핑
            text_color: 글자 색상 0~7 (미지정 시 class_name 기반 또는 config)
            back_color: 배경 색상 0~7 (미지정 시 config)
            text_size:  글자 크기 1~4 (미지정 시 config)
            text_speed: 속도 1~99     (미지정 시 config)
        """
        client = self._get_client()
        if client is None:
            return False

        cfg   = self.config
        color = text_color if text_color is not None else self._color_for(class_name)
        back  = back_color if back_color is not None else cfg.back_color
        size  = text_size  if text_size  is not None else cfg.text_size
        speed = text_speed if text_speed is not None else cfg.text_speed
        display_brightness = brightness if brightness is not None else cfg.brightness
        title = title or "CCTV 경보"

        slot: Tuple[str, str] = (title, class_name or text)
        try:
            with self._send_lock:
                # 대기 중 다른 요청이 성공했을 수 있어 전송 잠금 안에서 중복을 판단한다.
                elapsed = time.time() - self._last_display_ts.get(slot, 0.0)
                if elapsed < cfg.display_time:
                    logger.debug("[Signboard] 쿨다운 - 스킵 (남은 %ds)", int(cfg.display_time - elapsed))
                    return True
                # 다른 출력이 끼어들면 이전 요청의 종료를 추정하지 않는다.
                self._on_idle = None
                logger.info("[Signboard] 표시: title=%r text=%r color=%d back=%d", title, text, color, back)
                confirmed = client.set_brightness(display_brightness) is True
                confirmed = (client.send_title(title) is True) and confirmed
                has_content = False
                for line in text.splitlines():
                    if line.strip():
                        has_content = True
                        confirmed = (client.send_context(line, size, speed, back, color) is True) and confirmed
                # 느린 전송도 표시 시간을 보장하고, 실패한 요청의 재시도는 막지 않는다.
                self._last_display_ts[slot] = self._last_event_ts = time.time()
                if confirmed and has_content:
                    self._on_idle = on_idle
            logger.info("[Signboard] 방송 시작 완료")
            return True
        except Exception as exc:
            logger.error("[Signboard] display() 오류: %s", exc)
            return False

    def clear(self) -> bool:
        """기본 이미지로 복귀한다 (방송 종료)."""
        return self._simple_cmd(lambda c: c.show_basic(), "방송 종료")

    def power_on(self) -> bool:
        """화면을 켠다."""
        return self._simple_cmd(lambda c: c.turn_on(), "화면 ON")

    def power_off(self) -> bool:
        """화면을 끈다."""
        return self._simple_cmd(lambda c: c.turn_off(), "화면 OFF")

    def stop_idle(self) -> None:
        """idle 스레드를 종료한다."""
        self._idle_stop.set()

    @staticmethod
    def get_color_by_class(class_name: str) -> int:
        """탐지 클래스명 또는 이벤트 타입으로 Dabit 색상 코드를 반환한다."""
        key = class_name.lower()
        if key in CLASS_COLOR_MAP:
            return CLASS_COLOR_MAP[key]
        return _etm.color_code(key)

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[_DabitClient]:
        if not self.config.is_configured:
            logger.debug("[Signboard] host 미설정 - 비활성화")
            return None
        if self._client is None:
            self._client = _DabitClient(
                self.config.host, self.config.port, self.config.socket_timeout
            )
        return self._client

    def _color_for(self, class_name: Optional[str]) -> int:
        if class_name:
            return self.get_color_by_class(class_name)
        return self.config.text_color

    def _simple_cmd(self, fn: Callable[[_DabitClient], None], label: str) -> bool:
        """단순 명령(clear/power)을 실행하는 공통 헬퍼."""
        client = self._get_client()
        if client is None:
            return False
        try:
            with self._send_lock:
                self._on_idle = None
                fn(client)
            logger.info("[Signboard] %s 완료", label)
            return True
        except Exception as exc:
            logger.error("[Signboard] %s 오류: %s", label, exc)
            return False

    # ------------------------------------------------------------------
    # 유휴 스레드 (설정 주기에 따라 현재 시각 갱신)
    # ------------------------------------------------------------------

    def _start_idle_thread(self) -> None:
        self._idle_stop.clear()
        self._idle_thread = threading.Thread(
            target=self._idle_worker, daemon=True, name="SignboardIdleWorker"
        )
        self._idle_thread.start()

    def _idle_worker(self) -> None:
        last_second = -1
        while not self._idle_stop.wait(self.config.idle_refresh_interval):
            if time.time() - self._last_event_ts < self.config.display_time:
                last_second = -1    # 이벤트 수신 중 → 초 추적 초기화
                continue
            now = datetime.now(_KST)
            if now.second == last_second:
                continue            # 같은 초 → 재전송 생략
            last_second = now.second
            client = self._get_client()
            if client is not None:
                self._send_idle_frame(client, now)

    def _send_idle_frame(self, client: _DabitClient, now: datetime) -> None:
        """현재 시각을 전광판에 표시한다 (idle 전용)."""
        cfg = self.config
        weekday = _WEEKDAY_KO[now.weekday()]
        dt_text = (
            f"{now.year}{_CHAR_SLASH}{now.month:02d}{_CHAR_SLASH}{now.day:02d}"
            f"({weekday}) {now.hour:02d}시 {now.minute:02d}분 {now.second:02d}초"
        )
        try:
            with self._send_lock:
                # 유휴 판단 뒤 경보가 먼저 전송됐으면 시간 표시로 덮어쓰지 않는다.
                if time.time() - self._last_event_ts < cfg.display_time:
                    return
                confirmed = client.set_brightness(cfg.brightness) is True
                confirmed = (client.send_title("현재 시간") is True) and confirmed
                confirmed = (client.send_context(_center_pad(dt_text, width=32), 2, 1, 0, 5) is True) and confirmed
                # 타이머 경과가 아니라 유휴 화면 전체 ACK와 결과 저장 성공으로 소유권을 끝낸다.
                if confirmed and self._on_idle is not None and self._on_idle():
                    self._on_idle = None
            logger.debug("[Signboard] idle 표시: %s", dt_text)
        except Exception as exc:
            logger.debug("[Signboard] idle 표시 오류: %s", exc)


# ===========================================================================
# 테스트
# ===========================================================================

def _test_signboard() -> None:
    """전광판 연동 테스트."""
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    config = SignboardConfig(
        host="192.168.88.91",
        brightness=80,
        display_time=10,
        text_size=2,
        text_speed=10,
    )
    device = SignboardDevice(config)

    cases = [
        ("안전모 착용자 감지\n작업장 진입", "helmet"),
        ("안전모 미착용\n경고!",            "no_helmet"),
        ("사람 감지\n출입구",              "person"),
        ("차량 접근\n주의",               "car"),
        ("화재 감지\n긴급 대피",           "fire"),
        ("연기 감지\n환기 필요",           "smoke"),
        ("이벤트 감지\n기본 흰색",         None),
    ]

    for text, class_name in cases:
        color = SignboardDevice.get_color_by_class(class_name) if class_name else config.text_color
        print(f"[테스트] class={class_name or 'default':12s}  color={color}  text={text!r}")
        ok = device.display(text=text, title="CCTV 알림", class_name=class_name)
        print(f"         → {'성공' if ok else '실패'}")

    print("[테스트] clear()")
    device.clear()


if __name__ == "__main__":
    _test_signboard()
