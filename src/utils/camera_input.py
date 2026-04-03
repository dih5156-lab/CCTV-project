"""카메라 입력 추상화 모듈.

RTSP 스트림 연결과 재연결을 담당하는 `RTSPCamera`와
단순 카메라 입력을 위한 `CameraInput`을 제공한다.
"""

import logging
import time
from threading import Lock
from typing import Any, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class RTSPCamera:
    """재연결 백오프와 프레임 재시도를 내장한 RTSP 카메라 클라이언트.

    매개변수:
        camera_id: 카메라 식별자 (로그·통계용)
        source:    RTSP URL 또는 로컬 장치 인덱스 (int)
        config:    AppConfig 인스턴스 (재연결 설정 참조)
    """

    def __init__(self, camera_id: str, source: Any, config: Any) -> None:
        self.camera_id = camera_id
        self.source = source
        self._config = config

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = Lock()

        # 연결 상태
        self.connected: bool = False
        self.reconnect_attempts: int = 0
        self.last_frame_time: Optional[float] = None

        # 재연결 백오프
        self._reconnect_delay: float = 1.0
        self._reconnect_max_delay: float = 60.0

    # ------------------------------------------------------------------
    # 연결 관리
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """카메라에 연결을 시도하고 성공 여부를 반환한다."""
        with self._lock:
            return self._connect_internal()

    def _connect_internal(self) -> bool:
        """내부 연결 로직 (락 보유 상태에서 호출)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logger.warning("[%s] VideoCapture 열기 실패: %s", self.camera_id, self.source)
                cap.release()
                return False

            # 버퍼 크기 최소화 (지연 감소)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._cap = cap
            self.connected = True
            self.reconnect_attempts = 0
            self._reconnect_delay = 1.0
            logger.info("[%s] 카메라 연결 성공: %s", self.camera_id, self.source)
            return True

        except Exception as exc:
            logger.error("[%s] 카메라 연결 오류: %s", self.camera_id, exc)
            return False

    def _try_reconnect(self) -> bool:
        """백오프를 적용하여 재연결을 시도한다."""
        self.reconnect_attempts += 1
        logger.info(
            "[%s] 재연결 시도 #%d (%.1f초 대기)...",
            self.camera_id,
            self.reconnect_attempts,
            self._reconnect_delay,
        )
        time.sleep(self._reconnect_delay)
        self._reconnect_delay = min(
            self._reconnect_delay * 2.0, self._reconnect_max_delay
        )
        return self._connect_internal()

    # ------------------------------------------------------------------
    # 프레임 획득
    # ------------------------------------------------------------------

    def get_frame(self) -> Tuple[bool, Optional[Any]]:
        """프레임을 읽어 반환한다. 실패 시 재연결을 시도한다.

        반환값:
            (성공 여부, 프레임 또는 None)
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self.connected = False
                if not self._try_reconnect():
                    return False, None

            ret, frame = self._cap.read()
            if ret and frame is not None:
                self.last_frame_time = time.time()
                return True, frame

            # 프레임 읽기 실패 → 재연결
            logger.warning("[%s] 프레임 읽기 실패 — 재연결 시도", self.camera_id)
            self.connected = False
            if self._try_reconnect():
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self.last_frame_time = time.time()
                    return True, frame

            return False, None

    # ------------------------------------------------------------------
    # 해제
    # ------------------------------------------------------------------

    def release(self) -> None:
        """카메라 리소스를 해제한다."""
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as exc:
                    logger.debug("[%s] 카메라 해제 중 오류 (무시됨): %s", self.camera_id, exc)
                finally:
                    self._cap = None
            self.connected = False

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


class CameraInput:
    """단순 카메라 입력 래퍼 (로컬 웹캠 또는 파일).

    RTSPCamera 와 동일한 인터페이스를 제공하여
    VideoProcessor 에서 교체 가능하게 사용할 수 있다.

    매개변수:
        source: 장치 인덱스(int) 또는 파일 경로(str)
    """

    def __init__(self, source: Any = 0) -> None:
        self.source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self.connected: bool = False
        self.reconnect_attempts: int = 0
        self.last_frame_time: Optional[float] = None

    def connect(self) -> bool:
        """카메라에 연결한다."""
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logger.warning("CameraInput 열기 실패: %s", self.source)
                cap.release()
                return False
            self._cap = cap
            self.connected = True
            logger.info("CameraInput 연결 성공: %s", self.source)
            return True
        except Exception as exc:
            logger.error("CameraInput 연결 오류: %s", exc)
            return False

    def get_frame(self) -> Tuple[bool, Optional[Any]]:
        """프레임을 읽어 반환한다."""
        if self._cap is None or not self._cap.isOpened():
            self.connected = False
            return False, None
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self.last_frame_time = time.time()
            return True, frame
        return False, None

    def release(self) -> None:
        """카메라 리소스를 해제한다."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            finally:
                self._cap = None
        self.connected = False

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
