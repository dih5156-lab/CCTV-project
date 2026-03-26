"""카메라/비디오 입력 모듈.

RTSP 연결을 처리하는 `RTSPCamera`를
일반 입력용 `CameraInput`과 제공한다.
"""

import logging
import time
from threading import Lock
from typing import Any, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class RTSPCamera:
    """재연결 지원과 프레임 획득 기능을 갖춘 RTSP 카메라 입력 클래스."""

    def __init__(self, camera_id: str, source: Any, config: Any):
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.connected = False
        self.last_frame_time = 0.0
        self.reconnect_attempts = 0
        self._consecutive_read_failures = 0
        self._lock = Lock()

    @property
    def is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.startswith("rtsp://")

    def _camera_option(self, key: str, default: Any) -> Any:
        camera_cfg = getattr(self.config, "camera", None)
        if camera_cfg is not None and hasattr(camera_cfg, key):
            value = getattr(camera_cfg, key)
            if value is not None:
                return value
        return getattr(self.config, key, default)

    def _safe_release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.is_rtsp:
            timeout = int(self._camera_option("read_timeout", 10))
            buffer_size = int(self._camera_option("buffer_size", 1))
            cap = cv2.VideoCapture(
                self.source,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    timeout * 1000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    timeout * 1000,
                ],
            )
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            return cap

        logger.info("[%s] 로컬 소스 연결 시도 중: %s", self.camera_id, self.source)
        cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)
        return cap

    def connect(self) -> bool:
        """카메라 연결 후 첫 프레임 획득 시도."""
        with self._lock:
            try:
                logger.info("[%s] 카메라 연결 시도 중: %s", self.camera_id, self.source)

                self._safe_release()
                self.cap = self._create_capture()
                if self.cap is None:
                    self.connected = False
                    return False

                retry_seconds = float(getattr(self.config, "rtsp_connect_retry_seconds", 5))
                retry_interval = float(getattr(self.config, "rtsp_connect_retry_interval", 0.2))
                retry_interval = max(0.01, retry_interval)
                max_attempts = max(1, int(retry_seconds / retry_interval))

                ret, frame = False, None
                for attempt in range(max_attempts):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.connected = True
                        self.reconnect_attempts = 0
                        self._consecutive_read_failures = 0
                        logger.info(
                            f"[{self.camera_id}] 연결 성공 (해상도: {frame.shape[1]}x{frame.shape[0]}, "
                            f"시도 {attempt + 1}/{max_attempts})"
                        )
                        return True
                    time.sleep(retry_interval)

                self.connected = False
                logger.warning(
                    f"[{self.camera_id}] 프레임 읽기 실패 (최대 {max_attempts}회, "
                    f"결과 ret={ret}, frame={'None' if frame is None else 'exists'})"
                )

                is_opened = self.cap.isOpened() if self.cap else False
                logger.warning("[%s] VideoCapture.isOpened() = %s", self.camera_id, is_opened)

                if self.is_rtsp and is_opened:
                    self.connected = True
                    self.reconnect_attempts = 0
                    self._consecutive_read_failures = 0
                    logger.warning(
                        f"[{self.camera_id}] RTSP 초기 프레임 없지만 스트림 연결은 확인됨. "
                        "캡처를 계속 진행하므로 첫 프레임은 추후 도착할 수 있습니다."
                    )
                    return True

                self._safe_release()
                return False
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("[%s] 연결 오류: %s", self.camera_id, e)
                self.connected = False
                self._safe_release()
                return False

    def _try_reconnect(self) -> bool:
        max_retries = int(self._camera_option("max_retries", 5))
        if self.reconnect_attempts >= max_retries:
            logger.error("[%s] 최대 재연결 횟수 초과 (%s)", self.camera_id, max_retries)
            return False

        self.reconnect_attempts += 1
        base_interval = float(self._camera_option("reconnect_interval", 5))
        delay = min(base_interval * (2 ** (self.reconnect_attempts - 1)), 60)

        logger.info(
            f"[{self.camera_id}] 재연결 시도 {self.reconnect_attempts}/{max_retries} ({delay}초 후)"
        )
        time.sleep(delay)

        ok = self.connect()
        if ok:
            logger.info("[%s] 재연결 성공", self.camera_id)
        else:
            logger.warning("[%s] 재연결 실패", self.camera_id)
        return ok

    def get_frame(self) -> Tuple[bool, Optional[Any]]:
        """프레임 획득 (재연결 로직 내장)."""
        if not self.connected:
            self._try_reconnect()
            return False, None

        try:
            frame_retry_count = max(1, int(getattr(self.config, "rtsp_frame_retry_count", 5)))
            frame_retry_interval = float(getattr(self.config, "rtsp_frame_retry_interval", 0.05))

            ret, frame = False, None
            for _ in range(frame_retry_count):
                with self._lock:
                    if not self.connected or self.cap is None:
                        return False, None
                    ret, frame = self.cap.read()

                if ret and frame is not None:
                    self.last_frame_time = time.time()
                    self.reconnect_attempts = 0
                    self._consecutive_read_failures = 0
                    return True, frame
                time.sleep(frame_retry_interval)

            self._consecutive_read_failures += 1

            if self.is_rtsp:
                max_failures = int(getattr(self.config, "rtsp_max_read_failures", 20))
                if self._consecutive_read_failures < max_failures:
                    return False, None

            logger.warning("[%s] 프레임 획득 실패, 재연결 시도 중...", self.camera_id)
            self.connected = False
            return False, None
        except Exception as e:
            logger.error("[%s] 프레임 읽기 오류: %s", self.camera_id, e)
            self.connected = False
            return False, None

    def release(self) -> None:
        """카메라 장치 해제 후 리소스 정리."""
        with self._lock:
            self._safe_release()
            self.connected = False
            logger.info("[%s] 카메라 해제됨", self.camera_id)


__all__ = ["RTSPCamera"]
