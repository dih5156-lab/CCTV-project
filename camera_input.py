"""
[camera_input.py] 카메라/영상 입력 모듈
날짜 : 2025-11-12
내용 : AI 분석을 위한 카메라/영상 입력 모듈
특이상항 : 현재 준비된 카메라 및 NVR/VMS 장비가 없으므로 테스트용 코드 작성 추후 장비 입고 시 코드 수정 예정
버전 : Ver 0.1
제작자 : Hangibum
"""

import cv2
import time
from threading import Lock
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class RTSPCamera:
    """RTSP 카메라 관리 (자동 재연결)

    이 클래스는 `ProcessorConfig` 같은 설정 객체를 받아 사용합니다.
    processor.py에서 설정 객체를 전달하면 그대로 사용됩니다.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        config
    ):
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.cap = None
        self.connected = False
        self.last_frame_time = 0
        self.reconnect_attempts = 0
        self._lock = Lock()

    def connect(self) -> bool:
        """카메라 연결 (타임아웃 적용)"""
        with self._lock:
            try:
                logger.info(f"[{self.camera_id}] RTSP 연결 시도: {self.source}")
                
                # 타임아웃 설정 (기본 10초)
                timeout = getattr(self.config, 'rtsp_read_timeout', 10)
                
                # RTSP over TCP 사용 (UDP보다 안정적, 패킷 손실 방지)
                if isinstance(self.source, str) and self.source.startswith('rtsp://'):
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000,
                    ])
                    # TCP 전송 강제 (패킷 순서 보장)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                else:
                    self.cap = cv2.VideoCapture(self.source)
                
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 최신 프레임만 유지

                # 연결 확인 (타임아웃 내에 프레임 수신)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.connected = True
                    self.reconnect_attempts = 0
                    logger.info(f"[{self.camera_id}] ✅ 연결 성공")
                    return True
                else:
                    self.connected = False
                    logger.warning(f"[{self.camera_id}] ⚠️ 첫 프레임 수신 실패")
                    if self.cap:
                        self.cap.release()
                    return False
            except KeyboardInterrupt:
                raise  # Ctrl+C는 전파
            except Exception as e:
                logger.error(f"[{self.camera_id}] ❌ 연결 오류: {e}")
                self.connected = False
                if self.cap:
                    self.cap.release()
                return False

    def get_frame(self) -> Tuple[bool, any]:
        """프레임 획득 (지수 백오프 자동 재연결)"""
        if not self.connected:
            max_retries = getattr(self.config, 'rtsp_max_retries', 5)
            
            if self.reconnect_attempts < max_retries:
                self.reconnect_attempts += 1
                
                # 지수 백오프: 5초 → 10초 → 20초 → 40초 → 60초(최대)
                base_interval = getattr(self.config, 'rtsp_reconnect_interval', 5)
                delay = min(base_interval * (2 ** (self.reconnect_attempts - 1)), 60)
                
                logger.info(f"[{self.camera_id}] 재연결 시도 {self.reconnect_attempts}/{max_retries} (⏳ {delay}초 후)")
                time.sleep(delay)
                
                if self.connect():
                    logger.info(f"✅ [{self.camera_id}] 재연결 성공!")
                else:
                    logger.warning(f"⚠️ [{self.camera_id}] 재연결 실패")
            else:
                logger.error(f"❌ [{self.camera_id}] 최대 재시도 횟수 초과 ({max_retries}회)")
            
            return False, None

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame_time = time.time()
                self.reconnect_attempts = 0  # 성공하면 카운터 리셋
                return True, frame
            else:
                logger.warning(f"[{self.camera_id}] 프레임 수신 실패, 재연결 대기 중...")
                self.connected = False
                return False, None
        except Exception as e:
            logger.error(f"[{self.camera_id}] 프레임 획득 오류: {e}")
            self.connected = False
            return False, None

    def release(self):
        """연결 해제"""
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.connected = False
                logger.info(f"[{self.camera_id}] 연결 해제")

class CameraInput:
    def __init__(self, video_path=None):
        
        self.video_path = video_path
        if video_path:
            self.cap = cv2.VideoCapture(video_path)          
        else:
            self.cap = cv2.VideoCapture(0)              # 기본 카메라
            
        if not self.cap.isOpened():
            raise RecursionError("카메라나 비디오 파일을 열 수 없습니다.")
            
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


__all__ = ["CameraInput", "RTSPCamera"]