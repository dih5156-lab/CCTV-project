"""
camera_input.py - 카메라/비디오 입력 모듈
AI 분석 카메라/비디오 입력 모듈
"""

import cv2
import time
from threading import Lock
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class RTSPCamera:
    """자동 재연결 기능을 갖춘 RTSP 카메라 관리"""

    def __init__(self, camera_id: str, source: str, config):
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.cap = None
        self.connected = False
        self.last_frame_time = 0
        self.reconnect_attempts = 0
        self._lock = Lock()

    def connect(self) -> bool:
        """타임아웃을 적용하여 카메라 연결"""
        with self._lock:
            try:
                logger.info(f"[{self.camera_id}] 카메라 연결 중: {self.source}")
                
                # RTSP 스트림의 경우
                if isinstance(self.source, str) and self.source.startswith('rtsp://'):
                    timeout = getattr(self.config, 'rtsp_read_timeout', 10)
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000,
                    ])
                    # TCP 전송 강제 (패킷 순서 보장)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 최신 프레임만 유지
                else:
                    # 로컬 카메라 (정수 인덱스) 또는 비디오 파일
                    self.cap = cv2.VideoCapture(self.source)
                    # 로컬 카메라의 경우 BUFFERSIZE 설정 안함 (호환성 문제)

                # 연결 확인 (타임아웃 내에 프레임 수신)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.connected = True
                    self.reconnect_attempts = 0
                    logger.info(f"[{self.camera_id}] 연결 성공 (해상도: {frame.shape[1]}x{frame.shape[0]})")
                    return True
                else:
                    self.connected = False
                    logger.warning(f"[{self.camera_id}] 첫 번째 프레임 수신 실패")
                    if self.cap:
                        self.cap.release()
                    return False
            except KeyboardInterrupt:
                raise  # Ctrl+C 전파
            except Exception as e:
                logger.error(f"[{self.camera_id}] 연결 오류: {e}")
                self.connected = False
                if self.cap:
                    self.cap.release()
                return False

    def get_frame(self) -> Tuple[bool, any]:
        """프레임 가져오기 (지수 백오프를 사용한 자동 재연결)"""
        if not self.connected:
            max_retries = getattr(self.config, 'rtsp_max_retries', 5)
            
            if self.reconnect_attempts < max_retries:
                self.reconnect_attempts += 1
                
                # 지수 백오프: 5초 -> 10초 -> 20초 -> 40초 -> 60초(최대)
                base_interval = getattr(self.config, 'rtsp_reconnect_interval', 5)
                delay = min(base_interval * (2 ** (self.reconnect_attempts - 1)), 60)
                
                logger.info(f"[{self.camera_id}] 재연결 시도 {self.reconnect_attempts}/{max_retries} ({delay}초 대기 중)")
                time.sleep(delay)
                
                if self.connect():
                    logger.info(f"[{self.camera_id}] 재연결 성공!")
                else:
                    logger.warning(f"[{self.camera_id}] 재연결 실패")
            else:
                logger.error(f"[{self.camera_id}] 최대 재시도 횟수 초과 ({max_retries})")
            
            return False, None

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame_time = time.time()
                self.reconnect_attempts = 0  # 성공 시 카운터 초기화
                return True, frame
            else:
                logger.warning(f"[{self.camera_id}] 프레임 수신 실패, 재연결 대기 중...")
                self.connected = False
                return False, None
        except Exception as e:
            logger.error(f"[{self.camera_id}] 프레임 획듍 오류: {e}")
            self.connected = False
            return False, None

    def release(self):
        """연결 해제"""
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.connected = False
                logger.info(f"[{self.camera_id}] 연결 해제됨")

class CameraInput:
    """하위 호환성을 위한 간단한 카메라 입력 래퍼"""
    
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.cap = None
        self.opened = False
        
        try:
            if video_path:
                self.cap = cv2.VideoCapture(video_path)
            else:
                self.cap = cv2.VideoCapture(0)
                
            if not self.cap.isOpened():
                raise RuntimeError(f"{'비디오 파일을 열 수 없음: ' + video_path if video_path else '카메라를 열 수 없음'}")
            
            self.opened = True
        except Exception as e:
            logger.error(f"카메라 입력 초기화 실패: {e}")
            raise
            
    def get_frame(self):
        """단일 프레임 가져오기"""
        if not self.opened or self.cap is None:
            logger.warning("카메라가 열려있지 않음, 프레임을 가져올 수 없음")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("프레임 읽기 실패")
                return None
            return frame
        except Exception as e:
            logger.error(f"프레임 읽기 오류: {e}")
            return None
        
    def release(self):
        """카메라 리소스 해제"""
        if self.cap is not None:
            self.cap.release()
            self.opened = False
            logger.debug("카메라 해제됨")
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


__all__ = ["CameraInput", "RTSPCamera"]