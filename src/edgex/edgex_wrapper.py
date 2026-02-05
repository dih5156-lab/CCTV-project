"""
EdgeX CCTV 프로세서 래퍼
VideoProcessor를 EdgeX와 연동
"""

import asyncio
import logging
from typing import Dict, Optional
from src.core.processor import VideoProcessor
from src.core.events import DetectionEvent
from .device_service import CCTVDeviceService

logger = logging.getLogger(__name__)


class EdgeXCCTVProcessor:
    """EdgeX 기반 CCTV 프로세서"""
    
    def __init__(self, processor: VideoProcessor, edgex_config: Dict):
        """
        매개변수:
            processor: VideoProcessor 인스턴스
            edgex_config: EdgeX 설정 딕셔너리
        """
        self.processor = processor
        self.edgex_config = edgex_config
        self.edgex_service: Optional[CCTVDeviceService] = None
        self.event_loop = None
        self.use_edgex = True
    
    async def initialize(self):
        """EdgeX 서비스 초기화 및 카메라 등록"""
        try:
            logger.info("=" * 60)
            logger.info("EdgeX CCTV 프로세서 초기화")
            logger.info("=" * 60)
            
            # EdgeX Device Service 생성
            self.edgex_service = CCTVDeviceService(self.edgex_config)
            await self.edgex_service.initialize()
            
            # Device Profile 생성 (device-virtual 서비스 사용)
            await self.edgex_service.create_device_profile()
            
            # 모든 카메라를 EdgeX에 등록
            for camera_id, camera in self.processor.cameras.items():
                rtsp_source = camera.source
                if isinstance(rtsp_source, int):
                    # USB 웹캠인 경우
                    rtsp_source = f"usb-camera-{camera_id}"
                
                await self.edgex_service.add_camera(camera_id, rtsp_source)
            
            logger.info(f"✓ EdgeX 초기화 완료 ({len(self.processor.cameras)}개 카메라)")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"EdgeX 초기화 실패: {e}")
            raise
    
    async def send_events_to_edgex(self, camera_id: str, events: list) -> bool:
        """
        이벤트를 EdgeX로 전송
        
        매개변수:
            camera_id: 카메라 ID
            events: DetectionEvent 리스트
            
        반환값:
            전송 성공 여부
        """
        if not self.edgex_service or not events:
            return False
        
        try:
            return await self.edgex_service.send_detection_event(camera_id, events)
        except Exception as e:
            logger.error(f"EdgeX 이벤트 전송 실패 ({camera_id}): {e}")
            return False
    
    def start(self):
        """EdgeX와 함께 프로세서 시작"""
        try:
            # 비동기 루프 생성
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            # EdgeX 초기화 (동기적으로 대기)
            self.event_loop.run_until_complete(self.initialize())
            
            # VideoProcessor에 use_edgex 플래그 설정
            self.processor.use_edgex = True
            self.processor.edgex_processor = self
            
            # 프로세서 시작
            self.processor.start()
            
            logger.info("✓ EdgeX CCTV 프로세서 시작됨")
            
        except Exception as e:
            logger.error(f"프로세서 시작 실패: {e}")
            self.stop()
            raise
    
    def stop(self):
        """프로세서 종료"""
        try:
            self.processor.stop()
            
            if self.event_loop:
                self.event_loop.close()
            
            logger.info("✓ EdgeX CCTV 프로세서 종료됨")
            
        except Exception as e:
            logger.error(f"프로세서 종료 오류: {e}")
