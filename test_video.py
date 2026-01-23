"""
Test script for video file processing
"""

import sys
import time
from src.core import VideoProcessor
from src.config import AppConfig

# Video file path
VIDEO_PATH = "C:\\Users\\dih51\\OneDrive\\Desktop\\test_video\\video1.mp4"

# 설정 생성
config = AppConfig()
config.display = True
config.detection.device = "cpu"
config.detection.helmet_confidence = 0.35  # 헬멧 감지 임계값 낮춤 (더 많이 감지)
config.detection.pose_confidence = 0.35
config.detection.target_fps = 30
config.collect_dataset = False

processor = VideoProcessor(config)
processor.add_camera("video_test", VIDEO_PATH)

try:
    processor.start()
    while processor.running:
        time.sleep(10)
        processor.print_stats()
except KeyboardInterrupt:
    print("\nUser interrupted (Ctrl+C)")
finally:
    processor.stop()
