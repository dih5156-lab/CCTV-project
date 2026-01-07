"""
Test script for video file processing
"""

import sys
from processor import VideoProcessor, ProcessorConfig
from config import ModelPaths

# Video file path
VIDEO_PATH = "C:/Users/dih51/Desktop/Project/test_video/ai_test_video5.mp4"

# Load model paths from config
models = ModelPaths()

config = ProcessorConfig(
    helmet_model_path=models.helmet_model,
    pose_model_path=models.pose_model,
    display=True,
    target_fps=30,
    collect_dataset=False,
    helmet_confidence_threshold=0.45,  # 오탐지 방지를 위해 적절히 상향
    pose_confidence_threshold=0.35
)

processor = VideoProcessor(config)
processor.add_camera("video_test", VIDEO_PATH)

try:
    processor.start()
    import time
    while processor.running:
        time.sleep(10)
        processor.print_stats()
except KeyboardInterrupt:
    print("\nUser interrupted (Ctrl+C)")
finally:
    processor.stop()
