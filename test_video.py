"""
Test script for video file processing
"""

import sys
import time
from pathlib import Path
from src.core import VideoProcessor
from src.config import AppConfig

# Video file path - update this to your video location
VIDEO_PATH = "C:\\Users\\dih51\\OneDrive\\Desktop\\test_video\\video1.mp4"

def main():
    """Main test function"""
    # Validate video file exists
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in test_video.py to point to your test video.")
        sys.exit(1)
    
    print("=" * 60)
    print("Video Test Mode")
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    
    # Create configuration
    config = AppConfig()
    config.display = True
    config.detection.device = "cpu"
    config.detection.helmet_confidence = 0.35  # Lower threshold for more detections
    config.detection.pose_confidence = 0.35
    config.detection.target_fps = 30
    config.collect_dataset = False
    
    # Validate configuration
    if not config.validate():
        print("WARN: Configuration validation failed. Some features may not work.")
    
    print(config.summary())
    print("=" * 60)
    
    # Create processor
    processor = VideoProcessor(config)
    
    # Add video camera
    if not processor.add_camera("video_test", str(video_path)):
        print("ERROR: Failed to add video source")
        sys.exit(1)
    
    print("Starting video processing... Press Ctrl+C to stop.")
    print("=" * 60)
    
    try:
        processor.start()
        while processor.running:
            time.sleep(10)
            processor.print_stats()
    except KeyboardInterrupt:
        print("\nUser interrupted (Ctrl+C)")
    except Exception as e:
        print(f"\nERROR during processing: {e}")
    finally:
        print("Stopping processor...")
        processor.stop()
        print("Done.")

if __name__ == "__main__":
    main()
