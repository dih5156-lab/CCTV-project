"""동영상 파일 처리 테스트 스크립트

직접 실행: python tests/test_video.py
또는:      python -m tests.test_video
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가 (tests/ 에서 실행 시 필요)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import AppConfig
from src.core import VideoProcessor

# 동영상 파일 경로 - 로컬 환경에 맞게 수정하세요
VIDEO_PATH = "C:\\Users\\dih51\\OneDrive\\Desktop\\test_video\\video1.mp4"


def main():
    """테스트 실행 함수"""
    # 동영상 파일 존재 여부 확인
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"오류: 동영상 파일을 찾을 수 없습니다: {VIDEO_PATH}")
        print("tests/test_video.py 의 VIDEO_PATH를 실제 테스트 동영상 경로로 변경하세요.")
        sys.exit(1)

    print("=" * 60)
    print("동영상 테스트 모드")
    print("=" * 60)
    print(f"동영상: {VIDEO_PATH}")

    # 설정 생성
    config = AppConfig()
    config.display = True
    config.detection.device = "cpu"
    config.detection.helmet_confidence = 0.35  # 임계값을 낮춰 더 많은 탐지 허용
    config.detection.pose_confidence = 0.35
    config.detection.target_fps = 30
    config.collect_dataset = False

    # 설정 검증
    if not config.validate():
        print("경고: 설정 검증에 실패했습니다. 일부 기능이 정상 동작하지 않을 수 있습니다.")

    print(config.summary())
    print("=" * 60)

    # 프로세서 생성
    processor = VideoProcessor(config)

    # 동영상 소스 추가
    if not processor.add_camera("video_test", str(video_path)):
        print("오류: 동영상 소스 추가 실패")
        sys.exit(1)

    print("동영상 처리를 시작합니다... 중지하려면 Ctrl+C를 누르세요.")
    print("=" * 60)

    try:
        processor.start()
        while processor.running:
            time.sleep(10)
            processor.print_stats()
    except KeyboardInterrupt:
        print("\n사용자 중단 (Ctrl+C)")
    except Exception as e:
        print(f"\n처리 중 오류: {e}")
    finally:
        print("프로세서 중지 중...")
        processor.stop()
        print("완료.")


if __name__ == "__main__":
    main()
