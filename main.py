"""
main.py - 멀티 카메라 CCTV 시스템 실행 진입점
"""

import argparse
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any

from src.core import VideoProcessor
from src.config import default_config, AppConfig

# FFMPEG 경고 메시지 숨기기 (RTSP 스트리밍 노이즈 제거)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # OpenCV 로그 레벨 설정


def load_camera_list(path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 카메라 리스트 로드"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cameras = json.load(f)
            if not isinstance(cameras, list):
                print(f"❌ 잘못된 카메라 설정 형식: {path} (리스트 형식이어야 함)")
                return []
            return cameras
    except FileNotFoundError:
        print(f"❌ 카메라 설정 파일을 찾을 수 없습니다: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {path} - {e}")
        return []
    except Exception as e:
        print(f"❌ 카메라 리스트 로드 실패: {e}")
        return []


def start_processor(camera_list: List[Dict[str, Any]], cfg: AppConfig) -> None:
    """Start video processor with cameras"""
    processor = VideoProcessor(cfg)
    # 카메라 등록 (enabled=true인 것만)
    for cam in camera_list:
        # enabled 필드가 false이면 건너뛰기
        if not cam.get('enabled', True):
            print(f"⏭️  카메라 비활성화됨 (건너뛰기): {cam.get('id')} - {cam.get('name', 'N/A')}")
            continue
            
        cam_id = cam.get('id')
        source = cam.get('source')
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # 숫자 문자열이면 정수로 변환
            try:
                source = int(source)
            except Exception:
                pass
        added = processor.add_camera(cam_id, source)
        if not added:
            print(f"⚠️  카메라 등록 실패: {cam_id} ({source})")

    if not processor.cameras:
        print("❌ 등록된 카메라가 없습니다. 종료합니다.")
        return

    try:
        processor.start()
        while processor.running:
            time.sleep(10)
            processor.print_stats()
    except KeyboardInterrupt:
        print("사용자 중단 (Ctrl+C)")
    finally:
        processor.stop()


def apply_args_to_config(args: argparse.Namespace, config: AppConfig) -> AppConfig:
    """명령행 인자를 config에 적용"""
    # 모델 경로
    if args.helmet_model:
        config.models.helmet_model = args.helmet_model
    if args.pose_model:
        config.models.pose_model = args.pose_model
    
    # 서버 설정
    config.server.url = args.server
    
    # 탐지 설정
    config.detection.helmet_confidence = args.confidence
    config.detection.pose_confidence = args.pose_confidence
    config.detection.device = args.device
    config.detection.target_fps = args.fps
    
    # 이벤트 설정
    config.events.debounce_enabled = not args.no_debounce
    config.events.debounce_seconds = args.debounce
    
    # 표시 및 기능
    config.display = args.display
    config.zone_detection = args.zone_detection
    config.zones_config = args.zones_config
    config.collect_dataset = args.collect_dataset
    config.dataset_dir = args.dataset_dir
    
    return config


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='CCTV 헬멧 착용 및 낙상 감지 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 웹캠으로 실행 (기본)
  python main.py --display
  
  # 비디오 파일로 테스트
  python main.py --video test_video/sample.mp4 --display
  
  # RTSP 카메라로 실행
  python main.py --cameras cameras.json --server http://server.com/api
  
  # GPU 사용 및 데이터셋 수집
  python main.py --device cuda --collect-dataset --dataset-dir ./my_data
        """
    )
    
    # 입력 소스
    input_group = parser.add_argument_group('입력 소스')
    input_group.add_argument('--cameras', '-c', help='카메라 목록 JSON 파일 경로', default=None)
    input_group.add_argument('--mode', choices=['single', 'multi'], default='single', 
                            help='단일 카메라 또는 다중 카메라 모드')
    input_group.add_argument('--video', help='비디오 파일 경로 (웹캠 대신 사용)', default=None)
    
    # 모델 설정
    model_group = parser.add_argument_group('모델 설정')
    model_group.add_argument('--helmet_model', default=None, 
                            help='헬멧 감지 모델 경로 (기본: config.py에서 자동 탐지)')
    model_group.add_argument('--pose_model', default=None, 
                            help='Pose 모델 경로 (기본: yolov8n-pose.pt - 사람 + 관절)')
    model_group.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                            help='실행 디바이스 (cpu 또는 cuda)')
    model_group.add_argument('--confidence', type=float, default=0.5, 
                            help='헬멧 감지 신뢰도 임계값 (0.0-1.0)')
    model_group.add_argument('--pose-confidence', type=float, default=0.3, 
                            help='사람 감지 신뢰도 임계값 (0.0-1.0, 낮을수록 더 많이 감지)')
    
    # 서버 및 성능
    server_group = parser.add_argument_group('서버 및 성능')
    server_group.add_argument('--server', default='http://localhost:8000/api/events',
                             help='이벤트 전송 서버 URL')
    server_group.add_argument('--fps', type=int, default=30, help='목표 FPS')
    server_group.add_argument('--display', action='store_true', help='화면 표시 활성화')
    server_group.add_argument('--frame-skip', type=int, default=3,
                             help='프레임 스킵 (매 N프레임마다 AI 추론, 놓을수록 빠름. 권장: 2-5)')
    
    # 이벤트 설정
    event_group = parser.add_argument_group('이벤트 설정')
    event_group.add_argument('--no-debounce', action='store_true', 
                            help='이벤트 디바운싱 비활성화')
    event_group.add_argument('--debounce', type=float, default=3.0, 
                            help='디바운싱 시간 (초)')
    
    # 위험 구역 탐지
    zone_group = parser.add_argument_group('위험 구역 탐지')
    zone_group.add_argument('--zone-detection', action='store_true', 
                           help='위험 구역 감지 활성화')
    zone_group.add_argument('--zones-config', default='zones_config.json', 
                           help='구역 설정 JSON 파일 경로')
    
    # 데이터셋 수집
    dataset_group = parser.add_argument_group('데이터셋 수집')
    dataset_group.add_argument('--collect-dataset', action='store_true', 
                              help='탐지 데이터 자동 수집')
    dataset_group.add_argument('--dataset-dir', default='./collected_data', 
                              help='데이터셋 저장 디렉터리')
    
    args = parser.parse_args()
    
    # 설정 검증
    if args.confidence < 0.0 or args.confidence > 1.0:
        parser.error("--confidence 값은 0.0에서 1.0 사이여야 합니다")
    
    if args.fps <= 0:
        parser.error("--fps 값은 양수여야 합니다")
    
    if args.video and not Path(args.video).exists():
        parser.error(f"비디오 파일을 찾을 수 없습니다: {args.video}")
    
    # config에 명령행 인자 적용
    cfg = apply_args_to_config(args, default_config)
    
    # 모델 검증
    print("=" * 60)
    print("🚀 CCTV 헬멧 착용 감지 시스템 시작")
    print("=" * 60)
    if not cfg.models.helmet_model:
        print("⚠️  경고: 헬멧 모델 파일이 없습니다. config.py에서 경로를 확인하세요.")
    else:
        print(f"✅ 헬멧 모델: {cfg.models.helmet_model}")
    
    print(f"✅ Pose 모델: {cfg.models.pose_model} (사람 + 관절 감지)")
    print(f"🔧 디바이스: {cfg.detection.device}")
    print(f"🎯 신뢰도 임계값: {cfg.detection.helmet_confidence}")
    print("=" * 60)

    if args.cameras:
        cams = load_camera_list(args.cameras)
    else:
        if args.mode == 'single':
            # --video 옵션이 있으면 동영상 파일 사용
            source = args.video if args.video else 0
            source_name = 'video' if args.video else 'webcam'
            cams = [{'id': source_name, 'source': source}]
            print(f"📹 소스: {source_name} ({source})")
        else:
            # 기본 예시(사용자 수정 필요)
            print("⚠️  다중 카메라 모드: cameras.json 파일을 생성하거나 --cameras 옵션을 사용하세요")
            cams = [
                {'id': 'nvr_camera_1', 'source': 'rtsp://admin:password@192.168.1.100:554/stream1'},
                {'id': 'nvr_camera_2', 'source': 'rtsp://admin:password@192.168.1.100:554/stream2'},
            ]
    
    print("=" * 60)
    start_processor(cams, cfg)


if __name__ == '__main__':
    main()

