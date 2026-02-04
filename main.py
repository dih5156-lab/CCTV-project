"""
main.py - 다중 카메라 CCTV 시스템 진입점
"""

import argparse
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any

from src.core import VideoProcessor
from src.config import default_config, AppConfig

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


def load_camera_list(path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 카메라 목록 로드"""
    try:
        if not Path(path).exists():
            print(f"ERROR: 카메라 설정 파일을 찾을 수 없습니다: {path}")
            return []
        
        if Path(path).stat().st_size == 0:
            print(f"ERROR: 카메라 설정 파일이 비어있습니다: {path}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            cameras = json.load(f)
            
        if not isinstance(cameras, list):
            print(f"ERROR: 잘못된 카메라 설정 형식 (리스트 필요): {path}")
            return []
        
        # 각 카메라 항목 검증
        valid_cameras = []
        for idx, cam in enumerate(cameras):
            if not isinstance(cam, dict):
                print(f"WARN: 인덱스 {idx}의 잘못된 카메라 항목 건너뜀 (딕셔너리가 아님)")
                continue
            if 'id' not in cam or 'source' not in cam:
                print(f"WARN: 인덱스 {idx}의 카메라 건너뜀 ('id' 또는 'source' 누락)")
                continue
            valid_cameras.append(cam)
        
        print(f"{path}에서 {len(valid_cameras)}개 카메라 로드됨")
        return valid_cameras
        
    except FileNotFoundError:
        print(f"ERROR: 카메라 설정 파일을 찾을 수 없습니다: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"ERROR: {path}의 JSON 파싱 오류: {e}")
        return []
    except Exception as e:
        print(f"ERROR: 카메라 목록 로드 실패: {e}")
        return []


def start_processor(camera_list: List[Dict[str, Any]], cfg: AppConfig, use_edgex: bool = False, edgex_config: Dict = None) -> None:
    """카메라와 함께 비디오 프로세서 시작
    
    매개변수:
        camera_list: 카메라 목록
        cfg: 애플리케이션 설정
        use_edgex: EdgeX 사용 여부
        edgex_config: EdgeX 설정
    """
    if not camera_list:
        print("ERROR: 카메라가 제공되지 않았습니다. 프로세서를 시작할 수 없습니다.")
        return
    
    processor = VideoProcessor(cfg)
    
    added_count = 0
    for cam in camera_list:
        print(f"\n[DEBUG] 카메라 처리 중: {cam}")
        
        if not cam.get('enabled', True):
            print(f"SKIP: 카메라 비활성화됨: {cam.get('id')} - {cam.get('name', 'N/A')}")
            continue
            
        cam_id = cam.get('id')
        source = cam.get('source')
        
        print(f"[DEBUG] cam_id={cam_id}, source={source} (type={type(source).__name__})")
        
        if not cam_id or source is None:
            print(f"WARN: id 또는 source가 누락된 카메라 건너뜀: {cam}")
            continue
        
        # 웹캠 인덱스를 위한 숫자 문자열을 정수로 변환
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            try:
                source = int(source)
                print(f"[DEBUG] source를 정수로 변환: {source}")
            except (ValueError, TypeError):
                pass
        
        print(f"[DEBUG] 카메라 추가 시도: {cam_id}, source={source}")
        added = processor.add_camera(cam_id, source)
        if added:
            added_count += 1
            print(f"✓ 카메라 추가 성공: {cam_id} ({source})")
        else:
            print(f"✗ 카메라 추가 실패: {cam_id} ({source})")

    if added_count == 0:
        print("ERROR: 성공적으로 등록된 카메라가 없습니다. 종료합니다.")
        return
    
    print(f"\n{added_count}개 카메라로 프로세서 시작 중...")

    try:
        # EdgeX 연동 여부에 따라 시작
        if use_edgex and edgex_config:
            from src.edgex import EdgeXCCTVProcessor
            
            print("=" * 60)
            print("EdgeX Foundry 연동 모드로 시작합니다")
            print("=" * 60)
            
            edgex_processor = EdgeXCCTVProcessor(processor, edgex_config)
            edgex_processor.start()
        else:
            processor.start()
        
        print("프로세서가 성공적으로 시작되었습니다. 중지하려면 Ctrl+C를 누르세요.\n")
        while processor.running:
            time.sleep(10)
            processor.print_stats()
    except KeyboardInterrupt:
        print("\n사용자가 중단함 (Ctrl+C)")
    except Exception as e:
        print(f"\n처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("프로세서 중지 중...")
        processor.stop()
        print("프로세서가 중지되었습니다.")


def apply_args_to_config(args: argparse.Namespace, config: AppConfig) -> AppConfig:
    """명령줄 인자를 설정에 적용"""
    if args.helmet_model:
        config.models.helmet_model = args.helmet_model
    if args.pose_model:
        config.models.pose_model = args.pose_model
    
    config.server.url = args.server
    config.detection.helmet_confidence = args.confidence
    config.detection.pose_confidence = args.pose_confidence
    config.detection.device = args.device
    config.detection.target_fps = args.fps
    config.processing.frame_skip = args.frame_skip
    config.events.debounce_enabled = not args.no_debounce
    config.events.debounce_seconds = args.debounce
    config.display = args.display
    config.zone_detection = args.zone_detection
    config.zones_config = args.zones_config
    config.collect_dataset = args.collect_dataset
    config.dataset_dir = args.dataset_dir
    
    return config


def main() -> None:
    """메인 진입점"""
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
    
    # EdgeX 연동
    edgex_group = parser.add_argument_group('EdgeX Foundry 연동')
    edgex_group.add_argument('--edgex', action='store_true',
                            help='EdgeX Foundry 연동 활성화')
    edgex_group.add_argument('--edgex-metadata-url', default='http://localhost:59881',
                            help='EdgeX Core Metadata URL (기본: localhost:59881)')
    edgex_group.add_argument('--edgex-data-url', default='http://localhost:59880',
                            help='EdgeX Core Data URL (기본: localhost:59880)')
    edgex_group.add_argument('--edgex-service-name', default='cctv-device-service',
                            help='EdgeX Device Service 이름')
    
    args = parser.parse_args()
    
    if args.confidence < 0.0 or args.confidence > 1.0:
        parser.error("--confidence 값은 0.0에서 1.0 사이여야 합니다")
    
    if args.fps <= 0:
        parser.error("--fps 값은 양수여야 합니다")
    
    if args.video and not Path(args.video).exists():
        parser.error(f"비디오 파일을 찾을 수 없습니다: {args.video}")
    
    cfg = apply_args_to_config(args, default_config)
    
    # 설정 검증
    if not cfg.validate():
        print("\nWARN: 설정 검증 실패. 일부 기능이 작동하지 않을 수 있습니다.")
    
    print("=" * 60)
    print("CCTV 헬멧 감지 시스템")
    print("=" * 60)
    print(cfg.summary())
    print("=" * 60)

    if args.cameras:
        cams = load_camera_list(args.cameras)
    else:
        if args.mode == 'single':
            source = args.video if args.video else 0
            source_name = 'video' if args.video else 'webcam'
            cams = [{'id': source_name, 'source': source}]
            print(f"소스: {source_name} ({source})")
        else:
            print("WARN: 다중 카메라 모드는 cameras.json 파일이 필요합니다")
            cams = [
                {'id': 'nvr_camera_1', 'source': 'rtsp://admin:password@192.168.1.100:554/stream1'},
                {'id': 'nvr_camera_2', 'source': 'rtsp://admin:password@192.168.1.100:554/stream2'},
            ]
    
    print("=" * 60)
    
    # EdgeX 설정
    edgex_config = None
    if args.edgex:
        edgex_config = {
            "coreMetadataUrl": args.edgex_metadata_url,
            "coreDataUrl": args.edgex_data_url,
            "deviceServiceName": args.edgex_service_name,
            "baseUrl": "http://localhost:59999"
        }
    
    start_processor(cams, cfg, use_edgex=args.edgex, edgex_config=edgex_config)


if __name__ == '__main__':
    main()

