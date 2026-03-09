"""main.py - 다중 카메라 CCTV 시스템 진입점"""

import argparse
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path

from src.core import VideoProcessor
from src.config import AppConfig
from src.services.zone_api import start_zone_api_server
from src.utils.zone_drawer import ZoneDrawer

# ── OpenCV 환경 설정 ───────────────────────────────────────────────
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# ── 로거 설정 ──────────────────────────────────────────────────────
import sys
import io

# Windows 콘솔 코드페이지를 UTF-8(65001)로 변경 + Python 스트림 인코딩 통일
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    ctypes.windll.kernel32.SetConsoleCP(65001)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


# ──────────────────────────────────────────────────────────────────
# 카메라 설정 로드
# ──────────────────────────────────────────────────────────────────

def load_camera_list(path: str) -> list[dict]:
    """카메라 설정 파일에서 목록 로드"""
    p = Path(path)
    if not p.exists():
        logger.error("카메라 설정 파일을 찾을 수 없습니다: %s", path)
        return []
    if p.stat().st_size == 0:
        logger.error("카메라 설정 파일이 비어있습니다: %s", path)
        return []

    try:
        cameras = json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        logger.error("%s JSON 파싱 오류: %s", path, e)
        return []

    if not isinstance(cameras, list):
        logger.error("잘못된 카메라 설정 형식 (리스트 필요): %s", path)
        return []

    valid_cameras = []
    for idx, cam in enumerate(cameras):
        if not isinstance(cam, dict):
            logger.warning("인덱스 %d의 카메라 항목 건너뜀 (딕셔너리가 아님)", idx)
            continue
        if 'id' not in cam or 'source' not in cam:
            logger.warning("인덱스 %d의 카메라 건너뜀 ('id' 또는 'source' 누락)", idx)
            continue
        valid_cameras.append(cam)

    logger.info("%s에서 %d개 카메라 로드됨", path, len(valid_cameras))
    return valid_cameras


# ──────────────────────────────────────────────────────────────────
# 프로세서 실행
# ──────────────────────────────────────────────────────────────────

def _collect_active_cameras(camera_list: list[dict], processor: VideoProcessor) -> list[tuple]:
    """활성화된 카메라만 걸러 (cam_id, source, detections, model_paths, zones_data) 튜플 리스트로 반환"""
    active: list[tuple] = []
    for cam in camera_list:
        if not cam.get('enabled', True):
            logger.info("카메라 비활성화됨: %s (%s)", cam.get('id'), cam.get('name', 'N/A'))
            continue
        cam_id = cam.get('id')
        source = cam.get('source')
        if not cam_id or source is None:
            logger.warning("id 또는 source 누락 - 건너뜀: %s", cam)
            continue
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        active.append((
            cam_id,
            source,
            cam.get('detections') or cam.get('ai_models'),  # 하위 호환
            cam.get('model_paths') or None,
            cam.get('zones') or None,
        ))
    return active


def _connect_cameras_parallel(active_cams: list[tuple], processor: VideoProcessor) -> dict[str, bool]:
    """카메라 연결을 병렬로 수행하고 결과 딕셔너리 반환"""
    results: dict[str, bool] = {}
    lock = threading.Lock()

    def _try_add(cam_id: str, source, detections, model_paths, zones_data) -> None:
        ok = processor.add_camera(
            cam_id, source,
            detections=detections, model_paths=model_paths, zones_data=zones_data,
        )
        with lock:
            results[cam_id] = ok

    threads = [
        threading.Thread(target=_try_add, args=(cid, src, det, mp, zones), daemon=True)
        for cid, src, det, mp, zones in active_cams
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


def _initial_retry(active_cams: list[tuple], processor: VideoProcessor, max_attempts: int = 3) -> int:
    """연결 성공 카메라가 없을 때 블로킹 재시도 (최대 max_attempts회)

    반환값:
        추가 성공한 카메라 수
    """
    for attempt in range(1, max_attempts + 1):
        logger.info("초기 연결 재시도 %d/%d (30초 대기)...", attempt, max_attempts)
        time.sleep(30)
        for cam_id, source, detections, model_paths, zones_data in active_cams:
            if cam_id not in processor.cameras and processor.add_camera(
                cam_id, source,
                detections=detections, model_paths=model_paths, zones_data=zones_data,
            ):
                logger.info("재시도 성공: %s", cam_id)
                return 1
    return 0


def start_processor(
    camera_list: list[dict],
    cfg: AppConfig,
    cameras_json_path: str = 'cameras.json',
    api_port: int = 0,
    zone_presets_path: str = 'zone_presets.json',
) -> None:
    """카메라와 함께 비디오 프로세서를 시작

    매개변수:
        camera_list: 카메라 목록
        cfg: 애플리케이션 설정
        cameras_json_path: cameras.json 경로 (구역 API 저장용)
        api_port: Zone API HTTP 포트 (0이면 비활성화)
        zone_presets_path: 구역 프리셋 저장 파일 경로 (기본: zone_presets.json)
    """
    if not camera_list:
        logger.error("카메라가 제공되지 않았습니다. 프로세서를 시작할 수 없습니다.")
        return

    processor = VideoProcessor(cfg)

    active_cams = _collect_active_cameras(camera_list, processor)
    if not active_cams:
        logger.error("활성화된 카메라가 없습니다.")
        return

    results = _connect_cameras_parallel(active_cams, processor)

    added_count = 0
    for cam_id, source, _det, _mp, _zones in active_cams:
        if results.get(cam_id):
            added_count += 1
            logger.info("카메라 추가 성공: %s (%s)", cam_id, source)
        else:
            logger.warning("카메라 연결 실패: %s (%s) → 백그라운드 재시도 예약", cam_id, source)
            processor.enqueue_camera_retry(cam_id, source, delay_seconds=30)

    if added_count == 0:
        logger.warning("현재 연결된 카메라가 없습니다. 초기 재연결을 시도합니다.")
        added_count += _initial_retry(active_cams, processor)

    if added_count == 0:
        logger.error("카메라 연결에 최종 실패했습니다. 종료합니다.")
        return

    logger.info("%d개 카메라로 프로세서 시작 중...", added_count)

    if api_port > 0:
        start_zone_api_server(processor, cameras_json_path, api_port,
                              presets_path=zone_presets_path)

    if cfg.display:
        drawer = ZoneDrawer(processor, cameras_json_path)
        processor.set_zone_drawer(drawer)
        logger.info("구역 그리기 모드 사용 가능: 디스플레이 창에서 'd' 키를 누르세요")

    try:
        processor.start()
        logger.info("프로세서가 시작되었습니다. 중지하려면 Ctrl+C를 누르세요.")
        while processor.running:
            time.sleep(10)
            processor.print_stats()
    except KeyboardInterrupt:
        logger.info("사용자가 중단함 (Ctrl+C)")
    except Exception as e:
        logger.error("처리 중 오류 발생: %s", e)
        traceback.print_exc()
    finally:
        logger.info("프로세서 중지 중...")
        processor.stop()
        logger.info("프로세서가 중지되었습니다.")


# ──────────────────────────────────────────────────────────────────
# 인자 → 설정 적용
# ──────────────────────────────────────────────────────────────────

def apply_args_to_config(args: argparse.Namespace, config: AppConfig) -> AppConfig:
    """명령줄 인자를 AppConfig에 적용"""
    if args.helmet_model:
        config.models.helmet_model = args.helmet_model
    if args.pose_model:
        config.models.pose_model = args.pose_model

    config.detection.helmet_confidence = args.confidence
    config.detection.pose_confidence   = args.pose_confidence
    config.detection.device            = args.device
    config.detection.target_fps        = args.fps
    config.processing.frame_skip       = args.frame_skip
    config.events.debounce_enabled     = not args.no_debounce
    config.events.debounce_seconds     = args.debounce
    config.display                     = args.display
    config.zone_detection              = args.zone_detection
    config.zones_config                = args.zones_config
    config.collect_dataset             = args.collect_dataset
    config.dataset_dir                 = args.dataset_dir
    config.mqtt.broker                 = args.mqtt_broker
    config.mqtt.port                   = args.mqtt_port
    config.mqtt.topic_prefix           = args.mqtt_topic_prefix

    return config


# ──────────────────────────────────────────────────────────────────
# 인자 파서 구성
# ──────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='CCTV 헬멧 착용 및 낙상 감지 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 웹캠으로 실행 (기본)
  python main.py --display

  # 비디오 파일로 테스트
  python main.py --video sample.mp4 --display

  # RTSP 다중 카메라 실행
  python main.py --cameras cameras.json

  # CUDA 사용 및 데이터셋 수집
  python main.py --device cuda --collect-dataset --dataset-dir ./my_data
        """,
    )

    # 입력 소스
    g = parser.add_argument_group('입력 소스')
    g.add_argument('--cameras', '-c', default=None,
                   help='카메라 목록 JSON 파일 경로 (없으면 단일 소스 사용)')
    g.add_argument('--video', default=None,
                   help='비디오 파일 경로 (지정하지 않으면 웹캠 0번 사용)')

    # 모델 설정
    g = parser.add_argument_group('모델 설정')
    g.add_argument('--helmet-model', default=None,
                   help='헬멧 감지 모델 경로 (기본: config.py에서 자동 탐지)')
    g.add_argument('--pose-model', default=None,
                   help='Pose 모델 경로 (기본: yolov8n-pose.pt)')
    g.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                   help='실행 디바이스 (기본: cpu)')
    g.add_argument('--confidence', type=float, default=0.5,
                   help='헬멧 감지 신뢰도 임계값 0.0-1.0 (기본: 0.5)')
    g.add_argument('--pose-confidence', type=float, default=0.3,
                   help='사람 감지 신뢰도 임계값 0.0-1.0 (기본: 0.3)')

    # 성능
    g = parser.add_argument_group('성능')
    g.add_argument('--fps', type=int, default=30,
                   help='목표 FPS (기본: 30)')
    g.add_argument('--frame-skip', type=int, default=3,
                   help='AI 추론을 매 N프레임마다 실행 (기본: 3, 권장: 2-5)')
    g.add_argument('--display', action='store_true',
                   help='화면 표시 활성화')

    # MQTT
    g = parser.add_argument_group('MQTT 출력')
    g.add_argument('--mqtt-broker', default='localhost',
                   help='MQTT 브로커 호스트 (기본: localhost)')
    g.add_argument('--mqtt-port', type=int, default=1883,
                   help='MQTT 브로커 포트 (기본: 1883)')
    g.add_argument('--mqtt-topic-prefix', default='cctv/ai/events',
                   help='MQTT 이벤트 토픽 prefix (기본: cctv/ai/events)')

    # 이벤트
    g = parser.add_argument_group('이벤트 설정')
    g.add_argument('--no-debounce', action='store_true',
                   help='이벤트 디바운싱 비활성화')
    g.add_argument('--debounce', type=float, default=3.0,
                   help='디바운싱 간격(초) (기본: 3.0)')

    # 위험 구역
    g = parser.add_argument_group('위험 구역 탐지')
    g.add_argument('--zone-detection', action='store_true',
                   help='위험 구역 감지 활성화')
    g.add_argument('--zones-config', default='zones_config.json',
                   help='구역 설정 JSON 파일 경로 (기본: zones_config.json)')

    # 데이터셋
    g = parser.add_argument_group('데이터셋 수집')
    g.add_argument('--collect-dataset', action='store_true',
                   help='탐지 데이터 자동 수집')
    g.add_argument('--dataset-dir', default='./collected_data',
                   help='데이터셋 저장 디렉터리 (기본: ./collected_data)')
    # Zone API
    g = parser.add_argument_group('Zone 설정 API')
    g.add_argument('--api-port', type=int, default=0,
                   help='위험구역 설정 REST API 포트 (기본: 비활성화, 예: 8765)')
    g.add_argument('--zone-presets', default='zone_presets.json', metavar='FILE',
                   help='구역 프리셋 저장 파일 경로 (기본: zone_presets.json)')
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """인자 값 범위 검증"""
    if not (0.0 <= args.confidence <= 1.0):
        parser.error("--confidence 값은 0.0에서 1.0 사이여야 합니다")
    if not (0.0 <= args.pose_confidence <= 1.0):
        parser.error("--pose-confidence 값은 0.0에서 1.0 사이여야 합니다")
    if args.fps <= 0:
        parser.error("--fps 값은 양수여야 합니다")
    if args.mqtt_port <= 0:
        parser.error("--mqtt-port 값은 양수여야 합니다")
    if args.video and not Path(args.video).exists():
        parser.error(f"비디오 파일을 찾을 수 없습니다: {args.video}")


# ──────────────────────────────────────────────────────────────────
# 메인 진입점
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    """메인 진입점"""
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    cfg = apply_args_to_config(args, AppConfig.from_env())

    if not cfg.validate():
        logger.warning("설정 검증 실패. 일부 기능이 작동하지 않을 수 있습니다.")

    logger.info(SEPARATOR)
    logger.info("CCTV 헬멧 감지 시스템")
    logger.info(SEPARATOR)
    logger.info(cfg.summary())
    logger.info(SEPARATOR)

    if args.cameras:
        cams = load_camera_list(args.cameras)
    else:
        source = args.video if args.video else 0
        source_name = 'video' if args.video else 'webcam'
        cams = [{'id': source_name, 'source': source}]
        logger.info("단일 소스 모드: %s (%s)", source_name, source)

    cameras_json_path = args.cameras or 'cameras.json'
    start_processor(cams, cfg, cameras_json_path=cameras_json_path,
                    api_port=args.api_port, zone_presets_path=args.zone_presets)


if __name__ == '__main__':
    main()

