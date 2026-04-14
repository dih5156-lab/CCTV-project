"""CLI 인자 파싱과 AppConfig 반영."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    """메인 CLI 파서를 생성한다."""
    parser = argparse.ArgumentParser(
        description="CCTV 헬멧 착용 및 낙상 감지 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --display
  python main.py --video sample.mp4 --display
  python main.py --cameras cameras.json
  python main.py --device cuda --collect-dataset --dataset-dir ./my_data
        """,
    )

    group = parser.add_argument_group("입력 소스")
    group.add_argument("--cameras", "-c", default=None, help="카메라 목록 JSON 파일 경로")
    group.add_argument("--video", default=None, help="비디오 파일 경로")

    group = parser.add_argument_group("모델 설정")
    group.add_argument("--helmet-model", default=None, help="헬멧 감지 모델 경로")
    group.add_argument("--pose-model", default=None, help="Pose 모델 경로")
    group.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="실행 디바이스")
    group.add_argument("--confidence", type=float, default=0.5, help="헬멧 감지 신뢰도 임계값")
    group.add_argument("--pose-confidence", type=float, default=0.3, help="사람 감지 신뢰도 임계값")

    group = parser.add_argument_group("성능")
    group.add_argument("--fps", type=int, default=30, help="목표 FPS")
    group.add_argument("--frame-skip", type=int, default=3, help="AI 추론 프레임 스킵 간격")
    group.add_argument("--display", action="store_true", help="화면 표시 활성화")

    group = parser.add_argument_group("MQTT 출력")
    group.add_argument("--mqtt-broker", default="localhost", help="MQTT 브로커 호스트")
    group.add_argument("--mqtt-port", type=int, default=1883, help="MQTT 브로커 포트")
    group.add_argument("--mqtt-topic-prefix", default="cctv/ai/events", help="MQTT 이벤트 토픽 prefix")

    group = parser.add_argument_group("이벤트 설정")
    group.add_argument("--no-debounce", action="store_true", help="이벤트 디바운싱 비활성화")
    group.add_argument("--debounce", type=float, default=3.0, help="디바운싱 간격(초)")

    group = parser.add_argument_group("위험 구역 탐지")
    group.add_argument("--zone-detection", action="store_true", help="위험 구역 감지 활성화")
    group.add_argument("--zones-config", default="zones_config.json", help="구역 설정 JSON 파일 경로")

    group = parser.add_argument_group("데이터셋 수집")
    group.add_argument("--collect-dataset", action="store_true", help="탐지 데이터 자동 수집")
    group.add_argument("--dataset-dir", default="./collected_data", help="데이터셋 저장 디렉터리")

    group = parser.add_argument_group("Zone 설정 API")
    group.add_argument("--api-port", type=int, default=0, help="위험구역 설정 REST API 포트")
    group.add_argument("--zone-presets", default="zone_presets.json", metavar="FILE", help="구역 프리셋 저장 파일")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """CLI 인자 범위를 검증한다."""
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


def apply_args_to_config(args: argparse.Namespace, config: AppConfig) -> AppConfig:
    """명령줄 인자를 AppConfig에 반영한다."""
    if args.helmet_model:
        config.models.helmet_model = args.helmet_model
    if args.pose_model:
        config.models.pose_model = args.pose_model

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
    config.mqtt.broker = args.mqtt_broker
    config.mqtt.port = args.mqtt_port
    config.mqtt.topic_prefix = args.mqtt_topic_prefix
    return config
