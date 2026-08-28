"""애플리케이션 실행 환경 초기화와 프로세서 런타임 제어."""

from __future__ import annotations

import atexit
import json
import logging
import logging.handlers as log_handlers
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from ..config import AppConfig
from ..core import VideoProcessor
from ..core.base_processor import BaseProcessor
from ..services.camera_model_api import start_camera_model_api_server
from ..services.face_api import start_face_api_server
from ..services.processor_metrics import start_processor_metrics_server
from ..services.stream_api import start_stream_api_server
from ..services.zone_api import start_zone_api_server
from ..utils.env import get_env_bool, get_env_int, load_dotenv_file
from ..utils.zone_drawer import ZoneDrawer

logger = logging.getLogger(__name__)


VIDEO_FILE_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}


def _resolve_video_file_source(source: str) -> Path | None:
    """검증 가능한 로컬 비디오 파일 source를 Path로 변환한다."""
    if source.startswith("file://"):
        parsed = urlparse(source)
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            return None
        path = Path(unquote(parsed.path))
        return path if path.suffix.lower() in VIDEO_FILE_SUFFIXES else None

    path = Path(source)
    return path if path.suffix.lower() in VIDEO_FILE_SUFFIXES else None


def configure_runtime_environment() -> None:
    """OpenCV/콘솔/Jetson 실행 환경을 초기화한다."""
    load_dotenv_file()
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    if sys.platform == "win32":
        os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()
            if "orin" in device_name or "tegra" in device_name or "nvgpu" in device_name:
                torch.backends.cudnn.enabled = False
                logging.getLogger(__name__).info(
                    "Jetson Orin 통합 GPU 감지 → cuDNN 비활성화 (TensorRT .engine 사용 권장)"
                )
    except Exception:
        pass

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(log_dir: str = "logs") -> None:
    """루트 로거에 콘솔과 파일 핸들러를 등록한다."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = log_handlers.RotatingFileHandler(
        log_path / "cctv.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def load_camera_list(path: str) -> list[dict]:
    """카메라 설정 JSON 파일을 검증하면서 읽어 온다."""
    file_path = Path(path)
    if not file_path.exists():
        logger.error("카메라 설정 파일을 찾을 수 없습니다: %s", path)
        return []
    if file_path.stat().st_size == 0:
        logger.error("카메라 설정 파일이 비어있습니다: %s", path)
        return []

    try:
        cameras = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("%s JSON 파싱 오류: %s", path, exc)
        return []

    if not isinstance(cameras, list):
        logger.error("잘못된 카메라 설정 형식 (리스트 필요): %s", path)
        return []

    valid_cameras = []
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            logger.warning("인덱스 %d의 카메라 항목 건너뜀 (딕셔너리가 아님)", index)
            continue
        if "id" not in camera or "source" not in camera:
            logger.warning("인덱스 %d의 카메라 건너뜀 ('id' 또는 'source' 누락)", index)
            continue

        source = camera["source"]
        camera_id = camera["id"]
        if isinstance(source, str):
            if source.isdigit():
                pass
            elif source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                pass
            elif video_file_path := _resolve_video_file_source(source):
                if not video_file_path.exists():
                    logger.warning("[%s] 비디오 파일을 찾을 수 없습니다: %s — 건너뜀", camera_id, source)
                    continue
            else:
                logger.warning("[%s] 알 수 없는 source 형식: %s — 그대로 사용", camera_id, source)
        elif not isinstance(source, int):
            logger.warning(
                "[%s] source는 문자열 또는 정수여야 합니다 (받은 타입: %s) — 건너뜀",
                camera_id,
                type(source).__name__,
            )
            continue

        valid_cameras.append(camera)

    logger.info("%s에서 %d개 카메라 로드됨", path, len(valid_cameras))
    return valid_cameras


def build_single_source_camera(video_path: Optional[str]) -> list[dict]:
    """단일 소스 실행용 카메라 목록을 만든다."""
    source = video_path if video_path else 0
    source_name = "video" if video_path else "webcam"
    logger.info("단일 소스 모드: %s (%s)", source_name, source)
    return [{"id": source_name, "source": source}]


def collect_active_cameras(
    camera_list: list[dict], processor: BaseProcessor
) -> list[tuple]:
    """활성화된 카메라만 실행용 튜플로 정규화한다."""
    active = []
    for camera in camera_list:
        if not camera.get("enabled", True):
            logger.info("카메라 비활성화됨: %s (%s)", camera.get("id"), camera.get("name", "N/A"))
            continue
        camera_id = camera.get("id")
        source = camera.get("source")
        if not camera_id or source is None:
            logger.warning("id 또는 source 누락 - 건너뜀: %s", camera)
            continue
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        active.append(
            (
                camera_id,
                source,
                camera.get("model_settings") or camera.get("detections") or camera.get("ai_models"),
                camera.get("model_paths") or None,
                camera.get("zones") or None,
            )
        )
    return active


def connect_cameras_parallel(
    active_cameras: list[tuple], processor: BaseProcessor
) -> dict[str, bool]:
    """카메라 연결을 병렬로 시도한다."""
    results: dict[str, bool] = {}
    lock = threading.Lock()

    def _try_add(camera_id: str, source, detections, model_paths, zones_data) -> None:
        ok = processor.add_camera(
            camera_id,
            source,
            detections=detections,
            model_paths=model_paths,
            zones_data=zones_data,
        )
        with lock:
            results[camera_id] = ok

    connect_timeout = 30
    threads = [
        threading.Thread(target=_try_add, args=(cid, src, det, paths, zones), daemon=True)
        for cid, src, det, paths, zones in active_cameras
    ]
    for thread in threads:
        thread.start()

    try:
        for thread in threads:
            thread.join(timeout=connect_timeout)
            if thread.is_alive():
                camera_id = active_cameras[threads.index(thread)][0]
                logger.error("[%s] 카메라 연결이 %d초 초과 → 건너뜀", camera_id, connect_timeout)
                with lock:
                    results.setdefault(camera_id, False)
    except KeyboardInterrupt:
        logger.info("카메라 연결 중단 (Ctrl+C)")
        raise

    return results


def initial_retry(
    active_cameras: list[tuple], processor: BaseProcessor, max_attempts: int = 3
) -> int:
    """최초 연결 실패 시 블로킹 재시도를 수행한다."""
    for attempt in range(1, max_attempts + 1):
        logger.info("초기 연결 재시도 %d/%d (30초 대기)...", attempt, max_attempts)
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("초기 재시도 중단 (Ctrl+C)")
            raise

        for camera_id, source, detections, model_paths, zones_data in active_cameras:
            if camera_id not in processor.cameras and processor.add_camera(
                camera_id,
                source,
                detections=detections,
                model_paths=model_paths,
                zones_data=zones_data,
            ):
                logger.info("재시도 성공: %s", camera_id)
                return 1
    return 0


def create_processor(cfg: AppConfig) -> BaseProcessor:
    """환경 변수에 따라 적합한 프로세서 인스턴스를 생성해 반환한다.

    USE_DEEPSTREAM=1 로 설정하면 DeepStreamProcessor 를 사용한다.
    DeepStream 라이브러리가 없거나 USE_DEEPSTREAM 이 설정되지 않으면
    VideoProcessor 를 사용한다.
    """
    if os.environ.get("USE_DEEPSTREAM", "0") == "1":
        try:
            from ..core.deepstream_processor import (
                DEEPSTREAM_AVAILABLE,
                DeepStreamProcessor,
            )

            if not DEEPSTREAM_AVAILABLE:
                logger.warning(
                    "USE_DEEPSTREAM=1 이지만 DeepStream 라이브러리를 찾을 수 없습니다. "
                    "VideoProcessor 로 폴백합니다."
                )
                return VideoProcessor(cfg)

            logger.info("DeepStreamProcessor 모드로 시작합니다.")
            return DeepStreamProcessor(cfg)
        except Exception as exc:
            logger.error(
                "DeepStreamProcessor 초기화 실패 (%s) — VideoProcessor 로 폴백합니다.", exc
            )
            return VideoProcessor(cfg)

    logger.info("VideoProcessor 모드로 시작합니다.")
    return VideoProcessor(cfg)


def start_processor_runtime(
    camera_list: list[dict],
    cfg: AppConfig,
    cameras_json_path: str = "cameras.json",
    api_port: int = 0,
    zone_presets_path: str = "zone_presets.json",
    processor_refs: Optional[list] = None,
) -> None:
    """프로세서를 생성하고 카메라 연결부터 실행 루프까지 담당한다."""
    processor = create_processor(cfg)
    if processor_refs is not None:
        processor_refs.append(processor)

    if api_port > 0:
        start_zone_api_server(processor, cameras_json_path, api_port, presets_path=zone_presets_path)
        start_camera_model_api_server(processor, cameras_json_path, api_port + 1)
        start_face_api_server(processor, api_port + 2)

    stream_port = get_env_int("STREAM_PORT", 0, minimum=0, maximum=65535, logger=logger)
    stream_api_enabled = get_env_bool("STREAM_API_ENABLED", False)
    if stream_port > 0 or stream_api_enabled:
        start_stream_api_server(processor, stream_port or (api_port + 3 if api_port > 0 else 8769))
    elif api_port > 0:
        start_stream_api_server(processor, api_port + 3)

    metrics_port = get_env_int("METRICS_PORT", 0, minimum=0, maximum=65535, logger=logger)
    if metrics_port > 0:
        start_processor_metrics_server(processor, metrics_port)

    def _keep_api_runtime_alive(reason: str) -> None:
        logger.error("%s API 서버는 유지하고 추론 파이프라인은 시작하지 않습니다.", reason)
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("사용자가 중단함 (Ctrl+C)")

    if not camera_list:
        _keep_api_runtime_alive("카메라가 제공되지 않았습니다.")
        return

    active_cameras = collect_active_cameras(camera_list, processor)
    if not active_cameras:
        _keep_api_runtime_alive("활성화된 카메라가 없습니다.")
        return

    results = connect_cameras_parallel(active_cameras, processor)

    added_count = 0
    for camera_id, source, _det, _paths, _zones in active_cameras:
        if results.get(camera_id):
            added_count += 1
            logger.info("카메라 추가 성공: %s (%s)", camera_id, source)
        else:
            logger.warning("카메라 연결 실패: %s (%s) → 백그라운드 재시도 예약", camera_id, source)
            processor.enqueue_camera_retry(camera_id, source, delay_seconds=30)

    if added_count == 0:
        logger.warning("현재 연결된 카메라가 없습니다. 초기 재연결을 시도합니다.")
        added_count += initial_retry(active_cameras, processor)

    if added_count == 0:
        _keep_api_runtime_alive("카메라 연결에 최종 실패했습니다.")
        return

    logger.info("%d개 카메라로 프로세서 시작 중...", added_count)

    if cfg.display:
        drawer = ZoneDrawer(processor, cameras_json_path)
        processor.set_zone_drawer(drawer)
        logger.info("구역 그리기 모드 사용 가능: 디스플레이 창에서 'd' 키를 누르세요")

    restart_on_pipeline_error = get_env_bool("AI_ENGINE_RESTART_ON_PIPELINE_ERROR", True)
    restart_delay_sec = get_env_int(
        "AI_ENGINE_RESTART_DELAY_SEC",
        10,
        minimum=1,
        maximum=3600,
        logger=logger,
    )

    try:
        while True:
            next_restart_delay_sec = float(restart_delay_sec)
            try:
                processor.start()
                logger.info("프로세서가 시작되었습니다. 중지하려면 Ctrl+C를 누르세요.")

                if cfg.display:
                    logger.info("디스플레이 루프 시작 (메인 스레드)")
                    processor.start_display_loop()
                else:
                    last_stats = time.time()
                    while processor.running and not processor.stop_event.is_set():
                        time.sleep(0.5)
                        if time.time() - last_stats >= 10:
                            processor.print_stats()
                            last_stats = time.time()

                if processor.stop_event.is_set() and processor.running:
                    logger.warning("프로세서 중지 이벤트 감지 → 파이프라인 재시작 준비")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                source_retry_delay = (
                    processor.next_source_retry_delay()
                    if hasattr(processor, "next_source_retry_delay")
                    else None
                )
                if source_retry_delay is not None:
                    next_restart_delay_sec = max(float(restart_delay_sec), float(source_retry_delay))
                    logger.warning(
                        "활성 DeepStream 소스가 없습니다. %.0f초 후 소스 backoff 종료에 맞춰 재시도합니다.",
                        next_restart_delay_sec,
                    )
                else:
                    logger.error("처리 중 오류 발생: %s", exc)
                    traceback.print_exc()
            finally:
                logger.info("프로세서 중지 중...")
                processor.stop()
                logger.info("프로세서가 중지되었습니다.")

            if cfg.display or not restart_on_pipeline_error:
                break

            source_retry_delay = (
                processor.next_source_retry_delay()
                if hasattr(processor, "next_source_retry_delay")
                else None
            )
            if source_retry_delay is not None:
                next_restart_delay_sec = max(float(restart_delay_sec), float(source_retry_delay))

            logger.warning(
                "AI 엔진 프로세스는 유지하고 %.0f초 후 파이프라인만 재시작합니다.",
                next_restart_delay_sec,
            )
            time.sleep(next_restart_delay_sec)
    except KeyboardInterrupt:
        logger.info("사용자가 중단함 (Ctrl+C)")


def register_shutdown_handlers(processor_refs: list) -> None:
    """종료 시 카메라와 OpenCV 창을 정리하는 핸들러를 등록한다."""

    def _release_all() -> None:
        import cv2

        if processor_refs:
            try:
                processor_refs[0].release_all_cameras()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def _sig_handler(signum, frame) -> None:
        logger.info("시그널 %s 수신 → 종료", signum)
        _release_all()
        sys.exit(0)

    atexit.register(_release_all)
    signal.signal(signal.SIGTERM, _sig_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGHUP, _sig_handler)
