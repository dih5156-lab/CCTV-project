"""DeepStream 카메라 소스 backoff/오류 상태 헬퍼."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def build_source_entries(
    *,
    cameras: Dict[str, Dict],
    source_backoff_until: Dict[str, float],
    now: float,
    normalize_uri: Callable[[Union[str, int]], str],
) -> List[Tuple[int, str, Dict, str]]:
    """카메라 설정을 DeepStream source entry 목록으로 변환한다."""
    source_entries: List[Tuple[int, str, Dict, str]] = []
    for camera_id, info in cameras.items():
        backoff_until = source_backoff_until.get(camera_id)
        if backoff_until is not None:
            if now < backoff_until:
                remaining = backoff_until - now
                logger.warning(
                    "[%s] DeepStream 소스 backoff 중: %.1f초 후 재시도",
                    camera_id,
                    remaining,
                )
                continue
            source_backoff_until.pop(camera_id, None)
            logger.info("[%s] DeepStream 소스 backoff 종료 -> 파이프라인 포함", camera_id)

        try:
            source_uri = normalize_uri(info["source"])
        except ValueError as exc:
            logger.warning("[%s] DeepStream 소스 제외: %s", camera_id, exc)
            continue
        source_entries.append((len(source_entries), camera_id, info, source_uri))
    return source_entries


def mark_source_failed(
    *,
    cameras: Dict[str, Dict],
    source_backoff_until: Dict[str, float],
    source_last_error: Dict[str, str],
    source_failure_backoff_sec: float,
    camera_id: str,
    reason: str,
    now: float,
) -> None:
    """문제 소스를 일정 시간 제외해 다른 카메라 파이프라인을 보호한다."""
    if camera_id not in cameras:
        return
    source_last_error[camera_id] = reason
    source_backoff_until[camera_id] = now + max(1.0, source_failure_backoff_sec)
    info = cameras[camera_id]
    info["reconnect_attempts"] = int(info.get("reconnect_attempts", 0) or 0) + 1
    logger.warning(
        "[%s] DeepStream 소스 오류로 %.0f초 backoff 적용: %s",
        camera_id,
        source_failure_backoff_sec,
        reason,
    )


def next_source_retry_delay(
    source_backoff_until: Dict[str, float],
    *,
    now: float,
) -> Optional[float]:
    """backoff 중인 소스가 있으면 가장 가까운 재시도까지 남은 초를 반환한다."""
    remaining = [
        backoff_until - now
        for backoff_until in source_backoff_until.values()
        if backoff_until > now
    ]
    if not remaining:
        return None
    return max(1.0, min(remaining))


def camera_id_from_message(
    *,
    cameras: Dict[str, Dict],
    message: Any,
    debug: object,
) -> Optional[str]:
    """GStreamer error message에서 src-<camera_id> element명을 추출한다."""
    candidates = []
    try:
        src = getattr(message, "src", None)
        if src is not None and hasattr(src, "get_name"):
            candidates.append(str(src.get_name()))
    except Exception:
        pass
    candidates.append(str(debug or ""))

    for text in candidates:
        for camera_id in cameras:
            if f"src-{camera_id}" in text:
                return camera_id
    return None


def should_request_pipeline_restart(
    *,
    running: bool,
    restart_pending: bool,
    now: float,
    last_restart_at: float,
    min_interval_sec: float,
    reason: str,
) -> bool:
    """DeepStream 파이프라인 재시작 요청을 지금 수락해도 되는지 판단한다."""
    if not running:
        logger.debug("DeepStream 재시작 요청 무시 (running=False): %s", reason)
        return False

    if restart_pending:
        logger.info("DeepStream 재시작 이미 예약됨 — 요청 무시: %s", reason)
        return False

    elapsed = now - last_restart_at
    if elapsed < min_interval_sec:
        logger.warning(
            "DeepStream 재시작 요청 쿨다운 중(%.1fs 남음): %s",
            min_interval_sec - elapsed,
            reason,
        )
        return False

    return True


def mark_restart_pending_if_allowed(
    *,
    running: bool,
    restart_pending: bool,
    now: float,
    last_restart_at: float,
    min_interval_sec: float,
    reason: str,
    set_pending_cb: Callable[[bool], None],
) -> bool:
    """재시작 요청 허용 시 pending 상태를 True로 전이한다."""
    if not should_request_pipeline_restart(
        running=running,
        restart_pending=restart_pending,
        now=now,
        last_restart_at=last_restart_at,
        min_interval_sec=min_interval_sec,
        reason=reason,
    ):
        return False

    set_pending_cb(True)
    return True


def start_pipeline_restart_thread(
    *,
    pending: bool,
    reason: str,
    request_pipeline_restart_cb: Callable[[str], bool],
    restart_pipeline_cb: Callable[[str], None],
) -> None:
    """재시작 예약 상태를 확인하고 비동기 재시작 스레드를 시작한다."""
    if not pending:
        logger.debug(
            "_restart_pipeline_async 직접 호출 감지 — 재요청 게이트로 위임: %s",
            reason,
        )
        request_pipeline_restart_cb(reason)
        return

    thread = threading.Thread(
        target=restart_pipeline_cb,
        args=(reason,),
        daemon=True,
        name="ds-pipeline-restart",
    )
    thread.start()


def execute_pipeline_restart(
    *,
    reason: str,
    stop_cb: Callable[[], None],
    start_cb: Callable[[], None],
    monotonic_now: Callable[[], float],
    set_last_restart_at_cb: Callable[[float], None],
) -> None:
    """파이프라인 stop/start 재시작을 수행하고 결과를 로깅한다."""
    logger.info("DeepStream 파이프라인 재시작 시작: %s", reason)
    try:
        stop_cb()
        start_cb()
        set_last_restart_at_cb(monotonic_now())
        logger.info("DeepStream 파이프라인 재시작 완료: %s", reason)
    except Exception as exc:
        logger.exception("DeepStream 파이프라인 재시작 실패(%s): %s", reason, exc)


def build_camera_status_map(
    *,
    cameras: Dict[str, Dict],
    running: bool,
    source_backoff_until: Dict[str, float],
    source_last_error: Dict[str, str],
    preview_last_frame_at: Optional[float],
    now_monotonic: float,
    build_status_entry: Callable[..., Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """카메라별 연결/재시도/오류 상태 맵을 생성한다."""
    return {
        camera_id: build_status_entry(
            connected=running and camera_id not in source_backoff_until,
            source=info.get("source"),
            reconnect_attempts=int(info.get("reconnect_attempts", 0) or 0),
            last_frame_time=preview_last_frame_at,
            status="backoff" if camera_id in source_backoff_until else None,
            pad_id=info.get("pad_id"),
            source_backoff_remaining_sec=round(
                max(0.0, source_backoff_until.get(camera_id, 0.0) - now_monotonic),
                1,
            )
            if camera_id in source_backoff_until
            else None,
            last_error=source_last_error.get(camera_id),
        )
        for camera_id, info in cameras.items()
    }


def build_deepstream_stats_fields(
    *,
    cameras_count: int,
    frames_processed: int,
    frames_dropped: int,
    events_detected: int,
    events_sent: int,
    events_dropped: int,
    events_filtered: int,
    events_failed: int,
    output_mode: str,
    preview_enabled: bool,
    preview_max_fps: float,
    preview_ready: bool,
) -> Dict[str, Any]:
    """DeepStream get_stats용 공통 payload 필드를 생성한다."""
    return {
        "camera_count": cameras_count,
        "frames_processed": frames_processed,
        "frames_dropped": frames_dropped,
        "events_detected": events_detected,
        "events_sent": events_sent,
        "events_dropped": events_dropped,
        "events_filtered": events_filtered,
        "events_failed": events_failed,
        "output_mode": output_mode,
        "preview_enabled": preview_enabled,
        "preview_max_fps": preview_max_fps,
        "preview_ready": preview_ready,
        "cameras": cameras_count,
    }


def handle_bus_message(
    *,
    cameras: Dict[str, Dict],
    source_backoff_until: Dict[str, float],
    source_last_error: Dict[str, str],
    source_failure_backoff_sec: float,
    message: Any,
    gst_module: Any,
    monotonic_now: Callable[[], float],
    request_pipeline_restart_cb: Callable[[str], bool],
    stop_runtime_cb: Callable[[], None],
) -> bool:
    """DeepStream bus message를 해석하고 소스 복구/종료 정책을 적용한다."""
    msg_type = message.type

    if msg_type == gst_module.MessageType.EOS:
        camera_id = camera_id_from_message(
            cameras=cameras,
            message=message,
            debug=None,
        )
        if camera_id:
            logger.warning("[%s] DeepStream 소스 EOS 수신", camera_id)
            mark_source_failed(
                cameras=cameras,
                source_backoff_until=source_backoff_until,
                source_last_error=source_last_error,
                source_failure_backoff_sec=source_failure_backoff_sec,
                camera_id=camera_id,
                reason="source_eos",
                now=monotonic_now(),
            )
            request_pipeline_restart_cb(f"source_eos:{camera_id}")
        else:
            logger.warning("DeepStream EOS 수신")
            stop_runtime_cb()
        return True

    if msg_type == gst_module.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("DeepStream 오류: %s debug=%s", err, debug)
        camera_id = camera_id_from_message(
            cameras=cameras,
            message=message,
            debug=debug,
        )
        if camera_id:
            mark_source_failed(
                cameras=cameras,
                source_backoff_until=source_backoff_until,
                source_last_error=source_last_error,
                source_failure_backoff_sec=source_failure_backoff_sec,
                camera_id=camera_id,
                reason=str(err),
                now=monotonic_now(),
            )
            if request_pipeline_restart_cb(f"source_error:{camera_id}"):
                logger.warning("[%s] 소스 오류 감지 - 파이프라인 자동 복구 시도", camera_id)
            return True
        stop_runtime_cb()
        return True

    if msg_type == gst_module.MessageType.WARNING:
        warn, debug = message.parse_warning()
        logger.warning("DeepStream 경고: %s debug=%s", warn, debug)
        return True

    return True
