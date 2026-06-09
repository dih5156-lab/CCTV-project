"""base_processor.py — 프로세서 추상 기반 클래스.

VideoProcessor (현재 파이프라인) 와 DeepStreamProcessor (Jetson 전용) 가
동일한 공개 인터페이스를 구현하도록 강제하는 ABC.

공개 인터페이스:
    add_camera()            — 카메라 추가
    remove_camera()         — 카메라 제거
    update_zones()          — 구역 설정 갱신
    enqueue_camera_retry()  — 카메라 재연결 예약
    start()                 — 파이프라인 시작
    stop()                  — 파이프라인 중지
    start_display_loop()    — 디스플레이 루프 (선택적 오버라이드)
    set_zone_drawer()       — ZoneDrawer 연결 (선택적 오버라이드)
    get_stats()             — 통계 반환
    get_camera_status()     — 카메라 상태 반환
    cameras (property)      — 카메라 맵

필수 확장 인터페이스:
    get_camera_frame()      — 최신 카메라 프레임 반환
    get_detection_snapshot()— 최신 탐지 결과 반환

선택 확장 인터페이스 (기본 구현: NotImplementedError 또는 빈 반환):
    update_camera_model_settings()
    get_camera_model_settings()
    list_registered_faces()
    register_face()
    delete_face()
    reload_face_gallery()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..config import AppConfig
from ..utils.zone_drawer import ZoneDrawer


class BaseProcessor(ABC):
    """모든 프로세서 구현체가 따라야 하는 추상 기반 클래스."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # 필수 구현 메서드
    # ------------------------------------------------------------------

    @abstractmethod
    def add_camera(
        self,
        camera_id: str,
        source: Union[str, int],
        *,
        detections: Optional[List[str]] = None,
        model_paths: Optional[Dict[str, str]] = None,
        zones_data: Optional[List[Dict]] = None,
    ) -> bool:
        """처리 파이프라인에 카메라를 추가하고 성공 여부를 반환한다."""

    @abstractmethod
    def remove_camera(self, camera_id: str) -> None:
        """처리 파이프라인에서 카메라를 제거한다."""

    @abstractmethod
    def enqueue_camera_retry(
        self,
        camera_id: str,
        source: Union[str, int],
        delay_seconds: float = 30.0,
    ) -> None:
        """카메라 재연결을 지연 후 예약한다."""

    @abstractmethod
    def start(self) -> None:
        """파이프라인을 시작한다 (블로킹)."""

    @abstractmethod
    def stop(self) -> None:
        """파이프라인을 중지한다."""

    @abstractmethod
    def get_stats(self) -> Dict:
        """처리 통계를 딕셔너리로 반환한다."""

    @abstractmethod
    def get_camera_status(self) -> Dict[str, dict]:
        """카메라별 상태를 딕셔너리로 반환한다."""

    @property
    @abstractmethod
    def cameras(self) -> Dict:
        """현재 등록된 카메라 맵을 반환한다."""

    @abstractmethod
    def get_camera_frame(
        self, camera_id: str, *, annotated: bool = False, copy_frame: bool = True
    ) -> Optional[Any]:
        """특정 카메라의 최신 프레임을 반환한다."""

    @abstractmethod
    def get_detection_snapshot(self) -> Dict[str, dict]:
        """카메라별 최신 탐지 스냅샷을 반환한다."""

    # ------------------------------------------------------------------
    # 선택 오버라이드 메서드 (기본 구현 제공)
    # ------------------------------------------------------------------

    def update_zones(
        self,
        camera_id: str,
        zones_data: Optional[List[Dict]],
        cameras_json_path: str = "cameras.json",
    ) -> bool:
        """구역 설정을 갱신한다. 지원하지 않는 구현체는 False 반환."""
        return False

    def start_display_loop(self) -> None:
        """디스플레이 루프를 시작한다. GUI가 없으면 아무것도 하지 않는다."""

    def set_zone_drawer(self, drawer: ZoneDrawer) -> None:
        """ZoneDrawer를 연결한다. GUI가 없으면 아무것도 하지 않는다."""

    def release_all_cameras(self) -> None:
        """등록된 모든 카메라 리소스를 해제한다."""
        pass

    def print_stats(self) -> None:
        """처리 통계를 콘솔/로그에 출력한다."""
        pass

    # ------------------------------------------------------------------
    # 얼굴 인식 확장 (옵션)
    # ------------------------------------------------------------------

    def list_registered_faces(self) -> List[Dict[str, str]]:
        return []

    def register_face(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        raise NotImplementedError(f"{type(self).__name__}은 얼굴 인식을 지원하지 않습니다.")

    def delete_face(self, face_id: str) -> bool:
        raise NotImplementedError(f"{type(self).__name__}은 얼굴 인식을 지원하지 않습니다.")

    def reload_face_gallery(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 카메라별 모델 설정 확장 (옵션)
    # ------------------------------------------------------------------

    def get_camera_model_settings(self, camera_id: str) -> Optional[Dict[str, bool]]:
        return None

    def update_camera_model_settings(
        self,
        camera_id: str,
        model_settings: Dict,
        cameras_json_path: str = "cameras.json",
    ) -> bool:
        return False

    # ------------------------------------------------------------------
    # 공통 상태 응답 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _build_camera_status_entry(
        *,
        connected: bool,
        source: Optional[Union[str, int]] = None,
        reconnect_attempts: int = 0,
        last_frame_time: Optional[float] = None,
        status: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """프로세서 구현체 공통 카메라 상태 payload를 생성한다."""
        if status is None:
            if connected:
                status = "online"
            elif reconnect_attempts > 0:
                status = "reconnecting"
            else:
                status = "offline"

        now = time.time()
        payload: Dict[str, Any] = {
            "status": status,
            "connected": connected,
            "source": source,
            "reconnect_attempts": reconnect_attempts,
            "last_frame_time": last_frame_time,
            "last_frame_age_sec": round(now - last_frame_time, 1)
            if last_frame_time
            else None,
        }
        payload.update(extra)
        return payload

    @staticmethod
    def _build_stats_payload(
        *,
        backend: str,
        camera_count: int,
        frames_processed: int = 0,
        frames_dropped: int = 0,
        events_detected: int = 0,
        events_sent: int = 0,
        events_filtered: int = 0,
        events_dropped: int = 0,
        events_failed: int = 0,
        inference_errors: int = 0,
        camera_errors: int = 0,
        fps: float = 0.0,
        uptime_seconds: float = 0.0,
        avg_inference_ms: float = 0.0,
        **extra: Any,
    ) -> Dict[str, Any]:
        """프로세서 구현체 공통 통계 payload를 생성한다."""
        payload: Dict[str, Any] = {
            "backend": backend,
            "camera_count": camera_count,
            "frames_processed": frames_processed,
            "frames_dropped": frames_dropped,
            "events_detected": events_detected,
            "events_sent": events_sent,
            "events_filtered": events_filtered,
            "events_dropped": events_dropped,
            "events_failed": events_failed,
            "inference_errors": inference_errors,
            "camera_errors": camera_errors,
            "fps": fps,
            "uptime_seconds": uptime_seconds,
            "avg_inference_ms": avg_inference_ms,
        }
        payload.update(extra)
        return payload
