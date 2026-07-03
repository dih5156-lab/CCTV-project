"""ActionBridge 사이트 설정과 수동 승인 큐."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Dict, List, Optional, Tuple

from ..canonical_event import (
    get_payload_camera_id,
    get_payload_confidence,
    get_payload_display_message,
    get_payload_event_type,
    get_payload_severity,
    get_payload_tts_message,
)
from ..event_priority import event_priority, event_risk_level, event_risk_score
from ..time_utils import now_kst_iso
from ._action_bridge_models import AlarmDevice, ControlMode, SiteConfig

logger = logging.getLogger(__name__)


class _SiteRegistry:
    """사이트 설정과 수동 승인 큐를 관리한다."""

    def __init__(
        self,
        default_mode: ControlMode,
        initial_sites: Optional[List[SiteConfig]] = None,
    ) -> None:
        self.default_mode: ControlMode = default_mode
        self.default_alarm_devices: List[AlarmDevice] = list(AlarmDevice)
        self.default_confidence_threshold: Optional[float] = None
        self.default_display_message: str = ""
        self.default_tts_message: str = ""
        self._sites: Dict[str, SiteConfig] = {
            site.site_id: site for site in (initial_sites or [])
        }
        self._pending: Dict[str, Dict] = {}
        self._pending_lock = Lock()

    def add(self, site: SiteConfig) -> None:
        self._sites[site.site_id] = site
        logger.info(
            "사이트 등록: %s (%s) mode=%s",
            site.site_id,
            site.site_name,
            site.control_mode.value,
        )

    def remove(self, site_id: str) -> bool:
        if site_id in self._sites:
            del self._sites[site_id]
            logger.info("사이트 제거: %s", site_id)
            return True
        return False

    def list_all(self) -> List[Dict]:
        return [site.to_dict() for site in self._sites.values()]

    def find_by_camera(self, camera_id: str) -> Optional[SiteConfig]:
        return next(
            (site for site in self._sites.values() if camera_id in site.camera_ids),
            None,
        )

    def set_mode(self, mode: ControlMode, site_id: Optional[str] = None) -> None:
        if site_id:
            site = self._sites.get(site_id)
            if site:
                site.control_mode = mode
                logger.info("사이트 모드 변경: %s -> %s", site_id, mode.value)
            else:
                logger.warning("set_mode: 사이트 없음 (%s)", site_id)
        else:
            self.default_mode = mode
            logger.info("전역 기본 모드 변경 -> %s", mode.value)

    def default_settings(self) -> Dict:
        return {
            "mode": self.default_mode.value,
            "alarm_devices": [device.value for device in self.default_alarm_devices],
            "confidence_threshold": self.default_confidence_threshold,
            "display_message": self.default_display_message,
            "tts_message": self.default_tts_message,
        }

    def set_default_action_settings(
        self,
        *,
        alarm_devices: Optional[List[AlarmDevice]] = None,
        confidence_threshold: Optional[float] = None,
        display_message: Optional[str] = None,
        tts_message: Optional[str] = None,
    ) -> None:
        if alarm_devices is not None:
            self.default_alarm_devices = alarm_devices
        self.default_confidence_threshold = confidence_threshold
        if display_message is not None:
            self.default_display_message = display_message
        if tts_message is not None:
            self.default_tts_message = tts_message

    def resolve_alarm_devices(self, camera_id: str) -> List[AlarmDevice]:
        site = self.find_by_camera(camera_id)
        return site.alarm_devices if site else self.default_alarm_devices

    def resolve_action_settings(self, camera_id: str) -> Dict:
        site = self.find_by_camera(camera_id)
        if site:
            return {
                "site": site,
                "site_id": site.site_id,
                "confidence_threshold": site.confidence_threshold,
                "display_message": site.display_message,
                "tts_message": site.tts_message,
            }
        return {
            "site": None,
            "site_id": None,
            "confidence_threshold": self.default_confidence_threshold,
            "display_message": self.default_display_message,
            "tts_message": self.default_tts_message,
        }

    def resolve_mode(self, camera_id: str) -> Tuple[ControlMode, Optional[str]]:
        site = self.find_by_camera(camera_id)
        if site:
            return site.control_mode, site.site_id
        return self.default_mode, None

    def push_pending(
        self,
        event_id: str,
        topic: str,
        payload: Dict,
        site_id: Optional[str],
    ) -> None:
        with self._pending_lock:
            self._pending[event_id] = {
                "payload": payload,
                "topic": topic,
                "queued_at": now_kst_iso(),
                "site_id": site_id,
            }
        logger.info(
            "[수동 대기] event_id=%s camera=%s type=%s site=%s",
            event_id,
            payload.get("camera_id"),
            payload.get("type"),
            site_id,
        )

    def pop_pending(self, event_id: str) -> Optional[Dict]:
        with self._pending_lock:
            return self._pending.pop(event_id, None)

    def list_pending(self) -> List[Dict]:
        with self._pending_lock:
            return [
                {
                    "event_id": event_id,
                    "queued_at": info.get("queued_at"),
                    "site_id": info.get("site_id"),
                    "camera_id": get_payload_camera_id(info["payload"]),
                    "event_type": get_payload_event_type(info["payload"]),
                    "confidence": get_payload_confidence(info["payload"]),
                    "severity": get_payload_severity(info["payload"]),
                    "priority": event_priority(info["payload"]),
                    "risk_level": event_risk_level(info["payload"]),
                    "risk_score": event_risk_score(info["payload"]),
                    "display_message": get_payload_display_message(info["payload"]),
                    "tts_message": get_payload_tts_message(info["payload"]),
                    "topic": info.get("topic"),
                }
                for event_id, info in self._pending.items()
            ]
