"""구역 이벤트를 디스플레이용 이벤트 타입으로 변환하는 헬퍼."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

from ..utils.zone_detection import ZoneEvent, ZoneManager
from .events import DetectionEvent, EventType


class DisplayEventMapper:
    """구역 상태를 반영해 화면 표시용 이벤트를 만든다."""

    def __init__(self, zone_in_objects: Dict[str, set]) -> None:
        self._zone_in_objects = zone_in_objects

    def build(
        self,
        camera_id: str,
        events_for_display: List[DetectionEvent],
        zone_events: List[ZoneEvent],
        removed_ids: List[int],
        zone_manager: Optional[ZoneManager],
    ) -> List[DetectionEvent]:
        zone_set = self._zone_in_objects.setdefault(camera_id, set())
        zone_set_modes: Dict[int, str] = {}
        has_active_zones = bool(zone_manager and zone_manager.zones.get(camera_id))

        if not has_active_zones:
            zone_set.clear()
        else:
            for zone_event in zone_events:
                event_value = zone_event.event_type.value
                if event_value in ("zone_entered", "zone_dwelling", "zone_object_detected"):
                    zone_set.add(zone_event.object_id)
                    mode_value = (zone_event.metadata or {}).get("mode", "danger")
                    zone_set_modes[zone_event.object_id] = (
                        "zone_object" if mode_value == "object_watch" else "danger"
                    )
                elif event_value == "zone_exited":
                    zone_set.discard(zone_event.object_id)

        for removed_id in removed_ids:
            zone_set.discard(removed_id)

        display_events: List[DetectionEvent] = []
        for event in events_for_display:
            object_id = getattr(event, "object_id", None)
            if event.event_type != EventType.PERSON or object_id not in zone_set:
                display_events.append(event)
                continue

            mapped_type = (
                EventType.ZONE_OBJECT
                if zone_set_modes.get(object_id) == "zone_object"
                else EventType.DANGER_ZONE
            )
            display_events.append(replace(event, event_type=mapped_type))

        return display_events
