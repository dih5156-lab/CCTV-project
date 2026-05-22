"""최신 탐지 스냅샷을 스레드 안전하게 보관하는 저장소."""

import time
from threading import Lock
from typing import Dict, List

from .events import DetectionEvent


class DetectionSnapshotStore:
    """카메라별 최신 DetectionEvent 스냅샷을 저장한다."""

    def __init__(self) -> None:
        self._snapshots: Dict[str, dict] = {}
        self._lock = Lock()

    def record(self, camera_id: str, events: List[DetectionEvent]) -> None:
        with self._lock:
            self._snapshots[camera_id] = {
                "timestamp": time.time(),
                "detections": [event.to_dict() for event in events],
            }

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._snapshots)
