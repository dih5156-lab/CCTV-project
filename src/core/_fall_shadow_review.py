"""Fall shadow review logging helpers for DeepStream.

This module keeps optional falldata review bookkeeping out of the main
DeepStream processor loop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue
from typing import Any, Dict, Iterable, Optional, Tuple

from .events import DetectionEvent, EventType

logger = logging.getLogger(__name__)


@dataclass
class FallShadowReviewConfig:
    review_log_path: Path
    clip_dir: Path
    save_clips: bool = False
    near_miss_enabled: bool = False
    near_miss_cooldown_sec: float = 10.0


class FallShadowReviewRecorder:
    """Writes falldata shadow review records and throttles near-miss logs."""

    def __init__(
        self,
        config: FallShadowReviewConfig,
        *,
        falldata_aux: Any = None,
        near_miss_last_at: Optional[Dict[Tuple[str, int], float]] = None,
    ) -> None:
        self.config = config
        self.falldata_aux = falldata_aux
        self.near_miss_last_at = (
            near_miss_last_at if near_miss_last_at is not None else {}
        )

    def submit_aux_work(
        self,
        queue: Queue,
        camera_name: str,
        filtered_events: Iterable[DetectionEvent],
    ) -> Optional[DetectionEvent]:
        """발행 대기 낙상을 우선 선택하고 접수 시점의 Pose를 보조 큐에 보관한다."""
        if not self.falldata_aux or not self.falldata_aux.enabled:
            return None

        fall_events = [
            event
            for event in filtered_events
            if event.event_type == EventType.FALL_DETECTED
        ]
        fall_event = next(
            (
                event
                for event in fall_events
                if (event.metadata or {}).get("falldata_aux_publish_pending")
            ),
            fall_events[0] if fall_events else None,
        )
        if fall_event is None:
            return None

        payload = fall_event.to_dict()
        if self.falldata_aux.config.inline_pose_rf:
            # 원 경보 메타데이터에는 넣지 않아 외부 발행·저장 형식을 유지한다.
            payload["_falldata_pose_records"] = self.falldata_aux.snapshot_pose_records(
                camera_name, (payload.get("metadata") or {}).get("falldata_pose_scope")
            )
        try:
            queue.put_nowait((camera_name, payload))
            return fall_event
        except Full:
            logger.warning(
                "[%s] falldata shadow 워커 큐 가득 참 - 후보 검증 건너뜀",
                camera_name,
            )
            return None

    def submit_shadow_window(
        self,
        queue: Queue,
        camera_name: str,
    ) -> bool:
        """Queue a shadow-only verification for the current buffered window."""
        if not self.falldata_aux or not self.falldata_aux.enabled:
            return False
        payload = {
            "type": "fall_shadow_window",
            "metadata": {"falldata_aux_shadow_window": True},
        }
        try:
            queue.put_nowait((camera_name, payload))
            return True
        except Full:
            logger.debug("[%s] falldata shadow window 큐 가득 참", camera_name)
            return False

    def submit_near_miss_aux_work(
        self,
        queue: Queue,
        camera_name: str,
        filtered_events: Iterable[DetectionEvent],
        *,
        now_monotonic: float,
    ) -> Optional[DetectionEvent]:
        """Submit one near-miss event for shadow-only temporal review."""
        if not self.falldata_aux or not self.falldata_aux.enabled:
            return None
        for event in filtered_events:
            if event.event_type != EventType.PERSON:
                continue
            if not isinstance((event.metadata or {}).get("fall_near_miss"), dict):
                continue
            object_id = int(event.object_id) if event.object_id is not None else 0
            key = (camera_name, object_id)
            last_at = self.near_miss_last_at.get(key)
            if last_at is not None and now_monotonic - last_at < self.config.near_miss_cooldown_sec:
                continue
            payload = event.to_dict()
            payload["type"] = "fall_near_miss"
            metadata = dict(payload.get("metadata") or {})
            metadata["falldata_aux_near_miss_shadow"] = True
            payload["metadata"] = metadata
            try:
                queue.put_nowait((camera_name, payload))
                return event
            except Full:
                logger.warning("[%s] near-miss temporal shadow 큐 가득 참", camera_name)
                return None
        return None

    def write_near_miss_records(
        self,
        camera_name: str,
        filtered_events: Iterable[DetectionEvent],
        *,
        now_monotonic: float,
    ) -> None:
        """Write throttled review rows for person events carrying near-miss metadata."""
        if not self.config.near_miss_enabled:
            return

        for event in filtered_events:
            if event.event_type != EventType.PERSON:
                continue
            near_miss = (event.metadata or {}).get("fall_near_miss")
            if not isinstance(near_miss, dict):
                continue

            object_id = int(event.object_id) if event.object_id is not None else 0
            key = (camera_name, object_id)
            last_at = self.near_miss_last_at.get(key)
            if (
                last_at is not None
                and now_monotonic - last_at < self.config.near_miss_cooldown_sec
            ):
                continue
            self.near_miss_last_at[key] = now_monotonic

            event_payload = event.to_dict()
            event_payload["type"] = "fall_near_miss"
            result = {
                "status": "not_run",
                "reason": "near_miss_only",
                "confirmed": None,
            }
            self.write_record(camera_name, event_payload, result, near_miss=near_miss)
            logger.info(
                "[%s] fall near-miss shadow record: type=%s object_id=%s review_log=%s",
                camera_name,
                near_miss.get("type"),
                event.object_id,
                self.config.review_log_path,
            )

    def verify_and_write_aux_record(
        self,
        camera_name: str,
        event_payload: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if event_payload.get("type") in {"fall_near_miss", "fall_shadow_window"}:
            result = self.falldata_aux.verify_temporal_shadow()
        else:
            # 빈 스냅샷도 그대로 전달해 접수 후의 Pose로 과거 입력을 대체하지 않는다.
            pose_input = {}
            if "_falldata_pose_records" in event_payload:
                pose_input["pose_records"] = event_payload.pop("_falldata_pose_records")
            result = self.falldata_aux.verify(
                camera_name=camera_name,
                pose_scope=(event_payload.get("metadata") or {}).get(
                    "falldata_pose_scope"
                ),
                **pose_input,
            )
        record = self.write_record(camera_name, event_payload, result)
        return result, record

    def write_record(
        self,
        camera_name: str,
        event_payload: Dict[str, Any],
        result: Dict[str, Any],
        *,
        near_miss: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a JSONL record for later fall-label review."""
        now = datetime.now(timezone.utc)
        event_id = fall_shadow_event_id(camera_name, event_payload, now)
        clip_path: Optional[str] = None
        clip_frames: Optional[int] = None
        temporal_event = event_payload.get("type") in {
            "fall_near_miss",
            "fall_shadow_window",
        }
        temporal_result = result.get("temporal_compare_model") or {}
        temporal_has_pose = int(temporal_result.get("frames_with_pose") or 0) > 0
        should_save_clip = self.config.save_clips and (
            not temporal_event or temporal_has_pose
        )
        if should_save_clip and self.falldata_aux:
            clip_file = self.config.clip_dir / f"{event_id}.mp4"
            try:
                if temporal_event:
                    clip_frames = self.falldata_aux.save_last_verified_clip(clip_file)
                else:
                    clip_frames = self.falldata_aux.save_buffered_clip(clip_file)
                if clip_frames:
                    clip_path = str(clip_file)
            except Exception as exc:
                logger.warning("[%s] fall shadow clip 저장 실패: %s", camera_name, exc)
        elif temporal_event and self.falldata_aux:
            self.falldata_aux.discard_last_verified_clip()

        record: Dict[str, Any] = {
            "event_id": event_id,
            "created_at": now.isoformat(),
            "camera_id": camera_name,
            "event_type": event_payload.get("type"),
            "object_id": event_payload.get("object_id"),
            "bbox": event_payload.get("bbox"),
            "confidence": event_payload.get("confidence"),
            "fall_score": (event_payload.get("metadata") or {}).get("fall_score"),
            "fall_reasons": (event_payload.get("metadata") or {}).get("fall_reasons"),
            "falldata_aux_publish_pending": bool(
                (event_payload.get("metadata") or {}).get("falldata_aux_publish_pending")
            ),
            "review_source": "fall_near_miss" if near_miss is not None else "falldata_aux",
            "falldata_aux": result,
            "clip_path": clip_path,
            "clip_frames": clip_frames,
            "label": None,
            "review_status": "unlabeled",
            "note": "",
        }
        if near_miss is not None:
            record["near_miss"] = near_miss

        self.config.review_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.review_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record


def fall_shadow_event_id(
    camera_name: str,
    event_payload: Dict[str, Any],
    created_at: datetime,
) -> str:
    safe_camera = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(camera_name or "camera")
    )
    object_id = event_payload.get("object_id")
    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_camera}_{stamp}_obj{object_id if object_id is not None else 'unknown'}"
