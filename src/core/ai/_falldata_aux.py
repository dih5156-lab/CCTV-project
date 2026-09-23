"""Public falldata RF model auxiliary verifier.

The public model stack has incompatible runtime dependencies:

- MediaPipe feature extraction currently needs a numpy>=2 environment.
- The RandomForest model needs the legacy sklearn 1.3.x environment.

Keep this helper process-isolated and disabled by default.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Iterable, Optional

import numpy as np

from ..events import DetectionEvent, EventType
from .fall_temporal_model import FRAME_FEATURE_NAMES

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models/legacy/falldata_mediapipe/FNF_RF_SMOTE_CAM_1.pkl"
)
DEFAULT_COMPARE_MODEL_PATH = PROJECT_ROOT / "models/experiments/falldata_sample_rf_max120_guarded.pkl"
DEFAULT_MEDIAPIPE_PYTHON = PROJECT_ROOT / ".venv-mediapipe/bin/python"
DEFAULT_MODEL_PYTHON = PROJECT_ROOT / ".venv-falldata/bin/python"
DEFAULT_TEMPORAL_PYTHON = PROJECT_ROOT / ".venv-jetson-train/bin/python"
DEFAULT_TEMPORAL_POSE_MODEL = PROJECT_ROOT / "models/fall/yolov8n-pose.pt"
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts/datasets/extract_falldata_mediapipe_features.py"
SMOKE_SCRIPT = PROJECT_ROOT / "scripts/datasets/smoke_falldata_video_model.py"
TEMPORAL_SMOKE_SCRIPT = PROJECT_ROOT / "scripts/inference/smoke_fall_temporal_model.py"
YOLO_RF_SMOKE_SCRIPT = PROJECT_ROOT / "scripts/datasets/smoke_yolo_pose_fall_rf.py"


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _parse_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


@dataclass(frozen=True)
class FallDataAuxConfig:
    enabled: bool = False
    mode: str = "shadow"  # shadow | confirm
    threshold: float = 0.7
    min_nonzero_frames: int = 30
    fall_class_index: int = 0
    buffer_frames: int = 600
    max_extract_frames: int = 120
    sequence_transform: str = "postpad"
    timeout_seconds: float = 30.0
    cooldown_seconds: float = 10.0
    min_pose_frames: int = 0
    min_confirmed_results: int = 1
    fail_open_on_unavailable: bool = True
    mediapipe_python: Path = DEFAULT_MEDIAPIPE_PYTHON
    model_python: Path = DEFAULT_MODEL_PYTHON
    model_path: Path = DEFAULT_MODEL_PATH
    compare_model_path: Optional[Path] = None
    compare_model_paths_by_camera: dict[str, Path] | None = None
    compare_model_kind: str = "mediapipe_rf"
    compare_fall_class_index: int = 0
    compare_threshold: Optional[float] = None
    temporal_python: Path = DEFAULT_TEMPORAL_PYTHON
    temporal_compare_model_path: Optional[Path] = None
    temporal_pose_model_path: Path = DEFAULT_TEMPORAL_POSE_MODEL
    temporal_sliding_window_size: int = 0
    temporal_sliding_window_stride: int = 5
    temporal_min_confirmed_windows: int = 1
    inline_pose_rf: bool = False
    inline_feature_capture_path: Optional[Path] = None
    inline_feature_capture_min_probability: float = 0.0
    clip_buffer_enabled: bool = False
    clip_buffer_width: int = 640
    clip_buffer_height: int = 360

    @classmethod
    def from_env(cls) -> "FallDataAuxConfig":
        mode = os.environ.get("FALLDATA_AUX_MODE", "shadow").strip().lower()
        if mode not in {"shadow", "confirm"}:
            mode = "shadow"
        threshold = _parse_float(os.environ.get("FALLDATA_AUX_THRESHOLD"), 0.7)
        compare_threshold_raw = os.environ.get(
            "FALLDATA_AUX_COMPARE_THRESHOLD", ""
        ).strip()
        compare_model_raw = os.environ.get("FALLDATA_AUX_COMPARE_MODEL_PATH", "").strip()
        compare_model_map_raw = os.environ.get(
            "FALLDATA_AUX_COMPARE_MODEL_MAP", ""
        ).strip()
        compare_model_map: dict[str, Path] = {}
        if compare_model_map_raw:
            try:
                parsed_map = json.loads(compare_model_map_raw)
                if isinstance(parsed_map, dict):
                    compare_model_map = {
                        str(camera_id): Path(str(model_path))
                        for camera_id, model_path in parsed_map.items()
                        if str(camera_id).strip() and str(model_path).strip()
                    }
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("FALLDATA_AUX_COMPARE_MODEL_MAP JSON 파싱 실패")
        temporal_compare_model_raw = os.environ.get(
            "FALLDATA_AUX_TEMPORAL_COMPARE_MODEL_PATH", ""
        ).strip()
        inline_feature_capture_raw = os.environ.get(
            "FALLDATA_AUX_INLINE_FEATURE_CAPTURE_PATH", ""
        ).strip()
        sequence_transform = os.environ.get(
            "FALLDATA_AUX_SEQUENCE_TRANSFORM", "postpad"
        ).strip().lower()
        if sequence_transform not in {"postpad", "tail_align", "stretch"}:
            sequence_transform = "postpad"
        return cls(
            enabled=_parse_bool(os.environ.get("FALLDATA_AUX_ENABLED"), False),
            mode=mode,
            threshold=threshold,
            min_nonzero_frames=_parse_int(
                os.environ.get("FALLDATA_AUX_MIN_NONZERO_FRAMES"), 30
            ),
            fall_class_index=_parse_int(os.environ.get("FALLDATA_AUX_FALL_CLASS_INDEX"), 0),
            buffer_frames=_parse_int(os.environ.get("FALLDATA_AUX_BUFFER_FRAMES"), 600),
            max_extract_frames=_parse_int(
                os.environ.get("FALLDATA_AUX_MAX_EXTRACT_FRAMES"), 120
            ),
            sequence_transform=sequence_transform,
            timeout_seconds=_parse_float(
                os.environ.get("FALLDATA_AUX_TIMEOUT_SECONDS"), 30.0
            ),
            cooldown_seconds=_parse_float(
                os.environ.get("FALLDATA_AUX_COOLDOWN_SECONDS"), 10.0
            ),
            min_pose_frames=max(
                _parse_int(os.environ.get("FALLDATA_AUX_MIN_POSE_FRAMES"), 0),
                0,
            ),
            min_confirmed_results=max(
                _parse_int(os.environ.get("FALLDATA_AUX_MIN_CONFIRMED_RESULTS"), 1),
                1,
            ),
            fail_open_on_unavailable=_parse_bool(
                os.environ.get("FALLDATA_AUX_FAIL_OPEN_ON_UNAVAILABLE"), True
            ),
            mediapipe_python=Path(
                os.environ.get("FALLDATA_AUX_MEDIAPIPE_PYTHON", str(DEFAULT_MEDIAPIPE_PYTHON))
            ),
            model_python=Path(
                os.environ.get("FALLDATA_AUX_MODEL_PYTHON", str(DEFAULT_MODEL_PYTHON))
            ),
            model_path=Path(os.environ.get("FALLDATA_AUX_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            compare_model_path=Path(compare_model_raw) if compare_model_raw else None,
            compare_model_paths_by_camera=compare_model_map or None,
            compare_model_kind=os.environ.get(
                "FALLDATA_AUX_COMPARE_MODEL_KIND", "mediapipe_rf"
            ).strip().lower(),
            compare_fall_class_index=_parse_int(
                os.environ.get("FALLDATA_AUX_COMPARE_FALL_CLASS_INDEX"),
                _parse_int(os.environ.get("FALLDATA_AUX_FALL_CLASS_INDEX"), 0),
            ),
            compare_threshold=(
                _parse_float(compare_threshold_raw, threshold)
                if compare_threshold_raw
                else threshold
            ),
            temporal_python=Path(
                os.environ.get(
                    "FALLDATA_AUX_TEMPORAL_PYTHON",
                    str(DEFAULT_TEMPORAL_PYTHON),
                )
            ),
            temporal_compare_model_path=(
                Path(temporal_compare_model_raw) if temporal_compare_model_raw else None
            ),
            temporal_pose_model_path=Path(
                os.environ.get(
                    "FALLDATA_AUX_TEMPORAL_POSE_MODEL_PATH",
                    str(DEFAULT_TEMPORAL_POSE_MODEL),
                )
            ),
            temporal_sliding_window_size=_parse_int(
                os.environ.get("FALLDATA_AUX_TEMPORAL_SLIDING_WINDOW_SIZE"), 0
            ),
            temporal_sliding_window_stride=_parse_int(
                os.environ.get("FALLDATA_AUX_TEMPORAL_SLIDING_WINDOW_STRIDE"), 5
            ),
            temporal_min_confirmed_windows=_parse_int(
                os.environ.get("FALLDATA_AUX_TEMPORAL_MIN_CONFIRMED_WINDOWS"), 1
            ),
            inline_pose_rf=_parse_bool(
                os.environ.get("FALLDATA_AUX_INLINE_POSE_RF"), False
            ),
            inline_feature_capture_path=(
                Path(inline_feature_capture_raw)
                if inline_feature_capture_raw
                else None
            ),
            inline_feature_capture_min_probability=max(
                _parse_float(
                    os.environ.get("FALLDATA_AUX_INLINE_FEATURE_CAPTURE_MIN_PROBABILITY"),
                    0.0,
                ),
                0.0,
            ),
            clip_buffer_enabled=_parse_bool(
                os.environ.get("FALLDATA_AUX_CLIP_BUFFER_ENABLED"), False
            ),
            clip_buffer_width=max(
                0, _parse_int(os.environ.get("FALLDATA_AUX_CLIP_BUFFER_WIDTH"), 640)
            ),
            clip_buffer_height=max(
                0, _parse_int(os.environ.get("FALLDATA_AUX_CLIP_BUFFER_HEIGHT"), 360)
            ),
        )


class FallDataAuxVerifier:
    """Buffers frames and verifies pose fall candidates with the public RF model."""

    UNAVAILABLE_STATUSES = {
        "error",
        "missing_dependency",
        "no_frames",
        "skipped_cooldown",
    }

    def __init__(
        self,
        config: Optional[FallDataAuxConfig] = None,
        *,
        inline_pose_rf_bundle: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config or FallDataAuxConfig.from_env()
        self._frames: Deque[np.ndarray] = deque(maxlen=max(self.config.buffer_frames, 1))
        self._pose_records: dict[str, Deque[dict[str, Any]]] = {}
        self._pose_generations: dict[str, int] = {}
        self._next_pose_generation = 0
        self._inline_results: dict[tuple[str, int], tuple[float, dict]] = {}
        self._inline_metrics: dict[str, dict[str, float]] = {}
        self._temporal_metrics: dict[str, float] = {}
        self._last_verified_frames: list[np.ndarray] = []
        self._inline_pose_rf_bundle = inline_pose_rf_bundle
        self._inline_pose_rf_bundles: dict[str, dict[str, Any]] = {}
        self._last_run_at = 0.0
        self._last_result: dict | None = None
        self._last_result_by_camera: dict[str, dict] = {}
        self._lock = Lock()
        self._inline_feature_capture_lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def add_frame(self, frame: np.ndarray) -> None:
        if (
            not self.enabled
            or frame is None
            or not isinstance(frame, np.ndarray)
        ):
            return
        if self.config.inline_pose_rf and not self.config.clip_buffer_enabled:
            return
        buffered_frame = frame
        if self.config.clip_buffer_enabled and (
            self.config.clip_buffer_width > 0 or self.config.clip_buffer_height > 0
        ):
            import cv2

            source_height, source_width = frame.shape[:2]
            target_width = self.config.clip_buffer_width or source_width
            target_height = self.config.clip_buffer_height or source_height
            if (source_width, source_height) != (target_width, target_height):
                buffered_frame = cv2.resize(
                    frame, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
        with self._lock:
            self._frames.append(buffered_frame.copy())

    def add_pose_events(
        self,
        camera_name: str,
        events: Iterable[DetectionEvent],
    ) -> None:
        if not self.enabled or not self.config.inline_pose_rf:
            return
        from scripts.datasets.train_yolo_pose_fall_rf import _pose_geometry

        events = list(events)
        for event in events:
            metadata = dict(event.metadata or {})
            metadata.pop("falldata_pose_scope", None)
            event.metadata = metadata
        pose_ids_by_box: dict[tuple, set[int]] = {}
        records: list[dict[str, Any]] = []
        for event in events:
            if event.event_type != EventType.PERSON or not event.keypoints:
                continue
            if type(event.object_id) is not int or not 0 <= event.object_id < 2**64 - 1:
                continue
            metadata = dict(event.metadata or {})
            if type(metadata.get("frame_num")) is not int or metadata["frame_num"] < 0:
                continue
            fall_score = metadata.get("fall_score")
            if fall_score is None:
                continue
            keypoints = np.asarray(event.keypoints, dtype=np.float32)
            if keypoints.shape != (17, 3):
                continue
            frame_width = max(int(metadata.get("frame_width") or 0), 1)
            frame_height = max(int(metadata.get("frame_height") or 0), 1)
            bbox = np.asarray(
                [event.x, event.y, event.x + event.width, event.y + event.height],
                dtype=np.float32,
            )
            keypoint_confidences = keypoints[:, 2]
            pose_ids_by_box.setdefault(self._pose_box_key(event), set()).add(
                event.object_id
            )
            records.append(
                {
                    "object_id": event.object_id,
                    "frame_index": int(metadata.get("frame_num") or 0),
                    "timestamp": float(event.timestamp),
                    "fall_score": float(fall_score),
                    "fall_reasons": list(metadata.get("fall_reasons") or []),
                    "detection_confidence": float(event.confidence),
                    "bbox_aspect": float(event.width / max(event.height, 1)),
                    "bbox_area_ratio": float(
                        (event.width * event.height)
                        / max(frame_width * frame_height, 1)
                    ),
                    "visible_keypoints": int((keypoint_confidences >= 0.35).sum()),
                    "mean_keypoint_confidence": float(keypoint_confidences.mean()),
                    **_pose_geometry(
                        keypoints,
                        bbox=bbox,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        min_keypoint_confidence=0.35,
                    ),
                }
            )
        if not records:
            return
        records_by_frame: dict[tuple[int, int], dict[str, Any]] = {}
        for record in records:
            key = (record["frame_index"], record["object_id"])
            current = records_by_frame.get(key)
            if current is None or (
                record["fall_score"],
                record["detection_confidence"],
            ) > (
                current["fall_score"],
                current["detection_confidence"],
            ):
                records_by_frame[key] = record
        with self._lock:
            camera_records = self._pose_records.setdefault(
                camera_name,
                deque(maxlen=max(self.config.buffer_frames, 1)),
            )
            for record in sorted(
                records_by_frame.values(),
                key=lambda item: int(item["frame_index"]),
            ):
                frame_index = int(record["frame_index"])
                if camera_records:
                    last_record = camera_records[-1]
                    last_frame_index = int(last_record["frame_index"])
                    if frame_index < last_frame_index:
                        camera_records.clear()
                        self._reset_pose_generation(camera_name)
                if camera_name not in self._pose_generations:
                    self._reset_pose_generation(camera_name)
                duplicate = None
                for index in range(len(camera_records) - 1, -1, -1):
                    previous = camera_records[index]
                    if previous["frame_index"] != frame_index:
                        break
                    if previous["object_id"] == record["object_id"]:
                        duplicate = index
                        break
                if duplicate is None:
                    camera_records.append(record)
                elif (record["fall_score"], record["detection_confidence"]) > (
                    camera_records[duplicate]["fall_score"],
                    camera_records[duplicate]["detection_confidence"],
                ):
                    camera_records[duplicate] = record
            generation = self._pose_generations[camera_name]
            active_ids = {record["object_id"] for record in camera_records}
            for key in list(self._inline_results):
                if key[0] == camera_name and key[1] not in active_ids:
                    self._inline_results.pop(key)
            for event in events:
                metadata = dict(event.metadata or {})
                metadata.pop("falldata_pose_scope", None)
                matching_ids = pose_ids_by_box.get(self._pose_box_key(event), set())
                if len(matching_ids) == 1:
                    metadata["falldata_pose_scope"] = {
                        "camera_id": camera_name,
                        "object_id": next(iter(matching_ids)),
                        "generation": generation,
                        "frame_index": metadata["frame_num"],
                    }
                event.metadata = metadata

    @staticmethod
    def _pose_box_key(event: DetectionEvent) -> tuple:
        return (
            (event.metadata or {}).get("frame_num"),
            event.x,
            event.y,
            event.width,
            event.height,
        )

    def _reset_pose_generation(self, camera_name: str) -> None:
        """Caller holds _lock; invalidate both cached probabilities and streaks."""
        self._next_pose_generation += 1
        self._pose_generations[camera_name] = self._next_pose_generation
        for key in list(self._inline_results):
            if key[0] == camera_name:
                self._inline_results.pop(key)

    def reset_pose_records(self, camera_name: str) -> None:
        with self._lock:
            self._pose_records.pop(camera_name, None)
            self._reset_pose_generation(camera_name)

    def is_pose_scope_current(self, scope: dict) -> bool:
        with self._lock:
            return (
                type(scope.get("generation")) is int
                and self._pose_generations.get(scope.get("camera_id"))
                == scope["generation"]
            )

    def _write_inline_feature_capture(
        self,
        camera_name: str,
        summary: dict[str, Any],
        *,
        window_seconds: float,
    ) -> str | None:
        path = self.config.inline_feature_capture_path
        if path is None:
            return None

        record = {
            "schema_version": 2,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "runtime": "deepstream_pose_inline",
            "camera_id": camera_name,
            "pose_scope": summary.get("pose_scope"),
            "window_seconds": float(window_seconds),
            "frames_seen": int(summary["frames_seen"]),
            "frames_with_pose": int(summary["frames_with_pose"]),
            "sampled_frames": len(summary.get("frame_records", [])),
            "feature_names": list(summary["feature_names"]),
            "feature_vector": list(summary["feature_vector"]),
            "frame_feature_names": list(FRAME_FEATURE_NAMES),
            "frame_records": [
                dict(frame_record) for frame_record in summary.get("frame_records", [])
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            with self._inline_feature_capture_lock:
                with path.open("a", encoding="utf-8") as capture_file:
                    capture_file.write(line)
            return "written"
        except (OSError, TypeError, ValueError):
            logger.exception("DeepStream inline pose feature capture failed")
            return "error"

    def annotate_events(self, events: Iterable[DetectionEvent]) -> list[DetectionEvent]:
        events = list(events)
        if not self.enabled:
            return events

        annotated: list[DetectionEvent] = []
        for event in events:
            if event.event_type != EventType.FALL_DETECTED:
                annotated.append(event)
                continue

            metadata = dict(event.metadata or {})
            pose_scope = metadata.get("falldata_pose_scope")
            result = (
                self.verify(
                    camera_name=(
                        pose_scope.get("camera_id")
                        if isinstance(pose_scope, dict)
                        else metadata.get("camera_id")
                    ),
                    pose_scope=pose_scope,
                )
                if self.config.inline_pose_rf
                else self.verify()
            )
            metadata["falldata_aux"] = result
            event.metadata = metadata
            if self.config.mode == "confirm" and not result.get("confirmed", False):
                if self._should_fail_open(result):
                    metadata["falldata_aux_confirm_fallback"] = result.get("status")
                    event.metadata = metadata
                    logger.warning(
                        "falldata aux confirm unavailable; publishing original fall event: %s",
                        result,
                    )
                    annotated.append(event)
                    continue
                logger.info("falldata aux confirm mode rejected fall event: %s", result)
                continue
            annotated.append(event)
        return annotated

    def _should_fail_open(self, result: dict) -> bool:
        """검증기 자체 실패는 안전 알람 손실을 막기 위해 기본 fail-open 처리한다."""
        if not self.config.fail_open_on_unavailable:
            return False
        return str(result.get("status")) in self.UNAVAILABLE_STATUSES

    def verify(
        self, camera_name: Optional[str] = None, *, pose_scope: Optional[dict] = None
    ) -> dict:
        if self.config.inline_pose_rf:
            self._record_inline_metrics(camera_name, requests=1)
            try:
                result = self._verify_inline_pose_rf_with_cooldown(
                    camera_name, pose_scope
                )
            except Exception:
                self._record_inline_metrics(camera_name, error_results=1)
                raise
            status = result.get("status")
            outcome = (
                "ok_results"
                if status == "ok"
                else "error_results" if status == "error" else "unavailable_results"
            )
            self._record_inline_metrics(
                camera_name,
                **{outcome: 1, "cache_hits": int(bool(result.get("cache_reused")))},
            )
            return result
        now = time.time()
        if now - self._last_run_at < self.config.cooldown_seconds:
            return self._cooldown_result()
        self._last_run_at = now

        try:
            result = self._verify_once()
            if camera_name:
                result = self._apply_confirmation_streak(result, camera_name)
            self._last_result = dict(result)
            return result
        except Exception as exc:
            logger.warning("falldata aux verification failed: %s", exc)
            result = self._result(
                "error",
                confirmed=False,
                error=str(exc),
            )
            self._last_result = dict(result)
            return result

    def verify_temporal_shadow(self) -> dict:
        """Run buffered-video verification so near-miss events include TCN output."""
        frames = self.snapshot_frames()
        self._last_verified_frames = frames
        if not frames:
            return self._result("no_frames", confirmed=False, buffered_frames=0)
        try:
            with tempfile.TemporaryDirectory(prefix="falldata_temporal_shadow_") as tmp:
                video_path = Path(tmp) / "candidate.mp4"
                self._write_video(video_path, frames)
                temporal_result = self._run_temporal_compare_model(video_path)
            if temporal_result is None:
                result = self._result(
                    "temporal_unavailable",
                    confirmed=False,
                    buffered_frames=len(frames),
                    temporal_compare_model=None,
                )
            else:
                temporal_status = temporal_result.get("status")
                result = self._result(
                    (
                        temporal_status
                        if temporal_status in {"ok", "insufficient_pose_frames"}
                        else "error"
                    ),
                    confirmed=bool(temporal_result.get("confirmed")),
                    buffered_frames=len(frames),
                    temporal_compare_model=temporal_result,
                )
            self._last_result = dict(result)
            return result
        except Exception as exc:
            logger.warning("temporal shadow verification failed: %s", exc)
            return self._result(
                "error",
                confirmed=False,
                buffered_frames=len(frames),
                error=str(exc),
            )

    def _record_inline_metrics(self, camera_name: Optional[str], **increments) -> None:
        with self._lock:
            # Only cameras with pose input can create metric labels; never track IDs.
            if camera_name not in self._pose_generations:
                return
            metrics = self._inline_metrics.setdefault(camera_name, {})
            for field, amount in increments.items():
                metrics[field] = metrics.get(field, 0) + amount

    def inline_metrics_snapshot(self) -> dict:
        """Verifier-lifetime counters; pose/cache resets do not erase model work."""
        with self._lock:
            return {
                "enabled": self.enabled and self.config.inline_pose_rf,
                "cameras": {
                    camera: dict(values)
                    for camera, values in self._inline_metrics.items()
                },
            }

    def _verify_inline_pose_rf_with_cooldown(
        self,
        camera_name: Optional[str],
        pose_scope: Optional[dict],
    ) -> dict:
        if not camera_name:
            return self._result("missing_camera", confirmed=False)
        if (
            not isinstance(pose_scope, dict)
            or pose_scope.get("camera_id") != camera_name
            or any(
                type(pose_scope.get(key)) is not int or pose_scope[key] < 0
                for key in ("object_id", "generation", "frame_index")
            )
            or pose_scope["object_id"] >= 2**64 - 1
        ):
            return self._result("missing_pose_scope", confirmed=False)
        pose_scope = dict(pose_scope)
        key = (camera_name, pose_scope["object_id"])
        now = time.monotonic()
        with self._lock:
            if self._pose_generations.get(camera_name) != pose_scope["generation"]:
                return self._result("stale_pose_scope", confirmed=False)
            if not any(
                record["object_id"] == key[1]
                and record["frame_index"] == pose_scope["frame_index"]
                for record in self._pose_records.get(camera_name, ())
            ):
                return self._result("no_pose_records", confirmed=False)
            last_run_at, previous = self._inline_results.get(key, (0.0, {}))
            previous_frame = (previous.get("pose_scope") or {}).get("frame_index", -1)
            if previous and (
                previous_frame == pose_scope["frame_index"]
                or (
                    previous_frame < pose_scope["frame_index"]
                    and now - last_run_at < self.config.cooldown_seconds
                )
            ):
                return {
                    **previous,
                    "pose_scope": pose_scope,
                    "cache_reused": True,
                    "cached_pose_scope": dict(previous["pose_scope"]),
                }
        try:
            result = self._verify_inline_pose_rf(camera_name, pose_scope)
        except Exception as exc:
            logger.warning("inline pose RF verification failed: %s", exc)
            result = self._result("error", confirmed=False, error=str(exc))
        # An older queued candidate must not inherit a future window's streak.
        if previous_frame >= pose_scope["frame_index"]:
            previous = {}
        result = self._apply_confirmation_streak(result, camera_name, previous=previous)
        result.update(pose_scope=pose_scope, cache_reused=False)
        with self._lock:
            if self._pose_generations.get(camera_name) != pose_scope["generation"]:
                return self._result("stale_pose_scope", confirmed=False)
            if previous_frame < pose_scope["frame_index"]:
                self._inline_results[key] = (now, dict(result))
        return result

    def _apply_confirmation_streak(
        self, result: dict, camera_name: str, *, previous: Optional[dict] = None
    ) -> dict:
        """Require consecutive confirmed windows before publishing a result."""
        if not result.get("confirmed"):
            return {**result, "confirmation_streak": 0}
        if previous is None:
            previous = self._last_result_by_camera.get(camera_name) or {}
        previous_streak = int(previous.get("confirmation_streak") or 0)
        streak = previous_streak + 1
        required = max(int(self.config.min_confirmed_results), 1)
        return {
            **result,
            "confirmation_streak": streak,
            "confirmed": streak >= required,
            "min_confirmed_results": required,
        }

    def _verify_inline_pose_rf(self, camera_name: str, pose_scope: dict) -> dict:
        from scripts.datasets.smoke_yolo_pose_fall_rf import (
            _fall_probability_from_classifier,
            _select_model_features,
        )
        from scripts.datasets.train_yolo_pose_fall_rf import _summarize_frames

        model_path = (
            (self.config.compare_model_paths_by_camera or {}).get(camera_name)
            or self.config.compare_model_path
        )
        bundle_key = str(model_path or "")
        bundle = self._inline_pose_rf_bundles.get(bundle_key)
        if bundle is None and self._inline_pose_rf_bundle is not None and not bundle_key:
            bundle = self._inline_pose_rf_bundle
        if bundle is None:
            if model_path is None or not model_path.exists():
                return self._result(
                    "missing_dependency",
                    confirmed=False,
                    missing=str(model_path),
                )
            import joblib

            bundle = joblib.load(model_path)
            if not isinstance(bundle, dict):
                bundle = {"model": bundle}
            self._inline_pose_rf_bundles[bundle_key] = bundle

        with self._lock:
            if self._pose_generations.get(camera_name) != pose_scope["generation"]:
                return self._result("stale_pose_scope", confirmed=False)
            records = [
                record
                for record in self._pose_records.get(camera_name, ())
                if record["object_id"] == pose_scope["object_id"]
                and record["frame_index"] <= pose_scope["frame_index"]
            ]
        if not records:
            return self._result("no_pose_records", confirmed=False)

        inference_config = dict(bundle.get("inference_config") or {})
        training_config = dict(bundle.get("training_config") or {})
        window_seconds = float(
            inference_config.get("candidate_window_seconds") or 3.0
        )
        latest_timestamp = float(records[-1]["timestamp"])
        window_start = latest_timestamp - max(window_seconds, 0.1)
        window_records = [
            record for record in records if float(record["timestamp"]) >= window_start
        ]
        min_pose_frames = int(training_config.get("min_pose_frames") or 1)
        min_pose_frames = max(min_pose_frames, int(self.config.min_pose_frames))
        if len(window_records) < min_pose_frames:
            return self._result(
                "insufficient_pose_records",
                confirmed=False,
                frames_with_pose=len(window_records),
                min_pose_frames=min_pose_frames,
            )

        max_frames = max(int(inference_config.get("max_frames") or 48), 1)
        selected_indices = np.linspace(
            0,
            len(window_records) - 1,
            num=max_frames,
        ).astype(int)
        selected_records = [window_records[index] for index in selected_indices]
        summary = _summarize_frames(selected_records, max_frames)
        summary["pose_scope"] = dict(pose_scope)
        classifier = bundle.get("model", bundle)
        feature_names = bundle.get("feature_names")
        expected_features = getattr(
            classifier,
            "n_features_in_",
            len(summary["feature_vector"]),
        )
        features = _select_model_features(
            summary=summary,
            model_feature_names=feature_names,
            expected_feature_count=expected_features,
        )
        self._record_inline_metrics(camera_name, predict_calls=1)
        predict_started = time.monotonic()
        predict_completed = False
        try:
            prediction = classifier.predict_proba(features)
            predict_completed = True
        finally:
            self._record_inline_metrics(
                camera_name,
                predict_completed=int(predict_completed),
                predict_errors=int(not predict_completed),
                predict_seconds=max(0.0, time.monotonic() - predict_started),
            )
        probabilities = prediction.tolist()
        fall_probability = _fall_probability_from_classifier(
            classifier,
            probabilities,
        )
        threshold = float(
            self.config.compare_threshold
            if self.config.compare_threshold is not None
            else training_config.get("decision_threshold", self.config.threshold)
        )
        confirmed = (
            fall_probability is not None and fall_probability >= threshold
        )
        feature_capture_status = None
        capture_min_probability = self.config.inline_feature_capture_min_probability
        if (
            fall_probability is not None
            and fall_probability >= capture_min_probability
        ):
            feature_capture_status = self._write_inline_feature_capture(
                camera_name,
                summary,
                window_seconds=window_seconds,
            )
        capture_result = (
            {"feature_capture_status": feature_capture_status}
            if feature_capture_status is not None
            else {}
        )
        return self._result(
            "ok",
            confirmed=confirmed,
            runtime="deepstream_pose_inline",
            fall_probability=fall_probability,
            threshold=threshold,
            frames_with_pose=len(window_records),
            sampled_frames=len(selected_records),
            probability=probabilities[0],
            **capture_result,
        )

    def _cooldown_result(self) -> dict:
        if self._last_result:
            return {
                **self._last_result,
                "status": "skipped_cooldown",
                "previous_status": self._last_result.get("status"),
            }
        return self._result("skipped_cooldown", confirmed=False)

    def _verify_once(self) -> dict:
        if not self._frames:
            return self._result("no_frames", confirmed=False)

        missing = self._missing_dependency()
        if missing:
            return {
                **self._result("missing_dependency", confirmed=False),
                "missing": missing,
            }

        with tempfile.TemporaryDirectory(prefix="falldata_aux_") as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "candidate.mp4"
            feature_dir = tmp_path / "features"
            self._write_video(video_path, self.snapshot_frames())
            extract = self._run(
                [
                    str(self.config.mediapipe_python),
                    str(EXTRACT_SCRIPT),
                    "--video",
                    str(video_path),
                    "--output-dir",
                    str(feature_dir),
                    "--max-frames",
                    str(max(self.config.max_extract_frames, 1)),
                    "--sequence-transform",
                    self.config.sequence_transform,
                ]
            )
            nonzero_frames = self._parse_nonzero_frames(extract.stdout)
            infer = self._run(
                [
                    str(self.config.model_python),
                    str(SMOKE_SCRIPT),
                    "--model",
                    str(self.config.model_path),
                    "--sequence-dir",
                    str(feature_dir),
                ]
            )
            probability = self._parse_probability(infer.stdout)
            prediction = self._parse_prediction(infer.stdout)
            compare_result = self._run_compare_model(
                feature_dir,
                nonzero_frames,
                video_path,
            )
            temporal_compare_result = self._run_temporal_compare_model(video_path)

        fall_probability = (
            probability[self.config.fall_class_index]
            if probability and 0 <= self.config.fall_class_index < len(probability)
            else None
        )
        confirmed = self._is_confirmed(nonzero_frames, fall_probability)
        return self._result(
            "ok",
            confirmed=confirmed,
            prediction=prediction,
            probability=probability,
            fall_probability=fall_probability,
            threshold=self.config.threshold,
            fall_class_index=self.config.fall_class_index,
            nonzero_feature_frames=nonzero_frames,
            min_nonzero_feature_frames=self.config.min_nonzero_frames,
            compare_model=compare_result,
            temporal_compare_model=temporal_compare_result,
        )

    def _run_compare_model(
        self,
        feature_dir: Path,
        nonzero_frames: int,
        video_path: Path,
    ) -> Optional[dict]:
        compare_model_path = self.config.compare_model_path
        if compare_model_path is None:
            return None
        if not compare_model_path.exists():
            return {
                "status": "missing_dependency",
                "model_path": str(compare_model_path),
                "confirmed": False,
            }
        if self.config.compare_model_kind == "yolo_pose_rf":
            infer = self._run(
                [
                    str(self.config.temporal_python),
                    str(YOLO_RF_SMOKE_SCRIPT),
                    "--model",
                    str(compare_model_path),
                    "--pose-model",
                    str(self.config.temporal_pose_model_path),
                    "--video",
                    str(video_path),
                ]
            )
        else:
            infer = self._run(
                [
                    str(self.config.model_python),
                    str(SMOKE_SCRIPT),
                    "--model",
                    str(compare_model_path),
                    "--sequence-dir",
                    str(feature_dir),
                ]
            )
        probability = self._parse_probability(infer.stdout)
        prediction = self._parse_prediction(infer.stdout)
        frames_with_pose = None
        if self.config.compare_model_kind == "yolo_pose_rf":
            fall_probability = self._parse_named_float(
                infer.stdout,
                "fall_probability",
            )
            frames_with_pose = self._parse_named_int(
                infer.stdout,
                "frames_with_pose",
            )
        else:
            fall_probability = None
        if fall_probability is None:
            fall_probability = (
                probability[self.config.compare_fall_class_index]
                if probability
                and 0 <= self.config.compare_fall_class_index < len(probability)
                else None
            )
        compare_threshold = (
            self.config.compare_threshold
            if self.config.compare_threshold is not None
            else self.config.threshold
        )
        confirmation_frames = (
            frames_with_pose if frames_with_pose is not None else nonzero_frames
        )
        return {
            "status": "ok",
            "model_path": str(compare_model_path),
            "confirmed": self._is_confirmed(
                confirmation_frames,
                fall_probability,
                threshold=compare_threshold,
            ),
            "prediction": prediction,
            "probability": probability,
            "fall_probability": fall_probability,
            "threshold": compare_threshold,
            "fall_class_index": self.config.compare_fall_class_index,
            "frames_with_pose": frames_with_pose,
        }

    def _run_temporal_compare_model(self, video_path: Path) -> Optional[dict]:
        self._record_temporal_metrics(requests=1)
        model_path = self.config.temporal_compare_model_path
        if model_path is None:
            self._record_temporal_metrics(disabled=1)
            return None
        required_paths = (
            self.config.temporal_python,
            model_path,
            self.config.temporal_pose_model_path,
            TEMPORAL_SMOKE_SCRIPT,
        )
        missing = next((path for path in required_paths if not path.exists()), None)
        if missing is not None:
            self._record_temporal_metrics(missing_dependencies=1)
            return {
                "status": "missing_dependency",
                "model_path": str(model_path),
                "missing": str(missing),
                "confirmed": False,
            }
        command = [
            str(self.config.temporal_python),
            str(TEMPORAL_SMOKE_SCRIPT),
            "--model",
            str(model_path),
            "--pose-model",
            str(self.config.temporal_pose_model_path),
            "--video",
            str(video_path),
        ]
        if self.config.temporal_sliding_window_size > 0:
            command.extend(
                [
                    "--sliding-window-size",
                    str(self.config.temporal_sliding_window_size),
                    "--sliding-window-stride",
                    str(self.config.temporal_sliding_window_stride),
                    "--min-confirmed-windows",
                    str(self.config.temporal_min_confirmed_windows),
                ]
            )
        self._record_temporal_metrics(process_calls=1)
        started = time.monotonic()
        try:
            infer = self._run(command)
        except subprocess.TimeoutExpired:
            self._record_temporal_metrics(process_errors=1, process_timeouts=1)
            raise
        except Exception:
            self._record_temporal_metrics(process_errors=1)
            raise
        else:
            self._record_temporal_metrics(process_completed=1)
        finally:
            self._record_temporal_metrics(
                process_seconds=max(0.0, time.monotonic() - started)
            )
        prediction = self._parse_prediction(infer.stdout)
        fall_probability = self._parse_named_float(infer.stdout, "fall_probability")
        threshold = self._parse_named_float(infer.stdout, "threshold")
        frames_with_pose = self._parse_named_int(infer.stdout, "frames_with_pose")
        sliding_windows = self._parse_named_int(infer.stdout, "sliding_windows")
        min_pose_frames = max(int(self.config.min_pose_frames), 0)
        pose_frames_sufficient = min_pose_frames == 0 or (
            frames_with_pose is not None and frames_with_pose >= min_pose_frames
        )
        parsed = (
            prediction is not None
            and fall_probability is not None
            and threshold is not None
        )
        # 파싱 실패 시 실제 model forward 횟수를 알 수 없으므로 0으로 둔다.
        # sliding window가 없으면 전체 영상 1회 추론(단일 window)으로 센다.
        window_inferences = (sliding_windows if sliding_windows is not None else 1) if parsed else 0
        self._record_temporal_metrics(
            parsed_results=int(parsed),
            unparsed_results=int(not parsed),
            insufficient_pose_results=int(not pose_frames_sufficient),
            window_inferences=window_inferences,
        )
        confirmed = (
            pose_frames_sufficient
            and prediction == 1
            and fall_probability is not None
            and threshold is not None
            and fall_probability >= threshold
        )
        result = {
            "status": "ok" if pose_frames_sufficient else "insufficient_pose_frames",
            "model_path": str(model_path),
            "confirmed": confirmed,
            "prediction": prediction,
            "fall_probability": fall_probability,
            "threshold": threshold,
            "frames_with_pose": frames_with_pose,
        }
        if not pose_frames_sufficient:
            result["min_pose_frames"] = min_pose_frames
        return result

    def _record_temporal_metrics(self, **increments) -> None:
        with self._lock:
            for field, amount in increments.items():
                self._temporal_metrics[field] = (
                    self._temporal_metrics.get(field, 0) + amount
                )

    def temporal_metrics_snapshot(self) -> dict:
        """Scene-level subprocess counters, not per-camera or GPU inference counts."""
        with self._lock:
            return dict(self._temporal_metrics)

    def _result(self, status: str, *, confirmed: bool, **extra: object) -> dict:
        result = {
            "enabled": True,
            "mode": self.config.mode,
            "status": status,
            "confirmed": confirmed,
            "buffered_frames": len(self._frames),
        }
        result.update(extra)
        return result

    def _is_confirmed(
        self,
        nonzero_frames: int,
        fall_probability: Optional[float],
        *,
        threshold: Optional[float] = None,
    ) -> bool:
        if fall_probability is None:
            return False
        effective_threshold = self.config.threshold if threshold is None else threshold
        return (
            nonzero_frames >= self.config.min_nonzero_frames
            and fall_probability >= effective_threshold
        )

    def snapshot_frames(self) -> list[np.ndarray]:
        """현재 버퍼 프레임 복사본을 반환한다."""
        with self._lock:
            return list(self._frames)

    def save_buffered_clip(self, path: Path) -> int:
        """현재 버퍼를 mp4 클립으로 저장하고 저장 프레임 수를 반환한다."""
        frames = self.snapshot_frames()
        if not frames:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_video(path, frames)
        return len(frames[-self.config.buffer_frames :])

    def save_last_verified_clip(self, path: Path) -> int:
        """마지막 temporal 검증에 실제 사용한 프레임을 저장하고 참조를 해제한다."""
        frames = self._last_verified_frames
        self._last_verified_frames = []
        if not frames:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_video(path, frames)
        return len(frames)

    def discard_last_verified_clip(self) -> None:
        """클립 저장을 사용하지 않을 때 검증 프레임 참조를 해제한다."""
        self._last_verified_frames = []

    def _missing_dependency(self) -> Optional[str]:
        candidates = [
            self.config.mediapipe_python,
            self.config.model_python,
            EXTRACT_SCRIPT,
            SMOKE_SCRIPT,
            self.config.model_path,
        ]
        for candidate in candidates:
            if not candidate.exists():
                return str(candidate)
        if self.config.compare_model_kind == "yolo_pose_rf":
            for candidate in (
                YOLO_RF_SMOKE_SCRIPT,
                self.config.temporal_python,
                self.config.temporal_pose_model_path,
                self.config.compare_model_path,
            ):
                if candidate is not None and not candidate.exists():
                    return str(candidate)
        return None

    def _write_video(self, path: Path, frames: list[np.ndarray]) -> None:
        import cv2

        first = frames[0]
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {path}")
        try:
            for frame in frames[-self.config.buffer_frames :]:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        subprocess_env = os.environ.copy()
        subprocess_env["PYTHONPATH"] = str(PROJECT_ROOT)
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            cwd=str(PROJECT_ROOT),
            env=subprocess_env,
        )

    @staticmethod
    def _parse_nonzero_frames(output: str) -> int:
        match = re.search(r"nonzero_feature_frames:\s*(\d+)", output)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _parse_prediction(output: str) -> Optional[int]:
        match = re.search(r"prediction:\s*\[([0-9]+)\]", output)
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_probability(output: str) -> Optional[list[float]]:
        match = re.search(r"predict_proba:\s*\[\[([^\]]+)\]\]", output)
        if not match:
            return None
        return [float(part.strip()) for part in match.group(1).split(",")]

    @staticmethod
    def _parse_named_float(output: str, name: str) -> Optional[float]:
        match = re.search(rf"{re.escape(name)}:\s*([-+]?[0-9]*\.?[0-9]+)", output)
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_named_int(output: str, name: str) -> Optional[int]:
        match = re.search(rf"{re.escape(name)}:\s*(\d+)", output)
        return int(match.group(1)) if match else None
