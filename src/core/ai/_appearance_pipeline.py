"""외형 분석 파이프라인 전담 모듈."""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..events import DetectionEvent, EventType
from ._appearance_analyzer import AppearanceAnalyzer

if TYPE_CHECKING:
    from ...services.appearance_log import AppearanceLog

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))


class AppearancePipeline:
    """외형 속성 추출, 로그 저장, 조건 매칭을 담당한다."""

    def __init__(
        self,
        appearance: AppearanceAnalyzer,
        crop_dir: Path,
        *,
        save_crops: bool = False,
        crop_context_ratio: Optional[float] = None,
        color_smoothing_window: Optional[int] = None,
        color_min_samples: Optional[int] = None,
        bool_smoothing_window: Optional[int] = None,
        bool_min_samples: Optional[int] = None,
        bool_true_ratio: Optional[float] = None,
    ) -> None:
        self._appearance = appearance
        self._crop_dir = crop_dir
        self._save_crops = save_crops
        self._crop_context_ratio = max(0.0, float(
            crop_context_ratio
            if crop_context_ratio is not None
            else os.environ.get("APPEARANCE_CROP_CONTEXT_RATIO", "0.6")
        ))
        if self._save_crops:
            self._crop_dir.mkdir(parents=True, exist_ok=True)
        self._appearance_cooldown: Dict[str, float] = {}
        self._appearance_log: Optional["AppearanceLog"] = None
        self._color_smoothing_window = max(1, int(
            color_smoothing_window
            if color_smoothing_window is not None
            else os.environ.get("APPEARANCE_COLOR_SMOOTHING_WINDOW", "12")
        ))
        self._color_min_samples = max(1, int(
            color_min_samples
            if color_min_samples is not None
            else os.environ.get("APPEARANCE_COLOR_MIN_SAMPLES", "3")
        ))
        self._gender_min_samples = max(1, int(
            os.environ.get("APPEARANCE_GENDER_MIN_SAMPLES", "3")
        ))
        self._bool_smoothing_window = max(1, int(
            bool_smoothing_window
            if bool_smoothing_window is not None
            else os.environ.get("APPEARANCE_BOOL_SMOOTHING_WINDOW", "8")
        ))
        self._bool_min_samples = max(1, int(
            bool_min_samples
            if bool_min_samples is not None
            else os.environ.get("APPEARANCE_BOOL_MIN_SAMPLES", "3")
        ))
        self._bool_true_ratio = min(1.0, max(0.0, float(
            bool_true_ratio
            if bool_true_ratio is not None
            else os.environ.get("APPEARANCE_BOOL_TRUE_RATIO", "0.6")
        )))
        self._attribute_history: Dict[str, Dict[str, Deque]] = {}
        self._representative_crop_paths: Dict[str, str] = {}

    def ensure_log(self) -> None:
        """AppearanceLog를 지연 초기화한다."""
        if self._appearance_log is None:
            try:
                from ...services.appearance_log import AppearanceLog

                self._appearance_log = AppearanceLog()
            except Exception:
                logger.warning("AppearanceLog 초기화 실패 – DB 기록 비활성화")
                self._appearance_log = False  # type: ignore[assignment]

    def save_person_crop(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        camera_id: str,
        track_id: Optional[int],
        ts: float,
    ) -> Optional[str]:
        """person bbox 영역을 JPEG로 저장하고 경로를 반환한다."""
        if not self._save_crops:
            return None

        try:
            frame_h, frame_w = frame.shape[:2]
            x1, y1, x2, y2 = self._context_crop_bounds(frame_w, frame_h, x, y, w, h)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            safe_camera_id = "".join(
                ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(camera_id)
            )
            kst_timestamp = datetime.fromtimestamp(ts, tz=KST)
            timestamp_text = kst_timestamp.strftime("%Y%m%d_%H%M%S")
            milliseconds = int(ts * 1000) % 1000
            file_name = (
                f"{safe_camera_id}_{track_id}_{timestamp_text}_{milliseconds:03d}.jpg"
            )
            file_path = self._crop_dir / file_name
            self._crop_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(file_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 80]):
                logger.warning("person crop 저장 실패: %s", file_path)
                return None
            return str(file_path)
        except Exception:
            logger.debug("person crop 저장 실패", exc_info=True)
            return None

    def _context_crop_bounds(
        self,
        frame_w: int,
        frame_h: int,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Tuple[int, int, int, int]:
        """검수 이미지 저장용으로 bbox 주변 맥락을 포함한 crop 좌표를 계산한다."""
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        if self._crop_context_ratio > 0.0 and w > 0 and h > 0:
            pad_x = int(w * self._crop_context_ratio)
            pad_y = int(h * self._crop_context_ratio)
            x1 -= pad_x
            x2 += pad_x
            y1 -= pad_y
            y2 += pad_y
        return (
            max(0, x1),
            max(0, y1),
            min(frame_w, x2),
            min(frame_h, y2),
        )

    def _build_face_meta_map(
        self, face_events: List[DetectionEvent]
    ) -> Dict[int, Dict]:
        face_meta_map: Dict[int, Dict] = {}
        for face_event in face_events:
            if face_event.metadata and face_event.object_id is not None:
                face_meta_map[face_event.object_id] = face_event.metadata
        return face_meta_map

    @staticmethod
    def _build_log_parts(
        person: DetectionEvent,
        attrs: Dict,
        face_meta: Dict,
    ) -> List[str]:
        """운영 로그에 남길 외형 요약 문자열을 만든다."""
        gender = face_meta.get("gender") or attrs.get("gender")
        age_group = face_meta.get("age_group") or attrs.get("age_group")
        face_name = face_meta.get("face_name")

        parts = [f"track={person.object_id}"]
        if face_name and str(face_name).strip().lower() != "unknown":
            parts.append(f"이름={face_name}")
        if attrs.get("upper_color") not in (None, "unknown"):
            parts.append(f"상의={attrs['upper_color']}")
        if attrs.get("lower_color") not in (None, "unknown"):
            parts.append(f"하의={attrs['lower_color']}")
        parts.append(f"헬멧={bool(attrs.get('has_helmet'))}")
        if attrs.get("helmet_color") not in (None, "unknown"):
            parts.append(f"헬멧색={attrs['helmet_color']}")
        parts.extend([
            f"백팩={attrs.get('has_backpack', False)}",
            f"핸드백={attrs.get('has_handbag', False)}",
            f"캐리어={attrs.get('has_suitcase', False)}",
            f"성별={gender or '?'}",
            f"나이={age_group or '?'}",
        ])
        scores = attrs.get("attribute_scores")
        if isinstance(scores, dict) and scores.get("has_backpack") is not None:
            try:
                parts.append(f"백팩점수={float(scores['has_backpack']):.3f}")
            except (TypeError, ValueError):
                pass
        return parts

    @staticmethod
    def _coerce_positive_int(value: object) -> Optional[int]:
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def _frame_scale_for_person(
        self,
        frame: np.ndarray,
        person: DetectionEvent,
    ) -> Tuple[float, float]:
        metadata = person.metadata or {}
        ref_w = self._coerce_positive_int(metadata.get("frame_width"))
        ref_h = self._coerce_positive_int(metadata.get("frame_height"))
        if not ref_w or not ref_h:
            return 1.0, 1.0

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return 1.0, 1.0
        if abs(ref_w - frame_w) <= 1 and abs(ref_h - frame_h) <= 1:
            return 1.0, 1.0
        return frame_w / float(ref_w), frame_h / float(ref_h)

    @staticmethod
    def _scale_keypoints(
        keypoints: Optional[list],
        scale_x: float,
        scale_y: float,
    ) -> Optional[list]:
        if not keypoints or (scale_x == 1.0 and scale_y == 1.0):
            return keypoints
        scaled = []
        for point in keypoints:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                scaled.append(point)
                continue
            copied = list(point)
            try:
                copied[0] = float(copied[0]) * scale_x
                copied[1] = float(copied[1]) * scale_y
            except (TypeError, ValueError):
                pass
            scaled.append(copied)
        return scaled

    @staticmethod
    def _scale_nearby_objects(
        nearby_objects: List[Dict],
        scale_x: float,
        scale_y: float,
    ) -> List[Dict]:
        if scale_x == 1.0 and scale_y == 1.0:
            return nearby_objects
        scaled_objects: List[Dict] = []
        for obj in nearby_objects:
            copied = dict(obj)
            for key, scale in (("x", scale_x), ("width", scale_x), ("y", scale_y), ("height", scale_y)):
                if copied.get(key) is None:
                    continue
                try:
                    copied[key] = int(round(float(copied[key]) * scale))
                except (TypeError, ValueError):
                    pass
            scaled_objects.append(copied)
        return scaled_objects

    def _scaled_person_inputs(
        self,
        frame: np.ndarray,
        person: DetectionEvent,
        nearby_objects: List[Dict],
    ) -> Tuple[int, int, int, int, Optional[list], List[Dict]]:
        scale_x, scale_y = self._frame_scale_for_person(frame, person)
        if scale_x == 1.0 and scale_y == 1.0:
            return person.x, person.y, person.width, person.height, person.keypoints, nearby_objects
        return (
            int(round(person.x * scale_x)),
            int(round(person.y * scale_y)),
            max(1, int(round(person.width * scale_x))),
            max(1, int(round(person.height * scale_y))),
            self._scale_keypoints(person.keypoints, scale_x, scale_y),
            self._scale_nearby_objects(nearby_objects, scale_x, scale_y),
        )

    def _extract_person_attributes(
        self,
        frame: np.ndarray,
        person: DetectionEvent,
        nearby_objects: List[Dict],
    ) -> Dict:
        """사람 1명에 대한 외형 속성을 추출한다."""
        x, y, width, height, keypoints, scaled_nearby = self._scaled_person_inputs(
            frame,
            person,
            nearby_objects,
        )
        attrs = self._appearance.extract_attributes(
            frame,
            x,
            y,
            width,
            height,
            nearby_objects=scaled_nearby,
            keypoints=keypoints,
        )
        return self._merge_person_metadata_attributes(attrs, person)

    @staticmethod
    def _merge_person_metadata_attributes(attrs: Dict, person: DetectionEvent) -> Dict:
        """DeepStream SGIE가 person metadata에 붙인 외형 속성을 병합한다."""
        metadata = person.metadata if isinstance(person.metadata, dict) else {}
        sgie_attrs = metadata.get("appearance")
        if not isinstance(sgie_attrs, dict):
            return attrs

        merged = dict(attrs)
        attribute_metadata = dict(merged.get("attribute_metadata") or {})
        color_sources = dict(attribute_metadata.get("color_sources") or {})
        color_candidates = dict(attribute_metadata.get("color_candidates") or {})
        backend_name = str(metadata.get("appearance_backend") or "sgie")
        for key, value in sgie_attrs.items():
            if value in (None, "", "unknown"):
                continue
            merged[key] = value
            if key in ("upper_color", "lower_color", "helmet_color"):
                color_sources[key] = backend_name
                candidate = dict(color_candidates.get(key) or {})
                candidate["selected"] = value
                candidate["source"] = backend_name
                scores = sgie_attrs.get("attribute_scores")
                if isinstance(scores, dict) and scores.get(key) is not None:
                    candidate["confidence"] = scores[key]
                color_candidates[key] = candidate
        if metadata.get("appearance_backend"):
            merged["attribute_backend"] = metadata["appearance_backend"]
        if color_sources:
            attribute_metadata["color_sources"] = color_sources
        if color_candidates:
            attribute_metadata["color_candidates"] = color_candidates
        if attribute_metadata:
            merged["attribute_metadata"] = attribute_metadata
        return merged

    @staticmethod
    def _is_known_color(value: object) -> bool:
        return bool(value) and str(value).strip().lower() != "unknown"

    @staticmethod
    def _is_known_gender(value: object) -> bool:
        return str(value).strip().lower() in {"male", "female"}

    @staticmethod
    def _coerce_bool_observation(value: object) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
        return None

    @staticmethod
    def _track_history_key(camera_id: str, person: DetectionEvent) -> Optional[str]:
        if person.object_id is None:
            return None
        return f"{camera_id}:{int(person.object_id)}"

    @staticmethod
    def _majority_value(samples: Deque[str], current: object) -> str:
        counts = Counter(samples)
        if not counts:
            return str(current)
        highest = max(counts.values())
        candidates = {value for value, count in counts.items() if count == highest}
        current_value = str(current) if current else "unknown"
        if current_value in candidates:
            return current_value
        for value in reversed(samples):
            if value in candidates:
                return value
        return next(iter(candidates))

    def _smooth_track_attributes(
        self,
        camera_id: str,
        person: DetectionEvent,
        attrs: Dict,
    ) -> Dict:
        key = self._track_history_key(camera_id, person)
        if key is None:
            return dict(attrs)

        smoothed = dict(attrs)
        history = self._attribute_history.setdefault(key, {})
        observation_counts: Dict[str, int] = {}
        metadata = dict(attrs.get("attribute_metadata") or {})
        color_sources = dict(metadata.get("color_sources") or {})
        for field in ("upper_color", "lower_color", "helmet_color"):
            value = attrs.get(field)
            samples = history.setdefault(field, deque(maxlen=self._color_smoothing_window))
            if color_sources.get(field) == "not_visible":
                samples.clear()
                smoothed[field] = "unknown"
                observation_counts[field] = 0
                continue
            # 하의는 어두운 청바지에서 HSV/LAB가 black으로 치우치기 쉽다.
            # 색상 모델이 높은 확률로 명확한 유채색을 낸 경우 기존 black
            # 히스토리의 다수결이 새 관측을 영구히 덮어쓰지 않도록 한다.
            candidate = (metadata.get("color_candidates") or {}).get(field, {})
            model_confidence = candidate.get("model_confidence")
            model_color = candidate.get("model_color")
            if (
                field == "lower_color"
                and color_sources.get(field) == "color_yolov8n"
                and self._is_known_color(value)
                and model_color == value
                and str(value) not in {"black", "gray", "white"}
                and isinstance(model_confidence, (int, float))
                and float(model_confidence) >= 0.9
            ):
                samples.clear()
            if self._is_known_color(value):
                samples.append(str(value))
            observation_counts[field] = len(samples)
            if len(samples) >= self._color_min_samples:
                smoothed[field] = self._majority_value(samples, value)

        gender_samples = history.setdefault("gender", deque(maxlen=self._color_smoothing_window))
        gender = attrs.get("gender")
        if self._is_known_gender(gender):
            gender_samples.append(str(gender))
        if len(gender_samples) >= self._gender_min_samples:
            smoothed["gender"] = self._majority_value(gender_samples, gender)
        elif "gender" in smoothed:
            smoothed["gender"] = "unknown"

        bool_observation_counts: Dict[str, int] = {}
        bool_true_ratios: Dict[str, float] = {}
        for field in ("has_backpack", "has_handbag", "has_suitcase", "has_helmet"):
            samples = history.setdefault(field, deque(maxlen=self._bool_smoothing_window))
            observed = self._coerce_bool_observation(attrs.get(field))
            if observed is not None:
                samples.append(observed)
            bool_observation_counts[field] = len(samples)
            true_ratio = (sum(1 for value in samples if value) / float(len(samples))) if samples else 0.0
            bool_true_ratios[field] = round(true_ratio, 3)
            if len(samples) >= self._bool_min_samples:
                smoothed[field] = true_ratio >= self._bool_true_ratio
            elif field in smoothed:
                smoothed[field] = False

        metadata = dict(smoothed.get("attribute_metadata") or {})
        metadata["color_observations"] = observation_counts
        metadata["color_smoothing_window"] = self._color_smoothing_window
        metadata["gender_observations"] = len(gender_samples)
        metadata["gender_min_samples"] = self._gender_min_samples
        metadata["boolean_observations"] = bool_observation_counts
        metadata["boolean_true_ratios"] = bool_true_ratios
        metadata["boolean_smoothing_window"] = self._bool_smoothing_window
        metadata["boolean_min_samples"] = self._bool_min_samples
        metadata["boolean_true_ratio"] = self._bool_true_ratio
        smoothed["attribute_metadata"] = metadata
        return smoothed

    def _prune_attribute_history(self, active_keys: set[str]) -> None:
        if not active_keys:
            return
        max_entries = max(64, max(self._color_smoothing_window, self._bool_smoothing_window) * 32)
        if len(self._attribute_history) <= max_entries:
            return
        for key in list(self._attribute_history):
            if key not in active_keys:
                self._attribute_history.pop(key, None)
                self._representative_crop_paths.pop(key, None)
                if len(self._attribute_history) <= max_entries:
                    break

    def _build_log_payload(
        self,
        *,
        camera_id: str,
        person: DetectionEvent,
        attrs: Dict,
        face_meta: Dict,
        crop_path: Optional[str],
        timestamp: float,
    ) -> Dict:
        """AppearanceLog.insert()에 넘길 payload를 만든다."""
        return {
            "camera_id": camera_id,
            "event_id": f"appearance:{camera_id}:{person.object_id}:{int(timestamp * 1000)}",
            "track_id": person.object_id,
            "upper_color": attrs.get("upper_color"),
            "lower_color": attrs.get("lower_color"),
            "has_helmet": bool(attrs.get("has_helmet")),
            "helmet_color": attrs.get("helmet_color"),
            "has_backpack": bool(attrs.get("has_backpack")),
            "has_handbag": bool(attrs.get("has_handbag")),
            "has_suitcase": bool(attrs.get("has_suitcase")),
            "gender": face_meta.get("gender") or attrs.get("gender"),
            "age_group": face_meta.get("age_group") or attrs.get("age_group"),
            "face_name": face_meta.get("face_name"),
            "attribute_backend": attrs.get("attribute_backend"),
            "attribute_metadata": attrs.get("attribute_metadata"),
            "crop_path": crop_path,
            "bbox_x": person.x,
            "bbox_y": person.y,
            "bbox_w": person.width,
            "bbox_h": person.height,
            "timestamp": timestamp,
        }

    def _insert_log_payload(self, payload: Dict) -> None:
        """외형 로그 payload를 DB에 저장한다."""
        if not self._appearance_log:
            return
        self._appearance_log.insert(**payload)

    def log_person_appearance(
        self,
        frame: np.ndarray,
        person: DetectionEvent,
        now: float,
        camera_id: str,
        nearby_objects: List[Dict],
        face_meta: Dict,
        precomputed_attrs: Optional[Dict] = None,
    ) -> None:
        """외형 속성을 추출하고 로그/DB/crop 저장까지 처리한다."""
        log_key = f"_applog_{person.object_id}"
        last_log = self._appearance_cooldown.get(log_key, 0.0)
        if now - last_log < 3.0:
            return

        attrs = (
            dict(precomputed_attrs)
            if precomputed_attrs
            else self._extract_person_attributes(frame, person, nearby_objects)
        )
        logger.info("[외형] %s", "  ".join(self._build_log_parts(person, attrs, face_meta)))
        self._appearance_cooldown[log_key] = now

        crop_x, crop_y, crop_w, crop_h, _, _ = self._scaled_person_inputs(
            frame,
            person,
            nearby_objects,
        )
        track_key = self._track_history_key(camera_id, person)
        crop_path = (
            self._representative_crop_paths.get(track_key)
            if track_key is not None
            else None
        )
        if crop_path is None:
            crop_path = self.save_person_crop(
                frame,
                crop_x,
                crop_y,
                crop_w,
                crop_h,
                camera_id,
                person.object_id,
                now,
            )
            if track_key is not None and crop_path is not None:
                self._representative_crop_paths[track_key] = crop_path
        payload = self._build_log_payload(
            camera_id=camera_id,
            person=person,
            attrs=attrs,
            face_meta=face_meta,
            crop_path=crop_path,
            timestamp=now,
        )
        self._insert_log_payload(payload)

    def run_matching(
        self,
        frame: np.ndarray,
        person_events: List[DetectionEvent],
        camera_id: Optional[str] = None,
        cooldown: float = 5.0,
        nearby_objects: Optional[List[Dict]] = None,
        precomputed_attributes: Optional[Dict[int, Dict]] = None,
    ) -> List[DetectionEvent]:
        """외형 조건 매칭 이벤트를 생성한다."""
        if frame is None or not person_events:
            return []

        appearance_events: List[DetectionEvent] = []
        now = time.time()

        for person in person_events:
            precomputed = (
                precomputed_attributes.get(int(person.object_id))
                if precomputed_attributes and person.object_id is not None
                else None
            )
            if precomputed:
                matches = self._find_matches_from_attributes(precomputed, camera_id)
            else:
                attrs = self._extract_person_attributes(frame, person, nearby_objects or [])
                matches = self._find_matches_from_attributes(attrs, camera_id)
            for match in matches:
                cooldown_key = f"{person.object_id}:{match['condition_id']}"
                last_ts = self._appearance_cooldown.get(cooldown_key, 0.0)
                if now - last_ts < cooldown:
                    continue
                self._appearance_cooldown[cooldown_key] = now

                attributes = {
                    "upper_color": match["attributes"].get("upper_color"),
                    "lower_color": match["attributes"].get("lower_color"),
                    "has_helmet": bool(match["attributes"].get("has_helmet")),
                    "helmet_color": match["attributes"].get("helmet_color", "unknown"),
                    "has_backpack": match["attributes"].get("has_backpack", False),
                    "has_handbag": match["attributes"].get("has_handbag", False),
                    "has_suitcase": match["attributes"].get("has_suitcase", False),
                    "gender": match["attributes"].get("gender"),
                    "age_group": match["attributes"].get("age_group"),
                    "attribute_backend": match["attributes"].get("attribute_backend"),
                }
                attributes = {
                    key: value for key, value in attributes.items() if value is not None
                }
                metadata = {
                    "condition_id": match["condition_id"],
                    "condition_name": match["condition_name"],
                    **attributes,
                    "attributes": attributes,
                    "match_score": match["score"],
                }

                appearance_events.append(
                    DetectionEvent(
                        event_type=EventType.APPEARANCE_MATCH,
                        x=person.x,
                        y=person.y,
                        width=person.width,
                        height=person.height,
                        confidence=match["score"],
                        timestamp=now,
                        object_id=person.object_id,
                        class_name="person",
                        metadata=metadata,
                    )
                )

        stale = [
            key for key, timestamp in self._appearance_cooldown.items()
            if now - timestamp > cooldown * 5
        ]
        for key in stale:
            self._appearance_cooldown.pop(key, None)

        if appearance_events:
            logger.info("외형 매칭: %d건 발생", len(appearance_events))
        return appearance_events

    def _find_matches_from_attributes(
        self,
        attrs: Dict,
        camera_id: Optional[str],
    ) -> List[Dict]:
        """사전 계산된 속성으로 외형 조건 매칭 결과를 만든다."""
        matches: List[Dict] = []
        for condition in self._appearance.get_enabled_conditions(camera_id):
            score = self._appearance.match_conditions(attrs, condition)
            if score >= float(condition.get("threshold", 0.8)):
                matches.append({
                    "condition_id": condition["id"],
                    "condition_name": condition.get("name", ""),
                    "score": score,
                    "attributes": attrs,
                })
        return matches

    def run(
        self,
        frame: np.ndarray,
        person_events: List[DetectionEvent],
        face_events: List[DetectionEvent],
        *,
        camera_id: Optional[str],
        use_appearance: bool,
        nearby_objects: Optional[List[Dict]] = None,
        precomputed_attributes: Optional[Dict[int, Dict]] = None,
    ) -> List[DetectionEvent]:
        """외형 속성 추출, 로그 저장, 조건 매칭을 순서대로 수행한다."""
        if not use_appearance or not person_events:
            return []

        resolved_camera_id = camera_id or "unknown"
        now = time.time()
        nearby = nearby_objects or []
        self.ensure_log()
        face_meta_map = self._build_face_meta_map(face_events)
        smoothed_attributes: Dict[int, Dict] = {}
        active_history_keys: set[str] = set()

        for person in person_events:
            precomputed = (
                precomputed_attributes.get(int(person.object_id))
                if precomputed_attributes and person.object_id is not None
                else None
            )
            raw_attrs = (
                dict(precomputed)
                if precomputed
                else self._extract_person_attributes(frame, person, nearby)
            )
            attrs = self._smooth_track_attributes(resolved_camera_id, person, raw_attrs)
            history_key = self._track_history_key(resolved_camera_id, person)
            if history_key:
                active_history_keys.add(history_key)
            if person.object_id is not None:
                smoothed_attributes[int(person.object_id)] = attrs
            self.log_person_appearance(
                frame,
                person,
                now,
                resolved_camera_id,
                nearby,
                face_meta_map.get(person.object_id, {}),
                precomputed_attrs=attrs,
            )

        self._prune_attribute_history(active_history_keys)

        if not self._appearance.conditions:
            return []

        return self.run_matching(
            frame,
            person_events,
            camera_id=resolved_camera_id,
            nearby_objects=nearby,
            precomputed_attributes=smoothed_attributes,
        )
