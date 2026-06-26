"""Public falldata RF model auxiliary verifier.

The public model stack has incompatible runtime dependencies:

- MediaPipe feature extraction currently needs a numpy>=2 environment.
- The RandomForest model needs the legacy sklearn 1.3.x environment.

Keep this helper process-isolated and disabled by default.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Deque, Iterable, Optional

import numpy as np

from ..events import DetectionEvent, EventType

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "falldata/2. AI학습모델파일/영상/낙상분류/FNF_RF_SMOTE_CAM_1.pkl"
)
DEFAULT_MEDIAPIPE_PYTHON = PROJECT_ROOT / ".venv-mediapipe/bin/python"
DEFAULT_MODEL_PYTHON = PROJECT_ROOT / ".venv-falldata/bin/python"
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts/datasets/extract_falldata_mediapipe_features.py"
SMOKE_SCRIPT = PROJECT_ROOT / "scripts/datasets/smoke_falldata_video_model.py"


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
    timeout_seconds: float = 30.0
    cooldown_seconds: float = 10.0
    mediapipe_python: Path = DEFAULT_MEDIAPIPE_PYTHON
    model_python: Path = DEFAULT_MODEL_PYTHON
    model_path: Path = DEFAULT_MODEL_PATH

    @classmethod
    def from_env(cls) -> "FallDataAuxConfig":
        mode = os.environ.get("FALLDATA_AUX_MODE", "shadow").strip().lower()
        if mode not in {"shadow", "confirm"}:
            mode = "shadow"
        return cls(
            enabled=_parse_bool(os.environ.get("FALLDATA_AUX_ENABLED"), False),
            mode=mode,
            threshold=_parse_float(os.environ.get("FALLDATA_AUX_THRESHOLD"), 0.7),
            min_nonzero_frames=_parse_int(
                os.environ.get("FALLDATA_AUX_MIN_NONZERO_FRAMES"), 30
            ),
            fall_class_index=_parse_int(os.environ.get("FALLDATA_AUX_FALL_CLASS_INDEX"), 0),
            buffer_frames=_parse_int(os.environ.get("FALLDATA_AUX_BUFFER_FRAMES"), 600),
            timeout_seconds=_parse_float(
                os.environ.get("FALLDATA_AUX_TIMEOUT_SECONDS"), 30.0
            ),
            cooldown_seconds=_parse_float(
                os.environ.get("FALLDATA_AUX_COOLDOWN_SECONDS"), 10.0
            ),
            mediapipe_python=Path(
                os.environ.get("FALLDATA_AUX_MEDIAPIPE_PYTHON", str(DEFAULT_MEDIAPIPE_PYTHON))
            ),
            model_python=Path(
                os.environ.get("FALLDATA_AUX_MODEL_PYTHON", str(DEFAULT_MODEL_PYTHON))
            ),
            model_path=Path(os.environ.get("FALLDATA_AUX_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        )


class FallDataAuxVerifier:
    """Buffers frames and verifies pose fall candidates with the public RF model."""

    def __init__(self, config: Optional[FallDataAuxConfig] = None) -> None:
        self.config = config or FallDataAuxConfig.from_env()
        self._frames: Deque[np.ndarray] = deque(maxlen=max(self.config.buffer_frames, 1))
        self._last_run_at = 0.0
        self._last_result: dict | None = None
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def add_frame(self, frame: np.ndarray) -> None:
        if not self.enabled or frame is None or not isinstance(frame, np.ndarray):
            return
        with self._lock:
            self._frames.append(frame.copy())

    def annotate_events(self, events: Iterable[DetectionEvent]) -> list[DetectionEvent]:
        events = list(events)
        if not self.enabled:
            return events

        annotated: list[DetectionEvent] = []
        for event in events:
            if event.event_type != EventType.FALL_DETECTED:
                annotated.append(event)
                continue

            result = self.verify()
            metadata = dict(event.metadata or {})
            metadata["falldata_aux"] = result
            event.metadata = metadata
            if self.config.mode == "confirm" and not result.get("confirmed", False):
                logger.info("falldata aux confirm mode rejected fall event: %s", result)
                continue
            annotated.append(event)
        return annotated

    def verify(self) -> dict:
        now = time.time()
        if now - self._last_run_at < self.config.cooldown_seconds:
            return self._cooldown_result()
        self._last_run_at = now

        try:
            result = self._verify_once()
            self._last_result = dict(result)
            return result
        except Exception as exc:
            logger.warning("falldata aux verification failed: %s", exc)
            result = self._result(
                "error",
                confirmed=self.config.mode != "confirm",
                error=str(exc),
            )
            self._last_result = dict(result)
            return result

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
        )

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
    ) -> bool:
        if fall_probability is None:
            return False
        return (
            nonzero_frames >= self.config.min_nonzero_frames
            and fall_probability >= self.config.threshold
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
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            cwd=str(PROJECT_ROOT),
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
