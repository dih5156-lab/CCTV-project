"""등록 얼굴 스냅샷 저장 유틸리티."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def save_recognized_face_snapshot(
    *,
    frame: Any,
    camera_name: str,
    face_name: str,
    face_bbox: Dict[str, int],
    confidence: float,
    now: float,
    enabled: bool,
    snapshot_dir: Path,
    cooldown_sec: float,
    last_saved_at: Dict[Tuple[str, str], float],
) -> Optional[str]:
    """등록 얼굴 인식 시 현재 프레임을 증거용 스냅샷으로 저장한다."""
    if not enabled or frame is None:
        return None

    normalized_name = str(face_name or "").strip()
    if not normalized_name or normalized_name.lower() == "unknown":
        return None

    cooldown_key = (camera_name, normalized_name)
    if now - last_saved_at.get(cooldown_key, 0.0) < cooldown_sec:
        return None

    try:
        import cv2
    except ImportError:
        return None

    snapshot = frame.copy()
    x = max(0, int(face_bbox.get("x", 0)))
    y = max(0, int(face_bbox.get("y", 0)))
    width = max(0, int(face_bbox.get("width", 0)))
    height = max(0, int(face_bbox.get("height", 0)))
    if width > 0 and height > 0:
        cv2.rectangle(snapshot, (x, y), (x + width, y + height), (0, 200, 255), 2)
        cv2.putText(
            snapshot,
            f"{normalized_name} {confidence:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    safe_camera = re.sub(r"[^\w\-]", "_", camera_name)
    safe_name = re.sub(r"[^\w\-]", "_", normalized_name)
    timestamp = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S_%f")[:19]
    out_dir = snapshot_dir / safe_camera
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{timestamp}_{safe_name}.jpg"
    if not cv2.imwrite(str(dest), snapshot):
        return None

    last_saved_at[cooldown_key] = now
    return str(dest)
