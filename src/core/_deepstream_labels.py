"""DeepStream 라벨/이벤트 타입 매핑 헬퍼."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .events import EventType

logger = logging.getLogger(__name__)


def event_type_for_label(label: str) -> EventType:
    normalized = (label or "").strip().lower().replace("-", "_")
    if normalized == "person":
        return EventType.PERSON
    if normalized in {"helmet", "hardhat", "head_protected"}:
        return EventType.HELMET
    if normalized in {"head", "hardhat_off", "no_helmet", "helmet_off", "helmet_missing"}:
        return EventType.HEAD
    if normalized in {"fall", "fall_detected"}:
        return EventType.FALL_DETECTED
    return EventType.OTHER


def load_yolo_labels(
    labels_file: Path,
    env_name: str,
    fallback: Optional[List[str]] = None,
) -> List[str]:
    labels: List[str] = []
    if labels_file.exists():
        for line in labels_file.read_text(encoding="utf-8").splitlines():
            label = line.strip()
            if label and not label.startswith("#"):
                labels.append(label)

    if not labels:
        labels = list(fallback or [])
    if not labels:
        labels = [f"class_{idx}" for idx in range(80)]
        labels[0] = "person"

    env_labels = [label.strip() for label in os.environ.get(env_name, "").split(",")]
    env_labels = [label for label in env_labels if label]
    if env_labels:
        labels = env_labels
    return labels


def load_pphuman_label_map(label_map_path: Optional[str]) -> Dict[str, object]:
    candidates = [
        label_map_path,
        os.environ.get("APPEARANCE_LABEL_MAP_PATH"),
        "config/appearance_pphuman_labels.example.json",
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.exists():
            path = (Path.cwd() / str(value)).resolve()
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            logger.warning("PP-Human SGIE 라벨 맵 로드 실패: %s (%s)", path, exc)
    return {"labels": []}


def resolve_pphuman_sgie_backend_name(
    *,
    pphuman_infer_config: Path,
    pphuman_label_map: Dict[str, object],
) -> str:
    candidates = [
        str(pphuman_infer_config),
        os.environ.get("DS_PPHUMAN_INFER_CONFIG", ""),
        os.environ.get("APPEARANCE_LABEL_MAP_PATH", ""),
        str(pphuman_label_map.get("model") or ""),
    ]
    if any("pa100k" in value.lower() for value in candidates):
        return "pa100k_sgie"
    return "pphuman_sgie"
