#!/usr/bin/env python3
"""Train a lightweight temporal fall model from DeepStream pose captures."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ai.fall_temporal_model import (  # noqa: E402
    FRAME_FEATURE_NAMES,
    encode_frame_sequence,
)


@dataclass(frozen=True)
class TemporalCaptureDataset:
    sequences: np.ndarray
    labels: np.ndarray
    scene_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def load_temporal_capture_datasets(
    paths: Sequence[Path],
    *,
    sequence_length: int = 48,
) -> TemporalCaptureDataset:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    scene_ids: list[str] = []
    group_ids: list[str] = []
    metadata: list[dict[str, Any]] = []

    for path in paths:
        for row_index, row in enumerate(_read_jsonl(path)):
            row_name = f"{path}: row {row_index}"
            if int(row.get("schema_version") or 0) != 2:
                raise ValueError(f"{row_name}: schema_version must be 2")
            if row.get("runtime") != "deepstream_pose_inline":
                raise ValueError(f"{row_name}: unexpected runtime")
            if tuple(row.get("frame_feature_names") or ()) != FRAME_FEATURE_NAMES:
                raise ValueError(f"{row_name}: frame feature order mismatch")

            frame_records = row.get("frame_records")
            if not isinstance(frame_records, list) or not frame_records:
                raise ValueError(f"{row_name}: frame_records must be non-empty")

            label = row.get("label")
            if label not in {0, 1}:
                raise ValueError(f"{row_name}: label must be 0 or 1")
            scene_id = str(row.get("scene_id") or "").strip()
            group_id = str(row.get("group_id") or "").strip()
            if not scene_id:
                raise ValueError(f"{row_name}: scene_id is required")
            if not group_id:
                raise ValueError(f"{row_name}: group_id is required")

            sequence = encode_frame_sequence(
                frame_records,
                sequence_length=sequence_length,
            )
            if not np.isfinite(sequence).all():
                raise ValueError(f"{row_name}: non-finite sequence")

            sequences.append(sequence)
            labels.append(int(label))
            scene_ids.append(scene_id)
            group_ids.append(group_id)
            metadata.append(dict(row))

    if not sequences:
        raise ValueError("no temporal capture rows found")
    if set(labels) != {0, 1}:
        raise ValueError("dataset must contain both fall and non-fall rows")

    return TemporalCaptureDataset(
        sequences=np.stack(sequences).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scene_ids=tuple(scene_ids),
        group_ids=tuple(group_ids),
        metadata=tuple(metadata),
    )


def assert_validation_disjoint(
    training: TemporalCaptureDataset,
    validation: TemporalCaptureDataset,
) -> None:
    scene_overlap = sorted(set(training.scene_ids) & set(validation.scene_ids))
    if scene_overlap:
        raise ValueError(f"scene overlap: {', '.join(scene_overlap)}")

    group_overlap = sorted(set(training.group_ids) & set(validation.group_ids))
    if group_overlap:
        raise ValueError(f"group overlap: {', '.join(group_overlap)}")
