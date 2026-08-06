#!/usr/bin/env python3
"""Save sampled pose-selection frames for camera/debug review."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from scripts.datasets.train_yolo_pose_fall_rf import (
    _load_pose_model,
    _select_tracked_pose_index,
)
from src.core.ai._fall_detector import FallDetector


def _build_contact_sheet(
    frames: list[np.ndarray],
    *,
    columns: int,
    tile_height: int,
) -> np.ndarray:
    tile_h = min(tile_height, frames[0].shape[0])
    tile_w = int(frames[0].shape[1] * tile_h / frames[0].shape[0])
    tiles = [cv2.resize(frame, (tile_w, tile_h)) for frame in frames]
    blank_tile = np.zeros_like(tiles[0])
    rows: list[np.ndarray] = []
    for index in range(0, len(tiles), columns):
        row_tiles = tiles[index : index + columns]
        row_tiles.extend(blank_tile.copy() for _ in range(columns - len(row_tiles)))
        rows.append(cv2.hconcat(row_tiles))
    return cv2.vconcat(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    args = parser.parse_args()

    model = _load_pose_model(args.pose_model)
    detector = FallDetector()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = [int(v) for v in np.linspace(0, max(total - 1, 0), args.max_frames)]
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    results = model.predict(
        frames,
        imgsz=args.imgsz,
        conf=args.confidence_threshold,
        device=0,
        verbose=False,
    )
    previous_center: tuple[float, float] | None = None
    for frame, result in zip(frames, results):
        selected = None
        if result.boxes is not None and result.keypoints is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
            confidences = result.boxes.conf.detach().cpu().numpy()
            centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
            selected = _select_tracked_pose_index(
                confidences=confidences,
                centers=centers,
                previous_center=previous_center,
                frame_diagonal=float(np.hypot(frame.shape[1], frame.shape[0])),
            )
            previous_center = tuple(float(v) for v in centers[selected])
            box = boxes[selected]
            keypoints = result.keypoints.data[selected].detach().cpu().numpy()
            score = detector._score_fall(keypoints.astype(np.float32), int(box[2] - box[0]), int(box[3] - box[1]))
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 220, 0), 2)
            cv2.putText(frame, f"selected conf={confidences[selected]:.2f} fall={score.score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            for x, y, confidence in keypoints:
                if confidence >= 0.25:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 180, 255), -1)
        else:
            cv2.putText(frame, "no pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if not frames:
        raise RuntimeError("no frames decoded")
    sheet = _build_contact_sheet(frames, columns=3, tile_height=360)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
