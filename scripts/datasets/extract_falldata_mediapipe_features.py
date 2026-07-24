#!/usr/bin/env python3
"""Extract falldata-compatible MediaPipe Holistic features from a video.

The public fall-data video RandomForest models expect one sample shaped as
600 frames x 1662 features, flattened to 997200 features. The 1662 frame vector
comes from MediaPipe Holistic:

- pose: 33 landmarks x (x, y, z, visibility) = 132
- face: 468 landmarks x (x, y, z) = 1404
- left hand: 21 landmarks x (x, y, z) = 63
- right hand: 21 landmarks x (x, y, z) = 63
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FRAME_FEATURES = 1662
TARGET_FRAMES = 600


def _extract_keypoints(results: object) -> np.ndarray:
    pose_landmarks = getattr(results, "pose_landmarks", None)
    face_landmarks = getattr(results, "face_landmarks", None)
    left_hand_landmarks = getattr(results, "left_hand_landmarks", None)
    right_hand_landmarks = getattr(results, "right_hand_landmarks", None)

    pose = (
        np.array(
            [[res.x, res.y, res.z, res.visibility] for res in pose_landmarks.landmark],
            dtype=np.float32,
        ).reshape(-1)
        if pose_landmarks
        else np.zeros(33 * 4, dtype=np.float32)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in face_landmarks.landmark],
            dtype=np.float32,
        ).reshape(-1)
        if face_landmarks
        else np.zeros(468 * 3, dtype=np.float32)
    )
    left_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in left_hand_landmarks.landmark],
            dtype=np.float32,
        ).reshape(-1)
        if left_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )
    right_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in right_hand_landmarks.landmark],
            dtype=np.float32,
        ).reshape(-1)
        if right_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )
    keypoints = np.concatenate([pose, face, left_hand, right_hand]).astype(np.float32)
    if keypoints.shape != (FRAME_FEATURES,):
        raise RuntimeError(f"unexpected keypoint shape: {keypoints.shape}")
    return keypoints


def _pad_or_trim(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("no frames were decoded from the video")
    if len(frames) >= TARGET_FRAMES:
        return np.asarray(frames[:TARGET_FRAMES], dtype=np.float32)

    padded = list(frames)
    zero_frame = np.zeros(FRAME_FEATURES, dtype=np.float32)
    while len(padded) < TARGET_FRAMES:
        padded.append(zero_frame.copy())
    return np.asarray(padded, dtype=np.float32)


def _tail_start_frame(total_frames: int, max_frames: int | None) -> int:
    if max_frames is None or max_frames <= 0 or total_frames <= max_frames:
        return 0
    return total_frames - max_frames


def extract_video_features(
    video_path: Path,
    *,
    max_frames: int | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[np.ndarray, int]:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = _tail_start_frame(total_frames, max_frames)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    mp_holistic = mp.solutions.holistic
    frames: list[np.ndarray] = []
    decoded = 0
    with mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            frames.append(_extract_keypoints(results))
            if max_frames is not None and len(frames) >= max_frames:
                break
    cap.release()

    sequence = _pad_or_trim(frames)
    return sequence, decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where 000.npy ... 599.npy will be written.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Optional limit that extracts the most recent frames. "
            "Output is still padded to 600."
        ),
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.1)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence, decoded = extract_video_features(
        args.video,
        max_frames=args.max_frames,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(sequence):
        np.save(args.output_dir / f"{index}.npy", frame)

    nonzero_frames = int(np.count_nonzero(np.abs(sequence).sum(axis=1) > 0))
    print(f"video: {args.video}")
    print(f"decoded_frames: {decoded}")
    print(f"saved_frames: {sequence.shape[0]}")
    print(f"nonzero_feature_frames: {nonzero_frames}")
    print(f"frame_features: {sequence.shape[1]}")
    print(f"output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
