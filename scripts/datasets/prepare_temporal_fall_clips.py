#!/usr/bin/env python3
"""Create short, annotation-aligned fall clips for temporal model capture."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clip_window_frames(
    row: dict[str, Any], *, margin_frames: int
) -> tuple[int, int]:
    fall_start_frame = int(row.get("fall_start_frame") or 0)
    fall_end_frame = int(row.get("fall_end_frame") or 0)
    scene_length = int(row.get("scene_length") or 0)
    if (
        fall_start_frame <= 0
        or fall_end_frame <= fall_start_frame
        or scene_length <= 0
    ):
        raise ValueError(
            f"{row.get('scene_id', '<unknown>')}: valid fall frame annotation required"
        )

    clip_start_frame = max(0, fall_start_frame - margin_frames)
    clip_end_frame = min(scene_length, fall_end_frame + margin_frames)
    if clip_end_frame <= clip_start_frame:
        raise ValueError(
            f"{row.get('scene_id', '<unknown>')}: empty annotated clip window"
        )
    return clip_start_frame, clip_end_frame


def _clip_window_seconds(
    row: dict[str, Any], *, fps: float, margin_frames: int
) -> tuple[float, float]:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    clip_start_frame, clip_end_frame = _clip_window_frames(
        row, margin_frames=margin_frames
    )
    return (
        clip_start_frame / fps,
        (clip_end_frame - clip_start_frame) / fps,
    )


def _is_fall(row: dict[str, Any]) -> bool:
    value = row.get("is_fall")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "fall"}


def _select_rows_by_positions(
    rows: Sequence[dict[str, Any]],
    *,
    positions: Sequence[str],
    per_position: int,
) -> list[dict[str, Any]]:
    if per_position <= 0:
        raise ValueError("per_position must be greater than zero")

    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for position in positions:
        candidates: list[dict[str, Any]] = []
        for row in rows:
            scene_group = str(row.get("scene_group") or "")
            if (
                _is_fall(row)
                and str(row.get("scene_position") or "") == position
                and scene_group
                and scene_group not in used_groups
            ):
                candidates.append(row)
                used_groups.add(scene_group)
                if len(candidates) == per_position:
                    break
        if len(candidates) != per_position:
            raise ValueError(
                f"{position}: required {per_position}, found {len(candidates)} "
                "unique fall scene groups"
            )
        selected.extend(candidates)
    return selected


def _select_rows_by_scene_ids(
    rows: Sequence[dict[str, Any]], scene_ids: Sequence[str]
) -> list[dict[str, Any]]:
    fall_rows_by_scene = {
        str(row.get("scene_id") or ""): row for row in rows if _is_fall(row)
    }
    missing_scene_ids = [
        scene_id for scene_id in scene_ids if scene_id not in fall_rows_by_scene
    ]
    if missing_scene_ids:
        raise ValueError(f"fall scenes not found: {', '.join(missing_scene_ids)}")
    return [fall_rows_by_scene[scene_id] for scene_id in scene_ids]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _clip_output_path(output_dir: Path, row: dict[str, Any]) -> Path:
    scene_id = str(row.get("scene_id") or "").strip()
    if not scene_id:
        raise ValueError("scene_id is required")
    safe_scene_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in scene_id
    )
    return output_dir / f"{safe_scene_id}.mp4"


def _manifest_video_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


def _resolve_video_backend(preferred: str) -> str:
    if preferred == "ffmpeg":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg backend requested but ffmpeg is unavailable")
        return "ffmpeg"
    if preferred == "opencv":
        return "opencv"
    if preferred == "gstreamer":
        if shutil.which("gst-launch-1.0") is None:
            raise RuntimeError(
                "gstreamer backend requested but gst-launch-1.0 is unavailable"
            )
        return "gstreamer"
    if preferred != "auto":
        raise ValueError(f"unsupported video backend: {preferred}")
    if shutil.which("ffmpeg") is not None:
        return "ffmpeg"
    if shutil.which("gst-launch-1.0") is not None:
        return "gstreamer"
    return "opencv"


def _create_clip_with_ffmpeg(
    source_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    overwrite: bool,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_seconds:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _create_clip_with_opencv(
    source_path: Path,
    output_path: Path,
    *,
    clip_start_frame: int,
    clip_end_frame: int,
    fps: float,
) -> None:
    import cv2

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {source_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid video dimensions: {source_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot create video: {output_path}")
        written_frames = 0
        try:
            for _frame_index in range(clip_start_frame, clip_end_frame):
                success, frame = capture.read()
                if not success:
                    break
                writer.write(frame)
                written_frames += 1
        finally:
            writer.release()
    finally:
        capture.release()

    expected_frames = clip_end_frame - clip_start_frame
    if written_frames != expected_frames:
        raise RuntimeError(
            f"incomplete clip {output_path}: wrote {written_frames}/{expected_frames} frames"
        )


def _create_clip_with_gstreamer(
    source_path: Path,
    output_path: Path,
    *,
    clip_start_frame: int,
    clip_end_frame: int,
    fps: float,
) -> None:
    import cv2

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {source_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"invalid video dimensions: {source_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
    rounded_fps = max(1, round(fps))
    command = [
        "gst-launch-1.0",
        "-q",
        "-e",
        "fdsrc",
        "fd=0",
        "!",
        "rawvideoparse",
        "format=bgr",
        f"width={width}",
        f"height={height}",
        f"framerate={rounded_fps}/1",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=I420",
        "!",
        "x264enc",
        "speed-preset=ultrafast",
        "tune=zerolatency",
        "key-int-max=30",
        "!",
        "video/x-h264,stream-format=avc,alignment=au,profile=baseline",
        "!",
        "mp4mux",
        "faststart=true",
        "!",
        "filesink",
        f"location={output_path}",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        capture.release()
        process.kill()
        raise RuntimeError("failed to open GStreamer input pipe")
    written_frames = 0
    try:
        for _frame_index in range(clip_start_frame, clip_end_frame):
            success, frame = capture.read()
            if not success:
                break
            process.stdin.write(frame.tobytes())
            written_frames += 1
    finally:
        capture.release()
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    expected_frames = clip_end_frame - clip_start_frame
    if written_frames != expected_frames:
        raise RuntimeError(
            f"incomplete clip {output_path}: wrote {written_frames}/{expected_frames} frames"
        )


def _create_clip(
    row: dict[str, Any],
    *,
    output_path: Path,
    fps: float,
    margin_frames: int,
    overwrite: bool,
    video_backend: str,
) -> dict[str, Any]:
    source_path = Path(str(row.get("video_path") or "")).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"{row.get('scene_id', '<unknown>')}: video not found: {source_path}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    clip_start_frame, clip_end_frame = _clip_window_frames(
        row, margin_frames=margin_frames
    )
    start_seconds, duration_seconds = _clip_window_seconds(
        row, fps=fps, margin_frames=margin_frames
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if video_backend == "ffmpeg":
        _create_clip_with_ffmpeg(
            source_path,
            output_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            overwrite=overwrite,
        )
    elif video_backend == "gstreamer":
        _create_clip_with_gstreamer(
            source_path,
            output_path,
            clip_start_frame=clip_start_frame,
            clip_end_frame=clip_end_frame,
            fps=fps,
        )
    else:
        _create_clip_with_opencv(
            source_path,
            output_path,
            clip_start_frame=clip_start_frame,
            clip_end_frame=clip_end_frame,
            fps=fps,
        )

    output_row = dict(row)
    output_row.update(
        {
            "source_video_path": str(source_path.resolve()),
            "video_path": _manifest_video_path(output_path),
            "video_exists": True,
            "source_fall_start_frame": int(row["fall_start_frame"]),
            "source_fall_end_frame": int(row["fall_end_frame"]),
            "clip_source_start_frame": clip_start_frame,
            "clip_source_end_frame": clip_end_frame,
            "fall_start_frame": int(row["fall_start_frame"]) - clip_start_frame,
            "fall_end_frame": int(row["fall_end_frame"]) - clip_start_frame,
            "scene_length": clip_end_frame - clip_start_frame,
            "clip_fps": fps,
            "clip_margin_frames": margin_frames,
        }
    )
    return output_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--position",
        action="append",
        dest="positions",
        default=None,
        help="Scene position to sample; repeat for multiple positions.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        dest="scene_ids",
        default=None,
        help="Exact fall scene to clip; repeat for multiple scenes.",
    )
    parser.add_argument("--per-position", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--margin-frames", type=int, default=30)
    parser.add_argument(
        "--video-backend",
        choices=("auto", "ffmpeg", "gstreamer", "opencv"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.margin_frames < 0:
        raise ValueError("margin_frames must be zero or greater")
    if args.output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output_manifest}")

    video_backend = _resolve_video_backend(args.video_backend)
    input_rows = _read_jsonl(args.input_manifest)
    positions = tuple(args.positions or ("복도", "병실"))
    if args.scene_ids:
        selected_rows = _select_rows_by_scene_ids(input_rows, tuple(args.scene_ids))
    else:
        selected_rows = _select_rows_by_positions(
            input_rows,
            positions=positions,
            per_position=args.per_position,
        )
    output_rows = [
        _create_clip(
            row,
            output_path=_clip_output_path(args.output_dir, row),
            fps=args.fps,
            margin_frames=args.margin_frames,
            overwrite=args.overwrite,
            video_backend=video_backend,
        )
        for row in selected_rows
    ]
    _write_jsonl(args.output_manifest, output_rows)
    print(
        json.dumps(
            {
                "output_manifest": str(args.output_manifest),
                "output_dir": str(args.output_dir),
                "positions": positions if not args.scene_ids else None,
                "scene_ids": args.scene_ids,
                "rows": len(output_rows),
                "video_backend": video_backend,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
