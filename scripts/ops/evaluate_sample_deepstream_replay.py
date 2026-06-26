#!/usr/bin/env python3
"""Replay Sample fall videos through DeepStream and score shadow logs.

Typical flow on Jetson:

  1. Build the manifest:
     python scripts/datasets/build_sample_fall_manifest.py

  2. Run a quick file-source replay through the Jetson compose stack:
     python scripts/ops/evaluate_sample_deepstream_replay.py \
       --source-mode file --apply-camera-config --restart-ai-engine

In file mode, the script rewrites the evaluation camera to a `file:///app/...`
source and restarts `cctv-ai-engine` for each video. In RTSP mode, it publishes
each mp4 to MediaMTX with ffmpeg. Both modes wait for `fall_shadow_review.jsonl`
records and write per-video TP/FN/FP/TN results.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MANIFEST = Path("data/fall_eval/sample_manifest.jsonl")
DEFAULT_RESULTS_JSONL = Path("data/fall_eval/sample_deepstream_results.jsonl")
DEFAULT_RESULTS_CSV = Path("data/fall_eval/sample_deepstream_results.csv")
DEFAULT_REVIEW_LOG = Path("data/logs/fall_shadow_review.jsonl")
DEFAULT_EVAL_CAMERAS = Path("data/fall_eval/cameras.sample_eval.json")
DEFAULT_CAMERA_ID = "sample_eval"
DEFAULT_HOST_RTSP_URL = "rtsp://localhost:8554/sample_eval"
DEFAULT_CONTAINER_RTSP_URL = "rtsp://cctv-media-server:8554/sample_eval"
DEFAULT_CONTAINER_PROJECT_ROOT = Path("/app")
DEFAULT_COMPOSE_ENV_FILE = Path(".env.jetson")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_new_jsonl_records(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as fp:
        fp.seek(offset)
        data = fp.read().decode("utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _video_duration_seconds(video_path: Path, fallback_frames: int, fps: float) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            proc = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            return max(float(proc.stdout.strip()), 0.1)
        except Exception:
            pass
    if fallback_frames > 0 and fps > 0:
        return fallback_frames / fps
    return 30.0


def _write_eval_cameras(path: Path, camera_id: str, source: str) -> None:
    camera = {
        "id": camera_id,
        "name": "Sample Fall Evaluation",
        "source": source,
        "location": "sample",
        "enabled": True,
        "detections": ["fall", "person"],
        "model_settings": {
            "use_helmet": False,
            "use_pose": True,
            "use_person": False,
            "use_face": False,
            "use_appearance": False,
        },
        "model_paths": {
            "pose": "models/yolov8n-pose.engine",
            "person": "models/yolov8n.engine",
        },
        "zones": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([camera], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _apply_camera_config(eval_cameras: Path, cameras_json: Path) -> Path:
    backup = cameras_json.with_suffix(
        cameras_json.suffix + f".sample_eval_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    shutil.copy2(cameras_json, backup)
    shutil.copy2(eval_cameras, cameras_json)
    return backup


def _restart_ai_engine(compose_file: Path, compose_env_file: Path | None = None) -> None:
    command = ["docker", "compose"]
    if compose_env_file:
        command.extend(["--env-file", str(compose_env_file)])
    command.extend(["-f", str(compose_file), "restart", "cctv-ai-engine"])
    subprocess.run(command, check=True)


def _run_ffmpeg_replay(video_path: Path, rtsp_url: str, duration: float, timeout_grace: float) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install ffmpeg on the Jetson host first.")
    proc = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-re",
            "-i",
            str(video_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-f",
            "rtsp",
            rtsp_url,
        ]
    )
    try:
        return proc.wait(timeout=max(duration + timeout_grace, 5.0))
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            return proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait(timeout=5.0)


def _container_file_uri(video_path: Path, container_project_root: Path) -> str:
    return f"file://{container_project_root / video_path.as_posix()}"


def _score_result(expected_fall: bool, detected: bool) -> str:
    if expected_fall and detected:
        return "TP"
    if expected_fall and not detected:
        return "FN"
    if not expected_fall and detected:
        return "FP"
    return "TN"


def _summarize_shadow_records(records: list[dict[str, Any]], camera_id: str) -> dict[str, Any]:
    camera_records = [row for row in records if str(row.get("camera_id")) == camera_id]
    confirmed_records = [
        row
        for row in camera_records
        if isinstance(row.get("falldata_aux"), dict)
        and row["falldata_aux"].get("status") == "ok"
        and row["falldata_aux"].get("confirmed") is True
    ]
    probabilities = [
        row.get("falldata_aux", {}).get("fall_probability")
        for row in confirmed_records
        if isinstance(row.get("falldata_aux"), dict)
    ]
    numeric_probs = [float(value) for value in probabilities if isinstance(value, (int, float))]
    return {
        "detected": bool(confirmed_records),
        "shadow_record_count": len(camera_records),
        "confirmed_shadow_record_count": len(confirmed_records),
        "max_fall_probability": max(numeric_probs) if numeric_probs else None,
        "last_shadow_status": (
            camera_records[-1].get("falldata_aux", {}).get("status")
            if camera_records and isinstance(camera_records[-1].get("falldata_aux"), dict)
            else None
        ),
    }


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_jsonl(args.manifest)
    if args.label:
        rows = [row for row in rows if row.get("label") == args.label]
    if args.max_videos:
        rows = rows[: args.max_videos]

    initial_source = (
        _container_file_uri(Path(rows[0]["video_path"]), args.container_project_root)
        if args.source_mode == "file" and rows
        else args.container_rtsp_url
    )
    _write_eval_cameras(args.eval_cameras_json, args.camera_id, initial_source)
    print(f"eval cameras: {args.eval_cameras_json}")

    backup: Path | None = None
    if args.prepare_only:
        return []

    if args.apply_camera_config:
        backup = _apply_camera_config(args.eval_cameras_json, args.cameras_json)
        print(f"applied camera config: {args.cameras_json} (backup: {backup})")
        if args.restart_ai_engine:
            _restart_ai_engine(args.compose_file, args.compose_env_file)
            time.sleep(args.restart_wait_seconds)

    results: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(rows, start=1):
            video_path = Path(row["video_path"])
            if not video_path.exists():
                print(f"[{idx}/{len(rows)}] missing video: {video_path}", file=sys.stderr)
                continue

            offset = args.review_log.stat().st_size if args.review_log.exists() else 0
            duration = _video_duration_seconds(
                video_path,
                int(row.get("scene_length") or 0),
                args.assumed_fps,
            )
            print(f"[{idx}/{len(rows)}] replay {row['scene_id']} ({row['label']}, {duration:.1f}s)")
            if args.source_mode == "file":
                _write_eval_cameras(
                    args.eval_cameras_json,
                    args.camera_id,
                    _container_file_uri(video_path, args.container_project_root),
                )
                if args.apply_camera_config:
                    shutil.copy2(args.eval_cameras_json, args.cameras_json)
                if args.restart_ai_engine:
                    _restart_ai_engine(args.compose_file, args.compose_env_file)
                    time.sleep(args.restart_wait_seconds)
                time.sleep(duration + args.shadow_wait_seconds)
            else:
                _run_ffmpeg_replay(video_path, args.host_rtsp_url, duration, args.timeout_grace_seconds)
                time.sleep(args.shadow_wait_seconds)
            new_records = _read_new_jsonl_records(args.review_log, offset)
            shadow = _summarize_shadow_records(new_records, args.camera_id)
            expected_fall = bool(row.get("is_fall"))
            result = {
                "scene_id": row.get("scene_id"),
                "video_path": row.get("video_path"),
                "label": row.get("label"),
                "expected_fall": expected_fall,
                "detected": shadow["detected"],
                "result": _score_result(expected_fall, shadow["detected"]),
                "fall_start_frame": row.get("fall_start_frame"),
                "fall_end_frame": row.get("fall_end_frame"),
                "scene_length": row.get("scene_length"),
                "camera": row.get("camera"),
                "shadow_record_count": shadow["shadow_record_count"],
                "confirmed_shadow_record_count": shadow["confirmed_shadow_record_count"],
                "max_fall_probability": shadow["max_fall_probability"],
                "last_shadow_status": shadow["last_shadow_status"],
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append(result)
            _write_jsonl(results, args.results_jsonl)
            _write_csv(results, args.results_csv)
            print(
                "  -> {result} detected={detected} max_prob={prob}".format(
                    result=result["result"],
                    detected=result["detected"],
                    prob=result["max_fall_probability"],
                )
            )
    finally:
        if backup and args.restore_camera_config:
            shutil.copy2(backup, args.cameras_json)
            print(f"restored camera config from: {backup}")
            if args.restart_ai_engine:
                _restart_ai_engine(args.compose_file, args.compose_env_file)

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    counts = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    for row in results:
        counts[row["result"]] += 1
    total = sum(counts.values())
    precision = counts["TP"] / max(counts["TP"] + counts["FP"], 1)
    recall = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    print(f"total: {total}")
    print(f"TP: {counts['TP']} FN: {counts['FN']} FP: {counts['FP']} TN: {counts['TN']}")
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review-log", type=Path, default=DEFAULT_REVIEW_LOG)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS_JSONL)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--eval-cameras-json", type=Path, default=DEFAULT_EVAL_CAMERAS)
    parser.add_argument("--cameras-json", type=Path, default=Path("cameras.json"))
    parser.add_argument("--compose-file", type=Path, default=Path("docker-compose.jetson.yml"))
    parser.add_argument("--compose-env-file", type=Path, default=DEFAULT_COMPOSE_ENV_FILE)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--host-rtsp-url", default=DEFAULT_HOST_RTSP_URL)
    parser.add_argument("--container-rtsp-url", default=DEFAULT_CONTAINER_RTSP_URL)
    parser.add_argument("--container-project-root", type=Path, default=DEFAULT_CONTAINER_PROJECT_ROOT)
    parser.add_argument("--source-mode", choices=["file", "rtsp"], default="file")
    parser.add_argument("--label", choices=["fall", "not_fall"], default=None)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--assumed-fps", type=float, default=30.0)
    parser.add_argument("--timeout-grace-seconds", type=float, default=8.0)
    parser.add_argument("--shadow-wait-seconds", type=float, default=3.0)
    parser.add_argument("--restart-wait-seconds", type=float, default=20.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--apply-camera-config", action="store_true")
    parser.add_argument("--restart-ai-engine", action="store_true")
    parser.add_argument("--no-restore-camera-config", dest="restore_camera_config", action="store_false")
    parser.set_defaults(restore_camera_config=True)
    args = parser.parse_args()

    results = evaluate(args)
    if results:
        print_summary(results)
        print(f"results jsonl: {args.results_jsonl}")
        print(f"results csv: {args.results_csv}")
    elif args.prepare_only:
        print("prepare-only complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
