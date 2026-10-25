"""운영과 분리한 24FPS·2채널 낙상 주기 비교의 준비와 결과 판정."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import shutil
from fractions import Fraction
from pathlib import Path

FPS = 24
WORKTREE = Path(__file__).resolve().parents[2]
SOURCE = Path("/media/sawwave/Learning11/CCTV-project")
DEFAULT_OUTPUT = WORKTREE / "data/runtime/two_camera_cadence_20261023"
IMAGE = "sha256:ccbfe02abcbffe212c3ca09a2d08dd09983ee20fcfd7bf7b1ae7cda52a76cfe5"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def conditions():
    return [(hz, FPS // hz - 1) for hz in (12, 8, 6)]


def onset_seconds(original, native_fps):
    if not original["is_fall"]:
        return None
    frame = original.get("fall_start_frame")
    rate = float(Fraction(native_fps))
    if (
        not isinstance(frame, (int, float))
        or not math.isfinite(frame)
        or frame < 0
        or rate <= 0
    ):
        raise ValueError("유효한 낙상 시작 프레임·원본 FPS가 필요합니다")
    return frame / rate


def video_command(source, destination):
    # 재생 속도와 정답의 초 단위 시각은 유지하고 프레임만 재표본화한다.
    return [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-i",
        str(source),
        "-vf",
        "scale=1920:1080:flags=bicubic,fps=24",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        str(destination),
    ]


def overlap_ratio(first, second):
    a, b = first["processed"], second["processed"]
    duration = min(
        a[-1]["monotonic"] - a[0]["monotonic"], b[-1]["monotonic"] - b[0]["monotonic"]
    )
    overlap = min(a[-1]["monotonic"], b[-1]["monotonic"]) - max(
        a[0]["monotonic"], b[0]["monotonic"]
    )
    return max(0.0, overlap) / duration if duration > 0 else 0.0


def score_capture(capture, is_fall, onset, target_hz):
    invalid = {"verdict": "INVALID", "throughput_pass": False}
    camera = capture.get("camera_id")
    stats = capture.get("stats", {})
    primary = stats.get("primary_camera_stats", {}).get(camera, {})
    source, processed = capture.get("source", []), capture.get("processed", [])
    if (
        capture.get("errors")
        or not capture.get("all_frames_processed")
        or len(source) < 2
        or len(processed) != len(source)
        or stats.get("frames_processed") != len(source)
        or not primary
        or primary.get("frames_seen") != len(source)
        or capture.get("fps") != FPS
        or not isinstance(primary.get("frames_inferred"), (int, float))
        or not 0 < primary["frames_inferred"] <= len(source)
        or any(
            stats.get(k, 0)
            for k in (
                "primary_metrics_errors",
                "source_error_identified",
                "source_error_unidentified",
            )
        )
    ):
        return {**invalid, "reason": "재생·계측 완료 또는 카메라 식별 미확인"}
    timeline = processed[-1]["pts"] - processed[0]["pts"]
    elapsed = processed[-1]["monotonic"] - processed[0]["monotonic"]
    if (
        timeline <= 0
        or elapsed <= 0
        or len({row["frame_index"] for row in processed}) != len(source)
    ):
        return {**invalid, "reason": "유효한 프레임 시계가 필요함"}
    if any(
        b["pts"] <= a["pts"] or b["monotonic"] < a["monotonic"]
        for a, b in zip(processed, processed[1:])
    ):
        return {**invalid, "reason": "프레임 시각 역행"}
    frame_map = {row["frame_index"]: row for row in processed}
    alarms = [
        row
        for row in capture.get("events", [])
        if row["stage"] == "publish_queue" and row.get("accepted")
    ]
    if any(row.get("frame_index") not in frame_map for row in alarms):
        return {**invalid, "reason": "경보 프레임의 PTS를 찾을 수 없음"}
    early = [
        row
        for row in alarms
        if is_fall and frame_map[row["frame_index"]]["pts"] < onset - 1 / FPS
    ]
    detected = [row for row in alarms if row not in early]
    first = (
        min(detected, key=lambda row: frame_map[row["frame_index"]]["pts"])
        if detected
        else None
    )
    pts_delay = (
        max(0.0, frame_map[first["frame_index"]]["pts"] - onset)
        if first and is_fall
        else None
    )
    anchor = source[0]["monotonic"] - source[0]["pts"]
    wall_delay = first["monotonic"] - (anchor + onset) if first and is_fall else None
    inference_hz = primary["frames_inferred"] / (elapsed + 1 / FPS)
    processed_fps = (len(processed) - 1) / elapsed
    lag_growth = elapsed - timeline
    # 성능 미달과 실제 경보 누락을 별도로 기록해 느린 재생을 정상 처리로 숨기지 않는다.
    return {
        "verdict": ("TP" if detected else "FN")
        if is_fall
        else ("FP" if alarms else "TN"),
        "early_count": len(early),
        "duplicate_count": max(0, len(detected) - 1) if is_fall else 0,
        "pts_delay_seconds": pts_delay,
        "wall_delay_seconds": wall_delay,
        "over_5s": any(v is not None and v > 5 for v in (pts_delay, wall_delay)),
        "processed_fps": processed_fps,
        "inference_hz": inference_hz,
        "lag_growth_seconds": lag_growth,
        "alarm_count": len(alarms),
        "throughput_pass": processed_fps >= FPS * 0.95
        and target_hz * 0.95 <= inference_hz <= target_hz * 1.05
        and abs(lag_growth) <= 0.5
        and not stats.get("frames_dropped", 0)
        and not primary.get("frames_inference_unknown", 0),
    }


def bind(source, target, readonly=True):
    return {
        "type": "bind",
        "source": str(source),
        "target": target,
        "read_only": readonly,
        "bind": {"create_host_path": False},
    }


def prepare(output):
    output = Path(output).resolve()
    if not output.is_relative_to(WORKTREE / "data/runtime"):
        raise ValueError("출력은 전용 worktree의 data/runtime 아래여야 합니다")
    output.mkdir(parents=True, exist_ok=False)
    legacy = SOURCE / "data/runtime/p52_candidate_lock_20261016"
    previous = (
        WORKTREE
        / "data/runtime/two_camera_baseline_20261023_resume/channel_1.compose.json"
    )
    baseline = read(previous)["services"]["trial"]
    (output / "verification").mkdir()
    for name in (
        "two_camera_cadence.py",
        "capture_fall_cadence.py",
        "run_two_camera_cadence.py",
    ):
        shutil.copy2(WORKTREE / "scripts/ops" / name, output / "verification" / name)
    config = output / "config"
    shutil.copytree(
        SOURCE / "data/runtime/live_refresh_20261008/frozen/config/deepstream", config
    )
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(config / "config_infer_primary.txt")
    for key in ("onnx-file", "model-file", "proto-file", "tlt-encoded-model"):
        cfg["property"].pop(key, None)
    cfg["property"].update(
        {"batch-size": "1", "model-engine-file": "/app/models/fall/yolov8n-pose.engine"}
    )
    with (config / "config_infer_primary.txt").open("w") as stream:
        cfg.write(stream)
    engine = SOURCE / "models/fall/yolov8n-pose.engine"
    model_hash = digest(engine)
    scenes = []
    (output / "videos").mkdir()
    for trial in read(legacy / "plan.json")["trials"][:20]:
        old = read(legacy / f"trial_{trial['index']:03d}/manifest.json")
        original = old["original"]
        video = Path(original["video_path"])
        annotation = Path(original["label_path"])
        probe = original["probe"]
        onset = onset_seconds(original, probe["avg_frame_rate"])
        if onset is not None and abs(onset - old["fall_start_seconds"]) > 1e-6:
            raise ValueError("기존 검증 정답과 시작 시각이 다릅니다")
        duration = float(probe["duration"])
        if not 5 <= duration <= 60 or onset is not None and not 0 <= onset < duration:
            raise ValueError("시험 영상 길이·정답 시각을 확인하세요")
        destination = output / "videos" / f"{trial['scene']}.mp4"
        scenes.append(
            {
                "scene_id": trial["scene"],
                "is_fall": trial["is_fall"],
                "onset_seconds": onset,
                "source_video": str(video),
                "source_sha256": digest(video),
                "label_path": str(annotation),
                "label_sha256": digest(annotation),
                "duration_seconds": duration,
                "expected_frames": round(duration * FPS),
                "derived_video": str(destination),
                "convert_command": video_command(video, destination),
            }
        )
    positives = [s for s in scenes if s["is_fall"]]
    negatives = [s for s in scenes if not s["is_fall"]]
    assert len(positives) == len(negatives) == 10
    pairs = [
        list(pair if index % 2 == 0 else reversed(pair))
        for index, pair in enumerate(zip(positives, negatives))
    ]
    plan = {
        "image": IMAGE,
        "fps": FPS,
        "model_sha256": model_hash,
        "engine_path": str(engine),
        "scope": "two concurrent independent batch1 decode/primary/tracker/fall/event-queue pipelines",
        "model": "operating nano pose 640; heuristic fall; auxiliary shadow disabled",
        "source": "pinned operating image; no latest workspace product overlay",
        "dataset": "existing development regression20; 10 fall + 10 normal; not independent field validation",
        "capture_validation": "complete clips; real source PTS; actual Primary counters; overlap>=90%",
        "performance_gate": {
            "processed_fps_min": 22.8,
            "inference_target_ratio_min": 0.95,
            "lag_growth_max_seconds": 0.5,
        },
        "quality_gate": {
            "FN": 0,
            "FP": 0,
            "early": 0,
            "duplicates": 0,
            "latency_target_seconds": 3,
            "latency_limit_seconds": 5,
        },
        "operating_guard": {
            "processed_fps_min": 27,
            "primary_hz_min": 13.5,
            "frame_age_max_seconds": 1,
        },
        "cases": [],
        "scenes": scenes,
        "execution_status": "PREPARED_NOT_RUN",
    }
    for hz, interval in conditions():
        for index, pair in enumerate(pairs):
            case = output / f"hz{hz}/pair_{index:02d}"
            (case / "coordination").mkdir(parents=True)
            services = {}
            for slot, scene in enumerate(pair, 1):
                camera = f"replay_{slot}"
                folder = case / camera
                for sub in ("output", "data"):
                    (folder / sub).mkdir(parents=True)
                write(
                    folder / "cameras.json",
                    [
                        {
                            "id": camera,
                            "source": "file:///input/video.mp4",
                            "enabled": True,
                            "detections": ["person", "fall"],
                        }
                    ],
                )
                write(folder / "zones.json", {})
                write(
                    folder / "manifest.json",
                    {
                        **scene,
                        "camera_id": camera,
                        "fps": FPS,
                        "target_hz": hz,
                        "interval": interval,
                        "model_sha256": model_hash,
                        "config_sha256": digest(config / "config_infer_primary.txt"),
                    },
                )
                env = dict(baseline["environment"])
                env.update(
                    {
                        "USE_DEEPSTREAM": "1",
                        "AI_DEVICE": "cuda",
                        "APP_ENV": "development",
                        "DS_PRIMARY_INTERVAL": str(interval),
                        "DS_YOLO_TASK": "pose",
                        "DS_YOLO_CONFIDENCE": "0.5",
                        "DS_POSE_CONFIDENCE": "0.5",
                        "DS_YOLO_INPUT_SIZE": "640",
                        "DS_CROP_POSE_ENABLED": "0",
                        "DS_OUTPUT_MODE": "fakesink",
                        "DS_PREVIEW_ENABLED": "false",
                        "DS_HELMET_ENABLED": "0",
                        "DS_PPHUMAN_SGIE_ENABLED": "0",
                        "DS_APPEARANCE_ENABLED": "0",
                        "DS_FACE_ENABLED": "0",
                        "DS_STREAM_WIDTH": "1920",
                        "DS_STREAM_HEIGHT": "1080",
                        "DS_OSD_PROCESS_MODE": "0",
                        "DS_H264_FPS": "24",
                        "FALLDATA_AUX_ENABLED": "false",
                        "FALLDATA_AUX_CLIP_BUFFER_ENABLED": "false",
                        "APPEARANCE_ENABLED": "false",
                        "FACE_SNAPSHOT_ENABLED": "false",
                        "DS_EVENT_MIN_INTERVAL_SEC": "3",
                        "FALL_SUSTAINED_SECONDS": "2",
                        "FALL_GAP_RESET_SECONDS": "5",
                        "FALL_RESEND_COOLDOWN": "60",
                        "DB_PATH": "/app/data/runtime/events.db",
                    }
                )
                mounts = [
                    v
                    for v in baseline["volumes"]
                    if v["target"]
                    in (
                        "/opt/nvidia/deepstream",
                        "/usr/lib/aarch64-linux-gnu/nvidia",
                        "/app/.venv-jetson-train",
                        "/app/models",
                    )
                ]
                mounts += [
                    bind(config, "/app/config/deepstream"),
                    bind(output / "verification", "/verification"),
                    bind(scene["derived_video"], "/input/video.mp4"),
                    bind(folder / "manifest.json", "/input/manifest.json"),
                    bind(folder / "cameras.json", "/app/cameras.json"),
                    bind(folder / "zones.json", "/app/zones_config.json"),
                    bind(folder / "output", "/output", False),
                    bind(folder / "data", "/app/data", False),
                    bind(case / "coordination", "/coordination", False),
                ]
                services[camera] = {
                    "image": IMAGE,
                    "pull_policy": "never",
                    "network_mode": "none",
                    "runtime": "nvidia",
                    "restart": "no",
                    "cpus": 2,
                    "mem_limit": "8g",
                    "working_dir": "/app",
                    "environment": env,
                    "volumes": mounts,
                    "container_name": f"cctv-cadence24-{hz}-{index:02d}-{slot}",
                    "entrypoint": [
                        "timeout",
                        "--signal=TERM",
                        "--kill-after=10",
                        "150",
                        "python3",
                        "/verification/capture_fall_cadence.py",
                    ],
                }
            write(case / "compose.json", {"services": services})
            plan["cases"].append(
                {
                    "hz": hz,
                    "interval": interval,
                    "pair": index,
                    "directory": str(case),
                    "scenes": [x["scene_id"] for x in pair],
                }
            )
    plan["cases"].sort(key=lambda case: (case["pair"], -case["hz"]))
    plan["fixed_sha256"] = {
        str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
    }
    write(output / "plan.json", plan)
    return {
        "output": str(output),
        "cases": len(plan["cases"]),
        "unique_videos": len(scenes),
        "converted_videos": 0,
    }


def summarize(rows):
    summaries = []
    for hz, _ in conditions():
        selected = [row for row in rows if row["hz"] == hz]
        counts = {
            key: sum(row["verdict"] == key for row in selected)
            for key in ("TP", "FN", "TN", "FP", "INVALID", "NOT_RUN")
        }
        valid = [row for row in selected if row["verdict"] in ("TP", "FN", "TN", "FP")]
        complete = (
            len(valid) == 20
            and counts["TP"] + counts["FN"] == 10
            and counts["TN"] + counts["FP"] == 10
        )
        delays = [
            row["wall_delay_seconds"]
            for row in valid
            if row.get("wall_delay_seconds") is not None
        ]
        summaries.append(
            {
                "target_hz_per_camera": hz,
                "counts": counts,
                "complete": complete,
                "miss_rate": counts["FN"] / 10 if complete else None,
                "false_alarm_clip_rate": counts["FP"] / 10 if complete else None,
                "actual_hz_min": min(
                    (row["inference_hz"] for row in valid), default=None
                ),
                "processed_fps_min": min(
                    (row["processed_fps"] for row in valid), default=None
                ),
                "alarm_delay_max_seconds": max(delays, default=None),
                "throughput_pass": complete
                and all(row["throughput_pass"] for row in valid),
                "quality_pass": complete
                and not counts["FN"]
                and not counts["FP"]
                and all(
                    not row["early_count"]
                    and not row["duplicate_count"]
                    and not row["over_5s"]
                    for row in valid
                ),
            }
        )
    return summaries


def evaluate(output):
    output = Path(output)
    plan = read(output / "plan.json")
    rows = []
    for case in plan["cases"]:
        captures = []
        for slot in (1, 2):
            folder = Path(case["directory"]) / f"replay_{slot}"
            path = folder / "output/capture.json"
            manifest = read(folder / "manifest.json")
            row = {
                "hz": case["hz"],
                "pair": case["pair"],
                "camera": slot,
                "scene": manifest["scene_id"],
            }
            if not path.exists():
                rows.append({**row, "verdict": "NOT_RUN"})
                continue
            data = read(path)
            captures.append(data)
            if (
                data.get("camera_id") != manifest["camera_id"]
                or len(data.get("source", [])) != manifest["expected_frames"]
            ):
                score = {
                    "verdict": "INVALID",
                    "throughput_pass": False,
                    "reason": "배정 카메라 또는 영상 프레임 수 불일치",
                }
            else:
                score = score_capture(
                    data, manifest["is_fall"], manifest["onset_seconds"], case["hz"]
                )
            rows.append({**row, **score})
        simultaneous = (
            len(captures) == 2
            and all(c.get("processed") for c in captures)
            and overlap_ratio(*captures) >= 0.9
        )
        execution_path = Path(case["directory"]) / "execution.json"
        execution = read(execution_path) if execution_path.exists() else {}
        for row in rows[-2:]:
            row["two_channel_overlap_pass"] = bool(simultaneous)
            if row["verdict"] != "NOT_RUN" and (
                not simultaneous or not execution.get("valid_execution")
            ):
                row.update(
                    verdict="INVALID",
                    throughput_pass=False,
                    reason="2채널 동시 재생·운영 보호·정리 완료 미확인",
                )
    write(
        output / "comparison.json",
        {
            "fps": FPS,
            "summary": summarize(rows),
            "rows": rows,
            "scope": plan["scope"],
            "dataset": plan["dataset"],
        },
    )
    return {"rows": len(rows), "not_run": sum(r["verdict"] == "NOT_RUN" for r in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.output) if args.action == "prepare" else evaluate(args.output),
            ensure_ascii=False,
        )
    )
