#!/usr/bin/env python3
"""동일 현장 영상에서 temporal 낙상 모델 두 개를 순차 비교한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SMOKE_SCRIPT = Path("scripts/inference/smoke_fall_temporal_model.py")
RESULT_NAMES = ("TP", "FN", "FP", "TN", "NO_RESULT")


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "fall"}


def read_manifest(path: Path, *, max_videos: int = 0) -> list[dict[str, Any]]:
    """JSONL manifest를 읽어 평가에 필요한 최소 필드로 정규화한다."""
    if max_videos < 0:
        raise ValueError("max_videos must be zero or greater")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            source = json.loads(line)
            scene_id = str(source.get("scene_id") or "").strip()
            video_path = str(source.get("video_path") or "").strip()
            if not scene_id or not video_path:
                raise ValueError(
                    f"manifest line {line_number}: scene_id and video_path are required"
                )
            if "is_fall" in source:
                is_fall = _parse_bool(source["is_fall"])
            elif "label" in source:
                is_fall = _parse_bool(source["label"])
            else:
                raise ValueError(
                    f"manifest line {line_number}: label or is_fall is required"
                )
            rows.append(
                {
                    "scene_id": scene_id,
                    "video_path": video_path,
                    "is_fall": is_fall,
                }
            )
            if max_videos and len(rows) >= max_videos:
                break
    if not rows:
        raise ValueError("manifest contains no videos")
    return rows


def resolve_container_video_path(
    video_path: Path,
    *,
    project_root: Path,
    container_project_root: Path,
) -> Path:
    """호스트 프로젝트 내부의 영상 경로를 컨테이너 마운트 경로로 바꾼다."""
    resolved_project_root = project_root.resolve()
    resolved_video_path = (
        video_path.resolve()
        if video_path.is_absolute()
        else (resolved_project_root / video_path).resolve()
    )
    try:
        relative_path = resolved_video_path.relative_to(resolved_project_root)
    except ValueError as exc:
        raise ValueError(
            f"video path must be inside project root: {resolved_video_path}"
        ) from exc
    return container_project_root / relative_path


def parse_smoke_output(stdout: str) -> dict[str, int | float]:
    """smoke_fall_temporal_model.py의 안정적인 key:value stdout를 해석한다."""
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    try:
        prediction = int(values["prediction"].strip("[] "))
        if prediction not in {0, 1}:
            raise ValueError("prediction must be 0 or 1")
        return {
            "prediction": prediction,
            "fall_probability": float(values["fall_probability"]),
            "threshold": float(values["threshold"]),
            "frames_with_pose": int(values["frames_with_pose"]),
        }
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid smoke output: {exc}") from exc


def _classification(is_fall: bool, prediction: int | None) -> str:
    if prediction is None:
        return "NO_RESULT"
    if is_fall:
        return "TP" if prediction else "FN"
    return "FP" if prediction else "TN"


def _run_smoke(
    *,
    model_path: Path,
    pose_model: Path,
    container_video_path: Path | None,
    feature_json: Path | None,
    output_feature_json: Path | None,
    python_executable: str,
    project_root: Path,
    device: str,
    imgsz: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    command = [
        python_executable,
        str(project_root / SMOKE_SCRIPT),
        "--model",
        str(model_path),
        "--pose-model",
        str(pose_model),
    ]
    if feature_json is not None:
        command.extend(["--feature-json", str(feature_json)])
    elif container_video_path is not None:
        command.extend(["--video", str(container_video_path)])
    else:
        raise ValueError("container_video_path or feature_json is required")
    if output_feature_json is not None:
        command.extend(["--output-feature-json", str(output_feature_json)])
    command.extend([
        "--device",
        device,
        "--imgsz",
        str(imgsz),
        "--confidence-threshold",
        str(confidence_threshold),
    ])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {
            "prediction": None,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
            "command": command,
        }

    elapsed_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        if not error:
            error = f"smoke command exited with {completed.returncode}"
        return {
            "prediction": None,
            "error": error,
            "elapsed_seconds": elapsed_seconds,
            "command": command,
        }
    try:
        parsed = parse_smoke_output(completed.stdout)
    except ValueError as exc:
        return {
            "prediction": None,
            "error": str(exc),
            "elapsed_seconds": elapsed_seconds,
            "command": command,
        }
    return {**parsed, "error": None, "elapsed_seconds": elapsed_seconds, "command": command}


def summarize_model_results(
    rows: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    counts = {name: 0 for name in RESULT_NAMES}
    elapsed_seconds: list[float] = []
    attempted_elapsed_seconds: list[float] = []
    for row in rows:
        model_result = row[model_name]
        prediction = model_result.get("prediction")
        counts[_classification(bool(row["is_fall"]), prediction)] += 1
        elapsed = model_result.get("elapsed_seconds")
        if elapsed is not None:
            attempted_elapsed_seconds.append(float(elapsed))
            if prediction is not None:
                elapsed_seconds.append(float(elapsed))
    precision_denominator = counts["TP"] + counts["FP"]
    recall_denominator = counts["TP"] + counts["FN"]
    precision = counts["TP"] / precision_denominator if precision_denominator else None
    recall = counts["TP"] / recall_denominator if recall_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "counts": counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_execution_seconds": (
            sum(elapsed_seconds) / len(elapsed_seconds) if elapsed_seconds else None
        ),
        "attempted_average_execution_seconds": (
            sum(attempted_elapsed_seconds) / len(attempted_elapsed_seconds)
            if attempted_elapsed_seconds
            else None
        ),
    }


def _summary_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    return {
        "counts": {
            key: candidate["counts"][key] - baseline["counts"][key]
            for key in RESULT_NAMES
        },
        "precision": _metric_delta(baseline["precision"], candidate["precision"]),
        "recall": _metric_delta(baseline["recall"], candidate["recall"]),
        "f1": _metric_delta(baseline["f1"], candidate["f1"]),
        "average_execution_seconds": _metric_delta(
            baseline["average_execution_seconds"],
            candidate["average_execution_seconds"],
        ),
        "attempted_average_execution_seconds": _metric_delta(
            baseline["attempted_average_execution_seconds"],
            candidate["attempted_average_execution_seconds"],
        ),
    }


def _metric_delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    return round(candidate - baseline, 8)


def _video_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": _metric_delta(baseline.get("prediction"), candidate.get("prediction")),
        "fall_probability": _metric_delta(
            baseline.get("fall_probability"), candidate.get("fall_probability")
        ),
    }


def _result_for_video(
    manifest_row: dict[str, Any],
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    execution_order: tuple[str, str],
) -> dict[str, Any]:
    return {
        **manifest_row,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_result": _classification(manifest_row["is_fall"], baseline.get("prediction")),
        "candidate_result": _classification(manifest_row["is_fall"], candidate.get("prediction")),
        "execution_order": list(execution_order),
        "delta": _video_delta(baseline, candidate),
    }


def _csv_rows(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video in videos:
        row = {
            "scene_id": video["scene_id"],
            "video_path": video["video_path"],
            "is_fall": video["is_fall"],
            "execution_order": ",".join(video["execution_order"]),
            "baseline_result": video["baseline_result"],
            "candidate_result": video["candidate_result"],
            "prediction_delta": video["delta"]["prediction"],
            "fall_probability_delta": video["delta"]["fall_probability"],
        }
        for model_name in ("baseline", "candidate"):
            model = video[model_name]
            row.update(
                {
                    f"{model_name}_prediction": model.get("prediction"),
                    f"{model_name}_fall_probability": model.get("fall_probability"),
                    f"{model_name}_threshold": model.get("threshold"),
                    f"{model_name}_frames_with_pose": model.get("frames_with_pose"),
                    f"{model_name}_elapsed_seconds": model.get("elapsed_seconds"),
                    f"{model_name}_error": model.get("error"),
                }
            )
        rows.append(row)
    return rows


def write_outputs(
    result: dict[str, Any], *, output_json: Path, output_csv: Path
) -> None:
    """완료된 영상 쌍까지의 결과를 JSON과 CSV에 모두 기록한다."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows = _csv_rows(result["videos"])
    fieldnames = list(csv_rows[0]) if csv_rows else ["scene_id", "video_path", "is_fall"]
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def run_comparison(
    *,
    manifest: Path,
    baseline_model: Path,
    candidate_model: Path,
    pose_model: Path,
    output_json: Path,
    output_csv: Path,
    python_executable: str = sys.executable,
    project_root: Path = Path("."),
    container_project_root: Path = Path("/app"),
    device: str = "auto",
    imgsz: int = 640,
    confidence_threshold: float = 0.35,
    max_videos: int = 0,
    feature_cache_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_project_root = project_root.resolve()
    manifest_rows = read_manifest(manifest, max_videos=max_videos)
    result: dict[str, Any] = {
        "manifest": str(manifest),
        "baseline_model": str(baseline_model),
        "candidate_model": str(candidate_model),
        "pose_model": str(pose_model),
        "device": device,
        "pose_extraction": {
            "imgsz": imgsz,
            "confidence_threshold": confidence_threshold,
        },
        "videos": [],
        "summary": {},
        "candidate_minus_baseline": {},
    }
    if feature_cache_dir is not None:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)
        result["pose_extraction"]["feature_cache_dir"] = str(feature_cache_dir)
    model_paths = {"baseline": baseline_model, "candidate": candidate_model}
    for index, manifest_row in enumerate(manifest_rows):
        container_video_path = resolve_container_video_path(
            Path(manifest_row["video_path"]),
            project_root=resolved_project_root,
            container_project_root=container_project_root,
        )
        execution_order = (
            ("baseline", "candidate")
            if index % 2 == 0
            else ("candidate", "baseline")
        )
        feature_cache_path: Path | None = None
        if feature_cache_dir is not None:
            safe_scene_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", manifest_row["scene_id"])
            cache_key = hashlib.sha256(
                f"{manifest_row['scene_id']}\0{manifest_row['video_path']}".encode("utf-8")
            ).hexdigest()[:16]
            feature_cache_path = feature_cache_dir / f"{safe_scene_id}_{cache_key}.json"

        model_results: dict[str, dict[str, Any]] = {}
        for model_name in execution_order:
            cached_features = (
                feature_cache_path
                if feature_cache_path is not None and feature_cache_path.exists()
                else None
            )
            model_results[model_name] = _run_smoke(
                model_path=model_paths[model_name],
                pose_model=pose_model,
                container_video_path=(None if cached_features is not None else container_video_path),
                feature_json=cached_features,
                output_feature_json=(
                    feature_cache_path if cached_features is None else None
                ),
                python_executable=python_executable,
                project_root=resolved_project_root,
                device=device,
                imgsz=imgsz,
                confidence_threshold=confidence_threshold,
            )
        result["videos"].append(
            _result_for_video(
                manifest_row,
                baseline=model_results["baseline"],
                candidate=model_results["candidate"],
                execution_order=execution_order,
            )
        )
        baseline_summary = summarize_model_results(result["videos"], "baseline")
        candidate_summary = summarize_model_results(result["videos"], "candidate")
        result["summary"] = {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
        }
        result["candidate_minus_baseline"] = _summary_delta(
            baseline_summary, candidate_summary
        )
        write_outputs(result, output_json=output_json, output_csv=output_csv)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--container-project-root", type=Path, default=Path("/app"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help="Cache one full-video pose extraction and reuse it for both models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_comparison(
        manifest=args.manifest,
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
        pose_model=args.pose_model,
        output_json=args.output_json,
        output_csv=args.output_csv,
        python_executable=args.python_executable,
        project_root=args.project_root,
        container_project_root=args.container_project_root,
        device=args.device,
        imgsz=args.imgsz,
        confidence_threshold=args.confidence_threshold,
        max_videos=args.max_videos,
        feature_cache_dir=args.feature_cache_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
