"""Full-frame/crop-pose 비교 결과 JSON의 최소 계약을 검증한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "mode",
    "model_paths",
    "runtime_config",
    "metrics",
    "video_cases",
    "decision",
    "operator",
    "measured_at",
}
VALID_MODES = {"full-frame", "crop-pose"}
VALID_DECISIONS = {"promote", "rollback", "hold"}


def validate_report(report: Any) -> list[str]:
    """보고서 오류를 반환하며, 빈 목록이면 최소 계약을 만족한다."""
    if not isinstance(report, dict):
        return ["report must be a JSON object"]

    errors = [f"missing field: {field}" for field in sorted(REQUIRED_FIELDS - report.keys())]
    if report.get("mode") not in VALID_MODES:
        errors.append("mode must be one of: full-frame, crop-pose")
    if report.get("decision") not in VALID_DECISIONS:
        errors.append("decision must be one of: promote, rollback, hold")
    for field in ("model_paths", "runtime_config", "metrics"):
        if field in report and not isinstance(report[field], dict):
            errors.append(f"{field} must be an object")
    runtime_config = report.get("runtime_config")
    if isinstance(runtime_config, dict) and "pose_crop_mode" in runtime_config:
        crop_enabled = runtime_config["pose_crop_mode"] in {1, True, "1", "true", "yes", "on"}
        if report.get("mode") == "crop-pose" and not crop_enabled:
            errors.append("mode crop-pose requires runtime_config.pose_crop_mode enabled")
        if report.get("mode") == "full-frame" and crop_enabled:
            errors.append("mode full-frame requires runtime_config.pose_crop_mode disabled")
    if "video_cases" in report and not isinstance(report["video_cases"], list):
        errors.append("video_cases must be an array")
    for field in ("operator", "measured_at"):
        if field in report and not isinstance(report[field], str):
            errors.append(f"{field} must be a string")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="비교 결과 JSON 경로")
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid report: {exc}")
        return 2

    errors = validate_report(report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("OK: pose crop comparison report contract valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
