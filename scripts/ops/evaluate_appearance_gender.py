#!/usr/bin/env python3
"""Evaluate PP-Human gender threshold behavior on saved appearance crops."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ai._attribute_backend import AttributeCrop  # noqa: E402
from src.core.ai._attribute_backends import PPHumanAttributeBackend  # noqa: E402


def _parse_threshold_pair(value: str) -> tuple[float, float]:
    try:
        female_min, male_max = value.split(":", 1)
        return float(female_min), float(male_max)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "threshold pair must use '<female_min>:<male_max>', e.g. 0.75:0.25"
        ) from exc


def _recent_crops(crop_dir: Path, limit: int) -> list[Path]:
    files = [path for path in crop_dir.glob("*.jpg") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def _classify(score: float, *, female_min: float, male_max: float) -> str:
    if score >= female_min:
        return "female"
    if score <= male_max:
        return "male"
    return "unknown"


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(ratio: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
        return round(ordered[index], 4)

    return {
        "min": round(ordered[0], 4),
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": round(ordered[-1], 4),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PP-Human gender scoring over recent appearance crops.",
    )
    parser.add_argument(
        "--crop-dir",
        default=os.environ.get("APPEARANCE_CROP_DIR", "data/runtime/appearance_crops"),
        help="Directory containing saved appearance crop jpg files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of most recent crops to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "APPEARANCE_MODEL_PATH",
            "models/pphuman_attribute_src/PP-LCNet_x1_0_pedestrian_attribute_infer",
        ),
        help="PP-Human attribute model path.",
    )
    parser.add_argument(
        "--label-map",
        default=os.environ.get(
            "APPEARANCE_LABEL_MAP_PATH",
            "config/appearance_pphuman_labels.example.json",
        ),
        help="PP-Human label map JSON path.",
    )
    parser.add_argument(
        "--runtime",
        default=os.environ.get("APPEARANCE_RUNTIME", "paddle"),
        help="Attribute runtime: paddle, onnxruntime, or auto.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=int(os.environ.get("APPEARANCE_INPUT_SIZE", "224")),
        help="Fallback model input size.",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        type=_parse_threshold_pair,
        default=[],
        help="Threshold pair '<female_min>:<male_max>'. Can be repeated.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    crop_dir = Path(args.crop_dir)
    crops = _recent_crops(crop_dir, max(1, args.limit))
    backend = PPHumanAttributeBackend(
        model_path=args.model_path,
        label_map_path=args.label_map,
        runtime=args.runtime,
        input_size=args.input_size,
        score_threshold=float(os.environ.get("APPEARANCE_SCORE_THRESHOLD", "0.5")),
    )

    scores: list[float] = []
    unreadable: list[str] = []
    for path in crops:
        image = cv2.imread(str(path))
        if image is None:
            unreadable.append(str(path))
            continue
        height, width = image.shape[:2]
        attrs = backend.predict(AttributeCrop(image, 0, 0, width, height))
        score = (attrs.get("attribute_scores") or {}).get("gender")
        if score is None:
            continue
        scores.append(float(score))

    thresholds: Iterable[tuple[float, float]] = args.thresholds or [
        (0.65, 0.35),
        (0.75, 0.25),
        (0.85, 0.15),
    ]
    summaries = []
    for female_min, male_max in thresholds:
        counts = {"male": 0, "female": 0, "unknown": 0}
        for score in scores:
            counts[_classify(score, female_min=female_min, male_max=male_max)] += 1
        total = max(1, len(scores))
        summaries.append({
            "female_min_score": female_min,
            "male_max_score": male_max,
            "counts": counts,
            "ratios": {key: round(value / total, 4) for key, value in counts.items()},
        })

    return {
        "crop_dir": str(crop_dir),
        "requested_limit": args.limit,
        "evaluated": len(scores),
        "unreadable": unreadable,
        "female_score_percentiles": _percentiles(scores),
        "thresholds": summaries,
    }


def main() -> int:
    args = _build_parser().parse_args()
    result = evaluate(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"crop_dir: {result['crop_dir']}")
    print(f"evaluated: {result['evaluated']} / requested {result['requested_limit']}")
    print(f"female_score_percentiles: {result['female_score_percentiles']}")
    for item in result["thresholds"]:
        print(
            "threshold "
            f"female>={item['female_min_score']} male<={item['male_max_score']} "
            f"counts={item['counts']} ratios={item['ratios']}"
        )
    if result["unreadable"]:
        print(f"unreadable: {len(result['unreadable'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
