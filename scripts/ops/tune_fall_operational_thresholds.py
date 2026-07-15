#!/usr/bin/env python3
"""Tune fall-detection thresholds from DeepStream replay result JSONL files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS = [round(value / 10, 2) for value in range(0, 71)]


@dataclass(frozen=True)
class EvaluationRow:
    source: str
    scene_id: str
    video_path: str
    expected_fall: bool
    baseline_detected: bool
    score: float
    raw_score: Any


def load_result_rows(paths: list[Path], score_field: str) -> tuple[list[EvaluationRow], list[str]]:
    rows: list[EvaluationRow] = []
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            errors.append(f"missing file: {path}")
            continue
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_no}: {exc}")
                    continue
                if not isinstance(payload, dict):
                    errors.append(f"{path}:{line_no}: expected object")
                    continue
                expected_fall = payload.get("expected_fall")
                if not isinstance(expected_fall, bool):
                    errors.append(f"{path}:{line_no}: expected_fall must be bool")
                    continue
                raw_score = payload.get(score_field)
                rows.append(
                    EvaluationRow(
                        source=str(path),
                        scene_id=str(payload.get("scene_id") or ""),
                        video_path=str(payload.get("video_path") or ""),
                        expected_fall=expected_fall,
                        baseline_detected=bool(payload.get("detected")),
                        score=_score_or_zero(raw_score),
                        raw_score=raw_score,
                    )
                )
    return rows, errors


def _score_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def confusion_for_predictions(expected: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(1 for exp, pred in zip(expected, predicted) if exp and pred)
    fn = sum(1 for exp, pred in zip(expected, predicted) if exp and not pred)
    fp = sum(1 for exp, pred in zip(expected, predicted) if not exp and pred)
    tn = sum(1 for exp, pred in zip(expected, predicted) if not exp and not pred)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _f1(tp, fp, fn),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = (2 * tp) + fp + fn
    return (2 * tp) / denominator if denominator else None


def evaluate_thresholds(
    rows: list[EvaluationRow],
    thresholds: list[float],
    *,
    include_examples: bool = False,
) -> list[dict[str, Any]]:
    expected = [row.expected_fall for row in rows]
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted = [row.score >= threshold for row in rows]
        metrics = confusion_for_predictions(expected, predicted)
        metrics["threshold"] = threshold
        if include_examples:
            metrics["false_positive_examples"] = _examples(rows, predicted, expected_fall=False)
            metrics["false_negative_examples"] = _examples(rows, predicted, expected_fall=True)
        results.append(metrics)
    return results


def _examples(
    rows: list[EvaluationRow],
    predicted: list[bool],
    *,
    expected_fall: bool,
    limit: int = 8,
) -> list[dict[str, Any]]:
    examples: list[EvaluationRow] = []
    for row, pred in zip(rows, predicted):
        if expected_fall and row.expected_fall and not pred:
            examples.append(row)
        elif not expected_fall and not row.expected_fall and pred:
            examples.append(row)
    examples.sort(key=lambda row: row.score, reverse=not expected_fall)
    return [
        {
            "source": row.source,
            "scene_id": row.scene_id,
            "video_path": row.video_path,
            "score": row.score,
        }
        for row in examples[:limit]
    ]


def summarize_baseline(rows: list[EvaluationRow]) -> dict[str, Any]:
    return confusion_for_predictions(
        [row.expected_fall for row in rows],
        [row.baseline_detected for row in rows],
    )


def recommend_threshold(
    threshold_results: list[dict[str, Any]],
    *,
    max_false_positives: int,
) -> dict[str, Any] | None:
    allowed = [row for row in threshold_results if row["fp"] <= max_false_positives]
    if not allowed:
        return None
    return sorted(
        allowed,
        key=lambda row: (
            row["fn"],
            -(row["specificity"] or 0.0),
            -row["threshold"],
        ),
    )[0]


def build_payload(
    paths: list[Path],
    rows: list[EvaluationRow],
    errors: list[str],
    *,
    score_field: str,
    thresholds: list[float],
    max_false_positives: int,
    include_examples: bool = False,
) -> dict[str, Any]:
    threshold_results = evaluate_thresholds(
        rows,
        thresholds,
        include_examples=include_examples,
    )
    recommended = recommend_threshold(
        threshold_results,
        max_false_positives=max_false_positives,
    )
    return {
        "result_files": [str(path) for path in paths],
        "score_field": score_field,
        "parse_errors": errors,
        "sample_count": len(rows),
        "fall_count": sum(1 for row in rows if row.expected_fall),
        "non_fall_count": sum(1 for row in rows if not row.expected_fall),
        "baseline_detected_metrics": summarize_baseline(rows),
        "max_false_positives": max_false_positives,
        "recommended_threshold": recommended,
        "thresholds": threshold_results,
    }


def parse_thresholds(value: str | None) -> list[float]:
    if not value:
        return DEFAULT_THRESHOLDS
    thresholds: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    return sorted(set(thresholds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", nargs="+", type=Path)
    parser.add_argument("--score-field", default="max_fall_score")
    parser.add_argument("--thresholds", help="Comma-separated thresholds. Default: 0.0..7.0 step 0.1.")
    parser.add_argument("--max-false-positives", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--include-examples", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when parse errors exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, errors = load_result_rows(args.results_jsonl, args.score_field)
    payload = build_payload(
        args.results_jsonl,
        rows,
        errors,
        score_field=args.score_field,
        thresholds=parse_thresholds(args.thresholds),
        max_false_positives=args.max_false_positives,
        include_examples=args.include_examples,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict and errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
