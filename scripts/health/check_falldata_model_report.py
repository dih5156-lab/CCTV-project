#!/usr/bin/env python3
"""Validate falldata RF training metrics before model promotion."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    name: str
    actual: Any
    expected: Any
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("models/model_manifest.json"))
    parser.add_argument("--model-name", default="falldata_sample_rf")
    parser.add_argument("--required-group-by", default="scene_base")
    parser.add_argument("--min-class-groups", type=int, default=2)
    parser.add_argument("--max-false-negatives", type=int, default=0)
    parser.add_argument("--max-false-positives", type=int, default=0)
    parser.add_argument("--min-holdout-fall-precision", type=float)
    parser.add_argument("--min-holdout-fall-recall", type=float)
    parser.add_argument("--min-validation-fall-precision", type=float)
    parser.add_argument("--min-validation-fall-recall", type=float)
    parser.add_argument(
        "--require-cross-validation",
        action="store_true",
        help="Fail unless cross_validation.enabled is true.",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write/update the falldata model entry when checks pass.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"metrics JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _as_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def evaluate_report(
    report: dict[str, Any],
    *,
    required_group_by: str,
    min_class_groups: int,
    max_false_negatives: int,
    max_false_positives: int | None,
    require_cross_validation: bool,
    min_holdout_fall_precision: float | None = None,
    min_holdout_fall_recall: float | None = None,
    min_validation_fall_precision: float | None = None,
    min_validation_fall_recall: float | None = None,
) -> list[CheckResult]:
    split = report.get("holdout_split") or {}
    errors = report.get("holdout_errors") or {}
    dataset_summary = report.get("dataset_summary") or {}
    group_class_counts = dataset_summary.get("group_class_counts") or {}
    cross_validation = report.get("cross_validation") or {}
    holdout_report = (
        ((report.get("holdout") or {}).get("classification_report") or {}).get("fall")
        or {}
    )
    validation_report = (
        ((report.get("validation") or {}).get("classification_report") or {}).get("fall")
        or {}
    )
    train_groups = _as_set(split.get("train_groups"))
    test_groups = _as_set(split.get("test_groups"))
    overlap = sorted(train_groups & test_groups)

    checks = [
        CheckResult(
            "holdout_split.method",
            split.get("method"),
            "group_shuffle",
            split.get("method") == "group_shuffle",
        ),
        CheckResult(
            "holdout_split.group_by",
            split.get("group_by"),
            required_group_by,
            split.get("group_by") == required_group_by,
        ),
        CheckResult(
            "holdout_split.group_overlap",
            overlap,
            [],
            not overlap and bool(train_groups) and bool(test_groups),
        ),
        CheckResult(
            "holdout_errors.false_negative_count",
            int(errors.get("false_negative_count", 0)),
            f"<= {max_false_negatives}",
            int(errors.get("false_negative_count", 0)) <= max_false_negatives,
        ),
        CheckResult(
            "dataset_summary.group_class_counts.fall",
            int(group_class_counts.get("fall", 0)),
            f">= {min_class_groups}",
            int(group_class_counts.get("fall", 0)) >= min_class_groups,
        ),
        CheckResult(
            "dataset_summary.group_class_counts.non_fall",
            int(group_class_counts.get("non_fall", 0)),
            f">= {min_class_groups}",
            int(group_class_counts.get("non_fall", 0)) >= min_class_groups,
        ),
    ]

    if max_false_positives is not None:
        checks.append(
            CheckResult(
                "holdout_errors.false_positive_count",
                int(errors.get("false_positive_count", 0)),
                f"<= {max_false_positives}",
                int(errors.get("false_positive_count", 0)) <= max_false_positives,
            )
        )

    if require_cross_validation:
        checks.append(
            CheckResult(
                "cross_validation.enabled",
                bool(cross_validation.get("enabled")),
                True,
                bool(cross_validation.get("enabled")),
            )
        )

    for report_name, classification, requirements in (
        (
            "holdout",
            holdout_report,
            (
                ("precision", min_holdout_fall_precision),
                ("recall", min_holdout_fall_recall),
            ),
        ),
        (
            "validation",
            validation_report,
            (
                ("precision", min_validation_fall_precision),
                ("recall", min_validation_fall_recall),
            ),
        ),
    ):
        for metric_name, minimum in requirements:
            if minimum is None:
                continue
            actual = float(classification.get(metric_name, 0.0))
            checks.append(
                CheckResult(
                    f"{report_name}.classification_report.fall.{metric_name}",
                    actual,
                    f">= {minimum}",
                    actual >= minimum,
                )
            )

    return checks


def build_payload(path: Path, report: dict[str, Any], checks: list[CheckResult]) -> dict[str, Any]:
    failed = [check for check in checks if not check.passed]
    return {
        "passed": not failed,
        "metrics_json": str(path),
        "dataset_version": report.get("dataset_version"),
        "manifest": report.get("manifest"),
        "output_model": report.get("output_model"),
        "rows": report.get("rows"),
        "class_counts": report.get("class_counts"),
        "checks": [
            {
                "name": check.name,
                "actual": check.actual,
                "expected": check.expected,
                "passed": check.passed,
            }
            for check in checks
        ],
    }


def _summary_metric(report: dict[str, Any], label: str, metric: str) -> float:
    return float(
        ((report.get("classification_report") or {}).get(label) or {}).get(metric, 0.0)
    )


def build_manifest_entry(
    *,
    model_name: str,
    metrics_path: Path,
    report: dict[str, Any],
    checks: list[CheckResult],
) -> dict[str, Any]:
    holdout_errors = report.get("holdout_errors") or {}
    return {
        "name": model_name,
        "task": "falldata_video_fall_verifier",
        "primary_runtime": "sklearn_random_forest",
        "artifacts": {
            "pickle": report.get("output_model"),
            "metrics": str(metrics_path),
            "manifest": report.get("manifest"),
            "feature_cache": report.get("feature_cache"),
        },
        "input_shape": [600, 1662],
        "classes": ["fall", "non_fall"],
        "deployment_target": "falldata_aux_shadow_or_borderline_confirm",
        "acceptance_criteria": {
            "required_group_by": "scene_base",
            "max_false_negatives": 0,
            "max_false_positives": 0,
            "min_class_groups": 2,
            "requires_cross_validation": True,
        },
        "latest_evaluation": {
            "report": str(metrics_path),
            "dataset_version": report.get("dataset_version"),
            "rows": report.get("rows"),
            "class_counts": report.get("class_counts"),
            "fall_precision": _summary_metric(report, "fall", "precision"),
            "fall_recall": _summary_metric(report, "fall", "recall"),
            "non_fall_precision": _summary_metric(report, "non_fall", "precision"),
            "non_fall_recall": _summary_metric(report, "non_fall", "recall"),
            "false_positive_count": holdout_errors.get("false_positive_count"),
            "false_negative_count": holdout_errors.get("false_negative_count"),
            "holdout_split": report.get("holdout_split"),
            "checks": [
                {
                    "name": check.name,
                    "actual": check.actual,
                    "expected": check.expected,
                    "passed": check.passed,
                }
                for check in checks
            ],
        },
    }


def update_manifest(
    manifest_path: Path,
    *,
    model_name: str,
    metrics_path: Path,
    report: dict[str, Any],
    checks: list[CheckResult],
) -> None:
    manifest = load_json(manifest_path)
    models = manifest.setdefault("models", [])
    entry = build_manifest_entry(
        model_name=model_name,
        metrics_path=metrics_path,
        report=report,
        checks=checks,
    )
    for index, model in enumerate(models):
        if model.get("name") == model_name:
            models[index] = {**model, **entry}
            break
    else:
        models.append(entry)
    manifest["updated_at"] = datetime.now(timezone.utc).date().isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        report = load_json(args.metrics_json)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    checks = evaluate_report(
        report,
        required_group_by=args.required_group_by,
        min_class_groups=args.min_class_groups,
        max_false_negatives=args.max_false_negatives,
        max_false_positives=args.max_false_positives,
        require_cross_validation=args.require_cross_validation,
        min_holdout_fall_precision=args.min_holdout_fall_precision,
        min_holdout_fall_recall=args.min_holdout_fall_recall,
        min_validation_fall_precision=args.min_validation_fall_precision,
        min_validation_fall_recall=args.min_validation_fall_recall,
    )
    payload = build_payload(args.metrics_json, report, checks)
    if args.update_manifest and payload["passed"]:
        try:
            update_manifest(
                args.manifest,
                model_name=args.model_name,
                metrics_path=args.metrics_json,
                report=report,
                checks=checks,
            )
            payload["updated_manifest"] = str(args.manifest)
            payload["model_name"] = args.model_name
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            payload["passed"] = False
            payload["error"] = str(exc)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
