#!/usr/bin/env python3
"""Summarize falldata shadow review JSONL records for deployment decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_REVIEW_LOG = Path("data/logs/fall_shadow_review.jsonl")


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"review log not found: {path}"]
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: {exc}")
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                errors.append(f"line {line_no}: expected object")
    return rows, errors


def _bucket_probability(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    if value < 0.3:
        return "0.0-0.3"
    if value < 0.5:
        return "0.3-0.5"
    if value < 0.7:
        return "0.5-0.7"
    if value < 0.9:
        return "0.7-0.9"
    return "0.9-1.0"


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    review_sources: Counter[str] = Counter()
    review_statuses: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    clip_statuses: Counter[str] = Counter()
    confirmed: Counter[str] = Counter()
    probability_buckets: Counter[str] = Counter()
    pending_unconfirmed: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = []
    cooldown_skips: list[dict[str, Any]] = []
    labeling_candidates: list[dict[str, Any]] = []
    human_label_aux_confusion: Counter[str] = Counter()

    for row in rows:
        aux = row.get("falldata_aux")
        if not isinstance(aux, dict):
            aux = {}
        status = str(aux.get("status") or "missing")
        is_confirmed = aux.get("confirmed")
        fall_probability = aux.get("fall_probability")

        statuses[status] += 1
        review_sources[str(row.get("review_source") or "unknown")] += 1
        review_statuses[str(row.get("review_status") or "unknown")] += 1
        labels[str(row.get("label") or "unlabeled")] += 1
        clip_statuses["with_clip" if row.get("clip_path") else "without_clip"] += 1
        confirmed[str(is_confirmed)] += 1
        probability_buckets[_bucket_probability(fall_probability)] += 1

        if row.get("falldata_aux_publish_pending") is True and is_confirmed is not True:
            pending_unconfirmed.append(_compact_record(row, aux))
        if status in {"error", "missing_dependency", "no_frames"}:
            runtime_failures.append(_compact_record(row, aux))
        elif status == "skipped_cooldown":
            cooldown_skips.append(_compact_record(row, aux))
        if row.get("label") is None and row.get("clip_path"):
            labeling_candidates.append(_compact_record(row, aux))
        human_label = row.get("label")
        if status == "ok" and human_label in {"fall", "non_fall"}:
            predicted_fall = is_confirmed is True
            if human_label == "fall":
                human_label_aux_confusion["tp" if predicted_fall else "fn"] += 1
            else:
                human_label_aux_confusion["fp" if predicted_fall else "tn"] += 1

    return {
        "total_records": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "review_source_counts": dict(sorted(review_sources.items())),
        "review_status_counts": dict(sorted(review_statuses.items())),
        "label_counts": dict(sorted(labels.items())),
        "clip_counts": dict(sorted(clip_statuses.items())),
        "confirmed_counts": dict(sorted(confirmed.items())),
        "fall_probability_buckets": dict(sorted(probability_buckets.items())),
        "pending_unconfirmed_count": len(pending_unconfirmed),
        "pending_unconfirmed_examples": pending_unconfirmed[:10],
        "runtime_failure_count": len(runtime_failures),
        "runtime_failure_examples": runtime_failures[:10],
        "cooldown_skip_count": len(cooldown_skips),
        "cooldown_skip_examples": cooldown_skips[:10],
        "labeling_candidate_count": len(labeling_candidates),
        "labeling_candidate_examples": _top_labeling_candidates(labeling_candidates),
        "human_label_aux_evaluation": _confusion_metrics(human_label_aux_confusion),
        "recommendation": _recommendation(
            rows,
            pending_unconfirmed,
            runtime_failures,
            labels,
            labeling_candidates,
        ),
    }


def _confusion_metrics(counts: Counter[str]) -> dict[str, Any]:
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "evaluated_count": tp + fp + tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
    }


def _compact_record(row: dict[str, Any], aux: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row.get("event_id"),
        "created_at": row.get("created_at"),
        "camera_id": row.get("camera_id"),
        "object_id": row.get("object_id"),
        "fall_score": row.get("fall_score"),
        "status": aux.get("status"),
        "confirmed": aux.get("confirmed"),
        "fall_probability": aux.get("fall_probability"),
        "nonzero_feature_frames": aux.get("nonzero_feature_frames"),
        "clip_path": row.get("clip_path"),
        "label": row.get("label"),
        "review_status": row.get("review_status"),
    }


def _top_labeling_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(record: dict[str, Any]) -> tuple[int, float, float, str]:
        status_priority = 0 if record.get("status") == "ok" else 1
        probability = record.get("fall_probability")
        if not isinstance(probability, (int, float)):
            probability = -1.0
        fall_score = record.get("fall_score")
        if not isinstance(fall_score, (int, float)):
            fall_score = -1.0
        return (
            status_priority,
            -float(probability),
            -float(fall_score),
            str(record.get("created_at") or ""),
        )

    return sorted(records, key=sort_key)[:10]


def _recommendation(
    rows: list[dict[str, Any]],
    pending_unconfirmed: list[dict[str, Any]],
    runtime_failures: list[dict[str, Any]],
    labels: Counter[str],
    labeling_candidates: list[dict[str, Any]],
) -> str:
    if not rows:
        return "collect shadow records before enabling confirm or veto"
    if runtime_failures:
        return "fix aux runtime availability before policy changes"
    if pending_unconfirmed:
        return "review pending unconfirmed fall candidates before enabling confirm or veto"
    if labels.get("unlabeled", 0) and labeling_candidates:
        return "label available review clips before training or promotion"
    return "shadow records show no pending unconfirmed candidates in this log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-log", type=Path, default=DEFAULT_REVIEW_LOG)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on parse errors or risky records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, errors = _read_jsonl(args.review_log)
    payload = {
        "review_log": str(args.review_log),
        "parse_errors": errors,
        **summarize_records(rows),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        return 2
    if args.strict and (
        payload["total_records"] == 0
        or payload["pending_unconfirmed_count"] > 0
        or payload["runtime_failure_count"] > 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
