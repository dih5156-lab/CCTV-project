#!/usr/bin/env python3
"""완료된 낙상 오픈데이터 비교 배치를 검증하고 최종 보고서를 생성한다."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

MODEL_NAMES = ("baseline", "candidate")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("scene_id"):
                raise ValueError(f"manifest line {line_number}: scene_id is required")
            rows.append(row)
    if not rows:
        raise ValueError("manifest contains no videos")
    return rows


def _read_comparison(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("videos"), list):
        raise ValueError("comparison JSON must contain a videos list")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _result_name(row: dict[str, Any], model_name: str) -> str:
    prediction = row[model_name].get("prediction")
    if prediction is None:
        return "NO_RESULT"
    if bool(row["is_fall"]):
        return "TP" if prediction else "FN"
    return "FP" if prediction else "TN"


def _summarize_model_results(
    rows: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    counts = {name: 0 for name in ("TP", "FN", "FP", "TN", "NO_RESULT")}
    for row in rows:
        counts[_result_name(row, model_name)] += 1
    precision_denominator = counts["TP"] + counts["FP"]
    recall_denominator = counts["TP"] + counts["FN"]
    precision = counts["TP"] / precision_denominator if precision_denominator else None
    recall = counts["TP"] / recall_denominator if recall_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"counts": counts, "precision": precision, "recall": recall, "f1": f1}


def _group_summaries(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], field: str
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = metadata[row["scene_id"]].get(field, "unknown")
        groups[str(value)].append(row)
    return {
        group: {
            model_name: _summarize_model_results(group_rows, model_name)
            for model_name in MODEL_NAMES
        }
        for group, group_rows in sorted(groups.items())
    }


def _summary_table(summary: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| 모델 | TP | FN | FP | TN | NO_RESULT | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        model = summary[model_name]
        counts = model["counts"]
        lines.append(
            f"| {model_name} | {counts['TP']} | {counts['FN']} | {counts['FP']} | "
            f"{counts['TN']} | {counts['NO_RESULT']} | "
            f"{_format_metric(model['precision'])} | {_format_metric(model['recall'])} | "
            f"{_format_metric(model['f1'])} |"
        )
    return lines


def _group_table(groups: dict[str, dict[str, dict[str, Any]]], label: str) -> list[str]:
    lines = [
        f"| {label} | 모델 | TP | FN | FP | TN | NO_RESULT | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, summary in groups.items():
        for model_name in MODEL_NAMES:
            model = summary[model_name]
            counts = model["counts"]
            lines.append(
                f"| {group} | {model_name} | {counts['TP']} | {counts['FN']} | "
                f"{counts['FP']} | {counts['TN']} | {counts['NO_RESULT']} | "
                f"{_format_metric(model['recall'])} | {_format_metric(model['f1'])} |"
            )
    return lines


def _render_report(
    *,
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    status: dict[str, Any],
) -> str:
    summary = {
        model_name: _summarize_model_results(rows, model_name)
        for model_name in MODEL_NAMES
    }
    camera_groups = _group_summaries(rows, metadata, "camera")
    category_groups = _group_summaries(rows, metadata, "scene_category")
    disagreements = [
        row
        for row in rows
        if row["baseline"].get("prediction") != row["candidate"].get("prediction")
    ]
    issue_rows = [
        row
        for row in rows
        if any(row[name].get("prediction") is None for name in MODEL_NAMES)
    ]
    lines = [
        "# 낙상 오픈데이터 C1~C8 최종 비교",
        "",
        f"- 생성 시각: `{status['generated_at']}`",
        f"- 처리 영상: `{status['processed']}/{status['expected']}`",
        f"- 검증 상태: `{status['status']}`",
        "- 운영 모델 자동 교체는 수행하지 않았다. 결과 검토 후 사람이 결정한다.",
        "",
        "## 전체 결과",
        "",
        *_summary_table(summary),
        "",
        "## 카메라별 결과",
        "",
        *_group_table(camera_groups, "카메라"),
        "",
        "## 장면 방향별 결과",
        "",
        *_group_table(category_groups, "장면"),
        "",
        "## 모델 간 판정 불일치",
        "",
    ]
    if disagreements:
        lines.extend(
            [
                "| scene_id | 장면 | 카메라 | 정답 | baseline | candidate |",
                "|---|---|---:|---|---|---|",
            ]
        )
        for row in disagreements:
            meta = metadata[row["scene_id"]]
            lines.append(
                f"| {row['scene_id']} | {meta.get('scene_category', 'unknown')} | "
                f"C{meta.get('camera', 'unknown')} | "
                f"{'fall' if row['is_fall'] else 'not_fall'} | "
                f"{_result_name(row, 'baseline')} | {_result_name(row, 'candidate')} |"
            )
    else:
        lines.append("- 없음")
    lines.extend(["", "## 실행 오류", ""])
    if issue_rows:
        for row in issue_rows:
            errors = [
                f"{name}: {row[name].get('error') or 'no prediction'}"
                for name in MODEL_NAMES
                if row[name].get("prediction") is None
            ]
            lines.append(f"- `{row['scene_id']}`: {'; '.join(errors)}")
    else:
        lines.append("- 없음")
    lines.extend(
        [
            "",
            "## 다음 검토",
            "",
            "1. 실제 현장 holdout 결과와 이번 오픈데이터 결과를 함께 비교한다.",
            "2. 카메라별 FN/FP가 편중된 모델은 운영 승격하지 않는다.",
            "3. 모델 승격, 재학습, 임계값 변경은 담당자 검토 후 수행한다.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(
    *, manifest_path: Path, comparison_path: Path, report_path: Path, status_path: Path
) -> int:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        manifest_rows = _read_manifest(manifest_path)
        comparison = _read_comparison(comparison_path)
        rows = comparison["videos"]
        metadata = {row["scene_id"]: row for row in manifest_rows}
        result_ids = [str(row.get("scene_id", "")) for row in rows]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("comparison contains duplicate scene_id values")
        unknown_ids = sorted(set(result_ids) - set(metadata))
        if unknown_ids:
            raise ValueError(f"comparison contains unknown scene_id: {unknown_ids[0]}")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        _write_json(
            status_path,
            {"status": "invalid", "generated_at": generated_at, "error": str(exc)},
        )
        return 3

    expected = len(manifest_rows)
    processed = len(rows)
    no_result = {
        model_name: sum(row[model_name].get("prediction") is None for row in rows)
        for model_name in MODEL_NAMES
    }
    state = "incomplete" if processed < expected else "complete"
    if processed > expected:
        state = "invalid"
    elif processed == expected and any(no_result.values()):
        state = "completed_with_issues"
    status = {
        "status": state,
        "generated_at": generated_at,
        "processed": processed,
        "expected": expected,
        "progress_percent": round(processed / expected * 100, 2),
        "no_result": no_result,
        "report": str(report_path),
    }
    _write_json(status_path, status)
    if state == "incomplete":
        return 2
    if state == "invalid":
        return 3

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(rows=rows, metadata=metadata, status=status), encoding="utf-8"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return finalize(
        manifest_path=args.manifest,
        comparison_path=args.comparison,
        report_path=args.report,
        status_path=args.status,
    )


if __name__ == "__main__":
    raise SystemExit(main())
