#!/usr/bin/env python3
"""Compare baseline and candidate appearance-color models on later crops."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CSV_FIELDS = (
    "id",
    "timestamp",
    "camera_id",
    "track_id",
    "crop_path",
    "source_path",
    "roi_path",
    "runtime_upper_color",
    "baseline_color",
    "baseline_confidence",
    "candidate_color",
    "candidate_confidence",
    "models_disagree",
)
HORIZONTAL_MARGIN = 0.30
COLOR_OPTIONS = (
    "black",
    "white",
    "gray",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "purple",
    "navy",
    "orange",
    "other",
    "exclude",
)


def _color_boxes(image: np.ndarray, color: str) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == "blue":
        mask = cv2.inRange(
            hsv,
            np.array([80, 80, 80]),
            np.array([120, 255, 255]),
        )
    else:
        low = cv2.inRange(
            hsv,
            np.array([0, 120, 100]),
            np.array([12, 255, 255]),
        )
        high = cv2.inRange(
            hsv,
            np.array([170, 120, 100]),
            np.array([179, 255, 255]),
        )
        mask = cv2.bitwise_or(low, high)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) > 30
    ]


def _person_box(
    image: np.ndarray,
    bbox_x: int,
    bbox_y: int,
    bbox_width: int,
    bbox_height: int,
    *,
    bbox_frame_width: int,
    bbox_frame_height: int,
    saved_frame_width: int,
    saved_frame_height: int,
    context_ratio: float,
) -> tuple[int, int, int, int] | None:
    image_height, image_width = image.shape[:2]
    if min(
        bbox_frame_width,
        bbox_frame_height,
        saved_frame_width,
        saved_frame_height,
    ) <= 0:
        return None
    scale_x = saved_frame_width / float(bbox_frame_width)
    scale_y = saved_frame_height / float(bbox_frame_height)
    x = int(round(bbox_x * scale_x))
    y = int(round(bbox_y * scale_y))
    width = max(1, int(round(bbox_width * scale_x)))
    height = max(1, int(round(bbox_height * scale_y)))
    pad_x = int(width * context_ratio)
    pad_y = int(height * context_ratio)
    crop_x1 = max(0, x - pad_x)
    crop_y1 = max(0, y - pad_y)
    crop_x2 = min(saved_frame_width, x + width + pad_x)
    crop_y2 = min(saved_frame_height, y + height + pad_y)
    if (
        abs(image_width - (crop_x2 - crop_x1)) > 3
        or abs(image_height - (crop_y2 - crop_y1)) > 3
    ):
        return None
    local_x1 = max(0, x - crop_x1)
    local_y1 = max(0, y - crop_y1)
    local_x2 = min(image_width, local_x1 + width)
    local_y2 = min(image_height, local_y1 + height)
    if local_x2 <= local_x1 or local_y2 <= local_y1:
        return None
    return (
        local_x1,
        local_y1,
        local_x2,
        local_y2,
    )


def _upper_roi(
    image: np.ndarray,
    bbox_x: int,
    bbox_y: int,
    bbox_width: int,
    bbox_height: int,
    **person_box_kwargs: Any,
) -> np.ndarray | None:
    box = _person_box(
        image,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
        **person_box_kwargs,
    )
    if box is None:
        return None
    x1, y1, x2, y2 = box
    person = image[y1:y2, x1:x2]
    height, width = person.shape[:2]
    if height < 8 or width < 8:
        return None

    head_boxes = [
        box
        for box in _color_boxes(person, "red")
        if box[1] < height * 0.45 and box[3] < height * 0.65
    ]
    if head_boxes:
        head = max(head_boxes, key=lambda item: item[2] * item[3])
        start_ratio = min(0.65, (head[1] + head[3]) / height + 0.01)
        end_ratio = min(0.90, start_ratio + 0.30)
    else:
        start_ratio, end_ratio = 0.18, 0.42

    crop_y1 = max(0, min(height - 1, int(height * start_ratio)))
    crop_y2 = max(crop_y1 + 1, min(height, int(height * end_ratio)))
    crop_x1 = max(0, min(width - 1, int(width * HORIZONTAL_MARGIN)))
    crop_x2 = max(crop_x1 + 1, min(width, width - crop_x1))
    roi = person[crop_y1:crop_y2, crop_x1:crop_x2]
    if roi.size == 0 or min(roi.shape[:2]) < 8:
        return None
    return roi


def _resolve_crop_path(source: str, project_root: Path) -> Path:
    source_path = Path(source)
    if source_path.is_file():
        return source_path
    if source_path.is_absolute():
        try:
            return project_root / source_path.relative_to("/app")
        except ValueError:
            return source_path
    return project_root / source_path


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    max_per_track: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    track_counts: Counter[tuple[str, int]] = Counter()
    for row in sorted(rows, key=lambda item: int(item["id"]), reverse=True):
        track_key = (str(row["camera_id"]), int(row["track_id"]))
        if track_counts[track_key] >= max_per_track:
            continue
        selected.append(row)
        track_counts[track_key] += 1
        if len(selected) >= limit:
            break
    return selected


def _load_rows(
    db_path: Path,
    *,
    after_id: int,
    scan_limit: int,
) -> list[dict[str, Any]]:
    query = """
        SELECT id, timestamp, camera_id, track_id, crop_path,
               bbox_x, bbox_y, bbox_w, bbox_h, upper_color, attribute_metadata
        FROM appearance_log
        WHERE id > ? AND crop_path IS NOT NULL
        ORDER BY id DESC
        LIMIT ?
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(query, (after_id, scan_limit))
        ]


def _upper_color_observations(attribute_metadata: str | None) -> int:
    if not attribute_metadata:
        return 0
    try:
        payload = json.loads(attribute_metadata)
    except (TypeError, json.JSONDecodeError):
        return 0
    observations = payload.get("color_observations", {})
    try:
        return max(0, int(observations.get("upper_color", 0)))
    except (TypeError, ValueError):
        return 0


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prediction(model: Any, result: Any) -> tuple[str, float]:
    class_id = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    return str(model.names[class_id]), confidence


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    disagreements = sum(bool(item["models_disagree"]) for item in records)
    comparable = [
        item
        for item in records
        if item["runtime_upper_color"] not in (None, "", "unknown")
    ]
    baseline_matches = sum(
        item["baseline_color"] == item["runtime_upper_color"]
        for item in comparable
    )
    candidate_matches = sum(
        item["candidate_color"] == item["runtime_upper_color"]
        for item in comparable
    )
    return {
        "evaluated": total,
        "unique_tracks": len(
            {(item["camera_id"], item["track_id"]) for item in records}
        ),
        "model_disagreements": disagreements,
        "model_disagreement_rate": round(disagreements / max(1, total), 4),
        "baseline_prediction_counts": dict(
            sorted(Counter(item["baseline_color"] for item in records).items())
        ),
        "candidate_prediction_counts": dict(
            sorted(Counter(item["candidate_color"] for item in records).items())
        ),
        "runtime_comparable": len(comparable),
        "baseline_runtime_agreement": round(
            baseline_matches / max(1, len(comparable)), 4
        ),
        "candidate_runtime_agreement": round(
            candidate_matches / max(1, len(comparable)), 4
        ),
        "candidate_changes_to_black": sum(
            item["baseline_color"] != "black"
            and item["candidate_color"] == "black"
            for item in records
        ),
        "candidate_changes_from_black": sum(
            item["baseline_color"] == "black"
            and item["candidate_color"] != "black"
            for item in records
        ),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _write_html(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for item in sorted(
        records,
        key=lambda value: (not value["models_disagree"], -int(value["id"])),
    ):
        style = " style='background:#fff3cd'" if item["models_disagree"] else ""
        options = "<option value=''>검수 전</option>" + "".join(
            f"<option>{color}</option>" for color in COLOR_OPTIONS
        )
        rows.append(
            f"<tr{style}><td>{item['id']}</td>"
            f"<td>{html.escape(str(item['camera_id']))}:{item['track_id']}</td>"
            f"<td><img class='source' src='{html.escape(str(item['source_path']))}' "
            "loading='lazy'></td>"
            f"<td><img class='roi' src='{html.escape(str(item['roi_path']))}' "
            "loading='lazy'></td>"
            f"<td>{html.escape(str(item['runtime_upper_color']))}</td>"
            f"<td>{item['baseline_color']} ({item['baseline_confidence']:.3f})</td>"
            f"<td>{item['candidate_color']} ({item['candidate_confidence']:.3f})</td>"
            f"<td><select data-id='{item['id']}'>{options}</select></td></tr>"
        )
    document = """<!doctype html><html lang='ko'><head><meta charset='utf-8'>
<title>Appearance color shadow comparison</title><style>
body{font-family:sans-serif;margin:20px}table{border-collapse:collapse}
th,td{border:1px solid #bbb;padding:6px;vertical-align:top}
img.source{max-width:420px;max-height:300px}img.roi{min-width:180px;max-width:280px;max-height:220px}
</style></head><body><h1>상의 색상 모델 shadow 비교</h1>
<p>원본에서 대상 사람을 확인한 뒤, 모델 입력 ROI의 실제 상의 색상을 선택하세요.
대상이 다르거나 상의가 보이지 않거나 영상이 깨졌으면 exclude를 선택하세요.</p>
<button onclick='downloadLabels()'>검수 라벨 JSON 다운로드</button>
<table><thead><tr><th>ID</th><th>카메라:트랙</th><th>원본</th><th>모델 상의 ROI</th>
<th>운영 결과</th><th>기존 모델</th><th>후보 모델</th><th>상의 정답</th>
</tr></thead><tbody>
""" + "\n".join(rows) + """
</tbody></table><script>
function downloadLabels() {
  const items = Array.from(document.querySelectorAll('select[data-id]')).map(
    (element) => ({
      id: Number(element.dataset.id),
      upper_color: element.value || null,
      lower_color: 'exclude'
    })
  );
  const blob = new Blob(
    [JSON.stringify({schema_version: 1, items}, null, 2)],
    {type: 'application/json'}
  );
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'appearance_color_shadow_labels.json';
  link.click();
  URL.revokeObjectURL(link.href);
}
</script></body></html>
"""
    path.write_text(document, encoding="utf-8")


def compare(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    roi_dir = output_dir / "rois"
    roi_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "sources"
    source_dir.mkdir()

    rows = _load_rows(
        Path(args.db),
        after_id=args.after_id,
        scan_limit=max(args.limit, args.scan_limit),
    )
    selected = _select_rows(
        rows,
        limit=args.limit,
        max_per_track=args.max_per_track,
    )
    prepared: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    images: list[np.ndarray] = []
    for row in selected:
        observations = _upper_color_observations(row.get("attribute_metadata"))
        if observations < args.min_color_observations:
            skipped["insufficient_color_observations"] += 1
            continue
        crop_path = _resolve_crop_path(str(row["crop_path"]), project_root)
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            skipped["missing_or_unreadable_crop"] += 1
            continue
        roi = _upper_roi(
            image,
            int(row["bbox_x"]),
            int(row["bbox_y"]),
            int(row["bbox_w"]),
            int(row["bbox_h"]),
            bbox_frame_width=args.bbox_frame_width,
            bbox_frame_height=args.bbox_frame_height,
            saved_frame_width=args.saved_frame_width,
            saved_frame_height=args.saved_frame_height,
            context_ratio=args.context_ratio,
        )
        if roi is None:
            skipped["upper_roi_not_found"] += 1
            continue
        roi_height, roi_width = roi.shape[:2]
        if roi_width < args.min_roi_width or roi_height < args.min_roi_height:
            skipped["upper_roi_too_small"] += 1
            continue
        roi_name = f"upper_{row['id']}.jpg"
        source_name = f"source_{row['id']}{crop_path.suffix.lower() or '.jpg'}"
        if not cv2.imwrite(str(roi_dir / roi_name), roi):
            skipped["roi_write_failed"] += 1
            continue
        _link_or_copy(crop_path, source_dir / source_name)
        images.append(roi)
        prepared.append(
            {
                **row,
                "crop_path": str(crop_path),
                "source_path": f"sources/{source_name}",
                "roi_path": f"rois/{roi_name}",
            }
        )

    if not images:
        raise RuntimeError("No usable upper-body ROIs were found")
    baseline = YOLO(str(args.baseline_model))
    candidate = YOLO(str(args.candidate_model))
    predict_args = {
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "device": args.device,
        "verbose": False,
    }
    baseline_results = baseline.predict(images, **predict_args)
    candidate_results = candidate.predict(images, **predict_args)

    records: list[dict[str, Any]] = []
    for row, baseline_result, candidate_result in zip(
        prepared,
        baseline_results,
        candidate_results,
        strict=True,
    ):
        baseline_color, baseline_confidence = _prediction(baseline, baseline_result)
        candidate_color, candidate_confidence = _prediction(candidate, candidate_result)
        records.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "camera_id": row["camera_id"],
                "track_id": row["track_id"],
                "crop_path": row["crop_path"],
                "source_path": row["source_path"],
                "roi_path": row["roi_path"],
                "runtime_upper_color": row["upper_color"],
                "baseline_color": baseline_color,
                "baseline_confidence": round(baseline_confidence, 6),
                "candidate_color": candidate_color,
                "candidate_confidence": round(candidate_confidence, 6),
                "models_disagree": baseline_color != candidate_color,
            }
        )

    summary = {
        "after_id": args.after_id,
        "db_rows_scanned": len(rows),
        "db_rows_selected": len(selected),
        "max_per_track": args.max_per_track,
        "min_color_observations": args.min_color_observations,
        "min_roi_size": [args.min_roi_width, args.min_roi_height],
        "bbox_frame_size": [args.bbox_frame_width, args.bbox_frame_height],
        "saved_frame_size": [args.saved_frame_width, args.saved_frame_height],
        "context_ratio": args.context_ratio,
        "skipped": dict(sorted(skipped.items())),
        "baseline_model": str(args.baseline_model),
        "candidate_model": str(args.candidate_model),
        **_summarize(records),
        "note": "Runtime color is fused pipeline output, not human ground truth.",
    }
    _write_csv(output_dir / "comparison.csv", records)
    (output_dir / "comparison.json").write_text(
        json.dumps(
            {"schema_version": 1, "summary": summary, "items": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_html(
        output_dir / "review.html",
        [item for item in records if item["models_disagree"]],
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--after-id", type=int, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--scan-limit", type=int, default=3000)
    parser.add_argument("--max-per-track", type=int, default=10)
    parser.add_argument("--min-color-observations", type=int, default=3)
    parser.add_argument("--min-roi-width", type=int, default=32)
    parser.add_argument("--min-roi-height", type=int, default=32)
    parser.add_argument("--bbox-frame-width", type=int, default=1920)
    parser.add_argument("--bbox-frame-height", type=int, default=1080)
    parser.add_argument("--saved-frame-width", type=int, default=1280)
    parser.add_argument("--saved-frame-height", type=int, default=720)
    parser.add_argument("--context-ratio", type=float, default=0.6)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    print(json.dumps(compare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
