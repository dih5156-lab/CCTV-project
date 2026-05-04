"""Evaluate YOLO-style detection models on a fixed image/label dataset.

Expected dataset layout:

    data/eval/helmet/
      images/*.jpg
      labels/*.txt
      classes.txt              # optional

Labels use YOLO format: class_id x_center y_center width height, normalized.
The script writes a JSON report that can be referenced from models/model_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO detection model with precision/recall/latency metrics."
    )
    parser.add_argument("--model", required=True, help="Path to .pt, .onnx, or .engine model")
    parser.add_argument("--dataset", required=True, help="Dataset root with images/ and labels/")
    parser.add_argument("--output", required=True, help="JSON report output path")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching")
    parser.add_argument("--device", default=None, help="Ultralytics device value, e.g. cpu, cuda, 0")
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional classes.txt path. Defaults to <dataset>/classes.txt when present.",
    )
    parser.add_argument(
        "--target-classes",
        default=None,
        help="Comma-separated class names to include, e.g. helmet,head",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit image count for a quick smoke run")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup inference count excluded from latency metrics",
    )
    return parser.parse_args()


def load_class_names(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    names: dict[int, str] = {}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        name = line.strip()
        if name:
            names[idx] = name
    return names


def normalize_model_names(names: Any) -> dict[int, str]:
    if not isinstance(names, dict):
        return {}
    normalized: dict[int, str] = {}
    for key, value in names.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            continue
        normalized[class_id] = str(value)
    return normalized


def list_images(images_dir: Path, limit: int = 0) -> list[Path]:
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return images[:limit] if limit > 0 else images


def yolo_to_xyxy(values: list[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = values
    box_width = width * image_width
    box_height = height * image_height
    x1 = (x_center * image_width) - (box_width / 2)
    y1 = (y_center * image_height) - (box_height / 2)
    x2 = x1 + box_width
    y2 = y1 + box_height
    return (x1, y1, x2, y2)


def load_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
    class_names: dict[int, str],
    target_classes: set[str] | None,
) -> list[Box]:
    if not label_path.exists():
        return []

    boxes: list[Box] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_no}")
        class_id = int(float(parts[0]))
        class_name = class_names.get(class_id, str(class_id))
        if target_classes and class_name not in target_classes:
            continue
        xyxy = yolo_to_xyxy([float(value) for value in parts[1:5]], image_width, image_height)
        boxes.append(Box(class_name=class_name, xyxy=xyxy))
    return boxes


def box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(predictions: list[Box], ground_truth: list[Box], iou_threshold: float) -> dict[str, dict[str, int]]:
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    classes = sorted({box.class_name for box in predictions} | {box.class_name for box in ground_truth})

    for class_name in classes:
        class_predictions = [box for box in predictions if box.class_name == class_name]
        class_ground_truth = [box for box in ground_truth if box.class_name == class_name]
        matched_gt: set[int] = set()

        for prediction in sorted(class_predictions, key=lambda box: box.confidence or 0.0, reverse=True):
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(class_ground_truth):
                if idx in matched_gt:
                    continue
                current_iou = box_iou(prediction.xyxy, gt_box.xyxy)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= iou_threshold:
                by_class[class_name]["tp"] += 1
                matched_gt.add(best_idx)
            else:
                by_class[class_name]["fp"] += 1

        by_class[class_name]["fn"] += len(class_ground_truth) - len(matched_gt)

    return dict(by_class)


def summarize_counts(counts_by_class: dict[str, dict[str, int]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for class_name, counts in sorted(counts_by_class.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        summary[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    tp = totals["tp"]
    fp = totals["fp"]
    fn = totals["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "overall": {
            **totals,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "by_class": summary,
    }


def merge_counts(target: dict[str, dict[str, int]], source: dict[str, dict[str, int]]) -> None:
    for class_name, counts in source.items():
        target_counts = target.setdefault(class_name, {"tp": 0, "fp": 0, "fn": 0})
        for key in ("tp", "fp", "fn"):
            target_counts[key] += counts[key]


def result_to_boxes(result: Any, target_classes: set[str] | None) -> list[Box]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy_values = boxes.xyxy.cpu().tolist()
    class_values = boxes.cls.cpu().tolist()
    conf_values = boxes.conf.cpu().tolist()

    predictions: list[Box] = []
    for xyxy, class_id, confidence in zip(xyxy_values, class_values, conf_values):
        class_name = names.get(int(class_id), str(int(class_id)))
        if target_classes and class_name not in target_classes:
            continue
        predictions.append(
            Box(
                class_name=class_name,
                xyxy=tuple(float(value) for value in xyxy),
                confidence=float(confidence),
            )
        )
    return predictions


def percentile(values: Iterable[float], ratio: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * ratio))))
    return sorted_values[index]


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    output_path = Path(args.output)

    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    target_classes = (
        {item.strip() for item in args.target_classes.split(",") if item.strip()}
        if args.target_classes
        else None
    )

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.model)
    class_file = Path(args.classes) if args.classes else dataset_dir / "classes.txt"
    class_names = load_class_names(class_file if class_file.exists() else None)
    if not class_names:
        class_names = normalize_model_names(getattr(model, "names", {}))
    if target_classes and not class_names:
        raise ValueError(
            "target classes require class names. Provide <dataset>/classes.txt or --classes."
        )

    images = list_images(images_dir, args.limit)
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    for _ in range(max(0, args.warmup)):
        model.predict(
            source=str(images[0]),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

    counts_by_class: dict[str, dict[str, int]] = {}
    latencies_ms: list[float] = []
    per_image: list[dict[str, Any]] = []

    for image_path in images:
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        label_path = labels_dir / f"{image_path.stem}.txt"
        ground_truth = load_ground_truth(
            label_path,
            image_width,
            image_height,
            class_names,
            target_classes,
        )

        started_at = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        latencies_ms.append(latency_ms)

        predictions = result_to_boxes(result, target_classes)
        image_counts = match_detections(predictions, ground_truth, args.iou)
        merge_counts(counts_by_class, image_counts)
        per_image.append(
            {
                "image": str(image_path.relative_to(dataset_dir)),
                "ground_truth": len(ground_truth),
                "predictions": len(predictions),
                "latency_ms": round(latency_ms, 3),
                "counts": image_counts,
            }
        )

    metrics = summarize_counts(counts_by_class)
    report = {
        "model": args.model,
        "dataset": str(dataset_dir),
        "image_count": len(images),
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device or "auto",
            "target_classes": sorted(target_classes) if target_classes else None,
            "warmup": max(0, args.warmup),
        },
        "latency": {
            "avg_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
            "p50_ms": round(percentile(latencies_ms, 0.5), 3),
            "p95_ms": round(percentile(latencies_ms, 0.95), 3),
        },
        "metrics": metrics,
        "per_image": per_image,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": metrics["overall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
