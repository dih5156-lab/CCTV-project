"""Read-only audit of saved model evidence; no training, inference, or deployment."""

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def scene_base(scene_id):
    return scene_id.rsplit("_C", 1)[0]


def classify(row, model):
    prediction = row[model].get("prediction")
    if prediction not in (0, 1) or row[model].get("error"):
        return "NO_RESULT"
    if row["is_fall"]:
        return "TP" if prediction == 1 else "FN"
    return "FP" if prediction == 1 else "TN"


def summarize_comparison(relative_path):
    payload = read_json(relative_path)
    rows = payload["videos"]
    result = {"rows": len(rows), "scene_groups": len({scene_base(r["scene_id"]) for r in rows})}
    for model in ("baseline", "candidate"):
        counts = Counter(classify(row, model) for row in rows)
        true_positive = counts["TP"]
        false_positive = counts["FP"]
        false_negative = counts["FN"]
        true_negative = counts["TN"]
        result[model] = {
            "counts": dict(counts),
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
            "negative_clip_false_positive_rate": false_positive / max(false_positive + true_negative, 1),
            "camera_errors": dict(Counter(
                (row["scene_id"].rsplit("_", 1)[-1] if "_C" in row["scene_id"] else "unknown")
                for row in rows if row[f"{model}_result"] in {"FP", "FN"}
            )),
            "summary_counts_match": all(
                counts[key] == value
                for key, value in payload["summary"][model]["counts"].items()
            ),
            "row_results_match": all(classify(row, model) == row[f"{model}_result"] for row in rows),
        }
    return result


def main():
    split_groups = {}
    split_details = {}
    for split in ("train", "validation", "test"):
        path = ROOT / f"data/fall_eval/reconstructed_split_20260908_container/{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        split_groups[split] = {row.get("scene_group") or scene_base(row["scene_id"]) for row in rows}
        split_details[split] = {
            "rows": len(rows), "scene_groups": len(split_groups[split]),
            "labels": dict(Counter(str(row["is_fall"]) for row in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    output = {
        "splits": split_details,
        "scene_group_overlap": {
            f"{left}/{right}": len(split_groups[left] & split_groups[right])
            for left, right in combinations(split_groups, 2)
        },
    }
    output["saved_comparisons"] = {
        name: summarize_comparison(path)
        for name, path in {
            "open320": "data/fall_eval/open_data_test_20260914/balanced320_c1_c8_comparison.json",
            "field23": "data/fall_eval/yesterday_20260910_temporal_runtime_comparison_imgsz640.json",
        }.items()
    }
    output["training_reports"] = {}
    for model in ("fall_temporal_baseline_same_split_20260910", "fall_temporal_same_split_resume_20260908"):
        report = read_json(f"data/fall_eval/{model}_metrics.json")
        history = report["training"]["history"]
        groups = report["holdout_split"]
        output["training_reports"][model] = {
            "model_params": report["model_params"],
            "rows": report["rows"], "effective_rows": report["effective_rows"],
            "best_epoch": min(history, key=lambda row: row["holdout_loss"]),
            "last_epoch": history[-1],
            "internal_group_overlap": len(set(groups["train_groups"]) & set(groups["test_groups"])),
            "validation": report["validation"]["classification_report"],
            "holdout": report["holdout"]["classification_report"],
        }
    manifest = read_json("models/model_manifest.json")
    output["manifest_artifacts"] = {
        model["name"]: {
            kind: (ROOT / path).exists() for kind, path in model["artifacts"].items()
        } for model in manifest["models"]
    }
    color_model = next(model for model in manifest["models"] if model["name"] == "appearance_color_yolov8n")
    output["color_pytorch_hash_matches"] = (
        hashlib.sha256((ROOT / color_model["artifacts"]["pytorch"]).read_bytes()).hexdigest()
        == color_model["checksums"]["pytorch_sha256"]
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
