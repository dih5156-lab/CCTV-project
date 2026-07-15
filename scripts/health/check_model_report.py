"""Check an evaluation report against models/model_manifest.json criteria."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    metric: str
    actual: float
    expected: float
    passed: bool


@dataclass(frozen=True)
class ArtifactCheck:
    model_name: str
    artifact_type: str
    path: str
    exists: bool
    is_dir: bool
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a model evaluation JSON report against manifest acceptance criteria."
    )
    parser.add_argument("--manifest", default="models/model_manifest.json", help="Model manifest JSON path")
    parser.add_argument("--model-name", help="Model name in manifest")
    parser.add_argument("--report", help="Evaluation report JSON path")
    parser.add_argument(
        "--check-artifacts",
        action="store_true",
        help="Validate that every artifact path in the manifest exists",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write report path and summary metrics into latest_evaluation when checks pass",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_model(manifest: dict[str, Any], model_name: str) -> dict[str, Any]:
    for model in manifest.get("models", []):
        if model.get("name") == model_name:
            return model
    raise KeyError(f"model not found in manifest: {model_name}")


def iter_artifact_checks(manifest: dict[str, Any], base_dir: Path) -> list[ArtifactCheck]:
    checks: list[ArtifactCheck] = []
    for model in manifest.get("models", []):
        model_name = str(model.get("name", "unknown"))
        artifacts = model.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        for artifact_type, artifact_path in artifacts.items():
            path_text = str(artifact_path)
            path = Path(path_text)
            if not path.is_absolute():
                path = base_dir / path
            exists = path.exists()
            checks.append(
                ArtifactCheck(
                    model_name=model_name,
                    artifact_type=str(artifact_type),
                    path=path_text,
                    exists=exists,
                    is_dir=path.is_dir() if exists else False,
                    size_bytes=path.stat().st_size if exists and path.is_file() else 0,
                )
            )
    return checks


def build_artifact_payload(checks: list[ArtifactCheck]) -> dict[str, Any]:
    missing = [check for check in checks if not check.exists]
    return {
        "passed": not missing,
        "artifact_count": len(checks),
        "missing_count": len(missing),
        "artifacts": [
            {
                "model_name": check.model_name,
                "artifact_type": check.artifact_type,
                "path": check.path,
                "exists": check.exists,
                "is_dir": check.is_dir,
                "size_bytes": check.size_bytes,
            }
            for check in checks
        ],
    }


def get_report_values(report: dict[str, Any]) -> dict[str, float]:
    overall = report.get("metrics", {}).get("overall", {})
    latency = report.get("latency", {})
    return {
        "precision": float(overall.get("precision", 0.0)),
        "recall": float(overall.get("recall", 0.0)),
        "avg_latency_ms": float(latency.get("avg_ms", 0.0)),
    }


def evaluate_criteria(criteria: dict[str, Any], values: dict[str, float]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if "min_precision" in criteria:
        expected = float(criteria["min_precision"])
        actual = values["precision"]
        checks.append(CheckResult("precision", actual, expected, actual >= expected))
    if "min_recall" in criteria:
        expected = float(criteria["min_recall"])
        actual = values["recall"]
        checks.append(CheckResult("recall", actual, expected, actual >= expected))
    if "max_avg_latency_ms" in criteria:
        expected = float(criteria["max_avg_latency_ms"])
        actual = values["avg_latency_ms"]
        checks.append(CheckResult("avg_latency_ms", actual, expected, actual <= expected))
    return checks


def check_insightface_tensorrt_report(report: dict[str, Any]) -> list[str]:
    """InsightFace TensorRT POC report의 필수 측정값을 검증한다."""
    errors: list[str] = []
    model_id = report.get("model_id")
    if model_id != "arcface-w600k-r50-tensorrt-v1":
        errors.append(f"unexpected InsightFace model_id: {model_id}")

    for key, minimum in (
        ("gallery_images", 2),
        ("identities", 2),
        ("genuine_pairs", 1),
        ("impostor_pairs", 1),
    ):
        if int(report.get(key, 0)) < minimum:
            errors.append(f"InsightFace {key} must be at least {minimum}")

    if report.get("p95_latency_ms") is None:
        errors.append("InsightFace p95_latency_ms is required")
    return errors


def build_latest_evaluation(report_path: Path, report: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    return {
        "report": str(report_path),
        "image_count": report.get("image_count", 0),
        "precision": values["precision"],
        "recall": values["recall"],
        "avg_latency_ms": values["avg_latency_ms"],
        "settings": report.get("settings", {}),
    }


def update_manifest(manifest_path: Path, manifest: dict[str, Any], model_name: str, latest: dict[str, Any]) -> None:
    model = find_model(manifest, model_name)
    model["latest_evaluation"] = latest
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)

    try:
        manifest = load_json(manifest_path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    if args.check_artifacts:
        checks = iter_artifact_checks(manifest, manifest_path.resolve().parents[1])
        payload = build_artifact_payload(checks)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["passed"] else 1

    if not args.model_name or not args.report:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": "--model-name and --report are required unless --check-artifacts is used",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    report_path = Path(args.report)
    try:
        report = load_json(report_path)
        model = find_model(manifest, args.model_name)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    criteria = model.get("acceptance_criteria", {})
    values = get_report_values(report)
    checks = evaluate_criteria(criteria, values)

    failed = [check for check in checks if not check.passed]
    payload = {
        "model_name": args.model_name,
        "report": str(report_path),
        "passed": not failed,
        "checks": [
            {
                "metric": check.metric,
                "actual": check.actual,
                "expected": check.expected,
                "passed": check.passed,
            }
            for check in checks
        ],
    }

    if args.update_manifest and not failed:
        latest = build_latest_evaluation(report_path, report, values)
        update_manifest(manifest_path, manifest, args.model_name, latest)
        payload["updated_manifest"] = str(manifest_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
