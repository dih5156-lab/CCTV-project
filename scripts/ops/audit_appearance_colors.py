#!/usr/bin/env python3
"""최근 외형 로그의 색상 판정과 저장 crop 재분석 결과를 비교한다."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ai._appearance_analyzer import AppearanceAnalyzer  # noqa: E402

COLOR_FIELDS = ("upper_color", "lower_color")


def _resolve_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    text = str(raw_path)
    if text.startswith("/app/"):
        candidate = PROJECT_ROOT / text.removeprefix("/app/")
        if candidate.exists():
            return candidate
    candidate = PROJECT_ROOT / text
    if candidate.exists():
        return candidate
    return None


def _query_rows(db_path: Path, limit: int, backend: Optional[str]) -> list[sqlite3.Row]:
    where = "crop_path IS NOT NULL AND crop_path != ''"
    params: list[object] = []
    if backend:
        where += " AND attribute_backend = ?"
        params.append(backend)
    params.append(limit)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(appearance_log)").fetchall()
        }
        metadata_expr = (
            "attribute_metadata"
            if "attribute_metadata" in columns
            else "NULL AS attribute_metadata"
        )
        return list(
            conn.execute(
                f"""
                SELECT id, camera_id, track_id, attribute_backend,
                       upper_color, lower_color, crop_path, timestamp,
                       {metadata_expr}
                FROM appearance_log
                WHERE {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            )
        )


def _parse_metadata(raw_metadata: object) -> dict[str, object]:
    if not raw_metadata:
        return {}
    if isinstance(raw_metadata, dict):
        return raw_metadata
    try:
        decoded = json.loads(str(raw_metadata))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _color_candidates(metadata: dict[str, object]) -> dict[str, object]:
    candidates = metadata.get("color_candidates")
    return candidates if isinstance(candidates, dict) else {}


def _color_sources(metadata: dict[str, object]) -> dict[str, object]:
    sources = metadata.get("color_sources")
    return sources if isinstance(sources, dict) else {}


def _source_for(metadata: dict[str, object], field: str) -> str:
    source = _color_sources(metadata).get(field)
    return str(source or "unknown")


def _candidate_for(metadata: dict[str, object], field: str) -> dict[str, object]:
    candidate = _color_candidates(metadata).get(field)
    return candidate if isinstance(candidate, dict) else {}


def _collect_metadata_stats(rows: list[sqlite3.Row]) -> dict[str, object]:
    source_counts: dict[str, Counter[str]] = {field: Counter() for field in COLOR_FIELDS}
    hsv_lab_disagreements: Counter[str] = Counter()
    model_overrides: Counter[str] = Counter()
    rows_with_metadata = 0

    for row in rows:
        metadata = _parse_metadata(row["attribute_metadata"])
        if metadata:
            rows_with_metadata += 1
        for field in COLOR_FIELDS:
            source = _source_for(metadata, field)
            source_counts[field][source] += 1
            candidate = _candidate_for(metadata, field)
            hsv_color = str(candidate.get("hsv_color") or "unknown")
            lab_color = str(candidate.get("lab_color") or "unknown")
            selected = str(candidate.get("selected") or row[field] or "unknown")
            if hsv_color != "unknown" and lab_color != "unknown" and hsv_color != lab_color:
                hsv_lab_disagreements[field] += 1
            if source not in {"unknown", "hsv", "lab", "lab_fallback", "not_visible", "no_helmet"}:
                if selected != hsv_color and hsv_color != "unknown":
                    model_overrides[field] += 1

    return {
        "rows_with_metadata": rows_with_metadata,
        "source_counts": {
            field: dict(counts.most_common())
            for field, counts in source_counts.items()
        },
        "hsv_lab_disagreements": dict(hsv_lab_disagreements),
        "model_overrides": dict(model_overrides),
    }


def _build_report(db_path: Path, *, limit: int, backend: Optional[str]) -> dict[str, object]:
    analyzer = AppearanceAnalyzer()
    rows = _query_rows(db_path, limit, backend)
    stored_distribution: Counter[tuple[str, str]] = Counter()
    recalculated_distribution: Counter[tuple[str, str]] = Counter()
    backend_distribution: Counter[str] = Counter()
    mismatch_by_backend: Counter[str] = Counter()
    mismatches = []
    missing_crops = 0

    for row in rows:
        backend_name = str(row["attribute_backend"] or "unknown")
        backend_distribution[backend_name] += 1
        stored = (row["upper_color"] or "unknown", row["lower_color"] or "unknown")
        stored_distribution[stored] += 1
        crop_path = _resolve_path(row["crop_path"])
        if crop_path is None:
            missing_crops += 1
            continue
        image = cv2.imread(str(crop_path))
        if image is None:
            missing_crops += 1
            continue
        height, width = image.shape[:2]
        attrs = analyzer.extract_attributes(image, 0, 0, width, height)
        recalculated = (
            str(attrs.get("upper_color") or "unknown"),
            str(attrs.get("lower_color") or "unknown"),
        )
        recalculated_distribution[recalculated] += 1
        if stored != recalculated:
            mismatch_by_backend[backend_name] += 1
            mismatches.append((row, stored, recalculated, crop_path))

    return {
        "db_path": str(db_path),
        "checked_rows": len(rows),
        "missing_crops": missing_crops,
        "mismatches": len(mismatches),
        "stored_distribution": {
            f"{upper}/{lower}": count
            for (upper, lower), count in stored_distribution.most_common(10)
        },
        "backend_distribution": dict(backend_distribution.most_common()),
        "mismatch_by_backend": dict(mismatch_by_backend.most_common()),
        "recalculated_distribution": {
            f"{upper}/{lower}": count
            for (upper, lower), count in recalculated_distribution.most_common(10)
        },
        "metadata": _collect_metadata_stats(rows),
        "mismatch_samples": [
            {
                "id": row["id"],
                "track_id": row["track_id"],
                "backend": row["attribute_backend"],
                "stored": {"upper_color": stored[0], "lower_color": stored[1]},
                "recalculated": {"upper_color": recalculated[0], "lower_color": recalculated[1]},
                "crop_path": str(crop_path),
                "metadata": _parse_metadata(row["attribute_metadata"]),
            }
            for row, stored, recalculated, crop_path in mismatches[:10]
        ],
    }


def _print_report(report: dict[str, object]) -> None:
    print(f"db_path={report['db_path']}")
    print(
        f"checked_rows={report['checked_rows']} "
        f"missing_crops={report['missing_crops']} "
        f"mismatches={report['mismatches']}"
    )
    print("stored_distribution:")
    for color_pair, count in dict(report["stored_distribution"]).items():
        print(f"  {color_pair}: {count}")
    print("backend_distribution:")
    for backend_name, count in dict(report["backend_distribution"]).items():
        print(f"  {backend_name}: {count}")
    print("mismatch_by_backend:")
    for backend_name, count in dict(report["mismatch_by_backend"]).items():
        print(f"  {backend_name}: {count}")
    print("recalculated_distribution:")
    for color_pair, count in dict(report["recalculated_distribution"]).items():
        print(f"  {color_pair}: {count}")

    metadata = dict(report["metadata"])
    print(f"metadata_rows={metadata['rows_with_metadata']}")
    print("color_sources:")
    for field, counts in dict(metadata["source_counts"]).items():
        formatted = ", ".join(f"{source}={count}" for source, count in dict(counts).items())
        print(f"  {field}: {formatted or 'none'}")
    print("hsv_lab_disagreements:")
    for field in COLOR_FIELDS:
        print(f"  {field}: {dict(metadata['hsv_lab_disagreements']).get(field, 0)}")
    print("model_overrides:")
    for field in COLOR_FIELDS:
        print(f"  {field}: {dict(metadata['model_overrides']).get(field, 0)}")

    mismatch_samples = list(report["mismatch_samples"])
    if mismatch_samples:
        print("mismatch_samples:")
        for sample in mismatch_samples:
            stored = dict(sample["stored"])
            recalculated = dict(sample["recalculated"])
            metadata = dict(sample["metadata"])
            sources = _color_sources(metadata)
            print(
                "  "
                f"id={sample['id']} track={sample['track_id']} backend={sample['backend']} "
                f"stored={stored['upper_color']}/{stored['lower_color']} "
                f"recalculated={recalculated['upper_color']}/{recalculated['lower_color']} "
                f"sources={sources} crop={sample['crop_path']}"
            )


def audit_colors(db_path: Path, *, limit: int, backend: Optional[str], as_json: bool = False) -> int:
    report = _build_report(db_path, limit=limit, backend=backend)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("APPEARANCES_DB", "data/runtime/appearances.db")),
        help="appearance_log SQLite DB path",
    )
    parser.add_argument("--limit", type=int, default=80, help="최근 crop 재분석 개수")
    parser.add_argument("--backend", default=None, help="attribute_backend 필터")
    parser.add_argument("--json", action="store_true", help="JSON 리포트 출력")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = _resolve_path(str(args.db)) or args.db
    if not db_path.exists():
        print(f"DB를 찾을 수 없습니다: {db_path}", file=sys.stderr)
        return 2
    return audit_colors(db_path, limit=max(1, args.limit), backend=args.backend, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
