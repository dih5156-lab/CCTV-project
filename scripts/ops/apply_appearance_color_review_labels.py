#!/usr/bin/env python3
"""Apply reviewed upper/lower appearance colors to SQLite safely."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any

ALLOWED_COLORS = frozenset(
    {
        "black",
        "blue",
        "brown",
        "gray",
        "green",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    }
)
COLOR_FIELDS = ("upper_color", "lower_color")
SKIP_VALUES = {None, "exclude"}


def validate_labels(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported review label schema_version")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review label items must be a list")

    validated = []
    seen_ids: set[int] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("each review label item must be an object")
        item_id = raw_item.get("id")
        if (
            not isinstance(item_id, int)
            or isinstance(item_id, bool)
            or item_id <= 0
        ):
            raise ValueError(f"invalid id: {item_id!r}")
        if item_id in seen_ids:
            raise ValueError(f"duplicate id: {item_id}")
        seen_ids.add(item_id)

        item: dict[str, Any] = {"id": item_id}
        for field in COLOR_FIELDS:
            value = raw_item.get(field)
            if value not in ALLOWED_COLORS and value not in SKIP_VALUES:
                raise ValueError(f"invalid {field}: {value!r}")
            item[field] = value
        validated.append(item)
    return validated


def plan_updates(
    db_path: Path,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    item_ids = [item["id"] for item in items]
    if not item_ids:
        return []

    placeholders = ",".join("?" for _ in item_ids)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT id, upper_color, lower_color
            FROM appearance_log
            WHERE id IN ({placeholders})
            """,
            item_ids,
        ).fetchall()

    current_by_id = {int(row["id"]): dict(row) for row in rows}
    missing_ids = sorted(set(item_ids) - set(current_by_id))
    if missing_ids:
        raise ValueError(f"appearance ids not found: {missing_ids}")

    updates = []
    for item in items:
        current = current_by_id[item["id"]]
        after = {}
        for field in COLOR_FIELDS:
            value = item[field]
            if value not in SKIP_VALUES and value != current[field]:
                after[field] = value
        if after:
            updates.append(
                {
                    "id": item["id"],
                    "before": {
                        "upper_color": current["upper_color"],
                        "lower_color": current["lower_color"],
                    },
                    "after": after,
                }
            )
    return updates


def _default_backup_path(db_path: Path) -> Path:
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return db_path.with_name(f"{db_path.name}.{suffix}.bak")


def _backup_database(source_path: Path, destination_path: Path) -> None:
    if destination_path.exists():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        sqlite3.connect(source_path) as source,
        sqlite3.connect(destination_path) as destination,
    ):
        source.backup(destination)


def apply_review_labels(
    db_path: Path,
    labels_path: Path,
    apply: bool = False,
    backup_path: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    items = validate_labels(payload)
    updates = plan_updates(db_path, items)

    if not apply:
        return {
            "mode": "dry-run",
            "updates": len(updates),
            "changes": updates,
        }

    if not updates:
        return {
            "mode": "applied",
            "updates": 0,
            "changes": [],
            "backup": None,
        }

    resolved_backup_path = backup_path or _default_backup_path(db_path)
    _backup_database(db_path, resolved_backup_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for update in updates:
            assignments = ", ".join(
                f"{field} = ?" for field in update["after"]
            )
            values = [*update["after"].values(), update["id"]]
            connection.execute(
                f"UPDATE appearance_log SET {assignments} WHERE id = ?",
                values,
            )
        connection.commit()

    return {
        "mode": "applied",
        "updates": len(updates),
        "changes": updates,
        "backup": str(resolved_backup_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/runtime/appearances.db"),
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제 DB를 수정합니다. 생략하면 변경 예정 내용만 출력합니다.",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        help="적용 전 백업 경로. 생략하면 DB 옆에 시간 기반 이름을 사용합니다.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = apply_review_labels(
        db_path=args.db,
        labels_path=args.labels,
        apply=args.apply,
        backup_path=args.backup,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
