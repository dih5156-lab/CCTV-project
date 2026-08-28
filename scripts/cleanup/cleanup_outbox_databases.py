#!/usr/bin/env python3
"""전송 완료된 SQLite outbox 행을 보존 기간과 배치 한도에 맞춰 정리한다."""

from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutboxTarget:
    name: str
    path: Path
    table: str


def cleanup_target(
    target: OutboxTarget,
    *,
    cutoff_ms: int,
    batch_size: int,
    apply: bool,
) -> tuple[int, int]:
    """정리 가능 행 수와 실제 삭제 행 수를 반환한다."""
    if not target.path.is_file():
        print(f"[{target.name}] skipped: file not found ({target.path})")
        return 0, 0

    with sqlite3.connect(target.path, timeout=30) as connection:
        connection.execute("PRAGMA busy_timeout=30000")
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (target.table,),
        ).fetchone()
        if table_exists is None:
            raise RuntimeError(
                f"[{target.name}] required table not found: {target.table}"
            )

        eligible = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {target.table}
                WHERE status = 'sent'
                  AND sent_at_ms IS NOT NULL
                  AND sent_at_ms < ?
                """,
                (cutoff_ms,),
            ).fetchone()[0]
        )
        deleted = 0
        if apply and eligible:
            cursor = connection.execute(
                f"""
                DELETE FROM {target.table}
                WHERE id IN (
                    SELECT id
                    FROM {target.table}
                    WHERE status = 'sent'
                      AND sent_at_ms IS NOT NULL
                      AND sent_at_ms < ?
                    ORDER BY id ASC
                    LIMIT ?
                )
                """,
                (cutoff_ms, batch_size),
            )
            connection.commit()
            deleted = max(0, int(cursor.rowcount))

    print(
        f"[{target.name}] eligible={eligible}, deleted={deleted}, "
        f"batch_limit={batch_size}, path={target.path}"
    )
    return eligible, deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="전송 완료된 HTTP/MQTT outbox 행을 안전하게 정리"
    )
    parser.add_argument("--apply", action="store_true", help="실제 삭제 수행")
    parser.add_argument(
        "--http-db",
        default="data/runtime/action_http_outbox.db",
    )
    parser.add_argument(
        "--mqtt-db",
        default="data/runtime/mqtt_event_outbox.db",
    )
    parser.add_argument("--retention-days", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=25_000)
    args = parser.parse_args(argv)

    if args.retention_days < 1:
        parser.error("--retention-days must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    cutoff_ms = int(time.time() * 1000) - (
        args.retention_days * 24 * 60 * 60 * 1000
    )
    targets = (
        OutboxTarget("http", Path(args.http_db), "http_event_outbox"),
        OutboxTarget("mqtt", Path(args.mqtt_db), "mqtt_event_outbox"),
    )

    print(
        "=== SQLite outbox cleanup ===\n"
        f"mode={'apply' if args.apply else 'preview'}, "
        f"retention_days={args.retention_days}, batch_size={args.batch_size}"
    )
    try:
        for target in targets:
            cleanup_target(
                target,
                cutoff_ms=cutoff_ms,
                batch_size=args.batch_size,
                apply=args.apply,
            )
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
