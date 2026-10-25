#!/usr/bin/env python3
"""정지 시점 Action DB의 과거 ID를 새 장치 DB에 재실행 금지 상태로 옮긴다."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.edgex.request_idempotency import RequestIdempotencyCache  # noqa: E402
from src.services.command_identifiers import (  # noqa: E402
    build_device_command_id,
    resolve_logical_command_id,
)


def seed_legacy_requests(source_db: Path, destination_db: Path, device: str) -> dict:
    """원본은 읽기 전용으로 열고 완성된 새 DB만 노출하며 기존 저널은 덮지 않는다."""
    if device not in {"signboard", "speaker", "siren"}:
        raise ValueError("지원하지 않는 장치 유형입니다")
    if destination_db.exists() or destination_db.is_symlink():
        raise FileExistsError("대상 실행 이력 DB가 이미 있습니다; 덮어쓸 수 없습니다")
    identifiers: set[str] = set()
    statuses: Counter = Counter()
    selected = total = 0

    def identifier(value):
        if value is None or value == "":
            return ""
        if not isinstance(value, str) or value.strip() != value:
            raise ValueError("과거 식별자는 앞뒤 공백 없는 문자열이어야 합니다")
        return value

    with closing(sqlite3.connect(source_db.resolve().as_uri() + "?mode=ro", uri=True)) as source:
        source.execute("BEGIN")
        for command_id, topic, event_id, status, raw in source.execute(
            "SELECT command_id, topic, event_id, status, payload_json FROM action_commands"
        ):
            total += 1
            if not (topic == f"device/{device}" or
                    (topic.startswith("edgex/results/") and topic.endswith(f"/{device}"))):
                continue
            payload = json.loads(raw)
            command_id = identifier(command_id)
            if not isinstance(payload, dict) or not command_id:
                raise ValueError("과거 명령 본문/ID가 올바르지 않습니다")
            identifiers.add(command_id)
            identifiers.add(resolve_logical_command_id(command_id, identifier(payload.get("device_id"))))
            results = payload.get("device_results", [])
            if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
                raise ValueError("과거 장치별 결과 형식이 올바르지 않습니다")
            for item in [payload, *results]:
                request_id = identifier(item.get("request_id"))
                if request_id:
                    identifiers.add(request_id)
                    identifiers.add(resolve_logical_command_id(request_id, identifier(item.get("device_id"))))
                value = identifier(item.get("event_id"))
                if value:
                    identifiers.add(build_device_command_id(value, device))
            event_id = identifier(event_id)
            if event_id:
                identifiers.add(build_device_command_id(event_id, device))
            selected += 1
            statuses[status] += 1

    destination_db.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix=".legacy-", suffix=".sqlite3", dir=destination_db.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        RequestIdempotencyCache(database_path=temporary)
        with closing(sqlite3.connect(temporary)) as target, target:
            # SHA256는 빈 값이 될 수 없다. 내용 불명 이력을 ACK로 위조하지 않고 보류한다.
            target.executemany("INSERT INTO device_requests(request_id, request_hash) VALUES (?, '')",
                               ((value,) for value in sorted(identifiers)))
        with closing(sqlite3.connect(temporary)) as target:
            if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("새 실행 이력 DB 무결성 검사가 실패했습니다")
        # 같은 디렉터리의 완성본만 원자적으로 게시하며 경합 중 생긴 기존 DB도 덮지 않는다.
        os.link(temporary, destination_db)
    finally:
        temporary.unlink(missing_ok=True)
    return {"device": device, "source_rows": total, "selected_commands": selected,
            "held_identifiers": len(identifiers), "legacy_statuses": dict(statuses)}


def main() -> int:
    """전환 준비용 복사본을 이관한다. 서비스 중지나 운영 적용은 수행하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--destination-db", type=Path, required=True)
    parser.add_argument("--device", choices=["signboard", "speaker", "siren"], required=True)
    args = parser.parse_args()
    try:
        result = seed_legacy_requests(args.source_db, args.destination_db, args.device)
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        parser.exit(1, f"과거 요청 이관 실패: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
