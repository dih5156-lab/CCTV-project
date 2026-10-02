"""로컬 합성 입력으로 수신/저장 실패 경계를 재현한다. 결함이 있으면 종료 코드 2."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
SOURCES = (
    "src/core/_event_publish.py",
    "src/services/external_ingest.py",
    "src/protocols/mqtt_subscriber.py",
    "parser-python/service/sensor_service.py",
    "parser-python/database/processor.py",
    "parser-python/tlv/parser.py",
)
TLV_SAMPLE = "0065761000888a00c40041a80000c40142a00000e4668100000078e4158e68c0c24f"


def video_case(case_id: str, invalid_time: bool) -> dict:
    from src.core._event_publish import run_publish_loop
    from src.core.events import DetectionEvent, EventType

    queue = Queue()
    queue.put(
        DetectionEvent(
            EventType.PERSON,
            0,
            0,
            10,
            20,
            0.9,
            float("nan") if invalid_time else 1700000000,
            metadata={"camera_id": "synthetic-camera"},
        )
    )
    publish = Mock(side_effect=ConnectionError("synthetic publish failure"))
    counters = Counter()
    iterations = iter((True, False))
    run_publish_loop(
        is_running=lambda: next(iterations),
        stop_event=Event(),
        event_queue=queue,
        topic_prefix="synthetic/events",
        mqtt_publish=publish,
        event_publisher=None,
        increment_stat=lambda key: counters.update([key]),
        queue_timeout_sec=0,
    )
    return {
        "id": case_id,
        "kind": "failure_boundary",
        "scenario": (
            "video_invalid_timestamp" if invalid_time else "video_publish_exception"
        ),
        "observed": {
            "queue_remaining": queue.qsize(),
            "counters": dict(counters),
            "publish_calls": publish.call_count,
        },
        "expected": "Failed input must remain available for quarantine or retry.",
        "meets_target": queue.qsize() == 1,
        "issues": ["I12" if invalid_time else "I17"],
    }


def sensor_attempt(payload: str) -> tuple[list, dict]:
    from service.sensor_service import SensorService
    from tlv.parser import Parser

    records = []
    forwarder = Mock()
    service = SensorService.__new__(SensorService)
    service.device_info = SimpleNamespace(get_device_id=lambda _: "synthetic-sensor")
    service._tlv_parser = Parser()
    service._data_processor = SimpleNamespace(add_data=records.append)
    service._edgex_forwarder = forwarder
    service.event_service = Mock()
    error_type = None
    try:
        service.process_sensor_data(
            app_id="synthetic-app",
            dev_eui="synthetic-device",
            payload=payload,
            channel=0,
            frequency=0,
            received_at=1700000000000,
        )
    except Exception as exc:
        error_type = type(exc).__name__
    return records, {
        "queued_record_types": [type(record).__name__ for record in records],
        "forward_calls": forwarder.publish.call_count,
        "exception_to_caller": error_type,
    }


def sensor_cases() -> list[dict]:
    cases = []
    for case_id, name, payload in (
        ("B03", "tlv_invalid_base64", "%%%"),
        ("B04", "tlv_truncated_bytes", base64.b64encode(b"\x00").decode()),
    ):
        records, observed = sensor_attempt(payload)
        cases.append(
            {
                "id": case_id,
                "kind": "failure_boundary",
                "scenario": name,
                "observed": observed,
                "expected": "Malformed source must reach an error store or propagate a failure.",
                "meets_target": bool(records)
                or observed["exception_to_caller"] is not None,
                "issues": ["I12"],
            }
        )
    records, observed = sensor_attempt(
        base64.b64encode(bytes.fromhex(TLV_SAMPLE)).decode()
    )
    cases.append(
        {
            "id": "B05",
            "kind": "control",
            "scenario": "tlv_valid_fixture",
            "observed": observed,
            "expected": "Existing valid parser fixture queues typed and raw rows; this is not DB commit.",
            "meets_target": observed["queued_record_types"] == ["T34954", "SensorData"]
            and observed["forward_calls"] == 1
            and observed["exception_to_caller"] is None,
            "issues": [],
        }
    )
    if not cases[-1]["meets_target"]:
        raise RuntimeError("Valid TLV control failed; batch probe would be misleading")
    cases.append(batch_case(records))
    return cases


def batch_case(records: list) -> dict:
    from database import processor as module

    processor = module.DataProcessor.__new__(module.DataProcessor)
    processor._db = object()  # 실제 DB/worker를 생성하지 않는다.
    processor._lock = Lock()
    processor._flush_lock = Lock()
    attributes = [
        "_t3",
        "_t34950",
        "_t34952",
        "_t34954",
        "_t34955",
        "_t34956",
        "_t34957",
        "_t34958",
        "_sensor_data",
    ]
    for attribute in attributes:
        setattr(processor, attribute, [])
    for record in records:
        processor.add_data(record)
    with patch.object(
        module, "bulk_insert", side_effect=RuntimeError("synthetic DB failure")
    ) as insert:
        processor._process_all_tables()
        failed_attempts = insert.call_count
    remaining = sum(len(getattr(processor, name)) for name in attributes)
    with patch.object(module, "bulk_insert") as recovered_insert:
        processor._process_all_tables()
        retried_rows = sum(
            len(call.args[2]) for call in recovered_insert.call_args_list
        )
    return {
        "id": "B06",
        "kind": "failure_boundary",
        "scenario": "sensor_batch_db_failure_then_recovery",
        "observed": {
            "input_rows": len(records),
            "failed_insert_calls": failed_attempts,
            "queue_rows_after_failure": remaining,
            "retried_rows_after_recovery": retried_rows,
        },
        "expected": "Failed rows must still be retried after DB recovery.",
        "meets_target": remaining == len(records) and retried_rows == len(records),
        "issues": ["I01"],
    }


def external_cases(directory: Path) -> list[dict]:
    import sqlite3

    from src.protocols.mqtt_subscriber import MqttTopicSubscriber
    from src.services.external_ingest import (
        ExternalIngestService,
        IngestEventRepository,
        IngestStats,
    )

    cases = []
    for case_id, scenario in (
        ("B07", "db_failure"),
        ("B08", "invalid_quarantine"),
        ("B09", "publish_failure"),
    ):
        repo = IngestEventRepository(str(directory / f"{case_id}.db"))
        repo.init()
        service = ExternalIngestService.__new__(ExternalIngestService)
        service._repo = repo
        service._stats = IngestStats()
        calls = []

        def publish(payload):
            calls.append("publish")
            if scenario == "publish_failure":
                raise ConnectionError("synthetic publisher failure")
            return True

        def save(**kwargs):
            calls.append("save")
            if scenario == "db_failure":
                raise sqlite3.OperationalError("synthetic database failure")
            original_save(**kwargs)

        service._publisher = SimpleNamespace(publish_event=publish)
        original_save = repo.save
        subscriber = MqttTopicSubscriber.__new__(MqttTopicSubscriber)
        subscriber.message_handler = service.handle_message
        raw = {"camera_id": "synthetic-sensor", "timestamp": 1700000000}
        if scenario == "invalid_quarantine":
            raw["timestamp"] = "not-a-time"
        message = SimpleNamespace(
            topic="synthetic/inputs", payload=json.dumps(raw).encode()
        )
        error_type = None
        with patch.object(repo, "save", side_effect=save):
            try:
                subscriber._on_message(None, None, message)
            except Exception as exc:
                error_type = type(exc).__name__
        with sqlite3.connect(repo.db_path) as conn:
            rows = conn.execute(
                "SELECT raw_payload, normalized_payload, republished FROM ingest_events"
            ).fetchall()
        control = scenario == "invalid_quarantine"
        preserved = bool(rows) and json.loads(rows[0][0]) == raw
        meets_target = preserved
        if control:
            meets_target = preserved and calls == ["save"] and rows[0][2] == 0
            meets_target = (
                meets_target
                and json.loads(rows[0][1])["validation"]["status"] == "invalid"
            )
        cases.append(
            {
                "id": case_id,
                "kind": "control" if control else "failure_boundary",
                "scenario": "external_" + scenario,
                "observed": {
                    "call_order": calls,
                    "stored_rows": len(rows),
                    "raw_preserved": preserved,
                    "exception_from_subscriber": error_type,
                    "stats": service.get_stats(),
                },
                "expected": (
                    "Keep original input without normal publish."
                    if control
                    else "Original input must survive a DB/publisher failure; current callback has no retry store."
                ),
                "meets_target": meets_target,
                "issues": [] if control else ["I13"],
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "parser-python"))
    logging.disable(logging.CRITICAL)  # 합성 장애 로그 대신 구조화된 결과만 남긴다.
    with TemporaryDirectory(prefix="cctv-ingest-probe-") as directory:
        cases = [video_case("B01", True), video_case("B02", False)]
        cases.extend(sensor_cases())
        cases.extend(external_cases(Path(directory)))
    report = {
        "synthetic": True,
        "scope": "CPU, temporary SQLite, mock publisher/DB failure; no broker/GPU/device/service startup",
        "limits": "Probes current functions, not process restart, MQTT ACK, power loss or full pipeline durability.",
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        "cases": cases,
        "summary": dict(
            total=len(cases),
            controls=sum(case["kind"] == "control" for case in cases),
            unmet_targets=sum(not case["meets_target"] for case in cases),
        ),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 2 if report["summary"]["unmet_targets"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
