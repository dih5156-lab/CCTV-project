"""Send synthetic sensor/video records from a marked POC DB; never drives devices."""

import argparse
import json
import signal
import sqlite3
import ssl
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.event_ingest_store import EventIngestStore
from src.monitoring_contract import MonitoringBinding
from src.monitoring_delivery import MonitoringDeliveryStore
from src.monitoring_worker import MonitoringDeliveryWorker
from src.protocols.monitoring_http import HttpMonitoringReader
from src.protocols.record_mqtt import MqttRecordTransport
from src.sensor_record import SensorRecordStore, SensorRecordWorker


def build_workers(source_db, client_dir):
    source_db, client_dir = Path(source_db), Path(client_dir)
    config = json.loads((client_dir / "config.json").read_text())
    binding = MonitoringBinding(**config["binding"])
    marker = json.loads(source_db.with_suffix(".poc.json").read_text())
    if marker != binding.as_dict() or not source_db.is_file():
        raise ValueError("Client binding must match the dedicated POC source DB marker")
    source = MonitoringDeliveryStore(source_db)
    if source.origin_id != binding.origin_id:
        raise ValueError("POC source DB origin mismatch")
    ca = client_dir / "ca.crt"
    sender = MqttRecordTransport(
        config["mqtt_host"],
        port=config["mqtt_port"],
        transport="tcp",
        username=config["mqtt_username"],
        password=(client_dir / "sender-password").read_text().strip(),
        tls_context=ssl.create_default_context(cafile=str(ca)),
    )
    reader = HttpMonitoringReader(
        config["https_url"], (client_dir / "api-token").read_text().strip(), ca
    )
    workers = [
        MonitoringDeliveryWorker(
            source, sender, reader, namespace=namespace, binding=binding, ack_wait=1
        )
        for namespace in ("video_detection", "tlv_mqtt")
    ]
    return source, binding, workers


def seed_demo(source):
    # Repeat invocations resume the same two records, including partial seeding.
    marker = source.path.with_suffix(".demo.json")
    sensor = SensorRecordStore(source.path)
    if marker.exists():
        receipt = json.loads(marker.read_text())["sensor_ingest_id"]
    else:
        receipt = sensor.accept(
            "tlv_mqtt", "poc-sensor", {"demo": True, "temperature_c": 0}
        )
        marker.write_text(json.dumps({"sensor_ingest_id": receipt}) + "\n")
    sensor.prepare(
        receipt,
        json.dumps(
            {
                "sensor_data": [
                    {
                        "device_id": "poc-sensor",
                        "object_id": "34954",
                        "payload_tlv": {"temperature_c": 0, "created_at": 0},
                    }
                ]
            }
        ),
    )
    SensorRecordWorker(sensor).drain_once()
    events = EventIngestStore(source.path)
    event = {"event_id": "poc-video-1", "demo": True, "value": 0}
    received = events.accept_event_json("video_detection", event, destination=None)
    events.record_event(
        received, event, converter_version="remote-poc-v1", recovery=True
    )


def delivery_summary(source, binding):
    with sqlite3.connect(source.path) as connection:
        rows = connection.execute(
            "SELECT m.state,count(*) FROM monitoring_delivery m JOIN delivery_jobs j USING(job_id) "
            "WHERE j.destination=? GROUP BY m.state",
            (binding.destination,),
        ).fetchall()
    counts = dict(rows)
    return {
        "states": counts,
        "total": sum(counts.values()),
        "stored": counts.get("stored", 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--client-dir", type=Path, required=True)
    parser.add_argument(
        "--demo", action="store_true", help="Seed/resume two synthetic records"
    )
    parser.add_argument("--run-seconds", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.run_seconds <= 3600:
        parser.error("--run-seconds must be 1..3600")
    source, binding, workers = build_workers(args.source_db, args.client_dir)
    if args.demo:
        seed_demo(source)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    deadline = time.monotonic() + args.run_seconds
    while not stopped.is_set() and time.monotonic() < deadline:
        for worker in workers:
            worker.drain_once(10, stop_event=stopped)
        summary = delivery_summary(source, binding)
        if summary["total"] and summary["stored"] == summary["total"]:
            break
        stopped.wait(1)
    summary = delivery_summary(source, binding)
    print(json.dumps({"binding": binding.as_dict(), **summary}, indent=2))
    return 0 if summary["total"] > 0 and summary["stored"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
