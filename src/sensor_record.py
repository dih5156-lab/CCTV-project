"""Project frozen TLV data into common observations independently of PostgreSQL."""

import json
import logging
import threading
import time

from .event_ingest_store import EventIngestStore
from .sensor_observation import build_sensor_observation

logger = logging.getLogger(__name__)
CONVERTER_VERSION = "tlv-observation-v1"


class SensorRecordStore(EventIngestStore):
    def pending_sensor_records(self, limit: int = 100) -> list[str]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT r.ingest_id FROM ingest_receipts r "
                    "JOIN sensor_ingest_state p USING(ingest_id) "
                    "LEFT JOIN event_ingest_state s USING(ingest_id) "
                    "WHERE r.source_namespace='tlv_mqtt' AND p.projection_json IS NOT NULL "
                    "AND (s.ingest_id IS NULL OR (s.stage='pending' AND s.next_attempt_at<=?)) "
                    "ORDER BY r.rowid LIMIT ?",
                    (time.time(), limit),
                )
            ]

    def project_sensor(self, ingest_id: str) -> None:
        with self._connect() as connection:
            receipt = connection.execute(
                "SELECT r.*, p.projection_json, s.stage AS event_stage FROM ingest_receipts r "
                "JOIN sensor_ingest_state p USING(ingest_id) "
                "LEFT JOIN event_ingest_state s USING(ingest_id) "
                "WHERE r.ingest_id=? AND r.source_namespace='tlv_mqtt' "
                "AND p.projection_json IS NOT NULL",
                (ingest_id,),
            ).fetchone()
            if receipt is None:
                raise ValueError("A frozen TLV projection is required")
            # 완료된 접수는 읽기만으로 종료하고, 재시도 중인 상태도 불필요한 쓰기를 피한다.
            # 최초 동시 준비는 INSERT OR IGNORE와 정규화 transaction의 상태 재검사로 보호한다.
            if receipt["event_stage"] is not None:
                if receipt["event_stage"] != "pending":
                    return
            else:
                connection.execute(
                    "INSERT OR IGNORE INTO event_ingest_state(ingest_id) VALUES (?)",
                    (ingest_id,),
                )
        if self.get_event_receipt(ingest_id)["stage"] != "pending":
            return
        try:
            projection = json.loads(receipt["projection_json"])
            sensor_rows = projection["sensor_data"]
            if not isinstance(sensor_rows, list) or len(sensor_rows) != 1:
                raise ValueError("One sensor_data row is required")
            sensor = sensor_rows[0]
            if not isinstance(sensor, dict) or not isinstance(
                sensor["payload_tlv"], dict
            ):
                raise ValueError("Decoded TLV must be an object")
            observation = build_sensor_observation(
                {
                    "table": "t" + str(sensor["object_id"]),
                    "data": sensor["payload_tlv"],
                },
                namespace="tlv_mqtt",
                source_record_id=ingest_id,
                device_id=sensor["device_id"],
                received_at=receipt["received_at"],
            )
        except (ValueError, TypeError, KeyError, OverflowError):
            self.reject_event(
                ingest_id,
                [{"field": "projection", "code": "invalid_projection"}],
                converter_version=CONVERTER_VERSION,
            )
            return
        if observation is None:
            self.reject_event(
                ingest_id,
                [{"field": "profile", "code": "unsupported_profile"}],
                converter_version=CONVERTER_VERSION,
            )
            return
        self.record_observation(
            ingest_id, observation, converter_version=CONVERTER_VERSION
        )


class SensorRecordWorker:
    def __init__(self, store: SensorRecordStore):
        self.store = store
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SensorRecord"
        )

    def start(self):
        self._thread.start()

    def drain_once(self):
        with self._lock:
            for ingest_id in self.store.pending_sensor_records():
                if self._stop.is_set():
                    break
                try:
                    self.store.project_sensor(ingest_id)
                except Exception as error:
                    self.store.retry_event(ingest_id, type(error).__name__)
                    logger.warning(
                        "Sensor record pending retry: %s (%s)",
                        ingest_id,
                        type(error).__name__,
                    )

    def _run(self):
        while not self._stop.is_set():
            try:
                self.drain_once()
            except Exception:
                logger.exception("Sensor record worker could not update durable state")
            self._stop.wait(1)

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise TimeoutError("Sensor record worker has not stopped")
