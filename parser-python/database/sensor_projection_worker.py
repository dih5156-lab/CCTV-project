"""Retry persisted sensor inputs without replaying physical outputs."""

import logging
import threading

from psycopg2 import DataError, IntegrityError

from database.sensor_projection import ProjectionConflict, write_sensor_projection

logger = logging.getLogger(__name__)


class SensorProjectionWorker:
    def __init__(self, store, db, prepare_receipt):
        self.store = store
        self.db = db
        self.prepare_receipt = prepare_receipt
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SensorProjection"
        )

    def start(self):
        self._thread.start()

    def drain_once(self):
        with self._lock:
            for ingest_id in self.store.pending():
                if self._stop.is_set():
                    break
                try:
                    receipt = self.store.get(ingest_id)
                    if receipt["stage"] == "pending":
                        self.prepare_receipt(receipt)
                        receipt = self.store.get(ingest_id)
                    if receipt["stage"] == "ready":
                        write_sensor_projection(
                            self.db, ingest_id, receipt["projection_json"]
                        )
                        self.store.complete(ingest_id)
                except (DataError, IntegrityError, ProjectionConflict) as error:
                    self.store.quarantine(ingest_id, type(error).__name__)
                    logger.error(
                        "Sensor projection quarantined: %s (%s)",
                        ingest_id,
                        type(error).__name__,
                    )
                except Exception as error:
                    self.store.retry(ingest_id, type(error).__name__)
                    logger.warning(
                        "Sensor projection pending retry: %s (%s)",
                        ingest_id,
                        type(error).__name__,
                    )

    def _run(self):
        while not self._stop.is_set():
            try:
                self.drain_once()
            except Exception:
                logger.exception(
                    "Sensor projection worker could not update durable state"
                )
            self._stop.wait(1)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise TimeoutError(
                "Sensor projection worker has not stopped; preserve DB connection"
            )
