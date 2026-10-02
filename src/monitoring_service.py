"""Explicitly started periodic delivery; owns threads, not input/device pipelines."""

import math
import threading
import time


class MonitoringService:
    def __init__(self, workers, *, receiver=None, interval=1, batch_size=10):
        if not workers or not math.isfinite(interval) or interval <= 0:
            raise ValueError("workers and positive finite interval required")
        if type(batch_size) is not int or not 1 <= batch_size <= 1000:
            raise ValueError("batch_size must be 1..1000")
        self.workers, self.receiver = tuple(workers), receiver
        self.interval, self.batch_size = interval, batch_size
        self._stop = threading.Event()
        self._lifecycle = threading.Lock()
        self._status_lock = threading.Lock()
        self._thread = None
        self._state = "stopped"
        self._errors, self._completed = 0, 0
        self._last_error, self._last_cycle_at = None, None

    def start(self):
        with self._lifecycle:
            if self._state == "stopping":
                raise RuntimeError("Previous monitoring service has not stopped")
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            try:
                if self.receiver is not None:
                    self.receiver.start()
                self._thread = threading.Thread(
                    target=self._run, name="MonitoringDelivery", daemon=True
                )
                self._state = "running"
                self._thread.start()
            except Exception:
                self._state = "stopped"
                self._thread = None
                if self.receiver is not None:
                    self.receiver.close()
                raise

    def _run(self):
        while not self._stop.is_set():
            error_code, completed = None, 0
            try:
                if self.receiver is not None and not self.receiver.is_ready():
                    self.receiver.start()
            except Exception as error:
                error_code = type(error).__name__
            for worker in self.workers:
                if self._stop.is_set():
                    break
                try:
                    completed += worker.drain_once(
                        self.batch_size, stop_event=self._stop
                    )
                except Exception as error:
                    error_code = type(error).__name__
            with self._status_lock:
                self._completed += completed
                self._last_error = error_code
                self._errors += error_code is not None
                self._last_cycle_at = time.time()
            self._stop.wait(self.interval)

    def status(self):
        with self._status_lock:
            return {
                "state": self._state,
                "running": self._thread is not None and self._thread.is_alive(),
                "completed": self._completed,
                "failed_cycles": self._errors,
                "last_error": self._last_error,
                "last_cycle_at": self._last_cycle_at,
                "commit_observer_errors": sum(
                    worker.commit_observer_errors for worker in self.workers
                ),
            }

    def close(self, timeout=15):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite shutdown timeout required")
        with self._lifecycle:
            self._state = "stopping"
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout)
                if self._thread.is_alive():
                    raise TimeoutError(
                        "Monitoring worker is still finishing an operation"
                    )
            if self.receiver is not None:
                self.receiver.close()
            self._state = "stopped"


def build_local_service(
    source_path,
    receiver_path,
    consumer_id,
    destination,
    socket_path,
    *,
    namespaces,
    interval=1,
    timeout=5,
    lease_seconds=30,
    ack_wait=10,
    clock_sync="unknown",
):
    from pathlib import Path

    from .monitoring_contract import MonitoringBinding, require_string
    from .monitoring_delivery import MonitoringDeliveryStore
    from .monitoring_metrics import CommitObservations, MonitoringTimings
    from .monitoring_receiver import MonitoringReceiverStore
    from .monitoring_worker import MonitoringDeliveryWorker
    from .protocols.monitoring_local import LocalMonitoringReader, LocalMqttReceiver
    from .protocols.record_mqtt import MqttRecordTransport

    if Path(source_path).resolve() == Path(receiver_path).resolve():
        raise ValueError("Source and receiver databases must be separate")
    if (
        not math.isfinite(timeout)
        or timeout < 0.25
        or not math.isfinite(lease_seconds)
        or lease_seconds <= 2 * timeout + 2
        or not math.isfinite(ack_wait)
        or ack_wait <= 0
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError(
            "Invalid polling/lease/timeout; local reader needs timeout >= 0.25s"
        )
    if clock_sync not in {"confirmed", "unknown"} or isinstance(namespaces, str):
        raise ValueError("Invalid clock_sync or namespace collection")
    namespaces = tuple(
        dict.fromkeys(require_string(namespace) for namespace in namespaces)
    )
    if not namespaces:
        raise ValueError("At least one namespace required")
    source = MonitoringDeliveryStore(source_path)
    binding = MonitoringBinding(source.origin_id, consumer_id, destination)
    target = MonitoringReceiverStore(receiver_path, consumer_id=consumer_id)
    target.register_binding(binding)
    sender, reader = MqttRecordTransport(
        str(socket_path), transport="unix"
    ), LocalMonitoringReader(target)
    receiver = LocalMqttReceiver(target, binding, socket_path, timeout=timeout)
    timings, commits = MonitoringTimings(), CommitObservations(clock_sync=clock_sync)
    workers = [
        MonitoringDeliveryWorker(
            source,
            sender,
            reader,
            namespace=namespace,
            binding=binding,
            timeout=timeout,
            lease_seconds=lease_seconds,
            ack_wait=ack_wait,
            timings=timings,
            commit_observer=commits.committed,
        )
        for namespace in namespaces
    ]
    for namespace in namespaces:
        source.enqueue_monitoring(
            namespace, binding, sender.endpoint, reader.endpoint, limit=1
        )
    service = MonitoringService(workers, receiver=receiver, interval=interval)
    return service, source, target, binding, timings, commits
