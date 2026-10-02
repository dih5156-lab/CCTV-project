"""POC MQTT receiver (Unix by default, optional TCP/TLS) and local ACK lookup."""

import json
import math
import threading

import paho.mqtt.client as mqtt

from ..monitoring_contract import MonitoringContractError
from ..record_delivery import record_topic


class LocalMonitoringReader:
    def __init__(self, store):
        self.store = store
        self.endpoint = json.dumps(
            {
                "kind": "local-monitoring-sqlite",
                "consumer_id": store.consumer_id,
                "path": str(store.path.resolve()),
            },
            sort_keys=True,
        )

    def get_delivery_status(self, binding, delivery_id, *, timeout):
        if not math.isfinite(timeout) or timeout < 0.25:
            raise ValueError(
                "Local SQLite reader requires at least its 0.25s lock timeout"
            )
        return self.store.get_delivery_status(binding, delivery_id)


class LocalMqttReceiver:
    def __init__(
        self,
        store,
        binding,
        socket_path,
        *,
        timeout=5,
        transport="unix",
        port=1883,
        username=None,
        password=None,
        tls_context=None,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite connect timeout required")
        self.store, self.binding, self.socket_path = store, binding, str(socket_path)
        if (
            transport not in {"unix", "tcp"}
            or type(port) is not int
            or not 1 <= port <= 65535
        ):
            raise ValueError("A supported MQTT transport/port is required")
        self.transport, self.port = transport, port
        self.username, self.password, self.tls_context = username, password, tls_context
        self.timeout = timeout
        self.topic = record_topic(binding.destination)
        self._client = None
        self._ready = threading.Event()
        self._lock = threading.RLock()
        self._status_lock = threading.Lock()
        self._last_error = None
        self._stored, self._duplicates, self._rejected, self._failures = 0, 0, 0, 0

    def _on_connect(self, client, userdata, flags, reason, properties):
        if not reason.is_failure:
            client.subscribe(self.topic, qos=1)

    def _on_subscribe(self, client, userdata, mid, reasons, properties):
        if reasons and all(not reason.is_failure for reason in reasons):
            self._ready.set()

    def _on_disconnect(self, client, userdata, flags, reason, properties):
        self._ready.clear()

    def _on_message(self, client, userdata, message):
        if message.topic != self.topic or message.qos != 1 or message.retain:
            with self._status_lock:
                self._last_error = "unexpected_mqtt_message"
                self._failures += 1
            client.disconnect()
            self._ready.clear()
            return
        try:
            ack = self.store.accept(self.binding, bytes(message.payload))
            with self._status_lock:
                self._stored += ack["status"] == "stored"
                self._duplicates += ack["status"] == "duplicate"
                self._last_error = None
        except MonitoringContractError as error:
            # accept() raises these only after its rejection transaction succeeds.
            with self._status_lock:
                self._rejected += 1
                self._last_error = error.code
        except Exception as error:
            with self._status_lock:
                self._failures += 1
                self._last_error = type(error).__name__
            self._ready.clear()
            client.disconnect()
            return
        # This transport ACK never stands in for source-persisted consumer ACK.
        if client.ack(message.mid, message.qos) != mqtt.MQTT_ERR_SUCCESS:
            self._ready.clear()
            client.disconnect()

    def is_ready(self):
        client = self._client
        return self._ready.is_set() and client is not None and client.is_connected()

    def start(self):
        with self._lock:
            if self.is_ready():
                return
            self.close()
            with self.store._connect() as connection:
                self.store._check_binding(connection, self.binding)
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                transport=self.transport,
                protocol=mqtt.MQTTv311,
                clean_session=True,
                reconnect_on_failure=False,
                manual_ack=True,
            )
            client.connect_timeout = self.timeout
            if self.username is not None:
                client.username_pw_set(self.username, self.password)
            if self.tls_context is not None:
                client.tls_set_context(self.tls_context)
            client.on_connect, client.on_subscribe = (
                self._on_connect,
                self._on_subscribe,
            )
            client.on_message, client.on_disconnect = (
                self._on_message,
                self._on_disconnect,
            )
            self._client = client
            try:
                client.connect(self.socket_path, self.port)
                client.loop_start()
                if not self._ready.wait(self.timeout) or not client.is_connected():
                    raise TimeoutError("Monitoring receiver subscription was not ready")
            except Exception:
                self.close()
                raise

    def status(self):
        with self._status_lock:
            return {
                "ready": self.is_ready(),
                "stored": self._stored,
                "duplicates": self._duplicates,
                "rejected": self._rejected,
                "storage_or_transport_failures": self._failures,
                "last_error": self._last_error,
            }

    def close(self):
        with self._lock:
            if self._client is not None:
                self._client.disconnect()
                self._client.loop_stop()
                self._client = None
            self._ready.clear()
