"""Standalone TLS receiver. Run from the portable bundle's /app directory."""

import json
import logging
import os
import ssl
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from src.monitoring_api import create_monitoring_api
from src.monitoring_contract import MonitoringBinding
from src.monitoring_receiver import MonitoringReceiverStore
from src.protocols.monitoring_local import LocalMqttReceiver


def build_app(root):
    root = Path(root)
    binding = MonitoringBinding(**json.loads((root / "binding.json").read_text()))
    store = MonitoringReceiverStore(
        root / "data/receiver.db", consumer_id=binding.consumer_id
    )
    store.register_binding(binding)
    receiver = LocalMqttReceiver(
        store,
        binding,
        "broker",
        transport="tcp",
        port=8883,
        username="receiver",
        password=(root / "private/receiver-password").read_text().strip(),
        tls_context=ssl.create_default_context(cafile=str(root / "private/ca.crt")),
    )
    stopped = threading.Event()

    def supervise():
        try:
            while not stopped.is_set():
                try:
                    if not receiver.is_ready():
                        receiver.start()
                except Exception as error:
                    logging.warning("Receiver reconnect: %s", type(error).__name__)
                stopped.wait(2)
        finally:
            receiver.close()

    @asynccontextmanager
    async def lifespan(app):
        thread = threading.Thread(
            target=supervise, name="MonitoringReceiver", daemon=True
        )
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(15)
            if thread.is_alive():
                logging.error(
                    "Receiver shutdown timed out; process exit will close sockets"
                )

    return create_monitoring_api(
        store,
        binding,
        (root / "private/api-token").read_text().strip(),
        receiver=receiver,
        lifespan=lifespan,
    )


if __name__ == "__main__":
    os.umask(0o077)
    state = Path("/state")
    uvicorn.run(
        build_app(state),
        host="0.0.0.0",
        port=8443,
        access_log=False,
        ssl_keyfile=str(state / "private/server.key"),
        ssl_certfile=str(state / "private/server.crt"),
    )
