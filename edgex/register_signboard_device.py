#!/usr/bin/env python3
"""EdgeX Metadata에 검증용 Dabit Device Service/장치를 등록한다."""

from __future__ import annotations

import argparse
import logging

from register_aiot_devices import _api

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-url", default="http://127.0.0.1:59881")
    parser.add_argument("--service-url", default="http://cctv-device-dabit:59990")
    parser.add_argument("--device-name", default="cctv-signboard-01")
    args = parser.parse_args()
    base = args.metadata_url.rstrip("/")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    service = [{"apiVersion": "v3", "service": {
        "name": "cctv-device-dabit", "description": "Dabit TCP signboard service",
        "baseAddress": args.service_url, "adminState": "UNLOCKED",
        "labels": ["cctv", "output", "signboard"],
    }}]
    status, _ = _api(f"{base}/api/v3/deviceservice/name/cctv-device-dabit")
    if status != 200:
        status, response = _api(f"{base}/api/v3/deviceservice", "POST", service)
        if status not in (200, 201, 207):
            raise SystemExit(f"device service registration failed: {status} {response}")

    status, _ = _api(f"{base}/api/v3/device/name/{args.device_name}")
    if status == 200:
        logger.info("device already exists: %s", args.device_name)
        return
    device = [{"apiVersion": "v3", "device": {
        "name": args.device_name, "description": "CCTV Dabit signboard",
        "adminState": "UNLOCKED", "operatingState": "UP",
        "serviceName": "cctv-device-dabit", "profileName": "cctv-signboard-dabit",
        "protocols": {"tcp": {"host": "192.168.88.91", "port": "5000"}},
        "labels": ["cctv", "output", "signboard"],
    }}]
    status, response = _api(f"{base}/api/v3/device", "POST", device)
    if status not in (200, 201, 207):
        raise SystemExit(f"device registration failed: {status} {response}")
    logger.info("signboard registered: %s", args.device_name)


if __name__ == "__main__":
    main()
