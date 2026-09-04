#!/usr/bin/env python3
"""
edgex/register_aiot_devices.py
================================
EdgeX Metadata API에 aiot-parser 연동에 필요한
Device Service / Device Profile / Device 를 일괄 등록합니다.

실행:
  python edgex/register_aiot_devices.py [--metadata-url http://localhost:59881]
"""

import argparse
import json
import logging
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 설정 ─────────────────────────────────────────────────────────────
# device-rest: already-running EdgeX device service that responds to MQTT validation.
# aiot-parser is NOT an EdgeX SDK-based service, so we piggyback onto device-rest.
DEVICE_SERVICE_NAME = "device-rest"

# 프로필 → 디바이스 이름 접미사 매핑 (device_id + "-" + suffix)
TABLE_PROFILE_MAP = {
    "t34955": "aiot-t34955-inclinometer",
    "t34957": "aiot-t34957-tilt-temp",
    "t34958": "aiot-t34958-imu",
    "t34950": "aiot-t34950-river",
}

# ── 실제 디바이스 목록 (device_id, primary_table)
# 한 device_id 가 여러 table 을 가지므로 가장 많이 수신되는 table 을 primary 로 사용
DEVICES = [
    ("develop-05",  "t34957"),
    ("develop-09",  "t34957"),
    ("develop-11",  "t34957"),
    ("develop-15",  "t34957"),
    ("factory-12",  "t34957"),
    ("factory-14",  "t34957"),
    ("factory-15",  "t34957"),
    ("factory-16",  "t34958"),
    ("factory-21",  "t34957"),
    ("factory-24",  "t34957"),
    ("factory-26",  "t34958"),
    ("factory-27",  "t34957"),
    ("factory-34",  "t34958"),
    ("factory-35",  "t34957"),
]

PROFILES_DIR = Path(__file__).parent / "device-profiles"


def parse_device_spec(value: str) -> tuple[str, str]:
    """`DEVICE_ID:TABLE` 형식의 CLI 장치 지정자를 검증한다."""
    device_id, separator, primary_table = value.rpartition(":")
    if not separator or not device_id.strip() or not primary_table.strip():
        raise argparse.ArgumentTypeError("device must use DEVICE_ID:TABLE format")
    device_id = device_id.strip()
    primary_table = primary_table.strip()
    if primary_table not in TABLE_PROFILE_MAP:
        supported = ", ".join(sorted(TABLE_PROFILE_MAP))
        raise argparse.ArgumentTypeError(
            f"unsupported table {primary_table!r}; choose one of: {supported}"
        )
    return device_id, primary_table


def build_device_payload(device_id: str, primary_table: str) -> list[dict]:
    """EdgeX Metadata에 등록할 단일 AIoT 장치 payload를 만든다."""
    profile_name = TABLE_PROFILE_MAP.get(primary_table)
    if not profile_name:
        raise ValueError(f"unsupported table: {primary_table}")
    return [{"apiVersion": "v3", "device": {
        "name": f"aiot-{device_id}",
        "description": f"AIoT LoRa sensor {device_id}",
        "labels": ["aiot", "lora", primary_table],
        "adminState": "UNLOCKED",
        "operatingState": "UP",
        "serviceName": DEVICE_SERVICE_NAME,
        "profileName": profile_name,
        "protocols": {
            "lora": {
                "device_id": device_id,
                "primary_table": primary_table,
            }
        },
    }}]


def _api(url: str, method: str = "GET", body=None):
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, (json.loads(body) if body else {})
        except Exception:
            return e.code, {"raw": body.decode(errors="replace")}
    except (urllib.error.URLError, OSError) as exc:
        return 0, {"error": str(exc)}


def register_device_service(base: str, service_base_address: str = "http://device-rest:59986"):
    """EdgeX에 device-rest 서비스를 등록한다.

    *주의*: main()에서는 device-rest가 이미 실행 중인 것으로 간주하고
    이 함수를 호출하지 않는다. 재등록이 필요한 경우에만 사용한다.
    """
    url = f"{base}/api/v3/deviceservice"
    status, existing = _api(f"{base}/api/v3/deviceservice/name/{DEVICE_SERVICE_NAME}")
    if status == 200:
        # baseAddress 가 다르면 업데이트
        current_addr = existing.get("service", {}).get("baseAddress", "")
        if current_addr != service_base_address:
            patch = [{"apiVersion": "v3", "service": {
                "name": DEVICE_SERVICE_NAME,
                "baseAddress": service_base_address,
            }}]
            s2, _ = _api(url, "PATCH", patch)
            logger.info("Device Service baseAddress updated \u2192 %s (%s)", service_base_address, s2)
        else:
            logger.info("Device Service already exists: %s", DEVICE_SERVICE_NAME)
        return

    payload = [{ "apiVersion": "v3", "service": {
        "name": DEVICE_SERVICE_NAME,
        "description": "AIoT LoRa TLV Parser Device Service",
        "labels": ["aiot", "lora"],
        "baseAddress": service_base_address,
        "adminState": "UNLOCKED",
    }}]
    status, resp = _api(url, "POST", payload)
    if status in (200, 201, 207):
        logger.info("Device Service registered: %s", DEVICE_SERVICE_NAME)
    else:
        logger.error("Device Service registration failed: %s %s", status, resp)


def register_profiles(base: str):
    for profile_file in sorted(PROFILES_DIR.glob("*-profile.yaml")):
        content = profile_file.read_text()
        name_match = re.search(r'^name:\s*"?([^"\n]+)"?', content, re.MULTILINE)
        if not name_match:
            continue
        profile_name = name_match.group(1).strip()

        status, _ = _api(f"{base}/api/v3/deviceprofile/name/{profile_name}")
        if status == 200:
            logger.info("Profile already exists (skipping): %s", profile_name)
            continue

        url = f"{base}/api/v3/deviceprofile/uploadfile"
        boundary = uuid.uuid4().hex
        body_parts = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{profile_file.name}"\r\n'
            f"Content-Type: application/x-yaml\r\n\r\n"
        ).encode() + profile_file.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            url,
            data=body_parts,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                status = r.status
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            status = e.code
            resp = json.loads(e.read())

        if status in (200, 201, 207):
            logger.info("Profile registered: %s", profile_name)
        else:
            logger.error("Profile registration failed \u2014 %s: %s %s", profile_name, status, resp)


def register_devices(base: str, devices=None):
    for device_id, primary_table in devices or DEVICES:
        profile_name = TABLE_PROFILE_MAP.get(primary_table)
        if not profile_name:
            logger.warning("Unknown table %s for device %s \u2014 skipping", primary_table, device_id)
            continue

        edgex_device_name = f"aiot-{device_id}"
        status, _ = _api(f"{base}/api/v3/device/name/{edgex_device_name}")
        if status == 200:
            logger.info("Device already exists (skipping): %s", edgex_device_name)
            continue

        payload = build_device_payload(device_id, primary_table)
        status, resp = _api(f"{base}/api/v3/device", "POST", payload)
        if status in (200, 201, 207):
            logger.info("Device registered: %s (%s)", edgex_device_name, profile_name)
        else:
            logger.error("Device registration failed \u2014 %s: %s %s", edgex_device_name, status, resp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-url", default="http://localhost:59881")
    parser.add_argument(
        "--device",
        dest="devices",
        action="append",
        type=parse_device_spec,
        help="register DEVICE_ID:TABLE; repeat for multiple devices",
    )
    args = parser.parse_args()
    base = args.metadata_url.rstrip("/")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    logger.info("=== EdgeX AIoT Device Registration ===")
    logger.info("Metadata URL: %s", base)

    logger.info("[1] Device Service: using existing 'device-rest' (skip registration)")
    # Verify device-rest is reachable
    s, _ = _api(f"{base}/api/v3/deviceservice/name/device-rest")
    if s != 200:
        logger.error("device-rest not found in metadata (status %s) — is the service running?", s)
        sys.exit(1)
    logger.info("device-rest service confirmed in metadata")

    logger.info("[2] Device Profiles")
    register_profiles(base)

    logger.info("[3] Devices")
    register_devices(base, args.devices)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
