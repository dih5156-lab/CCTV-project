#!/usr/bin/env python3
"""EdgeX Metadata에 CCTV 출력 장치 서비스와 장치를 등록한다."""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
PROFILE_DIR = Path(__file__).parent / "device-profiles"

OUTPUT_DEVICES = (
    {
        "service_name": "cctv-device-speaker",
        "device_name": "cctv-speaker-01",
        "profile_name": "cctv-speaker",
        "service_url": "http://cctv-device-speaker:59991",
        "protocols": {"http": {"host": "cctv-device-speaker", "port": "59991"}},
        "labels": ["cctv", "output", "speaker"],
    },
    {
        "service_name": "cctv-device-siren",
        "device_name": "cctv-siren-01",
        "profile_name": "cctv-siren",
        "service_url": "http://cctv-device-siren:59992",
        "protocols": {"http": {"host": "cctv-device-siren", "port": "59992"}},
        "labels": ["cctv", "output", "siren"],
    },
    {
        "service_name": "cctv-device-signboard",
        "device_name": "cctv-signboard-01",
        "profile_name": "cctv-signboard-dabit",
        "service_url": "http://cctv-device-signboard:59993",
        "protocols": {"http": {"host": "cctv-device-signboard", "port": "59993"}},
        "labels": ["cctv", "output", "signboard"],
    },
)


def _api(url: str, method: str = "GET", body: Any = None) -> tuple[int, dict[str, Any]]:
    """EdgeX Metadata API를 호출하고 상태 코드와 JSON 응답을 반환한다."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except (json.JSONDecodeError, OSError):
            return exc.code, {}
    except (urllib.error.URLError, OSError) as exc:
            return 0, {"error": str(exc)}


def upload_profile(base: str, profile_path: Path) -> bool:
    """EdgeX Metadata에 YAML Device Profile을 없을 때만 업로드한다."""
    profile_name = profile_path.stem.removesuffix("-profile")
    status, _ = _api(f"{base}/api/v3/deviceprofile/name/{profile_name}")
    if status == 200:
        return True
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{profile_path.name}"\r\n'
        "Content-Type: application/x-yaml\r\n\r\n"
    ).encode() + profile_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{base}/api/v3/deviceprofile/uploadfile",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status in (200, 201, 207)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False


def build_device_payload(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """출력 장치 사양을 EdgeX 장치 등록 payload로 변환한다."""
    return [{"apiVersion": "v3", "device": {
        "name": spec["device_name"],
        "description": f"CCTV {spec['labels'][-1]} output device",
        "adminState": "UNLOCKED",
        "operatingState": "UP",
        "serviceName": spec["service_name"],
        "profileName": spec["profile_name"],
        "protocols": spec["protocols"],
        "labels": spec["labels"],
    }}]


def register_service(base: str, spec: dict[str, Any]) -> bool:
    """출력 Device Service를 등록하거나 baseAddress를 최신화한다."""
    name = spec["service_name"]
    service = {"name": name, "description": f"CCTV {spec['labels'][-1]} Device Service",
               "baseAddress": spec["service_url"], "adminState": "UNLOCKED",
               "labels": spec["labels"]}
    status, existing = _api(f"{base}/api/v3/deviceservice/name/{name}")
    if status == 200:
        current = existing.get("service", {}).get("baseAddress")
        if current != spec["service_url"]:
            status, _ = _api(f"{base}/api/v3/deviceservice", "PATCH", [{"apiVersion": "v3", "service": service}])
        return status in (200, 201, 207)
    status, _ = _api(f"{base}/api/v3/deviceservice", "POST", [{"apiVersion": "v3", "service": service}])
    return status in (200, 201, 207)


def register_device(base: str, spec: dict[str, Any]) -> bool:
    """출력 장치를 지정한 Device Service와 Profile에 연결해 등록한다."""
    status, existing = _api(f"{base}/api/v3/device/name/{spec['device_name']}")
    if status == 200:
        current = existing.get("device", {})
        if current.get("serviceName") == spec["service_name"] and current.get("profileName") == spec["profile_name"]:
            return True
        status, _ = _api(
            f"{base}/api/v3/device",
            "PATCH",
            build_device_payload(spec),
        )
        return status in (200, 201, 207)
    status, _ = _api(f"{base}/api/v3/device", "POST", build_device_payload(spec))
    return status in (200, 201, 207)


def main() -> int:
    """프로파일·서비스·장치를 순서대로 등록하고 결과를 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-url", default="http://127.0.0.1:59881")
    args = parser.parse_args()
    base = args.metadata_url.rstrip("/")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    failed = False
    for spec in OUTPUT_DEVICES:
        profile_path = PROFILE_DIR / f"{spec['profile_name']}-profile.yaml"
        if not profile_path.exists() or not upload_profile(base, profile_path):
            logger.error("프로파일 등록 실패: %s", profile_path.name)
            failed = True
            continue
        logger.info("프로파일 등록 확인: %s", profile_path.name)
        if not register_service(base, spec):
            logger.error("Device Service 등록 실패: %s", spec["service_name"])
            failed = True
            continue
        if not register_device(base, spec):
            logger.error("장치 등록 실패: %s", spec["device_name"])
            failed = True
        else:
            logger.info("출력 장치 등록 완료: %s", spec["device_name"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
