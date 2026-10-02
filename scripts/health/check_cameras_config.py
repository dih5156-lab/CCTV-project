#!/usr/bin/env python3
"""카메라 설정과 출력 장치 매핑을 확인하고 필요하면 영상 포트를 점검한다."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from urllib.parse import urlparse


def load_cameras(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cameras = raw.get("cameras", []) if isinstance(raw, dict) else raw
    if not isinstance(cameras, list):
        raise ValueError("cameras.json must contain a list or a cameras list")
    return [camera for camera in cameras if isinstance(camera, dict)]


def load_output_devices(path: Path) -> list[dict]:
    """출력 장치 JSON에서 장치 목록을 읽는다."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    devices = raw.get("devices", []) if isinstance(raw, dict) else raw
    if not isinstance(devices, list):
        raise ValueError("output device config must contain a list or a devices list")
    return [device for device in devices if isinstance(device, dict)]


def validate(cameras: list[dict]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    for index, camera in enumerate(cameras):
        camera_id = str(camera.get("id", "")).strip()
        source = str(camera.get("source", "")).strip()
        if not camera_id:
            errors.append(f"camera[{index}] missing id")
        elif camera_id in ids:
            errors.append(f"duplicate camera id: {camera_id}")
        ids.add(camera_id)
        if not source:
            errors.append(f"camera[{camera_id or index}] missing source")
        elif not source.startswith(("rtsp://", "rtmp://", "http://", "https://", "file://")):
            errors.append(f"camera[{camera_id or index}] unsupported source: {source}")
        settings = camera.get("model_settings", {})
        if settings and not isinstance(settings, dict):
            errors.append(f"camera[{camera_id or index}] model_settings must be an object")
    return errors


def validate_output_mappings(
    cameras: list[dict], devices: list[dict]
) -> list[str]:
    """활성 카메라가 하나 이상의 활성 출력 장치에 연결됐는지 확인한다."""
    enabled_camera_ids = {
        str(camera.get("id", "")).strip()
        for camera in cameras
        if bool(camera.get("enabled", True)) and str(camera.get("id", "")).strip()
    }
    mapped_camera_ids: set[str] = set()
    for device in devices:
        if not bool(device.get("enabled", True)):
            continue
        camera_ids = device.get("camera_ids") or []
        if not camera_ids:
            mapped_camera_ids.update(enabled_camera_ids)
            continue
        mapped_camera_ids.update(
            str(camera_id).strip()
            for camera_id in camera_ids
            if str(camera_id).strip()
        )

    return [
        f"camera[{camera_id}] has no enabled output device mapping"
        for camera_id in sorted(enabled_camera_ids - mapped_camera_ids)
    ]


def probe_source(source: str, timeout: float) -> tuple[bool, str]:
    parsed = urlparse(source)
    if parsed.scheme not in {"rtsp", "rtmp"} or not parsed.hostname:
        return True, "probe skipped"
    port = parsed.port or (554 if parsed.scheme == "rtsp" else 1935)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True, f"tcp {parsed.hostname}:{port} reachable"
    except OSError as exc:
        return False, f"tcp probe failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cameras.json"))
    parser.add_argument(
        "--output-devices",
        type=Path,
        default=Path("config/output_devices.json"),
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    try:
        cameras = load_cameras(args.config)
        output_devices = load_output_devices(args.output_devices)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    errors = validate(cameras)
    errors.extend(validate_output_mappings(cameras, output_devices))
    print(f"cameras={len(cameras)}")
    print(f"output_devices={len(output_devices)}")
    for camera in cameras:
        source = str(camera.get("source", ""))
        if args.probe:
            ok, detail = probe_source(source, args.timeout)
            print(f"{camera.get('id', '?')}: {'ok' if ok else 'failed'} {detail}")
            if not ok:
                errors.append(f"{camera.get('id', '?')}: {detail}")
        else:
            print(f"{camera.get('id', '?')}: configured")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
