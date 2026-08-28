#!/usr/bin/env python3
"""EdgeX Metadata/Core Data/Device Profile 계약을 읽기 전용으로 점검한다."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def audit_device_contracts(
    *, devices: list[dict], profiles: list[dict], events: list[dict]
) -> dict:
    """API snapshot 사이의 장치·프로파일·reading 계약을 비교한다."""
    devices_by_name = {device.get("name"): device for device in devices}
    profiles_by_name = {profile.get("name"): profile for profile in profiles}
    issues_by_key: dict[tuple[Any, ...], dict] = {}

    def add_issue(code: str, **details: Any) -> None:
        key = (code, *sorted(details.items()))
        issues_by_key[key] = {"code": code, **details}

    latest_events: dict[tuple[Any, ...], tuple[int, dict]] = {}
    for event in events:
        stream_key = (
            event.get("deviceName"),
            event.get("profileName"),
            event.get("sourceName"),
        )
        origin = int(event.get("origin") or 0)
        previous = latest_events.get(stream_key)
        if previous is None or origin > previous[0]:
            latest_events[stream_key] = (origin, event)

    for profile_name, profile in profiles_by_name.items():
        if not str(profile_name).startswith("aiot-"):
            continue
        exposed_resources = sorted(
            resource.get("name")
            for resource in profile.get("deviceResources", [])
            if resource.get("isHidden") is not True
        )
        exposed_commands = sorted(
            command.get("name")
            for command in profile.get("deviceCommands", [])
            if command.get("isHidden") is not True
        )
        if exposed_resources or exposed_commands:
            add_issue(
                "uplink_command_exposed",
                profile=profile_name,
                resources=tuple(exposed_resources),
                commands=tuple(exposed_commands),
            )

    for _, event in latest_events.values():
        device_name = event.get("deviceName")
        profile_name = event.get("profileName")
        device = devices_by_name.get(device_name)
        profile = profiles_by_name.get(profile_name)

        if device is None:
            add_issue("missing_metadata_device", device=device_name)
        elif device.get("profileName") != profile_name:
            add_issue(
                "device_profile_mismatch",
                device=device_name,
                metadata_profile=device.get("profileName"),
                event_profile=profile_name,
            )

        if profile is None:
            add_issue("missing_device_profile", profile=profile_name)
            continue

        declared_resources = {
            resource.get("name") for resource in profile.get("deviceResources", [])
        }
        for reading in event.get("readings", []):
            resource_name = reading.get("resourceName")
            if resource_name not in declared_resources:
                add_issue(
                    "unknown_profile_resource",
                    device=device_name,
                    profile=profile_name,
                    resource=resource_name,
                )

    issues = list(issues_by_key.values())
    return {"ok": not issues, "issues": issues}


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read() or b"{}")
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GET failed: {url}: {exc}") from exc


def check_live_contracts(
    *, metadata_url: str, core_data_url: str, event_limit: int
) -> dict:
    """실행 중인 EdgeX API snapshot을 읽어 계약 감사 결과를 반환한다."""
    metadata_url = metadata_url.rstrip("/")
    core_data_url = core_data_url.rstrip("/")
    query = urllib.parse.urlencode({"limit": max(1, event_limit)})
    devices = _get_json(f"{metadata_url}/api/v3/device/all?limit=1000").get(
        "devices", []
    )
    profiles = _get_json(
        f"{metadata_url}/api/v3/deviceprofile/all?limit=1000"
    ).get("profiles", [])
    events = _get_json(f"{core_data_url}/api/v3/event/all?{query}").get(
        "events", []
    )

    result = audit_device_contracts(
        devices=devices,
        profiles=profiles,
        events=events,
    )
    result["checked"] = {
        "devices": len(devices),
        "profiles": len(profiles),
        "events": len(events),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-url", default="http://127.0.0.1:59881"
    )
    parser.add_argument("--core-data-url", default="http://127.0.0.1:59880")
    parser.add_argument("--event-limit", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        result = check_live_contracts(
            metadata_url=args.metadata_url,
            core_data_url=args.core_data_url,
            event_limit=args.event_limit,
        )
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, default=list))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
