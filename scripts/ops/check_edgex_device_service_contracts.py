#!/usr/bin/env python3
"""세 출력 장치의 EdgeX Core Command 계약을 장치 없이 검증한다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 스크립트를 저장소 어느 위치에서 실행해도 src 패키지를 찾도록 루트를 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.devices.signboard import SignboardConfig
from src.devices.siren import SensorConfig
from src.devices.speaker import SpeakerConfig
from src.edgex.command_http import handle_command_request
from src.edgex.dabit_device_service import DabitDeviceService
from src.edgex.siren_device_service import SirenDeviceService
from src.edgex.speaker_device_service import SpeakerDeviceService


def run_contract_checks() -> list[dict[str, Any]]:
    """세 장치를 dry-run으로 실행해 공통 HTTP 계약 결과를 반환한다."""
    cases = [
        (
            "speaker",
            "cctv-speaker-01",
            "/api/v3/device/name/cctv-speaker-01/play",
            {"event_id": "uat-speaker", "text": "EdgeX 계약 점검"},
            SpeakerDeviceService(
                device_id="cctv-speaker-01", config=SpeakerConfig(), dry_run=True
            ),
        ),
        (
            "siren",
            "cctv-siren-01",
            "/api/v3/device/name/cctv-siren-01/trigger",
            {"event_id": "uat-siren", "event_type": "intrusion", "camera_id": "cam-01"},
            SirenDeviceService(
                device_id="cctv-siren-01", config=SensorConfig(), dry_run=True
            ),
        ),
        (
            "signboard",
            "cctv-signboard-01",
            "/api/v3/device/name/cctv-signboard-01/display",
            {"event_id": "uat-signboard", "text": "EdgeX 계약 점검"},
            DabitDeviceService(
                device_id="cctv-signboard-01", config=SignboardConfig(), dry_run=True
            ),
        ),
    ]
    results = []
    for device, device_id, path, payload, service in cases:
        status_code, body = handle_command_request(
            service,
            device_id,
            path,
            payload,
            f"uat-{device}",
            device_type=device,
        )
        results.append(
            {
                "device": device,
                "status_code": status_code,
                "result_status": body.get("status"),
                "error_code": body.get("error_code"),
            }
        )
        service.close()
    return results


def main() -> int:
    """명령행에서 계약 점검 결과를 출력하고 성공 여부를 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    args = parser.parse_args()
    results = run_contract_checks()
    passed = all(item["status_code"] == 200 and item["result_status"] == "simulated" for item in results)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for item in results:
            print(
                f"{item['device']}: HTTP {item['status_code']} / "
                f"{item['result_status'] or item['error_code']}"
            )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
