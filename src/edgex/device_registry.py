"""다중 출력 장치의 식별자와 설치 위치를 관리하는 레지스트리."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DeviceTarget:
    """이벤트 라우팅에 사용할 물리 장치 정보를 표현한다."""

    device_id: str
    device_type: str
    site_id: str = ""
    zone_id: str = ""
    camera_ids: tuple[str, ...] = field(default_factory=tuple)
    enabled: bool = True
    connection: Mapping[str, Any] = field(default_factory=dict)


class DeviceRegistry:
    """출력 장치 목록을 읽고 이벤트 대상 장치를 조회한다."""

    def __init__(self, targets: list[DeviceTarget] | None = None) -> None:
        """장치 대상 목록을 정규화해 메모리에 보관한다."""
        self._targets = tuple(targets or [])

    @classmethod
    def from_file(cls, path: str | Path) -> "DeviceRegistry":
        """JSON 레지스트리 파일을 읽어 DeviceRegistry를 생성한다."""
        registry_path = Path(path)
        if not registry_path.exists():
            return cls()
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        raw_targets = data.get("devices", [])
        if not isinstance(raw_targets, list):
            raise ValueError("devices는 JSON 배열이어야 합니다")
        return cls([_parse_target(item) for item in raw_targets])

    def resolve(
        self,
        device_type: str,
        *,
        camera_id: str = "",
        site_id: str = "",
        zone_id: str = "",
    ) -> list[DeviceTarget]:
        """장치 유형과 현장 조건에 맞는 활성 장치 목록을 반환한다."""
        normalized_type = device_type.strip().lower()
        return [
            target
            for target in self._targets
            if target.enabled
            and target.device_type == normalized_type
            and (not site_id or target.site_id in ("", site_id))
            and (not zone_id or target.zone_id in ("", zone_id))
            and (not camera_id or not target.camera_ids or camera_id in target.camera_ids)
        ]

    def targets(self, device_type: str) -> list[DeviceTarget]:
        """유형별 활성 장치를 위치 조건 없이 반환한다."""
        normalized_type = device_type.strip().lower()
        return [
            target
            for target in self._targets
            if target.enabled and target.device_type == normalized_type
        ]


def _parse_target(raw: Mapping[str, Any]) -> DeviceTarget:
    """JSON 장치 항목을 검증하고 DeviceTarget으로 변환한다."""
    device_id = str(raw.get("device_id") or "").strip()
    device_type = str(raw.get("device_type") or "").strip().lower()
    if not device_id or not device_type:
        raise ValueError("device_id와 device_type은 필수입니다")
    camera_ids = raw.get("camera_ids") or []
    if not isinstance(camera_ids, list):
        raise ValueError("camera_ids는 JSON 배열이어야 합니다")
    return DeviceTarget(
        device_id=device_id,
        device_type=device_type,
        site_id=str(raw.get("site_id") or "").strip(),
        zone_id=str(raw.get("zone_id") or "").strip(),
        camera_ids=tuple(str(value).strip() for value in camera_ids if str(value).strip()),
        enabled=bool(raw.get("enabled", True)),
        connection=dict(raw.get("connection") or {}),
    )
