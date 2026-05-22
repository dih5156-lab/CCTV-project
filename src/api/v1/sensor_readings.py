"""GET/POST /api/v1/sensor-readings - TLV/AIoT 센서 로그 조회와 시연 입력."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Body, Depends, Query, Request, status
from pydantic import BaseModel

from ..dependencies._settings import (
    ALERT_API_URL as _ALERT_API_URL,
    INTERNAL_SERVICE_TOKEN as _INTERNAL_TOKEN,
    SENSOR_DEVICE_MAP_PATH as _SENSOR_DEVICE_MAP,
    SENSOR_LOG_PATH as _SENSOR_LOG,
)
from ..dependencies.auth import verify_api_key
from ..dependencies.rate_limit import limiter
from ..schemas.common import BaseResponse, PaginatedResponse, success_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sensor-readings", tags=["sensor-readings"])

_INTERNAL_HEADERS: dict[str, str] = (
    {"X-Internal-Token": _INTERNAL_TOKEN} if _INTERNAL_TOKEN else {}
)

# 요청마다 새 TCP 연결을 만들지 않도록 모듈 레벨 공유 클라이언트를 사용한다.
_shared_sensor_client: httpx.AsyncClient | None = None


def _get_sensor_client() -> httpx.AsyncClient:
    global _shared_sensor_client
    if _shared_sensor_client is None or _shared_sensor_client.is_closed:
        _shared_sensor_client = httpx.AsyncClient(
            timeout=5.0,
            headers=_INTERNAL_HEADERS,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
    return _shared_sensor_client


class SensorReadingOut(BaseModel):
    """시연 대시보드에서 보여줄 TLV/센서 로그 1건."""

    received_at: Optional[str] = None
    timestamp: float = 0.0
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    dev_eui: Optional[str] = None
    table: Optional[str] = None
    data: dict[str, Any] = {}
    payload: dict[str, Any] = {}


class SensorReadingAccepted(BaseModel):
    accepted: bool = True
    device_id: Optional[str] = None
    table: Optional[str] = None


def _coerce_timestamp(value: object, fallback: object = None) -> float:
    for candidate in (value, fallback):
        if candidate in (None, ""):
            continue
        if isinstance(candidate, (int, float)):
            numeric = float(candidate)
            if abs(numeric) >= 1e11:
                numeric /= 1000.0
            return numeric
        if isinstance(candidate, str):
            try:
                numeric = float(candidate)
                if abs(numeric) >= 1e11:
                    numeric /= 1000.0
                return numeric
            except ValueError:
                try:
                    return datetime.fromisoformat(candidate.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
    return 0.0


def _payload_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    payload = entry.get("payload", entry)
    return payload if isinstance(payload, dict) else {}


def _extract_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    decoded = payload.get("decoded")
    if isinstance(decoded, dict):
        return decoded
    return {}


# 장비 이름 매핑은 프로세스 기동 중에 바뀌지 않으므로 한 번만 읽고 캐시한다.
_device_name_map_cache: dict[str, str] | None = None


def _load_device_name_map() -> dict[str, str]:
    """장비 ID/DevEUI를 시연용 표시 이름으로 변환하는 매핑을 읽는다."""
    global _device_name_map_cache
    if _device_name_map_cache is not None:
        return _device_name_map_cache
    if not _SENSOR_DEVICE_MAP.exists():
        _device_name_map_cache = {}
        return _device_name_map_cache
    try:
        raw = json.loads(_SENSOR_DEVICE_MAP.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("센서 장비명 매핑 읽기 실패: %s", exc)
        return {}

    entries = raw.get("devices", raw) if isinstance(raw, dict) else raw
    mapping: dict[str, str] = {}
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("device_name") or "").strip()
            if not name:
                continue
            for key in ("device_id", "dev_eui", "id"):
                value = str(item.get(key) or "").strip()
                if value:
                    mapping[value] = name
                    mapping[value.lower()] = name
    elif isinstance(entries, dict):
        for key, value in entries.items():
            name = str(value or "").strip()
            if key and name:
                mapping[str(key)] = name
                mapping[str(key).lower()] = name
    _device_name_map_cache = mapping
    return mapping


def _lookup_device_name(device_id: Optional[str], dev_eui: Optional[str]) -> Optional[str]:
    mapping = _load_device_name_map()
    for value in (device_id, dev_eui):
        if not value:
            continue
        text = str(value)
        name = mapping.get(text) or mapping.get(text.lower())
        if name:
            return name
    return None


def _normalize_entry(entry: dict[str, Any]) -> SensorReadingOut:
    payload = _payload_from_entry(entry)
    data = _extract_data(payload)
    device_info = payload.get("device") if isinstance(payload.get("device"), dict) else {}
    table = (
        payload.get("table")
        or payload.get("tableName")
        or data.get("tableName")
        or payload.get("type")
    )
    received_at = entry.get("receivedAt") or entry.get("received_at") or payload.get("received_at")
    dev_eui = (
        payload.get("dev_eui")
        or device_info.get("dev_eui")
        or payload.get("deviceEui")
        or payload.get("devEui")
    )
    device_id = (
        payload.get("device_id")
        or device_info.get("device_id")
        or device_info.get("camera_id")
        or dev_eui
    )
    return SensorReadingOut(
        received_at=str(received_at) if received_at else None,
        timestamp=_coerce_timestamp(
            payload.get("received_at") or payload.get("timestamp") or payload.get("occurred_at"),
            received_at,
        ),
        device_id=str(device_id) if device_id else None,
        device_name=_lookup_device_name(
            str(device_id) if device_id else None,
            str(dev_eui) if dev_eui else None,
        ),
        dev_eui=str(dev_eui) if dev_eui else None,
        table=str(table) if table else None,
        data=data,
        payload=payload,
    )


def _append_fallback(payload: dict[str, Any]) -> None:
    try:
        _SENSOR_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {"receivedAt": datetime.now(timezone.utc).isoformat(), "payload": payload}
        with _SENSOR_LOG.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("센서 로그 fallback 저장 실패: %s", exc)


def _read_sensor_log(
    limit: int,
    offset: int,
    device_id: Optional[str],
    table: Optional[str],
) -> tuple[list[SensorReadingOut], int]:
    """센서 로그를 tail 방식으로 읽어 필터링 후 반환한다 (동기, executor에서 실행)."""
    _TAIL_MAX = max(5000, (limit + offset) * 10)
    try:
        with _SENSOR_LOG.open("r", encoding="utf-8") as fh:
            last_lines = deque(fh, maxlen=_TAIL_MAX)
    except OSError as exc:
        logger.error("센서 로그 읽기 실패: %s", exc)
        return [], 0

    items: list[SensorReadingOut] = []
    for raw_line in reversed(list(last_lines)):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            item = _normalize_entry(json.loads(raw_line))
        except json.JSONDecodeError:
            continue
        if device_id and item.device_id != device_id:
            continue
        if table and item.table != table:
            continue
        items.append(item)
    return items, len(items)


@router.get(
    "",
    response_model=PaginatedResponse[SensorReadingOut],
    summary="TLV/센서 수신 로그 조회",
    description="cctv-alert-api가 저장한 sensor_readings.jsonl을 최신순으로 반환합니다.",
)
@limiter.limit("60/minute")
async def list_sensor_readings(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    device_id: Optional[str] = Query(default=None),
    table: Optional[str] = Query(default=None),
    _: None = Depends(verify_api_key),
) -> PaginatedResponse[SensorReadingOut]:
    if not _SENSOR_LOG.exists():
        return PaginatedResponse(items=[], total=0, limit=limit, offset=offset)

    loop = asyncio.get_event_loop()
    items, total = await loop.run_in_executor(
        None,
        lambda: _read_sensor_log(limit, offset, device_id, table),
    )
    return PaginatedResponse(items=items[offset : offset + limit], total=total, limit=limit, offset=offset)


@router.post(
    "",
    response_model=BaseResponse[SensorReadingAccepted],
    status_code=status.HTTP_202_ACCEPTED,
    summary="TLV/센서 로그 시연 입력",
    description="시연용 센서 payload를 내부 cctv-alert-api의 /api/sensor-readings로 중계합니다.",
)
@limiter.limit("30/minute")
async def receive_sensor_reading(
    request: Request,
    payload: dict[str, Any] = Body(...),
    _: None = Depends(verify_api_key),
) -> BaseResponse[SensorReadingAccepted]:
    target = f"{_ALERT_API_URL.rstrip('/')}/api/sensor-readings"
    try:
        client = _get_sensor_client()
        response = await client.post(target, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("내부 sensor reading 중계 실패 (%s) -> fallback 저장", exc)
        _append_fallback(payload)

    data = _extract_data(payload)
    return success_response(
        SensorReadingAccepted(
            accepted=True,
            device_id=payload.get("device_id") or payload.get("dev_eui"),
            table=payload.get("table") or payload.get("tableName") or data.get("tableName"),
        )
    )
