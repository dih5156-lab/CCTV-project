"""GET/POST /api/v1/sensor-readings - TLV/AIoT 센서 로그 조회와 시연 입력."""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Body, Depends, Query, Request, status
from pydantic import BaseModel

from ..dependencies._settings import (
    ACTION_LAYER_URL as _ACTION_LAYER_URL,
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


async def close_sensor_client() -> None:
    """Public API 종료 시 센서 중계 HTTP 클라이언트를 닫는다."""
    global _shared_sensor_client
    if _shared_sensor_client is not None and not _shared_sensor_client.is_closed:
        await _shared_sensor_client.aclose()
    _shared_sensor_client = None


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
    status: str = "normal"
    severity: str = "normal"
    event_type: Optional[str] = None
    reason: Optional[str] = None


class SensorReadingAccepted(BaseModel):
    accepted: bool = True
    device_id: Optional[str] = None
    table: Optional[str] = None
    status: str = "normal"
    severity: str = "normal"
    event_type: Optional[str] = None
    action_dispatched: bool = False


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_sensor_payload(payload: dict[str, Any]) -> dict[str, Optional[str]]:
    """TLV 센서값을 시연/관제용 위험 상태로 정규화한다."""
    data = _extract_data(payload)
    event_type = str(payload.get("type") or payload.get("event_type") or "").strip()
    severity = str(payload.get("severity") or "").strip().lower()
    if event_type:
        return {
            "status": "alert" if severity in {"critical", "warning", "warn"} else "normal",
            "severity": "warning" if severity == "warn" else (severity or "normal"),
            "event_type": event_type,
            "reason": str(payload.get("reason") or event_type),
        }

    temperature = _as_float(data.get("temperature") or data.get("temp"))
    angle_x = abs(_as_float(data.get("angle_x") or data.get("tilt_x") or data.get("x_angle")) or 0.0)
    angle_y = abs(_as_float(data.get("angle_y") or data.get("tilt_y") or data.get("y_angle")) or 0.0)
    event_code = _as_float(data.get("event_code") or data.get("code"))

    if temperature is not None and temperature >= 50.0:
        severity = "critical" if temperature >= 70.0 else "warning"
        return {
            "status": "alert",
            "severity": severity,
            "event_type": "temperature_alert",
            "reason": f"temperature={temperature:g} >= {'70' if severity == 'critical' else '50'}",
        }
    if max(angle_x, angle_y) >= 30.0:
        return {
            "status": "alert",
            "severity": "warning",
            "event_type": "tilt_alert",
            "reason": f"tilt={max(angle_x, angle_y):g} >= 30",
        }
    if event_code is not None and event_code != 0:
        return {
            "status": "alert",
            "severity": "warning",
            "event_type": "sensor_event",
            "reason": f"event_code={event_code:g}",
        }
    return {"status": "normal", "severity": "normal", "event_type": None, "reason": None}


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
    risk = _classify_sensor_payload(payload)
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
        status=risk["status"] or "normal",
        severity=risk["severity"] or "normal",
        event_type=risk["event_type"],
        reason=risk["reason"],
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

    items, total = _read_sensor_log(limit, offset, device_id, table)
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
    client = _get_sensor_client()
    target = f"{_ALERT_API_URL.rstrip('/')}/api/sensor-readings"
    try:
        response = await client.post(target, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("내부 sensor reading 중계 실패 (%s) -> fallback 저장", exc)
        _append_fallback(payload)

    data = _extract_data(payload)
    risk = _classify_sensor_payload(payload)
    action_dispatched = False
    if risk["status"] == "alert" and risk["event_type"]:
        device_id = payload.get("device_id") or payload.get("dev_eui") or "sensor"
        action_payload = {
            "camera_id": str(device_id),
            "type": risk["event_type"],
            "severity": risk["severity"] or "warning",
            "confidence": 1.0,
            "metadata": {
                "source": "sensor_readings",
                "table": payload.get("table") or payload.get("tableName") or data.get("tableName"),
                "reason": risk["reason"],
                "data": data,
            },
        }
        action_target = f"{_ACTION_LAYER_URL.rstrip('/')}/events"
        try:
            action_response = await client.post(action_target, json=action_payload)
            action_dispatched = action_response.status_code in (200, 202)
            if not action_dispatched:
                logger.warning("센서 위험 이벤트 action layer 전달 실패 (status=%s)", action_response.status_code)
        except httpx.HTTPError as exc:
            logger.warning("센서 위험 이벤트 action layer 전달 실패 (%s)", exc)

    return success_response(
        SensorReadingAccepted(
            accepted=True,
            device_id=payload.get("device_id") or payload.get("dev_eui"),
            table=payload.get("table") or payload.get("tableName") or data.get("tableName"),
            status=risk["status"] or "normal",
            severity=risk["severity"] or "normal",
            event_type=risk["event_type"],
            action_dispatched=action_dispatched,
        )
    )
