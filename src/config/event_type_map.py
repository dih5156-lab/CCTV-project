"""event_type → 장치별 문구·우선순위·경보 레벨 매핑 로더.

config/event_type_map.json 을 읽어 런타임에 사용한다.
파일이 없거나 유효하지 않으면 패닉 없이 내장 기본값으로 폴백한다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_MAP_PATH = PROJECT_ROOT / "config" / "event_type_map.json"

# ── 내장 기본값 (파일 로드 실패 시 사용) ─────────────────────────────────────
_BUILTIN_DEFAULTS = {
    "tts_message":  "안전 이벤트가 감지되었습니다.",
    "display_text": "안전 이벤트 감지",
    "color_code":   7,
    "priority":     1,
    "alert_level":  "info",
}


@dataclass(frozen=True)
class EventTypeEntry:
    tts_message:  str
    display_text: str
    color_code:   int
    priority:     int
    alert_level:  str


class EventTypeMap:
    """event_type 문자열 → EventTypeEntry 조회 테이블.

    싱글턴 패턴 없이 모듈 수준 인스턴스(_map)를 사용한다.
    """

    def __init__(self, path: Path = _MAP_PATH) -> None:
        self._entries: Dict[str, EventTypeEntry] = {}
        self._defaults = EventTypeEntry(**_BUILTIN_DEFAULTS)
        self._load(path)

    def _load(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("event_type_map.json 없음 (%s) — 내장 기본값 사용", path)
            return
        except json.JSONDecodeError as exc:
            logger.error("event_type_map.json 파싱 오류: %s — 내장 기본값 사용", exc)
            return

        defaults = raw.get("defaults", {})
        self._defaults = EventTypeEntry(
            tts_message =defaults.get("tts_message",  _BUILTIN_DEFAULTS["tts_message"]),
            display_text=defaults.get("display_text", _BUILTIN_DEFAULTS["display_text"]),
            color_code  =int(defaults.get("color_code",  _BUILTIN_DEFAULTS["color_code"])),
            priority    =int(defaults.get("priority",    _BUILTIN_DEFAULTS["priority"])),
            alert_level =defaults.get("alert_level",  _BUILTIN_DEFAULTS["alert_level"]),
        )

        for key, val in raw.get("event_types", {}).items():
            try:
                self._entries[key.lower()] = EventTypeEntry(
                    tts_message =val.get("tts_message",  self._defaults.tts_message),
                    display_text=val.get("display_text", self._defaults.display_text),
                    color_code  =int(val.get("color_code",  self._defaults.color_code)),
                    priority    =int(val.get("priority",    self._defaults.priority)),
                    alert_level =val.get("alert_level",  self._defaults.alert_level),
                )
            except Exception as exc:
                logger.warning("event_type '%s' 항목 로드 실패: %s", key, exc)

        logger.debug("event_type_map 로드 완료: %d개 항목", len(self._entries))

    def get(self, event_type: str) -> EventTypeEntry:
        """event_type 문자열로 항목을 조회한다. 미등록 시 defaults를 반환."""
        return self._entries.get(event_type.lower(), self._defaults)

    def tts_message(self, event_type: str, severity: str = "") -> str:
        entry = self._entries.get(event_type.lower())
        if entry:
            return entry.tts_message
        if "fall" in event_type.lower():
            return self._entries.get("fall_detected", self._defaults).tts_message
        if severity.lower() == "critical":
            return self._entries.get("critical", self._defaults).tts_message
        return self._defaults.tts_message

    def display_text(self, event_type: str, severity: str = "", camera_id: str = "") -> str:
        entry = self._entries.get(event_type.lower())
        if entry is None:
            if "fall" in event_type.lower():
                entry = self._entries.get("fall_detected", self._defaults)
            elif severity.lower() == "critical":
                entry = self._entries.get("critical", self._defaults)
            else:
                entry = self._defaults
        base = entry.display_text
        return f"[{camera_id}] {base}" if camera_id else base

    def color_code(self, event_type: str) -> int:
        return self._entries.get(event_type.lower(), self._defaults).color_code


# 모듈 수준 인스턴스 — speaker/signboard에서 import하여 사용
event_type_map = EventTypeMap()
