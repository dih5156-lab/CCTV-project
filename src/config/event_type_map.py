"""event_type → 장치별 문구·우선순위·경보 레벨 매핑 로더.

config/event_type_map.json 을 읽어 런타임에 사용한다.
파일이 없거나 유효하지 않으면 패닉 없이 내장 기본값으로 폴백한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
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
        self._routing_levels: Dict[str, int] = {}
        self._risk_order: Dict[str, int] = {
            "critical": 0,
            "warning": 1,
            "normal": 2,
            "info": 3,
            "low": 4,
        }
        # 유형 순서가 0~9이므로 설정이 없어도 위험 등급 사이에 10칸을 확보한다.
        self._risk_levels = {name: rank * 10 for name, rank in self._risk_order.items()}
        self._load(path)

    @staticmethod
    def _normalize_risk(value: str) -> str:
        """별칭의 표시와 정렬이 서로 다른 등급을 쓰지 않도록 통일한다."""
        name = str(value or "").strip().lower()
        return {"emergency": "critical", "warn": "warning", "error": "warning"}.get(name, name)

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
        routing = raw.get("routing", {})
        configured_risk_levels = routing.get("risk_levels", {})
        if isinstance(configured_risk_levels, dict):
            candidate_levels = self._risk_levels.copy()
            for name, value in configured_risk_levels.items():
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    continue
                normalized_risk = self._normalize_risk(name)
                if 0 <= numeric <= 99 and normalized_risk in candidate_levels:
                    candidate_levels[normalized_risk] = numeric
            # 구형 설정의 warning=2는 critical=3보다 앞서므로 구간 전체를 검증한다.
            ordered_levels = [candidate_levels[name] for name in self._risk_order]
            if all(lower - higher >= 10 for higher, lower in zip(ordered_levels, ordered_levels[1:])):
                self._risk_levels = candidate_levels
            else:
                logger.warning("라우팅 위험 등급 구간 겹침/역전 — 안전한 기본 구간 사용")
        configured_levels = routing.get("levels", {})
        if isinstance(configured_levels, dict):
            for name, value in configured_levels.items():
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= numeric <= 9:
                    self._routing_levels[str(name).lower()] = numeric
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

    def routing_priority(self, event_type: str, severity: str = "") -> int:
        """위험도 그룹과 그룹 내부 레벨을 결합한 장치 라우팅 우선순위."""
        normalized_type = event_type.lower()
        entry = self.get(normalized_type)
        configured_risk = self._normalize_risk(entry.alert_level or "normal")
        configured_order = self._risk_order.get(configured_risk, self._risk_order["normal"])
        severity_name = self._normalize_risk(severity)
        severity_order = self._risk_order.get(severity_name)
        effective_risk = configured_risk
        if severity_name and severity_order is None and configured_order >= self._risk_order["normal"]:
            effective_risk = "normal"
            severity_order = self._risk_order["normal"]
        if severity_order is not None and severity_order < configured_order:
            effective_risk = severity_name
        risk_base = self._risk_levels.get(effective_risk, self._risk_levels["normal"])
        level = self._routing_levels.get(normalized_type, 9)
        if severity_order is not None and severity_order < configured_order:
            level = 0
        return risk_base + level

    def routing_risk_level(self, event_type: str, severity: str = "") -> str:
        entry = self.get(event_type)
        configured_risk = self._normalize_risk(entry.alert_level or "normal")
        configured_order = self._risk_order.get(configured_risk, self._risk_order["normal"])
        severity_name = self._normalize_risk(severity)
        severity_order = self._risk_order.get(severity_name)
        if severity_name and severity_order is None:
            if configured_order < self._risk_order["normal"]:
                return "critical" if configured_order == 0 else "warning"
            return "normal"
        if severity_order is not None and severity_order < configured_order:
            return "critical" if severity_order == 0 else "warning"
        if severity_order is not None and severity_order > configured_order:
            if severity_order >= self._risk_order["low"]:
                return "low"
            if severity_order >= self._risk_order["normal"]:
                return "normal"
            return "info"
        return configured_risk if configured_risk in self._risk_order else "normal"

    def routing_snapshot(self, event_type: str, severity: str) -> dict:
        """영속 큐의 위험/유형 순서를 고정하고 미등록 입력은 임의로 승격하지 않는다."""
        if event_type.lower() not in self._entries or self._normalize_risk(severity) not in self._risk_order:
            raise ValueError("priority_policy_unset")
        priority = self.routing_priority(event_type, severity)
        risk = max((name for name, base in self._risk_levels.items() if base <= priority),
                   key=self._risk_levels.__getitem__)
        policy = {"entries": {name: asdict(entry) for name, entry in self._entries.items()},
                  "levels": self._routing_levels, "risk_levels": self._risk_levels}
        return {"risk_rank": self._risk_order[risk], "within_risk_rank": priority - self._risk_levels[risk],
                "policy_hash": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()}

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
