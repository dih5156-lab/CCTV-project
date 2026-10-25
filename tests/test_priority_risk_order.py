"""위험 등급 구간이 유형 순서 때문에 역전되지 않는지 확인한다."""

import json
from itertools import combinations
from pathlib import Path

import pytest

from src.config.event_type_map import EventTypeMap
from src.event_priority import event_priority
from src.services._action_bridge_alarm import _AlarmCoordinator
from src.services.action_policy import ActionPolicy


def test_critical_event_passes_after_warning_slot():
    policy = ActionPolicy(_AlarmCoordinator(set(), 30))
    assert policy.should_execute(
        "rest/inbound", {"camera_id": "cam1", "type": "head", "severity": "warning"}
    )
    assert policy.should_execute(
        "rest/inbound",
        {"camera_id": "cam1", "type": "zone_dwelling", "severity": "critical"},
    )


def test_all_configured_critical_types_precede_warning_types():
    raw = json.loads(Path("config/event_type_map.json").read_text())
    groups = {"critical": [], "warning": []}
    for name, entry in raw["event_types"].items():
        risk = entry["alert_level"]
        if risk in {"warn", "error"}:
            risk = "warning"
        if risk in groups:
            groups[risk].append({"type": name, "severity": risk})
    for critical in groups["critical"]:
        for warning in groups["warning"]:
            assert event_priority(critical) < event_priority(warning), (
                critical,
                warning,
            )


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("emergency", "critical"),
        ("warn", "warning"),
        ("error", "warning"),
        (" WARNING ", "warning"),
    ],
)
def test_risk_aliases_have_identical_priority(alias, canonical):
    assert event_priority({"type": "person", "severity": alias}) == event_priority(
        {"type": "person", "severity": canonical}
    )


@pytest.mark.parametrize(
    "risk_levels",
    [
        {},
        {"critical": 0, "warning": 2, "normal": 20, "info": 30, "low": 30},
        {"critical": 99, "warning": 0},
        {"warning": "invalid"},
    ],
)
def test_missing_or_overlapping_config_keeps_risk_groups_separate(
    tmp_path, risk_levels
):
    risks = ["critical", "warning", "normal", "info", "low"]
    entries = {
        f"{risk}_{level}": {"alert_level": risk} for risk in risks for level in [0, 9]
    }
    levels = {f"{risk}_{level}": level for risk in risks for level in [0, 9]}
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps(
            {
                "routing": {"risk_levels": risk_levels, "levels": levels},
                "event_types": entries,
            }
        )
    )
    mapping = EventTypeMap(path)
    for higher, lower in combinations(risks, 2):
        assert mapping.routing_priority(f"{higher}_9") < mapping.routing_priority(
            f"{lower}_0"
        )


def test_missing_config_uses_separate_risk_bands(tmp_path):
    mapping = EventTypeMap(tmp_path / "absent.json")
    assert mapping.routing_priority("unknown", "critical") < mapping.routing_priority(
        "unknown", "warning"
    )


def test_valid_configured_bands_and_within_risk_order_are_retained(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps(
            {
                "routing": {
                    "risk_levels": {
                        "critical": 5,
                        "warning": 15,
                        "normal": 25,
                        "info": 35,
                        "low": 45,
                    },
                    "levels": {"early": 0, "late": 9},
                },
                "event_types": {
                    "early": {"alert_level": "warning"},
                    "late": {"alert_level": "warning"},
                },
            }
        )
    )
    mapping = EventTypeMap(path)
    assert mapping.routing_priority("early") == 15
    assert mapping.routing_priority("late") == 24


def test_warning_cannot_replace_critical_slot():
    coordinator = _AlarmCoordinator(set(), 30)
    critical = {"type": "zone_dwelling", "severity": "critical"}
    warning = {"type": "head", "severity": "warning"}
    assert coordinator.try_acquire_slot(
        "cam1", "zone_dwelling", priority=event_priority(critical)
    )
    assert not coordinator.try_acquire_slot(
        "cam1", "head", priority=event_priority(warning)
    )
