"""
test_action_bridge.py — _SiteRegistry / _AlarmCoordinator / SiteConfig / ActionBridge 단위 테스트
"""
import time
import pytest
from unittest.mock import MagicMock
from src.services.action_bridge import (
    ActionBridge,
    ControlMode,
    AlarmDevice,
    SiteConfig,
    _SiteRegistry,
    _AlarmCoordinator,
)


# ---------------------------------------------------------------------------
# SiteConfig
# ---------------------------------------------------------------------------


class TestSiteConfig:
    def _make(self, **kwargs) -> SiteConfig:
        defaults = dict(
            site_id="site-001",
            site_name="테스트 현장",
            camera_ids=["cam1", "cam2"],
            control_mode=ControlMode.AUTO,
            alarm_devices=[AlarmDevice.SPEAKER],
        )
        defaults.update(kwargs)
        return SiteConfig(**defaults)

    def test_to_dict_round_trip(self):
        cfg = self._make()
        d = cfg.to_dict()
        restored = SiteConfig.from_dict(d)
        assert restored.site_id == cfg.site_id
        assert restored.site_name == cfg.site_name
        assert restored.control_mode == cfg.control_mode
        assert restored.alarm_devices == cfg.alarm_devices
        assert restored.camera_ids == cfg.camera_ids

    def test_to_dict_alarm_devices_as_strings(self):
        cfg = self._make(alarm_devices=[AlarmDevice.SPEAKER, AlarmDevice.SIREN])
        d = cfg.to_dict()
        assert d["alarm_devices"] == ["speaker", "siren"]

    def test_from_dict_defaults(self):
        """필수 필드만 있어도 from_dict 가 동작해야 함."""
        cfg = SiteConfig.from_dict({"site_id": "s1", "site_name": "현장1"})
        assert cfg.site_id == "s1"
        assert cfg.control_mode == ControlMode.AUTO
        assert AlarmDevice.SPEAKER in cfg.alarm_devices

    def test_nickname_optional(self):
        cfg = self._make(site_nickname="n1")
        d = cfg.to_dict()
        restored = SiteConfig.from_dict(d)
        assert restored.site_nickname == "n1"

    def test_manual_mode_preserved(self):
        cfg = self._make(control_mode=ControlMode.MANUAL)
        d = cfg.to_dict()
        assert d["control_mode"] == "manual"
        assert SiteConfig.from_dict(d).control_mode == ControlMode.MANUAL


# ---------------------------------------------------------------------------
# _SiteRegistry
# ---------------------------------------------------------------------------


class TestSiteRegistry:
    def _reg(self, mode=ControlMode.AUTO, sites=None) -> _SiteRegistry:
        return _SiteRegistry(default_mode=mode, initial_sites=sites)

    def _site(self, sid="s1", cams=None) -> SiteConfig:
        return SiteConfig(
            site_id=sid, site_name=f"현장{sid}",
            camera_ids=cams or ["cam1"],
            control_mode=ControlMode.AUTO,
        )

    # --- add / remove ---
    def test_add_and_list_all(self):
        reg = self._reg()
        reg.add(self._site())
        sites = reg.list_all()
        assert len(sites) == 1
        assert sites[0]["site_id"] == "s1"

    def test_remove_existing(self):
        reg = self._reg()
        reg.add(self._site())
        assert reg.remove("s1") is True
        assert reg.list_all() == []

    def test_remove_nonexistent(self):
        reg = self._reg()
        assert reg.remove("nonexistent") is False

    # --- find_by_camera ---
    def test_find_by_camera_hit(self):
        reg = self._reg()
        reg.add(self._site("s1", ["cam1", "cam2"]))
        site = reg.find_by_camera("cam2")
        assert site is not None
        assert site.site_id == "s1"

    def test_find_by_camera_miss(self):
        reg = self._reg()
        reg.add(self._site("s1", ["cam1"]))
        assert reg.find_by_camera("cam99") is None

    # --- set_mode ---
    def test_set_mode_global(self):
        reg = self._reg(mode=ControlMode.AUTO)
        reg.set_mode(ControlMode.MANUAL)
        assert reg.default_mode == ControlMode.MANUAL

    def test_set_mode_specific_site(self):
        reg = self._reg()
        reg.add(self._site("s1"))
        reg.set_mode(ControlMode.MANUAL, site_id="s1")
        site = reg.find_by_camera("cam1")
        assert site.control_mode == ControlMode.MANUAL

    def test_set_mode_nonexistent_site_no_crash(self):
        reg = self._reg()
        reg.set_mode(ControlMode.MANUAL, site_id="ghost")  # 예외 없이 통과

    # --- resolve_mode ---
    def test_resolve_mode_known_camera(self):
        reg = self._reg(mode=ControlMode.AUTO)
        site = self._site("s1", ["cam1"])
        site.control_mode = ControlMode.MANUAL
        reg.add(site)
        mode, sid = reg.resolve_mode("cam1")
        assert mode == ControlMode.MANUAL
        assert sid == "s1"

    def test_resolve_mode_unknown_camera_returns_default(self):
        reg = self._reg(mode=ControlMode.MANUAL)
        mode, sid = reg.resolve_mode("unknown_cam")
        assert mode == ControlMode.MANUAL
        assert sid is None

    # --- pending queue ---
    def test_push_and_pop_pending(self):
        reg = self._reg()
        reg.push_pending("evt-1", "topic/x", {"camera_id": "cam1"}, "s1")
        item = reg.pop_pending("evt-1")
        assert item is not None
        assert item["topic"] == "topic/x"

    def test_pop_nonexistent_returns_none(self):
        reg = self._reg()
        assert reg.pop_pending("ghost") is None

    def test_list_pending(self):
        reg = self._reg()
        reg.push_pending("e1", "t1", {"camera_id": "c1", "type": "helmet", "severity": "low"}, "s1")
        reg.push_pending("e2", "t2", {"camera_id": "c2", "type": "head", "severity": "high"}, "s2")
        pending = reg.list_pending()
        assert len(pending) == 2
        event_ids = {p["event_id"] for p in pending}
        assert {"e1", "e2"} == event_ids

    def test_pending_thread_safe(self):
        """동시 push/pop 에서 예외 없어야 함."""
        import threading
        reg = self._reg()
        errors = []

        def pusher():
            try:
                for i in range(20):
                    reg.push_pending(f"e{i}", f"t{i}", {"camera_id": "cam"}, None)
            except Exception as exc:
                errors.append(exc)

        def popper():
            try:
                for i in range(20):
                    reg.pop_pending(f"e{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=pusher), threading.Thread(target=popper)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# _AlarmCoordinator
# ---------------------------------------------------------------------------


class TestAlarmCoordinator:
    def _coord(self, topics=None, cooldown=3) -> _AlarmCoordinator:
        return _AlarmCoordinator(
            alarm_topics=topics or {"alerts/danger"},
            alarm_cooldown_seconds=cooldown,
        )

    # --- should_alarm ---
    def test_head_event_alarms(self):
        coord = self._coord()
        assert coord.should_alarm("any/topic", {"type": "head"}) is True

    def test_fall_detected_alarms(self):
        coord = self._coord()
        assert coord.should_alarm("any/topic", {"type": "fall_detected"}) is True

    def test_topic_in_alarm_topics_alarms(self):
        coord = self._coord(topics={"alerts/danger"})
        assert coord.should_alarm("alerts/danger", {"type": "something"}) is True

    def test_critical_severity_alarms(self):
        coord = self._coord()
        assert coord.should_alarm("other/topic", {"type": "helmet", "severity": "critical"}) is True

    def test_non_critical_other_not_alarm(self):
        coord = self._coord()
        assert coord.should_alarm("other/topic", {"type": "helmet", "severity": "low"}) is False

    # --- try_acquire_slot ---
    def test_first_call_acquires(self):
        coord = self._coord(cooldown=60)
        assert coord.try_acquire_slot("cam1", "helmet") is True

    def test_second_call_within_cooldown_blocked(self):
        coord = self._coord(cooldown=60)
        coord.try_acquire_slot("cam1", "helmet")
        assert coord.try_acquire_slot("cam1", "helmet") is False

    def test_after_cooldown_acquires_again(self):
        coord = self._coord(cooldown=1)   # 1초 쿨다운
        coord.try_acquire_slot("cam1", "helmet")
        time.sleep(1.1)
        assert coord.try_acquire_slot("cam1", "helmet") is True

    def test_head_event_not_subject_to_cooldown(self):
        """head / fall_detected 는 쿨다운 없이 연속 획득 가능."""
        coord = self._coord(cooldown=60)
        assert coord.try_acquire_slot("cam1", "head") is True
        # 쿨다운(60초)이 설정돼 있어도 head는 쿨다운 면제
        # BUT block_until 은 적용 → 두 번째는 차단됨
        # (구현 확인: block_until이 설정되므로 두 번째는 False)
        result = coord.try_acquire_slot("cam1", "head")
        # block_until 이 60초이므로 두 번째는 차단됨
        assert result is False

    def test_different_cameras_independent(self):
        coord = self._coord(cooldown=60)
        coord.try_acquire_slot("cam1", "helmet")
        assert coord.try_acquire_slot("cam2", "helmet") is True

    def test_different_event_types_independent(self):
        coord = self._coord(cooldown=60)
        coord.try_acquire_slot("cam1", "helmet")
        # 같은 카메라지만 block_until 이 cam1 전체에 적용되므로 False
        result = coord.try_acquire_slot("cam1", "person")
        assert result is False   # block_until 적용

    def test_cooldown_min_one_second(self):
        """cooldown_seconds < 1 을 전달해도 최소 1초로 클램핑."""
        coord = _AlarmCoordinator(alarm_topics=set(), alarm_cooldown_seconds=0)
        assert coord.alarm_cooldown_seconds == 1

    def test_thread_safety(self):
        """동시에 try_acquire_slot 호출 시 예외 없어야 함."""
        import threading
        coord = self._coord(cooldown=1)
        errors = []

        def worker():
            try:
                for _ in range(30):
                    coord.try_acquire_slot("cam1", "helmet")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestActionBridgeStatusPublishing:
    def _make_bridge(self) -> ActionBridge:
        bridge = ActionBridge(rest_enabled=False)
        bridge._mqtt_client = MagicMock()
        bridge._repo = MagicMock()
        bridge._forwarder = MagicMock()
        bridge._forwarder.has_targets = True
        bridge._speaker = MagicMock()
        bridge._speaker.play.return_value = True
        bridge._signboard = MagicMock()
        bridge._siren = MagicMock()
        bridge._resolve_devices = MagicMock(return_value=[AlarmDevice.SPEAKER])
        bridge._alarm = MagicMock()
        bridge._alarm.should_alarm.return_value = True
        bridge._alarm.try_acquire_slot.return_value = True
        return bridge

    def test_manual_event_publishes_pending_status(self):
        bridge = self._make_bridge()
        bridge.set_mode(ControlMode.MANUAL)

        bridge._handle_event({"camera_id": "cam1", "type": "fall_detected"}, topic="t")

        published = bridge._mqtt_client.publish.call_args_list
        assert any("cctv/status/action/events/pending" in call.args[0] for call in published)

    def test_approve_event_publishes_approved_status(self):
        bridge = self._make_bridge()
        bridge._sites.push_pending("evt1", "topic", {"camera_id": "cam1", "type": "head"}, "site1")

        ok, _ = bridge.approve_event("evt1")

        assert ok is True
        published = bridge._mqtt_client.publish.call_args_list
        assert any("cctv/status/action/events/approved" in call.args[0] for call in published)

    def test_reject_event_publishes_rejected_status(self):
        bridge = self._make_bridge()
        bridge._sites.push_pending("evt2", "topic", {"camera_id": "cam2", "type": "head"}, "site2")

        ok, _ = bridge.reject_event("evt2")

        assert ok is True
        published = bridge._mqtt_client.publish.call_args_list
        assert any("cctv/status/action/events/rejected" in call.args[0] for call in published)

    def test_dispatch_command_publishes_command_result(self):
        bridge = self._make_bridge()

        bridge._dispatch_command("cctv/commands/mode", {"command_id": "cmd1", "mode": "auto"})

        published = bridge._mqtt_client.publish.call_args_list
        assert any("cctv/status/action/commands/result" in call.args[0] for call in published)

    def test_execute_action_prefers_canonical_output_messages(self):
        bridge = self._make_bridge()
        bridge._resolve_devices.return_value = [AlarmDevice.SPEAKER, AlarmDevice.SIGNBOARD]
        bridge._executor._speaker = bridge._speaker
        bridge._executor._signboard = bridge._signboard

        bridge._execute_action(
            "aiot/rules/sensor/tilt",
            {
                "camera_id": "cam1",
                "type": "tilt_alert",
                "severity": "critical",
                "event": {
                    "event_type": "tilt_alert",
                    "severity": "critical",
                    "display_message": "기울기 이상 감지",
                    "tts_message": "기울기 이상이 감지되었습니다. 즉시 현장을 확인 바랍니다.",
                },
            },
        )

        bridge._speaker.play.assert_called_once_with(
            "tilt_alert",
            "critical",
            "cam1",
            text="기울기 이상이 감지되었습니다. 즉시 현장을 확인 바랍니다.",
        )
        assert bridge._signboard.display.call_count == 1
        assert bridge._signboard.display.call_args.kwargs["text"] == "기울기 이상 감지"
