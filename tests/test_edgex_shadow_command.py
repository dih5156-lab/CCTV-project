import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.services.action_bridge import ActionBridge


def test_action_bridge_publishes_shadow_command_as_json(tmp_path):
    bridge = ActionBridge(db_path=str(tmp_path / "action.db"))
    mqtt_client = MagicMock()
    mqtt_client.publish.return_value = SimpleNamespace(rc=0)
    bridge._mqtt_client = mqtt_client

    published = bridge._publish_edgex_command(
        "edgex/commands/cctv/jetson-01/speaker",
        {"event_id": "event-1", "action": "play"},
    )

    assert published is True
    mqtt_client.publish.assert_called_once()
    topic, body = mqtt_client.publish.call_args.args[:2]
    assert topic == "edgex/commands/cctv/jetson-01/speaker"
    assert json.loads(body)["event_id"] == "event-1"


def test_action_bridge_does_not_fail_when_shadow_broker_is_unavailable(tmp_path):
    bridge = ActionBridge(db_path=str(tmp_path / "action.db"))
    bridge._mqtt_client = None

    assert bridge._publish_edgex_command("edgex/commands/cctv/jetson-01/siren", {}) is False


def test_shadow_publisher_is_opt_in_and_injected_into_executor(tmp_path):
    disabled = ActionBridge(db_path=str(tmp_path / "disabled.db"))
    enabled = ActionBridge(
        db_path=str(tmp_path / "enabled.db"),
        edgex_shadow_enabled=True,
        edgex_jetson_id="jetson-test",
    )

    assert disabled._executor._publish_edgex_command is None
    assert enabled._executor._publish_edgex_command is not None
    assert enabled._executor._edgex_jetson_id == "jetson-test"


def test_registry_resolves_shadow_target_device_ids(tmp_path):
    bridge = ActionBridge(
        db_path=str(tmp_path / "registry.db"),
        edgex_shadow_enabled=True,
        edgex_device_registry_path="config/output_devices.json",
    )

    assert bridge._resolve_edgex_device_ids("speaker", "cam-01") == ["cctv-speaker-01"]
    assert bridge._resolve_edgex_device_ids("speaker", "cam-99") == []


def test_registry_adds_device_id_to_shadow_topic_and_request(tmp_path):
    bridge = ActionBridge(
        db_path=str(tmp_path / "fanout.db"),
        edgex_shadow_enabled=True,
        edgex_device_registry_path="config/output_devices.json",
    )
    mqtt_client = MagicMock()
    mqtt_client.publish.return_value = SimpleNamespace(rc=0)
    bridge._mqtt_client = mqtt_client

    bridge._executor._publish_shadow_command(
        event_id="event-1",
        device="speaker",
        action="play",
        payload={"text": "대상 장치"},
        command_id="event-1:speaker",
        camera_id="cam-01",
    )

    topic, body = mqtt_client.publish.call_args.args[:2]
    assert topic.endswith("/speaker/cctv-speaker-01")
    assert json.loads(body)["device_id"] == "cctv-speaker-01"
