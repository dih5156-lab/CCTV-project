from types import SimpleNamespace
from unittest.mock import Mock

import paho.mqtt.client as mqtt
import pytest

from src.protocols import _mqtt_factory
from src.protocols.mqtt_subscriber import MqttTopicSubscriber


def subscriber(handler=None, **kwargs):
    return MqttTopicSubscriber(
        topics=("input/test",),
        client_id="test-durable-v1",
        durable=True,
        message_handler=handler or Mock(),
        **kwargs,
    )


def message(qos=1):
    return SimpleNamespace(topic="input/test", payload=b"raw", mid=17, qos=qos)


def test_durable_factory_requires_stable_id_and_paho2(monkeypatch):
    with pytest.raises(ValueError, match="client_id"):
        _mqtt_factory.create_mqtt_client("test", durable=True)
    monkeypatch.setattr(_mqtt_factory, "_PAHO_V2", False)
    with pytest.raises(RuntimeError, match="Paho"):
        _mqtt_factory.create_mqtt_client("test", client_id="stable", durable=True)


def test_durable_factory_never_falls_back_to_auto_ack(monkeypatch):
    constructor = Mock(side_effect=TypeError("unsupported manual ack"))
    monkeypatch.setattr(_mqtt_factory.mqtt, "Client", constructor)
    with pytest.raises(TypeError, match="unsupported"):
        _mqtt_factory.create_mqtt_client("test", client_id="stable", durable=True)
    constructor.assert_called_once()


def test_durable_factory_options(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(_mqtt_factory.mqtt, "Client", constructor)
    _mqtt_factory.create_mqtt_client("test", client_id="stable", durable=True)
    assert constructor.call_args.kwargs["clean_session"] is False
    assert constructor.call_args.kwargs["manual_ack"] is True
    assert constructor.call_args.kwargs["client_id"] == "stable"


def test_durable_subscriber_rejects_missing_id():
    with pytest.raises(ValueError, match="client_id"):
        MqttTopicSubscriber(message_handler=Mock(), durable=True)


def test_ack_occurs_only_after_handler_returns():
    order = []
    instance = subscriber(lambda topic, raw: order.append(("commit", raw)))
    client = Mock()
    client.ack.side_effect = lambda mid, qos: order.append(("ack", mid, qos)) or 0
    instance._on_message(client, None, message())
    assert order == [("commit", b"raw"), ("ack", 17, 1)]
    client.disconnect.assert_not_called()


def test_failure_skips_ack_disconnects_and_requests_reconnect():
    instance = subscriber(Mock(side_effect=OSError("disk full")))
    client = Mock()
    instance._on_message(client, None, message())
    client.ack.assert_not_called()
    client.disconnect.assert_called_once()
    assert instance.is_connected is False
    assert instance._restart_requested is True


def test_ack_error_requests_reconnect_after_saved_receipt():
    handler = Mock()
    instance = subscriber(handler)
    client = Mock()
    client.ack.return_value = mqtt.MQTT_ERR_NO_CONN
    instance._on_message(client, None, message())
    handler.assert_called_once()
    client.disconnect.assert_called_once()
    assert instance._restart_requested is True


def test_qos0_is_saved_without_claiming_ack(caplog):
    handler = Mock()
    instance = subscriber(handler)
    client = Mock()
    instance._on_message(client, None, message(qos=0))
    handler.assert_called_once()
    client.ack.assert_not_called()
    assert "QoS 0" in caplog.text


def test_subscription_ready_only_after_qos1_suback():
    instance = subscriber()
    client = Mock()
    client.subscribe.return_value = (0, 10)
    instance._on_connect(client, None, {"session present": True}, 0)
    client.subscribe.assert_called_once_with("input/test", qos=1)
    assert instance.is_connected is False
    assert instance.session_present is True
    instance._on_subscribe(client, None, 10, [1])
    assert instance.is_connected is True


@pytest.mark.parametrize("granted", [[0], [128]])
def test_rejected_or_downgraded_subscription_is_not_ready(granted):
    instance = subscriber()
    client = Mock()
    client.subscribe.return_value = (0, 10)
    instance._on_connect(client, None, {}, 0)
    instance._on_subscribe(client, None, 10, granted)
    assert instance.is_connected is False
    assert instance._restart_requested is True
    client.disconnect.assert_called_once()


def test_reconnect_waits_and_joins_old_loop_outside_callback(monkeypatch):
    instance = subscriber(Mock(side_effect=OSError("full")))
    client = Mock()
    instance._client = client
    instance._loop_running = True
    monkeypatch.setattr("src.protocols.mqtt_subscriber.time.monotonic", lambda: 100)
    instance._on_message(client, None, message())
    client.loop_stop.assert_not_called()
    instance.connect = Mock(return_value=True)
    assert instance.reconnect_if_needed() is False
    instance.connect.assert_not_called()
    monkeypatch.setattr("src.protocols.mqtt_subscriber.time.monotonic", lambda: 200)
    assert instance.reconnect_if_needed() is True
    client.loop_stop.assert_called_once()
    assert instance._client is None
    assert instance._loop_running is False
    instance.connect.assert_called_once()


def test_legacy_subscriber_keeps_qos0_and_auto_ack_behavior():
    handler = Mock(side_effect=OSError("legacy"))
    instance = MqttTopicSubscriber(topics=("input/test",), message_handler=handler)
    client = Mock()
    instance._on_connect(client, None, {}, 0)
    client.subscribe.assert_called_once_with("input/test", qos=0)
    instance._on_message(client, None, message(qos=0))
    client.ack.assert_not_called()
    client.disconnect.assert_not_called()
