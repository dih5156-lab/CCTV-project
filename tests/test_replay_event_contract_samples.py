from scripts.smoke.replay_event_contract_samples import replay_samples


def test_replay_samples_defaults_to_safe_dry_run():
    report = replay_samples()

    assert report["valid"] is True
    assert report["total"] == 13
    assert report["published"] == 0


def test_replay_samples_does_not_publish_critical_events_by_default(monkeypatch):
    class UnexpectedClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MQTT client must not be created in dry-run mode")

    monkeypatch.setitem(__import__("sys").modules, "paho.mqtt.client", UnexpectedClient)
    report = replay_samples(publish=False)

    assert report["valid"] is True
    assert report["published"] == 0
