from dataclasses import dataclass

from src.services.action_policy import ActionPolicy


@dataclass
class FakeAlarm:
    should_alarm_result: bool = True
    slot_result: bool = True
    demo_result: bool = False

    def should_alarm(self, topic, payload):
        return self.should_alarm_result

    def try_acquire_slot(self, camera_id, event_type, **kwargs):
        return self.slot_result

    def is_demo_event(self, payload):
        return self.demo_result


def test_action_policy_rejects_event_when_alarm_rule_is_false():
    policy = ActionPolicy(FakeAlarm(should_alarm_result=False))

    assert policy.should_execute("events/detected", {"camera_id": "camera-1"}) is False


def test_action_policy_acquires_event_slot_with_normalized_context():
    alarm = FakeAlarm()
    policy = ActionPolicy(alarm)
    payload = {
        "camera_id": "camera-1",
        "event_type": "Fall",
        "object_id": "person-1",
        "event_id": "event-1",
    }

    assert policy.should_execute("events/detected", payload) is True

