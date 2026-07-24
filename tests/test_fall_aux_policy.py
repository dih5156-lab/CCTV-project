from src.core._fall_aux_policy import (
    aux_result_confirms_fall,
    should_confirm_fall_with_aux,
)
from src.core.events import DetectionEvent, EventType


def _fall_event(score: float) -> DetectionEvent:
    return DetectionEvent(
        EventType.FALL_DETECTED,
        0,
        0,
        10,
        20,
        0.9,
        1.0,
        metadata={"fall_score": score},
    )


def test_should_confirm_borderline_fall_at_configured_max_score():
    assert should_confirm_fall_with_aux(
        _fall_event(4.5),
        confirm_borderline=True,
        aux_enabled=True,
        max_fall_score=4.5,
        detector_score_threshold=3.0,
    )


def test_should_not_confirm_non_fall_event_with_aux():
    event = _fall_event(2.0)
    event.event_type = EventType.PERSON

    assert not should_confirm_fall_with_aux(
        event,
        confirm_borderline=True,
        aux_enabled=True,
        max_fall_score=4.5,
        detector_score_threshold=3.0,
    )


def test_compare_model_can_veto_confirmed_high_score_fall():
    event_payload = {"metadata": {"fall_score": 5.0}}
    result = {
        "status": "ok",
        "confirmed": True,
        "compare_model": {"status": "ok", "confirmed": False},
    }

    assert not aux_result_confirms_fall(
        event_payload,
        result,
        compare_veto_enabled=True,
        compare_veto_min_fall_score=4.0,
    )


def test_compare_veto_does_not_reject_score_below_minimum():
    event_payload = {"metadata": {"fall_score": 3.5}}
    result = {
        "status": "ok",
        "confirmed": True,
        "compare_model": {"status": "ok", "confirmed": False},
    }

    assert aux_result_confirms_fall(
        event_payload,
        result,
        compare_veto_enabled=True,
        compare_veto_min_fall_score=4.0,
    )
