import numpy as np

from src.core.ai._falldata_aux import FallDataAuxConfig, FallDataAuxVerifier
from src.core.events import DetectionEvent, EventType


def _fall_event() -> DetectionEvent:
    return DetectionEvent(
        event_type=EventType.FALL_DETECTED,
        x=1,
        y=2,
        width=3,
        height=4,
        confidence=0.9,
        timestamp=1.0,
    )


def test_disabled_verifier_keeps_events_unchanged() -> None:
    verifier = FallDataAuxVerifier(FallDataAuxConfig(enabled=False))
    event = _fall_event()

    assert verifier.annotate_events([event]) == [event]
    verifier.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    assert verifier.annotate_events([event]) == [event]


def test_shadow_mode_keeps_event_and_adds_metadata(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="shadow", cooldown_seconds=0)
    )
    monkeypatch.setattr(
        verifier,
        "verify",
        lambda: {
            "enabled": True,
            "mode": "shadow",
            "status": "ok",
            "confirmed": False,
            "fall_probability": 0.2,
        },
    )

    [event] = verifier.annotate_events([_fall_event()])

    assert event.metadata["falldata_aux"]["status"] == "ok"
    assert event.metadata["falldata_aux"]["confirmed"] is False


def test_confirm_mode_drops_unconfirmed_fall(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="confirm", cooldown_seconds=0)
    )
    monkeypatch.setattr(
        verifier,
        "verify",
        lambda: {
            "enabled": True,
            "mode": "confirm",
            "status": "ok",
            "confirmed": False,
        },
    )

    assert verifier.annotate_events([_fall_event()]) == []


def test_cooldown_without_previous_result_is_not_confirmed(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="shadow", cooldown_seconds=60)
    )
    monkeypatch.setattr("src.core.ai._falldata_aux.time.time", lambda: 100.0)
    verifier._last_run_at = 99.0

    result = verifier.verify()

    assert result["status"] == "skipped_cooldown"
    assert result["confirmed"] is False


def test_cooldown_reuses_previous_result_but_marks_status(monkeypatch) -> None:
    verifier = FallDataAuxVerifier(
        FallDataAuxConfig(enabled=True, mode="shadow", cooldown_seconds=60)
    )
    monkeypatch.setattr("src.core.ai._falldata_aux.time.time", lambda: 100.0)
    verifier._last_run_at = 99.0
    verifier._last_result = {
        "enabled": True,
        "mode": "shadow",
        "status": "ok",
        "confirmed": True,
        "fall_probability": 0.91,
    }

    result = verifier.verify()

    assert result["status"] == "skipped_cooldown"
    assert result["previous_status"] == "ok"
    assert result["confirmed"] is True
    assert result["fall_probability"] == 0.91


def test_parse_smoke_outputs() -> None:
    output = """
    nonzero_feature_frames: 40
    prediction: [0]
    predict_proba: [[0.9185, 0.0815]]
    """

    assert FallDataAuxVerifier._parse_nonzero_frames(output) == 40
    assert FallDataAuxVerifier._parse_prediction(output) == 0
    assert FallDataAuxVerifier._parse_probability(output) == [0.9185, 0.0815]
