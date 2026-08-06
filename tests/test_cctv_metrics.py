from src.services.cctv_metrics import REGISTRY, events_handled_by_type


def test_event_type_metric_is_registered_with_bounded_labels():
    events_handled_by_type.labels(event_type="helmet", mode="auto").inc()

    metric = REGISTRY.get_sample_value(
        "cctv_events_handled_by_type_total",
        {"event_type": "helmet", "mode": "auto"},
    )
    assert metric is not None
    assert metric >= 1
