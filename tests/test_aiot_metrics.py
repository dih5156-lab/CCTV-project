from prometheus_client import CollectorRegistry, generate_latest

from src.aiot.metrics import AiotMetrics


def test_aiot_metrics_use_bounded_labels():
    registry = CollectorRegistry()
    metrics = AiotMetrics(registry=registry)
    metrics.commands_received.labels(message_type="ai_query_request").inc()
    metrics.query_duration.labels(search_mode="both", status="completed").observe(0.2)
    output = generate_latest(registry).decode()
    assert "aiot_commands_received_total" in output
    assert 'search_mode="both"' in output
    assert "request_id" not in output
