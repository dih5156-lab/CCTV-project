from scripts.health.check_internal_port_exposure import check_compose


def test_missing_host_ip_is_reported_for_internal_services() -> None:
    result = check_compose(
        """
services:
  cctv-alert-api:
    ports:
      - target: 8000
  cctv-action-layer:
    ports:
      - target: 8080
        host_ip: 127.0.0.1
"""
    )

    assert result["passed"] is False
    assert result["findings"] == [
        {"service": "cctv-alert-api", "port_index": "0", "exposure": "missing_host_ip"}
    ]


def test_localhost_and_configurable_bindings_pass() -> None:
    result = check_compose(
        """
services:
  cctv-alert-api:
    ports:
      - target: 8000
        host_ip: 127.0.0.1
  cctv-ai-engine:
    ports:
      - target: 8769
        host_ip: ${AI_BIND_HOST:-127.0.0.1}
"""
    )

    assert result["passed"] is True
    assert result["findings"] == []


def test_string_external_binding_is_reported() -> None:
    result = check_compose(
        """
services:
  cctv-action-layer:
    ports:
      - 0.0.0.0:8080:8080
"""
    )

    assert result["passed"] is False
    assert result["findings"][0]["exposure"] == "missing_host_ip"
