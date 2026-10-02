import json
import secrets
from unittest.mock import MagicMock

import pytest

from src.monitoring_contract import MonitoringContractError
from src.protocols.monitoring_http import HttpMonitoringReader
from tests.test_monitoring_delivery import prepared


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (404, {"error": {"code": "not_found"}}, "ack_mismatch"),
        (401, {}, "unauthorized"),
        (403, {}, "binding_mismatch"),
        (200, [], "ack_mismatch"),
        (200, "invalid-json", "ack_mismatch"),
    ],
)
def test_lookup_does_not_trust_unscoped_or_invalid_responses(
    tmp_path, monkeypatch, status, body, expected
):
    store, binding = prepared(tmp_path / "source.db")
    ca = tmp_path / "ca.crt"
    ca.write_text("not-used-by-mock")
    reader = HttpMonitoringReader("https://laptop:18443", secrets.token_urlsafe(32), ca)
    response = MagicMock(status_code=status)
    response.__enter__.return_value = response
    response.iter_content.return_value = [
        body.encode() if isinstance(body, str) else json.dumps(body).encode()
    ]
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = response
    monkeypatch.setattr(
        "src.protocols.monitoring_http.requests.Session", lambda: session
    )
    with pytest.raises(MonitoringContractError, match=expected):
        reader.get_delivery_status(binding, "job-1", timeout=1)
    assert session.trust_env is False
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert session.get.call_args.kwargs["verify"] == str(ca)


def test_plaintext_url_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        HttpMonitoringReader(
            "http://laptop:18443", secrets.token_urlsafe(32), tmp_path / "ca.crt"
        )


@pytest.mark.parametrize("status", [301, 302, 500, 503])
def test_redirects_and_server_errors_are_retryable(tmp_path, monkeypatch, status):
    _, binding = prepared(tmp_path / "source.db")
    ca = tmp_path / "ca.crt"
    ca.write_text("mock-ca")
    response = MagicMock(status_code=status)
    response.__enter__.return_value = response
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = response
    monkeypatch.setattr(
        "src.protocols.monitoring_http.requests.Session", lambda: session
    )
    reader = HttpMonitoringReader("https://laptop:18443", secrets.token_urlsafe(32), ca)
    with pytest.raises(ConnectionError):
        reader.get_delivery_status(binding, "job-1", timeout=1)
    response.iter_content.assert_not_called()


@pytest.mark.parametrize("case", ["missing", "ack", "oversized"])
def test_scoped_absence_ack_and_size_limit(tmp_path, monkeypatch, case):
    _, binding = prepared(tmp_path / "source.db")
    ca = tmp_path / "ca.crt"
    ca.write_text("mock-ca")
    body = (
        {"status": "not_found", "binding": binding.as_dict(), "delivery_id": "job-1"}
        if case == "missing"
        else {"delivery_id": "job-1"}
    )
    response = MagicMock(status_code=404 if case == "missing" else 200)
    response.__enter__.return_value = response
    response.iter_content.return_value = [
        b"x" * 65537 if case == "oversized" else json.dumps(body).encode()
    ]
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = response
    monkeypatch.setattr(
        "src.protocols.monitoring_http.requests.Session", lambda: session
    )
    reader = HttpMonitoringReader("https://laptop:18443", secrets.token_urlsafe(32), ca)
    if case == "oversized":
        with pytest.raises(MonitoringContractError, match="ack_mismatch"):
            reader.get_delivery_status(binding, "job-1", timeout=1)
    else:
        assert reader.get_delivery_status(binding, "job-1", timeout=1) == (
            None if case == "missing" else body
        )
