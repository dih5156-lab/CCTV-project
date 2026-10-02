"""Action Layer 합성 낙상 추적 도구 테스트."""

from scripts.ops.run_action_layer_fall_trace import (
    build_fall_payload,
    main,
    trace_fall_event,
)


def test_build_fall_payload_marks_synthetic_source():
    """합성 이벤트가 실제 영상 이벤트와 구분되는지 확인한다."""
    payload = build_fall_payload(
        camera_id="cam-01",
        event_id="fall-1",
        request_id="request-1",
    )

    assert payload["type"] == "fall_detected"
    assert payload["event_id"] == "fall-1"
    assert payload["request_id"] == "request-1"
    assert payload["metadata"] == {"source": "synthetic_uat", "test": True}


def test_trace_defaults_to_preview_without_http_request():
    """기본 추적은 장치 명령을 보내지 않고 미리보기로 끝나는지 확인한다."""
    result = trace_fall_event(
        base_url="http://127.0.0.1:1",
        token="unused",
        camera_id="cam-01",
        event_id="fall-preview-1",
        request_id="fall-preview-1:input",
        send=False,
        wait_seconds=0,
    )

    assert result["status"] == "preview"
    assert result["sent"] is False
    assert result["event_id"] == "fall-preview-1"


def test_send_requires_physical_control_confirmation(monkeypatch):
    """실제 전송 전에 물리 장치 제어 승인 옵션을 요구하는지 확인한다."""
    monkeypatch.setattr(
        "sys.argv",
        ["run_action_layer_fall_trace.py", "--send", "--token", "test-token"],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("물리 장치 제어 승인 없이 전송되면 안 된다")
