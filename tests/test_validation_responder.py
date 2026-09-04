from src.edgex.validation_responder import build_validation_response


def test_validation_response_supports_edgex_request_id_and_topic():
    response = build_validation_response(
        {"requestID": "request-1", "correlationID": "correlation-1"},
        "edgex/cctv-device-siren/validate/device",
    )

    assert response["requestID"] == "request-1"
    assert response["correlationID"] == "correlation-1"
    assert response["receivedTopic"].endswith("validate/device")
    assert response["errorCode"] == 0
