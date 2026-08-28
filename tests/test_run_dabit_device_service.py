from src.edgex.dabit_device_service import DabitCommandResult


def test_dabit_service_contract_result_shape():
    result = DabitCommandResult("cmd-1", "signboard-01", "acknowledged")
    assert result.__dict__ == {
        "command_id": "cmd-1",
        "device_id": "signboard-01",
        "status": "acknowledged",
        "error_code": None,
    }
