from runners.run_dabit_device_service import resolve_device_id
from src.edgex.dabit_device_service import DabitCommandResult


def test_dabit_service_contract_result_shape():
    result = DabitCommandResult("cmd-1", "signboard-01", "acknowledged")
    assert result.__dict__ == {
        "command_id": "cmd-1",
        "device_id": "signboard-01",
        "status": "acknowledged",
        "error_code": None,
    }


def test_default_device_id_matches_edgex_metadata_name():
    assert resolve_device_id({}) == "cctv-signboard-01"


def test_explicit_device_id_is_preserved():
    assert resolve_device_id({"SIGNBOARD_DEVICE_ID": "signboard-west"}) == "signboard-west"
