from scripts.ops.check_edgex_device_service_contracts import run_contract_checks


def test_all_output_device_contracts_pass_in_dry_run():
    results = run_contract_checks()

    assert [result["device"] for result in results] == ["speaker", "siren", "signboard"]
    assert all(result["status_code"] == 200 for result in results)
    assert all(result["result_status"] == "simulated" for result in results)
