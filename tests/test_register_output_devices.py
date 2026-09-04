from edgex.register_output_devices import OUTPUT_DEVICES, build_device_payload


def test_output_device_specs_use_dedicated_services_and_http_ports():
    assert [spec["service_name"] for spec in OUTPUT_DEVICES] == [
        "cctv-device-speaker",
        "cctv-device-siren",
        "cctv-device-signboard",
    ]
    assert [spec["protocols"]["http"]["port"] for spec in OUTPUT_DEVICES] == [
        "59991",
        "59992",
        "59993",
    ]


def test_build_device_payload_links_profile_and_service():
    payload = build_device_payload(OUTPUT_DEVICES[1])[0]["device"]

    assert payload["name"] == "cctv-siren-01"
    assert payload["serviceName"] == "cctv-device-siren"
    assert payload["profileName"] == "cctv-siren"
