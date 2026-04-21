from src.services.sensor_rule_bridge import (
    SensorRuleBridgeService,
    build_rule_topic,
    build_sensor_bridge_inputs,
)


def test_build_rule_topic_removes_alert_suffix():
    assert build_rule_topic("tilt_alert") == "aiot/rules/sensor/tilt"
    assert build_rule_topic("temperature_alert") == "aiot/rules/sensor/temperature"


def test_build_sensor_bridge_inputs_keeps_table_and_data():
    uplink, decoded = build_sensor_bridge_inputs(
        {
            "app_eui": "app-1",
            "dev_eui": "dev-1",
            "device_id": "factory-24",
            "table": "t34957",
            "data": {
                "temperature": 72.5,
                "angle_x": 12.0,
            },
            "received_at": 1710000000,
        }
    )

    assert uplink["device_id"] == "factory-24"
    assert decoded["tableName"] == "t34957"
    assert decoded["data"]["temperature"] == 72.5


def test_process_sensor_message_creates_temperature_alert():
    service = SensorRuleBridgeService()

    events = service.process_sensor_message(
        {
            "app_eui": "app-1",
            "dev_eui": "dev-1",
            "device_id": "factory-24",
            "table": "t34957",
            "data": {
                "temperature": 80.0,
            },
            "received_at": 1710000000,
        }
    )

    assert len(events) == 1
    assert events[0]["camera_id"] == "dev-1"
    assert events[0]["type"] == "temperature_alert"
    assert events[0]["severity"] == "critical"


def test_process_sensor_message_creates_tilt_alert():
    service = SensorRuleBridgeService()

    events = service.process_sensor_message(
        {
            "app_eui": "app-1",
            "dev_eui": "dev-1",
            "device_id": "factory-24",
            "table": "t34955",
            "data": {
                "angle_x": 50.0,
                "angle_y": 10.0,
            },
            "received_at": 1710000000,
        }
    )

    assert len(events) == 1
    assert events[0]["type"] == "tilt_alert"
    assert events[0]["severity"] == "critical"
