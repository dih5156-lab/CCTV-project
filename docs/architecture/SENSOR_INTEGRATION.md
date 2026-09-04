# 센서 연동 형식과 방법

## 1. 입력 MQTT 형식

LoRa uplink 토픽은 다음 형식이다.

```text
{app_eui}/{dev_eui}/up
```

payload 예시:

```json
{
  "message_id": "messageID_uplink",
  "f_port": 1,
  "payload": "SGkhCg==",
  "f_cnt_up": 21,
  "is_confirmed": false,
  "rx_metadata": [{"channel": 3, "rssi": -56, "snr": 6.8}]
}
```

`payload`는 표준 Base64이고, 파서가 디코드한 byte buffer를 table별 TLV 규칙으로 해석한다. `join`, `downlink_event`, `error`, `$$server`는 센서 uplink가 아니므로 일반 TLV 파이프라인에 넣지 않는다.

## 2. 파싱 후 내부 형식

```text
aiot/sensors/{dev_eui}/{table_name}
```

```json
{
  "dev_eui": "0D0D33330D0D3333",
  "app_eui": "0000AAAA0000AAAA",
  "device_id": "0D0D33330D0D3333",
  "table": "t34957",
  "data": {"temperature_c": 25.1, "angle_x_deg": 1.2, "event_code": false},
  "received_at": 1562746105470,
  "uplink": {"f_port": 1, "f_cnt_up": 21, "radio": {"rssi": -56, "snr": 6.8}}
}
```

`SensorReading`은 `device_id`, `app_eui`, `dev_eui`, `table_name`, `telemetry`, `received_at`, `source`, `metadata`를 가진다. `tableName`은 telemetry 값에 중복 저장하지 않고 식별 필드로 승격한다.

## 3. 새 센서 추가 절차

1. 실제 uplink의 topic, Base64, `f_port`, table 번호, 센서 매뉴얼을 확보한다.
2. `parser-python/tlv/`의 table·TLV 타입 해석 규칙을 추가한다.
3. `parser-python/tests/`에 정상·길이 오류·알 수 없는 TLV 테스트를 추가한다.
4. `src/devices/sensor_device.py`에서 표준 telemetry 필드가 유지되는지 확인한다.
5. `src/services/sensor_rule_bridge.py`와 eKuiper rule의 sensor type 이름을 맞춘다.
6. EdgeX device profile과 등록 스크립트가 필요한 센서인지 결정한다.
7. 실제 MQTT를 구독해 원본·파싱 후·규칙 결과를 각각 확인한다.

## 4. 확인 명령

```bash
mosquitto_sub -h <MQTT 주소> -p 1883 -t '+/+/up' -v
mosquitto_sub -h <MQTT 주소> -p 1883 -t 'aiot/sensors/#' -v
mosquitto_sub -h <MQTT 주소> -p 1883 -t 'aiot/rules/sensor/#' -v
```

개발 장비가 없으면 parser 테스트 fixture로 Base64/TLV를 먼저 검증한다. 실제 운영 payload의 table 번호와 필드명은 센서 제조사·현장 설정에 따라 다를 수 있으므로 문서 예시를 그대로 장비 설정값으로 사용하지 않는다.

## 5. 자주 발생하는 문제

| 증상 | 우선 확인 |
|---|---|
| 메시지가 안 보임 | 외부 MQTT host/port/계정, topic의 EUI 순서 |
| Base64 오류 | padding 포함 표준 Base64인지 |
| 값이 누락됨 | table 번호, TLV type byte, payload 길이 |
| 규칙이 동작하지 않음 | `sensor_type`, topic, eKuiper rule 상태 |
| EdgeX에 안 쌓임 | EdgeX metadata/profile, forwarder outbox, Core Data health |

