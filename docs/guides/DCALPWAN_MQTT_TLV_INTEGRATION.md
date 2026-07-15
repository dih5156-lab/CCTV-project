# dcaLPWAN MQTT · TLV 연동

## 결론

TLV 센서는 네트워크 제어기의 `{app_eui}/{dev_eui}/up` 토픽으로 수신한다. JSON의
`payload`는 Base64 문자열이며, 파서가 Base64 디코딩 후 LwM2M TLV를 해석한다.
파싱 결과는 기존 내부 토픽 `aiot/sensors/{dev_eui}/{table_name}`으로 발행한다.

## 수신 예시

```text
topic: 0000AAAA0000AAAA/0D0D33330D0D3333/up
payload: {"message_id":"messageID_uplink","f_port":1,"payload":"SGkhCg==","is_confirmed":false,"is_ack":false,"f_cnt_up":21,"rx_metadata":[{"channel":3,"time":1562746105470}]}
```

필수값은 토픽의 `app_eui`, `dev_eui`와 JSON의 `payload`다. `rx_metadata`가 없으면
채널·주파수·수신 시각은 0으로 처리한다. `payload`는 표준 Base64여야 한다.

## 내부 발행 예시

```text
topic: aiot/sensors/0D0D33330D0D3333/t34957
payload: {
  "dev_eui":"0D0D33330D0D3333",
  "app_eui":"0000AAAA0000AAAA",
  "device_id":"0D0D33330D0D3333",
  "table":"t34957",
  "data":{"temperature_c":25.1,"angle_x_deg":1.2,"angle_y_deg":0.4,"event_code":false},
  "received_at":1562746105470,
  "uplink":{"message_id":"messageID_uplink","f_port":1,"f_cnt_up":21,"is_confirmed":false,"is_ack":false,"radio":{"gateway_id":"0A1B2C3D4E5F6789","data_rate":"SF7BW125","channel":3,"frequency":0,"rssi":-56,"snr":6.8}}
}
```

게이트웨이가 여러 개면 DB 처리 비용과 내부 MQTT 크기를 줄이기 위해 첫 번째 수신
메타데이터의 무선 품질만 내부 이벤트에 포함한다.

`join`, `downlink_event`, `error`, `$$server`는 센서 TLV 입력이 아니므로 파서가
구독하거나 재발행하지 않는다. 특히 하향 데이터는 서버의 REST 인터페이스로 등록하고,
`downlink_event`는 그 전송 결과를 읽는 용도다.

## 확인 명령

```bash
mosquitto_sub -h <서버 MQTT 주소> -p 1883 -t '+/+/up' -v
mosquitto_sub -h <내부 MQTT 주소> -p 1883 -t 'aiot/sensors/#' -v
```
