# External MQTT Ingest

외부 MQTT 데이터를 Windows에서 먼저 수신하고, 이후 같은 이벤트를 EdgeX로 확장하기 위한 1단계 MVP 문서입니다.

## 목적

- 외부 브로커에서 MQTT payload 수신
- 프로젝트 내부 표준 이벤트로 정규화
- SQLite에 원시/정규화 이벤트 저장
- 필요 시 내부 토픽 `cctv/ai/events/...` 로 재발행

## 실행

Python-only 수신만 확인:

```bash
python run_external_ingest.py \
  --mqtt-broker 192.168.88.30 \
  --mqtt-port 1883 \
  --topic "#" \
  --mqtt-client-id my-test-client \
  --mqtt-username a000000000000001
```

내부 MQTT로 재발행까지 확인:

```bash
python run_external_ingest.py \
  --mqtt-broker 192.168.88.30 \
  --mqtt-port 1883 \
  --topic "#" \
  --mqtt-username a000000000000001 \
  --republish \
  --republish-broker localhost \
  --republish-port 1883 \
  --republish-topic-prefix cctv/ai/events
```

## 환경 변수

- `EXTERNAL_MQTT_BROKER`
- `EXTERNAL_MQTT_PORT`
- `EXTERNAL_MQTT_TOPICS`
- `EXTERNAL_MQTT_CLIENT_ID_PREFIX`
- `EXTERNAL_MQTT_USERNAME`
- `EXTERNAL_MQTT_PASSWORD`
- `EXTERNAL_INGEST_DB_PATH`
- `EXTERNAL_REPUBLISH_ENABLED`

## 저장 형식

SQLite `ingest_events.db` 테이블:

- `id`
- `event_id` (중복 수신 방지용 unique index)
- `received_at`
- `topic`
- `raw_payload`
- `normalized_payload`
- `republished`

## 다음 단계

1. 실제 외부 payload 샘플 확보
2. 이미지 필드(`image_path`, `image_url`, `image_ref`) 중 하나로 표준화
3. 내부 재발행 이벤트를 EdgeX 어댑터에 연결
4. Jetson에서는 동일 코드 유지, 브로커/경로만 환경 변수로 분리
