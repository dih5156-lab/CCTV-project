# Action Layer (speaker-bridge)

## 역할

Action Layer는 Rule Engine(Kuiper) 출력 이벤트를 실제 조치로 연결합니다.

- 알람 재생 (외부 스피커 코드 연동)
- DB 저장 (SQLite)
- 외부 API 호출

## 입력 토픽

기본 구독 토픽:

- `cctv/rules/intrusion/filtered`
- `cctv/rules/intrusion/persisted`
- `cctv/rules/intrusion/critical`

## 액션 동작

1. 이벤트 수신
2. DB 저장 (`action_events` 테이블)
3. 외부 API 호출 (`--external-api-url` 설정 시)
4. 알람 재생 (`--alarm-topics`에 해당 시)

## 스피커 연동 방식

둘 중 하나를 선택해서 사용합니다.

1) Speaker-edgex 코드 직접 연동 (권장)
- `--speaker-code-root /opt/Speaker-edgex`
- `--speaker-host 192.168.88.92 --speaker-port 5000 --speaker-user admin --speaker-password ...`
- 내부적으로 `control.Speaker.start_broadcast(SpeakerOption)` 호출

2) Webhook 방식
- `--speaker-webhook-url http://localhost:9100/play`
- Action Layer가 `POST`로 알람 요청 전송

3) Command 방식
- `--speaker-command "python speaker.py --camera {camera_id}"`
- 이벤트 수신 시 명령 실행

## 실행

```bash
python runners/run_action_bridge.py --mqtt-broker localhost --mqtt-port 1883 --db-path data/runtime/action_events.db --external-api-url http://localhost:8000/api/alerts --speaker-webhook-url http://localhost:9100/play --alarm-cooldown 3
```

Speaker-edgex 직접 연동 예시:

```bash
python runners/run_action_bridge.py \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --db-path data/runtime/action_events.db \
  --external-api-url http://localhost:8000/api/alerts \
  --speaker-code-root "${SPEAKER_CODE_ROOT:-/opt/Speaker-edgex}" \
  --speaker-host "${SPEAKER_HOST}" \
  --speaker-port "${SPEAKER_PORT:-5000}" \
  --speaker-user "${SPEAKER_USER:-admin}" \
  --speaker-password "${SPEAKER_PASSWORD}" \
  --speaker-volume 1 \
  --alarm-cooldown 3
```

## DB 스키마

SQLite 파일(`--db-path`)에 아래 테이블을 사용합니다.

- `action_events(id, received_at, topic, camera_id, event_type, confidence, severity, payload_json)`
