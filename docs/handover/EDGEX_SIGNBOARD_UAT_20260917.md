# 전광판 EdgeX 운영 경로 UAT (2026-09-17)

## 결론

- 판정: PASS (로그 및 결과 DB 기준)
- 시험 경로: Action Layer → EdgeX MQTT → 전광판 Device Service → Dabit 전광판
- 시험 후 운영 설정: `speaker,signboard,siren` 전체 EdgeX 라우팅 복귀
- 제한: 현장 사진 또는 육안 확인 기록은 확보하지 않았으며 물리 출력 판정은 장치 호출 로그 기준이다.

## 사전 조건

- 전광판: `192.168.88.91:5000` TCP 연결 성공
- Action Layer: healthy, MQTT connected
- UAT 중 장치 제한: `EDGEX_ALLOWED_DEVICES=signboard`

## 시험 이벤트

```text
event_id=uat-edgex-signboard-20260917-0914
request_id=uat-edgex-signboard-20260917-0914:input
event_type=fall_detected
camera_id=cam-01
```

## 확인 결과

```text
command_id=uat-edgex-signboard-20260917-0914:signboard
device_id=cctv-signboard-01
status=acknowledged
requested_at=2026-09-17T09:14:39.532806+09:00
result_received_at=2026-09-17T09:14:40.566147+09:00
request_to_result_latency=약 1.03초
```

전광판 Device Service 로그:

```text
[Signboard] 표시: title='경고!' text='[cam-01] 낙상 감지 - 즉시 확인' color=7 back=0
[Signboard] 방송 시작 완료
```

## 시험 후 운영 상태

```text
EDGEX_COMMAND_MODE=edgex
EDGEX_ALLOWED_DEVICES=speaker,signboard,siren
EDGEX_DEVICE_REGISTRY_PATH=/app/config/output_devices.json
Action Layer health=up
MQTT=connected
```

스피커는 마지막 연결 점검에서 timeout이었고 사이렌은 접속정보가 아직
미설정이다. 두 장치도 EdgeX 명령 발행 대상이지만 실제 출력 성공은 별도 UAT가
필요하다.
