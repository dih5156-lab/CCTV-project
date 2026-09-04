# 전광판(Dabit) API 인수인계 문서

## 1. 문서 목적

이 문서는 Dabit Metrix 전광판의 TCP 제어 방식과 CCTV 프로젝트의 HTTP/EdgeX Device Service 경계를 설명합니다.

## 2. 구현 위치

| 항목 | 경로 |
|---|---|
| TCP 전광판 클라이언트 | `src/devices/signboard.py` |
| EdgeX 변환 서비스 | `src/edgex/dabit_device_service.py` |
| HTTP 경계 프로세스 | `runners/run_dabit_device_service.py` |
| EdgeX 등록 | `edgex/register_signboard_device.py` |
| Device Profile | `edgex/device-profiles/cctv-signboard-dabit-profile.yaml` |

## 3. 통신 구조

```text
Action Layer → SignboardDevice → Dabit TCP socket → 전광판
```

EdgeX 검증 경로:

```text
EdgeX Command → cctv-device-dabit:59990 → DabitDeviceService → Dabit TCP
```

현재 Action Layer 운영 기본 경로는 `tcp`입니다. EdgeX 경로는 Device Service 배포와 등록이 완료된 경우에만 사용합니다.

## 4. 환경변수

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `SIGNBOARD_CONTROL_BACKEND` | `tcp` | 전광판 제어 경로 |
| `SIGNBOARD_HOST` | 빈 값 | 전광판 IP. 비어 있으면 비활성화 |
| `SIGNBOARD_PORT` | `5000` | Dabit TCP 포트 |
| `SIGNBOARD_DEVICE_ID` | `cctv-signboard-01` | EdgeX 장치 ID |
| `SIGNBOARD_BRIGHTNESS` | `10` | 밝기 |
| `SIGNBOARD_TEXT_COLOR` | `7` | 기본 글자 색상 |
| `SIGNBOARD_BACK_COLOR` | `0` | 배경 색상 |
| `SIGNBOARD_TEXT_SIZE` | `2` | 글자 크기 1~4 |
| `SIGNBOARD_TEXT_SPEED` | `10` | 표시 속도 1~99 |
| `SIGNBOARD_IDLE_REFRESH_INTERVAL` | `10` | 유휴 시각 갱신 주기(초) |

## 5. Dabit TCP 프로토콜

- 명령마다 새 TCP 소켓을 연결하는 stateless 방식
- EUC-KR 인코딩
- 버퍼 형식: `![00<payload>!]`
- 전송 실패 시 최대 3회 재시도
- 기본 socket timeout 3초

### 내부 명령 코드

| 기능 | 코드/형식 |
|---|---|
| 밝기 | `50<amount>` |
| 기본 화면 | `621` |
| 전원 켜기 | `211` |
| 전원 끄기 | `210` |
| 제목 위치 | `/P0000/Y0004` |
| 본문 위치 | `/P0001/Y0408` |

### 색상 코드

| 코드 | 색상 |
|---:|---|
| 0 | 검정 |
| 1 | 빨강 |
| 2 | 녹색 |
| 3 | 노랑 |
| 4 | 파랑 |
| 5 | 자주 |
| 6 | 하늘 |
| 7 | 흰색 |

한글/전각 문자는 표시 너비를 2칸으로 계산하여 가운데 정렬합니다. 여러 줄은 줄바꿈으로 구분합니다.

## 5-1. Action Layer 이벤트 Payload

MQTT 토픽은 `cctv/ai/events/{camera_id}/{event_type}`입니다. 전광판은 `display_message`를 우선 표시합니다.

```json
{
  "event_id": "evt-20260903-0001",
  "camera_id": "camera_1",
  "type": "fall_detected",
  "severity": "critical",
  "display_message": "낙상 사고가 감지되었습니다.",
  "metadata": {"fall_score": 5.2}
}
```

`display_message`가 없으면 `message`, 그 다음 이벤트 타입별 기본 문구를 사용합니다. `type`은 `class_name`과 색상 매핑에 사용될 수 있습니다. Action Layer는 `{event_id}:signboard` command ID를 만들고 TCP 또는 Dabit Device Service로 `display`를 전달합니다. 전체 계약은 [디바이스 이벤트 Payload 계약](EVENT_PAYLOADS.md)을 참고합니다.

## 6. 프로젝트 내부 API

### 이벤트 표시

```python
signboard.display(
    text="낙상 사고가 감지되었습니다.",
    title="경고!",
    class_name="fall_detected",
    text_color=1,
    back_color=0,
    text_size=2,
    text_speed=10,
)
```

| 매개변수 | 설명 |
|---|---|
| `text` | 본문. 여러 줄 가능 |
| `title` | 상단 제목. 기본 `CCTV 경보` |
| `class_name` | 이벤트/탐지 클래스. 색상 자동 매핑 |
| `text_color` | 0~7 |
| `back_color` | 0~7 |
| `text_size` | 1~4 |
| `text_speed` | 1~99 |

같은 제목과 클래스 조합은 `display_time` 동안 재전송하지 않습니다. 이벤트가 없으면 idle worker가 현재 시각을 표시합니다.

### 화면/전원 제어

```python
signboard.clear()
signboard.power_on()
signboard.power_off()
signboard.stop_idle()
```

## 7. Dabit Device Service HTTP API

기본 주소는 `http://<device-service-host>:59990`입니다.

### 상태 확인

```http
GET /health
```

```json
{"service":"cctv-device-dabit","status":"up"}
```

### 직접 명령

```http
POST /command
Content-Type: application/json
```

표시:

```json
{
  "command_id": "manual-001",
  "command": "display",
  "parameters": {
    "display_text": "안전사고 주의",
    "title": "CCTV 알림",
    "display_color": 1,
    "back_color": 0,
    "text_size": 2,
    "text_speed": 10
  }
}
```

기본 화면:

```json
{"command_id":"manual-002","command":"clear","parameters":{}}
```

전원:

```json
{"command_id":"manual-003","command":"power","parameters":{"power":true}}
```

허용 명령은 `display`, `clear`, `power`입니다.

```json
{
  "command_id": "manual-001",
  "device_id": "cctv-signboard-01",
  "status": "acknowledged",
  "error_code": null
}
```

실패 시 `unsupported_command`, `device_unreachable`, `device_error`가 사용될 수 있습니다.

### EdgeX 호환 경로

```http
PUT /api/v3/device/name/{device_name}/{command}
Content-Type: application/json
X-Command-Id: <command-id>
```

지원 command는 `display`, `clear`, `power`입니다. resource 타입은 Device Profile을 기준으로 합니다.

## 8. EdgeX 등록

```bash
python edgex/register_signboard_device.py \
  --metadata-url http://127.0.0.1:59881 \
  --service-url http://cctv-device-dabit:59990 \
  --device-name cctv-signboard-01
```

등록 전 스크립트 안의 장치 TCP host가 실제 전광판 주소와 일치하는지 확인합니다.

## 9. 장애 대응

| 증상 | 확인 방법 |
|---|---|
| 비활성화 | `SIGNBOARD_HOST` 확인 |
| TCP 연결 실패 | IP/포트와 네트워크 확인 |
| 글자 깨짐 | EUC-KR 지원 여부 확인 |
| 문구 반복 안 됨 | 10초 cooldown 확인 |
| 시각 표시 안 됨 | idle worker, 갱신 주기, TCP 상태 확인 |
| EdgeX 명령 실패 | Device Service health, profile/장치 등록 확인 |
